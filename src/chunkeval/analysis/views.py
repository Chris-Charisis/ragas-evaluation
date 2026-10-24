"""The report's tables, computed from the checked inputs (inputs.py). Each function returns a DataFrame
(or a dict of them) and writes nothing; __init__.run writes them and report.py turns them into Markdown.

Conventions: "setup" is a variant under its report name (fixed-size_no_overlap, ...; inputs.label) or
gold_passage; the retrieval
views use the primary embedder at k = retrieval.top_k unless they say otherwise; every interval is a
bootstrap over questions (stats.py), and every comparison between setups is paired.
"""
from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from chunkeval.analysis.inputs import Inputs, label, run_path
from chunkeval.analysis.stats import (TIE, detectable_difference, holm, paired_diff, paired_family, questions_needed,
                                      rng_for, summary_table, verdict)
from chunkeval.core.schema import Config
from chunkeval.evaluation.answering import GOLD_PASSAGE
from chunkeval.evaluation.retrieval_metrics import METRICS as SPAN_METRICS, evidence_intervals, overlap_len

# The span metrics shown in the Markdown tables (all of SPAN_METRICS are in the CSVs).
HEADLINE_SPAN = ["span_hit", "span_recall", "mrr"]
# The generation table's metrics, in report order. Derived here: abstention (1 = abstained) and
# faithfulness_abst1 (abstentions count as faithful: they make no unsupported claim). fc_recall_ragas
# is a diagnostic only (Ragas' recall mixes the two claim sets): CSV only.
GEN_METRICS = ["fc_recall", "factual_correctness", "fc_precision", "fc_f1", "faithfulness", "faithfulness_abst1",
               "abstention", "answer_relevancy", "context_precision", "context_recall", "rouge_l", "token_f1",
               "answer_similarity"]
GEN_DIAGNOSTIC = ["fc_recall_ragas"]
HEADLINE_GEN = ["fc_recall", "factual_correctness", "faithfulness", "abstention", "answer_relevancy",
                "context_precision", "context_recall"]
# The retrieval table's columns joined onto each generation row (at top_k, primary embedder).
JOINED_SPAN = ["doc_hit", "span_hit", "all_evidence", "span_recall", "mrr", "context_tokens"]

# Where an answer ends up (`outcome`), in display order: with the evidence shown, best to worst; then
# without it; then no usable answer. "Correct" means every gold claim is in the answer (fc_recall 1).
OUTCOMES = {
    "correct": "evidence shown: every gold claim",
    "partial": "evidence shown: some gold claims",
    "wrong": "evidence shown: no gold claim",
    "abstained_shown": "evidence shown: abstained",
    "abstained_missed": "evidence not shown: abstained",
    "elsewhere": "evidence not shown: answered with gold claims (found elsewhere, or a guess)",
    "unsupported": "evidence not shown: answered without a gold claim (hallucination risk)",
    "error": "no usable answer or score",
}


def boot(cfg: Config) -> dict[str, Any]:
    return {"n_iter": cfg.analysis.bootstrap_iterations, "ci": cfg.analysis.ci, "seed": cfg.seed}


def top_k_rows(retrieval: pd.DataFrame, cfg: Config, embedder: str | None = None, k_label: str | None = None) -> pd.DataFrame:
    """The retrieval rows of one embedder (default: the primary) at one k (default: top_k)."""
    return retrieval[(retrieval["embedder"] == (embedder or cfg.embedding.primary))
                     & (retrieval["k_label"] == (k_label or str(cfg.retrieval.top_k)))]


# --- per-question frames -----------------------------------------------------------------------------

BIN_NAMES = ("short", "medium", "long")


def bin_order(label: str) -> int:
    """Sort key of a `size_bins` label: short, medium, long (then "bin 2", ...)."""
    name = label.split(" (")[0]
    return BIN_NAMES.index(name) if name in BIN_NAMES else 9


def size_bins(values: pd.Series, names: tuple[str, ...], unit: str) -> pd.Series:
    """Equal-count bins (terciles for 3 names), labelled with their name and the range they hold,
    e.g. "short (120–410 chars)". Fewer bins when ties leave fewer distinct edges."""
    if values.nunique() < 2:
        return pd.Series([f"all ({values.min():.0f} {unit})"] * len(values), index=values.index)
    codes = pd.qcut(values.rank(method="first"), len(names), labels=False)
    labels = {}
    for code, group in values.groupby(codes):
        name = names[int(code)] if codes.nunique() == len(names) else f"bin {int(code) + 1}"
        labels[code] = f"{name} ({group.min():.0f}–{group.max():.0f} {unit})"
    return codes.map(labels)


def question_frame(inputs: Inputs) -> pd.DataFrame:
    """One row per question: its attributes and the breakdown labels (evidence quotes, evidence length
    terciles per corpus), and whether it is in the Ragas table."""
    in_ragas = set() if inputs.ragas is None else set(inputs.ragas["question_id"])
    rows = []
    for qa in inputs.questions.values():
        evidence = evidence_intervals(qa)
        rows.append({"question_id": qa.id, "corpus": qa.corpus, "doc_id": qa.doc_id, "question_type": qa.question_type,
                     "kind": qa.kind, "n_evidence": len(evidence), "evidence_chars": sum(e - s for s, e in evidence),
                     "in_ragas": qa.id in in_ragas})
    q = pd.DataFrame(rows)
    q["evidence_quotes"] = np.where(q["n_evidence"] > 1, "2+ quotes", "1 quote")
    q["evidence_length"] = pd.concat([size_bins(g["evidence_chars"], BIN_NAMES, "chars")
                                      for _, g in q.groupby("corpus", sort=False)])
    return q


def outcome(evidence_shown: bool, abstained: bool, answer_error: bool, fc_recall: float) -> str:
    """Where one answer ends up (OUTCOMES), from the retrieval flag, the abstention and fc_recall."""
    if answer_error or (not abstained and math.isnan(fc_recall)):
        return "error"
    if evidence_shown:
        if abstained:
            return "abstained_shown"
        return "correct" if fc_recall >= 1 - TIE else "partial" if fc_recall > 0 else "wrong"
    if abstained:
        return "abstained_missed"
    return "elsewhere" if fc_recall > 0 else "unsupported"


def generation_frame(inputs: Inputs, cfg: Config, questions: pd.DataFrame) -> pd.DataFrame:
    """The Ragas table with the derived columns: abstention, faithfulness_abst1, judge_useful (some
    chunk useful to the judge), the retrieval columns at top_k (NaN for the gold passage), the rank of
    the first chunk that overlaps the evidence (0: none in the top k), the outcome and the breakdown labels."""
    g = inputs.ragas.copy()
    usable = ~g["answer_error"]
    g["answered"] = usable & ~g["abstained"]
    g["abstention"] = np.where(usable, g["abstained"].astype(float), np.nan)
    g["faithfulness_abst1"] = np.where(usable & g["abstained"], 1.0, g["faithfulness"])
    g["judge_useful"] = g["cp_useful"] > 0
    top = top_k_rows(inputs.retrieval, cfg)[["question_id", "corpus", "setup", *JOINED_SPAN]]
    g = g.merge(top, on=["question_id", "corpus", "setup"], how="left")
    g["evidence_rank"] = np.where(g["mrr"] > 0, np.round(1 / g["mrr"].where(g["mrr"] > 0)), 0)
    g["outcome"] = [outcome(e, a, err, f) for e, a, err, f in
                    zip(g["evidence_shown"], g["abstained"], g["answer_error"], g["fc_recall"])]
    labels = questions.set_index("question_id")[["evidence_quotes", "evidence_length"]]
    return g.join(labels, on="question_id")


def doc_of(questions: pd.DataFrame) -> pd.Series:
    return questions.set_index("question_id")["doc_id"]


# --- retrieval ---------------------------------------------------------------------------------------

def retrieval_means(inputs: Inputs, cfg: Config) -> pd.DataFrame:
    """Mean and CI of every span metric (and the k used) per corpus, retriever, setup and k label."""
    return summary_table(inputs.retrieval, [*SPAN_METRICS, "k"], ["corpus", "embedder", "setup", "k_label"], **boot(cfg))


# The columns of a paired comparison (stats.paired_family), so a table with nothing to compare still has them.
COMPARISON_COLUMNS = ["setup", "against", "n", "dropped", "diff", "low", "high", "better", "equal", "worse", "p",
                      "doc_low", "doc_high", "docs", "p_holm", "verdict", "doc_ci_reliable", "few_docs"]


def setup_comparisons(df: pd.DataFrame, metrics: list[str], against: str, others: list[str], docs: pd.Series,
                      cfg: Config, keys: list[str]) -> pd.DataFrame:
    """Paired comparisons of `others` with `against` for each metric, one Holm family per (keys, metric):
    df has one row per (question_id, setup, keys...)."""
    rows = []
    for key, group in df.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            wide = group.pivot(index="question_id", columns="setup", values=m)
            for r in paired_family(wide, against, others, docs, key=(*key, m), **boot(cfg)):
                rows.append({**dict(zip(keys, key)), "metric": m, **r})
    return pd.DataFrame(rows, columns=[*keys, "metric", *COMPARISON_COLUMNS])


def retrieval_vs_baseline(inputs: Inputs, cfg: Config, docs: pd.Series) -> pd.DataFrame:
    """Each setup − the baseline on every span metric, at top_k and at the token budget (primary embedder)."""
    base = label(cfg.analysis.baseline)
    df = inputs.retrieval[(inputs.retrieval["embedder"] == cfg.embedding.primary)
                          & inputs.retrieval["k_label"].isin([str(cfg.retrieval.top_k), "budget"])]
    return setup_comparisons(df, SPAN_METRICS, base, [s for s in inputs.setups if s != base], docs, cfg,
                             ["corpus", "k_label"])


def discriminating(inputs: Inputs, cfg: Config, questions: pd.DataFrame, g: pd.DataFrame | None) -> pd.DataFrame:
    """Per question and metric, its score in each setup, the spread and a class: the setups agree
    ("all <value>") or differ. Differences between chunkers come only from the questions that differ.
    Metrics: span_hit and span_recall at top_k (every question), fc_recall and abstention (Ragas sample)."""
    frames = [(m, top_k_rows(inputs.retrieval, cfg)) for m in ("span_hit", "span_recall")]
    if g is not None:
        frames += [(m, g[g["setup"] != GOLD_PASSAGE]) for m in ("fc_recall", "abstention")]
    attrs = questions.set_index("question_id")[["doc_id", "question_type"]]
    out = []
    for m, df in frames:
        wide = df.pivot_table(index=["corpus", "question_id"], columns="setup", values=m, aggfunc="first")
        wide = wide[[s for s in inputs.setups if s in wide]]
        spread = wide.max(axis=1) - wide.min(axis=1)
        cls = np.where(spread > TIE, "setups differ", "all " + wide.min(axis=1).round(2).astype(str))
        block = wide.reset_index().assign(metric=m, spread=spread.to_numpy(), **{"class": cls})
        out.append(block.join(attrs, on="question_id"))
    cols = ["corpus", "question_id", "doc_id", "question_type", "metric", "class", "spread", *inputs.setups]
    return pd.concat(out, ignore_index=True)[cols]


def per_document(inputs: Inputs, cfg: Config, questions: pd.DataFrame, g: pd.DataFrame | None) -> pd.DataFrame:
    """Per document: its questions, its share of the corpus's chunks (baseline chunker), and how its
    questions fare, averaged over the setups: span metrics at top_k; fc_recall and abstention over the
    retrieved setups, and fc_recall from the gold passage."""
    base = label(cfg.analysis.baseline)
    rows = []
    for corpus in inputs.corpora:
        counts = inputs.chunks_per_doc[(corpus, base)]
        total = sum(counts.values())
        qs = questions[questions["corpus"] == corpus]
        top = top_k_rows(inputs.retrieval, cfg)
        top = top[top["corpus"] == corpus]
        for doc_id, dq in qs.groupby("doc_id"):
            r = top[top["question_id"].isin(dq["question_id"])]
            row = {"corpus": corpus, "doc_id": doc_id, "questions": len(dq), "chunks": counts.get(doc_id, 0),
                   "chunk_share": counts.get(doc_id, 0) / total if total else math.nan,
                   "doc_hit": r["doc_hit"].mean(), "span_hit": r["span_hit"].mean(), "span_recall": r["span_recall"].mean()}
            if g is not None:
                gd = g[(g["corpus"] == corpus) & g["question_id"].isin(dq["question_id"])]
                setups, gold = gd[gd["setup"] != GOLD_PASSAGE], gd[gd["setup"] == GOLD_PASSAGE]
                row.update(ragas_questions=gold["question_id"].nunique(), fc_recall=setups["fc_recall"].mean(),
                           abstention=setups["abstention"].mean(), gold_fc_recall=gold["fc_recall"].mean())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["corpus", "questions", "doc_id"], ascending=[True, False, True])


def distractors(inputs: Inputs, cfg: Config) -> pd.DataFrame:
    """Which documents' chunks fill the top-k slots of other documents' questions: per (corpus, setup,
    question's document, distractor document), the slots taken and their share of that document's
    question slots. Near-duplicate documents and very long documents show up here."""
    k = cfg.retrieval.top_k
    rows = []
    for (corpus, setup), hits in inputs.hits.items():
        slots, taken = Counter(), Counter()
        for qid, ranked in hits.items():
            doc = inputs.questions[qid].doc_id
            slots[doc] += min(k, len(ranked))
            taken.update((doc, h.doc_id) for h in ranked[:k] if h.doc_id != doc)
        rows += [{"corpus": corpus, "setup": setup, "question_doc": qd, "distractor_doc": dd, "slots": n,
                  "share_of_slots": n / slots[qd]} for (qd, dd), n in taken.items()]
    cols = ["corpus", "setup", "question_doc", "distractor_doc", "slots", "share_of_slots"]
    return pd.DataFrame(rows, columns=cols).sort_values(["corpus", "setup", "slots"], ascending=[True, True, False])


def embedder_robustness(inputs: Inputs, cfg: Config, docs: pd.Series,
                        vs_baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do the chunker conclusions hold under the secondary embedder(s)? On the span metrics at top_k:
      - per setup: both embedders' means and ranks, and the paired difference secondary − primary
        on the same questions (one Holm family per corpus and metric): what changing the embedder buys;
      - per corpus and metric ("levers"): Kendall's tau between the two rankings (with 6 setups, one
        swapped pair of near-tied setups is not a contradiction: read the rankings too), the best
        chunker's gain over the baseline under the primary embedder against the mean embedder gain
        (which is the bigger lever?), and the spread between the best and worst setup under each
        embedder (does a better embedder shrink the differences between chunkers?).
    """
    per_setup, levers = [], []
    primary = cfg.embedding.primary
    for secondary in [e for e in cfg.embedding.secondary if (inputs.retrieval["embedder"] == e).any()]:
        df = inputs.retrieval[inputs.retrieval["embedder"].isin([primary, secondary])
                              & (inputs.retrieval["k_label"] == str(cfg.retrieval.top_k))]
        for corpus in inputs.corpora:
            dc = df[df["corpus"] == corpus]
            for m in HEADLINE_SPAN:
                means = dc.pivot_table(index="setup", columns="embedder", values=m).reindex(inputs.setups)
                ranks = means.rank(ascending=False, method="average")
                diffs = [paired_diff(*(dc[(dc["setup"] == s) & (dc["embedder"] == e)].set_index("question_id")[m]
                                       for e in (secondary, primary)),
                                     cfg.analysis.bootstrap_iterations, cfg.analysis.ci,
                                     rng_for(cfg.seed, corpus, m, s, secondary), clusters=docs)
                         for s in inputs.setups]
                for s, d, p in zip(inputs.setups, diffs, holm([d["p"] for d in diffs])):
                    per_setup.append({"corpus": corpus, "embedder": secondary, "metric": m, "setup": s,
                                      "mean_primary": means.loc[s, primary], "rank_primary": ranks.loc[s, primary],
                                      "mean_secondary": means.loc[s, secondary], "rank_secondary": ranks.loc[s, secondary],
                                      **d, "p_holm": p, "verdict": verdict(d["low"], d["high"], p, 1 - cfg.analysis.ci)})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")              # constant rankings: tau is NaN
                    tau = kendalltau(means[primary], means[secondary]).statistic
                vb = vs_baseline[(vs_baseline["corpus"] == corpus) & (vs_baseline["metric"] == m)
                                 & (vs_baseline["k_label"] == str(cfg.retrieval.top_k))].dropna(subset=["diff"])
                best = vb.loc[vb["diff"].idxmax()] if len(vb) else None
                gains = [d["diff"] for d in diffs]
                levers.append({"corpus": corpus, "embedder": secondary, "metric": m, "kendall_tau": tau,
                               "best_setup": None if best is None else best["setup"],
                               "chunker_gain": math.nan if best is None else best["diff"],
                               "chunker_low": math.nan if best is None else best["low"],
                               "chunker_high": math.nan if best is None else best["high"],
                               "chunker_verdict": "" if best is None else best["verdict"],
                               "embedder_gain_mean": float(np.mean(gains)),
                               "embedder_gain_min": float(np.min(gains)), "embedder_gain_max": float(np.max(gains)),
                               "spread_primary": means[primary].max() - means[primary].min(),
                               "spread_secondary": means[secondary].max() - means[secondary].min()})
    return pd.DataFrame(per_setup), pd.DataFrame(levers)


# --- generation --------------------------------------------------------------------------------------

def generation_means(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mean and CI of every generation metric per corpus and setup, the gold passage included."""
    return summary_table(g, [*GEN_METRICS, *GEN_DIAGNOSTIC], ["corpus", "setup"], **boot(cfg))


def generation_vs_baseline(g: pd.DataFrame, inputs: Inputs, cfg: Config, docs: pd.Series) -> pd.DataFrame:
    base = label(cfg.analysis.baseline)
    return setup_comparisons(g[g["setup"] != GOLD_PASSAGE], GEN_METRICS, base, [s for s in inputs.setups if s != base],
                             docs, cfg, ["corpus"])


def generation_vs_gold(g: pd.DataFrame, inputs: Inputs, cfg: Config, docs: pd.Series) -> pd.DataFrame:
    """Each setup − the gold passage: what retrieval with that chunker costs, per metric."""
    return setup_comparisons(g, GEN_METRICS, GOLD_PASSAGE, inputs.setups, docs, cfg, ["corpus"])


def metric_response(g: pd.DataFrame, cfg: Config, docs: pd.Series) -> pd.DataFrame:
    """Does a metric respond to retrieval? Per question, the gold passage's score − the mean of its
    setups' scores, paired over questions (one Holm family per corpus). A metric whose gap isn't shown
    can't rank chunkers (the previous round's answer relevancy was ~0.54 everywhere)."""
    rows = []
    for corpus, gc in g.groupby("corpus", sort=False):
        block = []
        for m in GEN_METRICS:
            gold = gc[gc["setup"] == GOLD_PASSAGE].set_index("question_id")[m]
            setups = gc[gc["setup"] != GOLD_PASSAGE].groupby("question_id")[m].mean()
            d = paired_diff(gold, setups, cfg.analysis.bootstrap_iterations, cfg.analysis.ci,
                            rng_for(cfg.seed, corpus, m, "gold-gap"), clusters=docs)
            block.append({"corpus": corpus, "metric": m, "gold_mean": gold.mean(), "setups_mean": setups.mean(), **d})
        for r, p in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = p
            r["verdict"] = verdict(r["low"], r["high"], p, 1 - cfg.analysis.ci)
            r["responds"] = r["verdict"] != ""
        rows += block
    return pd.DataFrame(rows)


def abstentions(g: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    """Per corpus and setup: the abstention counts (with and without the evidence shown), and
    faithfulness over the answered rows next to the view where abstentions count as 1."""
    counts = g.groupby(["corpus", "setup"], sort=False).agg(
        questions=("question_id", "size"), answered=("answered", "sum"), abstained=("abstained", "sum"),
        errors=("answer_error", "sum"),
        abstained_with_evidence=("abstained", lambda s: int((s & g.loc[s.index, "evidence_shown"]).sum())),
        answered_without_evidence=("answered", lambda s: int((s & ~g.loc[s.index, "evidence_shown"]).sum())),
    ).reset_index()
    for m in ("abstention", "faithfulness", "faithfulness_abst1"):
        part = means[means["metric"] == m].set_index(["corpus", "setup"])[["mean", "low", "high", "n"]]
        counts = counts.join(part.add_prefix(f"{m}_"), on=["corpus", "setup"])
    return counts


def failure_funnel(g: pd.DataFrame) -> pd.DataFrame:
    """Per corpus and setup: how many answers end up in each OUTCOMES category, their shares, and how
    many answered rows have a statement the contexts don't support (faithfulness < 1)."""
    table = pd.crosstab([g["corpus"], g["setup"]], g["outcome"]).reindex(columns=list(OUTCOMES), fill_value=0)
    shares = table.div(table.sum(axis=1), axis=0).add_suffix("_share")
    unfaithful = g[g["answered"]].assign(u=lambda d: d["faithfulness"] < 1 - TIE).groupby(["corpus", "setup"])["u"].sum()
    out = table.join(shares).join(unfaithful.rename("answered_unfaithful")).fillna({"answered_unfaithful": 0})
    out = out.astype({"answered_unfaithful": int}).reset_index()
    out.insert(2, "rows", table.sum(axis=1).to_numpy())
    return out


def judge_vs_spans(g: pd.DataFrame, inputs: Inputs, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Context precision (the judge) and span_hit (the character spans) measure the same thing two
    ways: "some top-k chunk is useful" vs "some top-k chunk overlaps the evidence". Per setup, their
    agreement and Cohen's kappa; and every row where they disagree, with where the judge's useful
    chunks come from: the evidence, the same document outside it, or another document (which the span
    metrics count as a miss even when it holds the same fact)."""
    r = g[g["setup"] != GOLD_PASSAGE]
    rows, cases = [], []
    for (corpus, setup), grp in r.groupby(["corpus", "setup"], sort=False):
        j, s = grp["judge_useful"].astype(bool), grp["span_hit"] > 0
        n, agree = len(grp), float((j == s).mean())
        pe = j.mean() * s.mean() + (1 - j.mean()) * (1 - s.mean())
        rows.append({"corpus": corpus, "setup": setup, "rows": n, "both": int((j & s).sum()),
                     "judge_only": int((j & ~s).sum()), "spans_only": int((~j & s).sum()),
                     "neither": int((~j & ~s).sum()), "agreement": agree,
                     "kappa": (agree - pe) / (1 - pe) if pe < 1 else math.nan})
        for row in grp[j != s].itertuples():
            qa = inputs.questions[row.question_id]
            evidence = evidence_intervals(qa)
            hits = inputs.hits[(corpus, setup)][row.question_id][:cfg.retrieval.top_k]
            verdicts = [v.get("verdict") for v in inputs.details[(row.question_id, setup)]["context_precision"]]
            origins = []
            for rank, (h, v) in enumerate(zip(hits, verdicts), start=1):
                if v != 1:
                    continue
                where = ("evidence" if h.doc_id == qa.doc_id and overlap_len(evidence, [(h.start, h.end)])
                         else "same document" if h.doc_id == qa.doc_id else f"other: {h.doc_id}")
                origins.append(f"{rank}: {where}")
            cases.append({"corpus": corpus, "question_id": row.question_id, "setup": setup,
                          "kind": "judge only" if row.judge_useful else "spans only",
                          "useful_chunks": "; ".join(origins), "cp_verdicts": row.cp_verdicts,
                          "evidence_rank": int(row.evidence_rank), "context_recall": row.context_recall,
                          "fc_recall": row.fc_recall})
    cols = ["corpus", "question_id", "setup", "kind", "useful_chunks", "cp_verdicts", "evidence_rank", "context_recall",
            "fc_recall"]
    return pd.DataFrame(rows), pd.DataFrame(cases, columns=cols)


TEXT_METRICS = ["token_f1", "answer_similarity", "rouge_l", "factual_correctness"]


def text_vs_judge(g: pd.DataFrame) -> pd.DataFrame:
    """Per setup, over the answered rows: Spearman's correlation of fc_recall (the judge) with the
    judge-free text metrics and with Ragas' factual correctness. A low correlation where the others are
    high points at the judge (or the gold answer), not at the chunker."""
    rows = []
    for (corpus, setup), grp in g[g["answered"]].groupby(["corpus", "setup"], sort=False):
        row = {"corpus": corpus, "setup": setup, "rows": len(grp)}
        for other in TEXT_METRICS:
            row[other] = spearman(grp["fc_recall"], grp[other])
        rows.append(row)
    return pd.DataFrame(rows, columns=["corpus", "setup", "rows", *TEXT_METRICS])


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return math.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def by_evidence(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The generation metrics of the retrieved setups split by evidence_shown: with the evidence, the
    setups should come near the gold passage; without it, an answer with gold claims was found
    elsewhere (or guessed), and an abstention is the right behaviour."""
    r = g[g["setup"] != GOLD_PASSAGE].sort_values("evidence_shown", ascending=False, kind="stable")
    return summary_table(r, HEADLINE_GEN, ["corpus", "setup", "evidence_shown"], **boot(cfg))


def score_distributions(g: pd.DataFrame | None, retrieval_top: pd.DataFrame) -> pd.DataFrame:
    """Per corpus, setup and metric: the share of scores at exactly 0, exactly 1 and in between, and
    the median. Means hide these lumps (context recall on a 1–2 sentence gold answer is 0, 0.5 or 1)."""
    frames = [retrieval_top.melt(id_vars=["corpus", "setup"], value_vars=["span_recall", "precision", "iou"],
                                 var_name="metric")]
    if g is not None:
        frames.append(g.melt(id_vars=["corpus", "setup"], value_vars=GEN_METRICS, var_name="metric"))
    long = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    out = long.groupby(["corpus", "setup", "metric"], sort=False)["value"].agg(
        n="size", zero=lambda v: (v.abs() < TIE).mean(), one=lambda v: (v > 1 - TIE).mean(), median="median").reset_index()
    out["between"] = 1 - out["zero"] - out["one"]
    return out


def audit_lists(g: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    """Questions to read by hand before trusting the generation results:
      - the gold passage abstains: mostly weak Q-A pairs (7 of 35 on the dev docs);
      - the gold passage's answer misses gold claims, or its context recall / precision is below 1
        (its one context holds all the evidence): a judge error, or a gold answer the passage doesn't support;
      - a correct answer (fc_recall 1) that Ragas' factual correctness scores 0."""
    gold = g[g["setup"] == GOLD_PASSAGE]
    checks = [
        ("gold passage abstains", gold[gold["abstained"]]),
        ("gold passage: answer misses gold claims", gold[gold["answered"] & (gold["fc_recall"] < 1 - TIE)]),
        ("gold passage: context recall < 1", gold[gold["context_recall"] < 1 - TIE]),
        ("gold passage: context precision < 1", gold[gold["context_precision"] < 1 - TIE]),
        ("correct answer, Ragas factual correctness 0",
         g[g["answered"] & (g["fc_recall"] >= 1 - TIE) & (g["factual_correctness"].abs() < TIE)]),
    ]
    rows = []
    for reason, df in checks:
        for (corpus, qid), grp in df.groupby(["corpus", "question_id"], sort=False):
            qa = inputs.questions[qid]
            setup = grp["setup"].iloc[0]
            rows.append({"corpus": corpus, "question_id": qid, "doc_id": qa.doc_id, "reason": reason,
                         "setups": ", ".join(grp["setup"]), "question": qa.question, "gold_answer": qa.answer,
                         "response": inputs.details[(qid, setup)]["response"], "response_setup": setup})
    return pd.DataFrame(rows, columns=["corpus", "question_id", "doc_id", "reason", "setups", "question",
                                       "gold_answer", "response", "response_setup"])


# --- breakdowns, question set, planning ----------------------------------------------------------------

BREAKDOWN_DIMENSIONS = ["question_type", "evidence_quotes", "evidence_length", "kind"]


def breakdowns(inputs: Inputs, cfg: Config, questions: pd.DataFrame, g: pd.DataFrame | None) -> pd.DataFrame:
    """The main metrics per setup, split by question type, number of evidence quotes (1 vs 2+: where
    chunk boundaries should matter most), evidence length (terciles per corpus) and kind (single /
    two-span, only when both exist; two-span units are on again since 2026-10-19). Decided 2026-10-12: no split by
    section (nearly one value per question). Cells under LOW_N questions are flagged `low_n`."""
    dims = [d for d in BREAKDOWN_DIMENSIONS if d != "kind" or questions["kind"].nunique() > 1]
    labels = questions.set_index("question_id")[dims]
    parts = [("retrieval", top_k_rows(inputs.retrieval, cfg).drop(columns=["question_type", "kind"]),
              ["span_hit", "span_recall", "all_evidence"])]
    if g is not None:
        parts.append(("generation", g.drop(columns=["question_type", "kind", "evidence_quotes", "evidence_length"]),
                      ["fc_recall", "factual_correctness", "faithfulness", "abstention"]))
    out = []
    for side, df, metrics in parts:
        df = df.join(labels, on="question_id")
        for dim in dims:
            block = df.assign(dimension=dim, value=df[dim])
            t = summary_table(block, metrics, ["corpus", "dimension", "value", "setup"], **boot(cfg))
            out.append(t.assign(side=side))
    return pd.concat(out, ignore_index=True)


def question_set(questions: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The question set's make-up per corpus (documents, types, evidence) as counts, and the yield of
    generation and filtering from their stage summaries (when the run has them)."""
    rows = []
    for dim in ("doc_id", "question_type", "kind", "evidence_quotes", "evidence_length"):
        counts = questions.groupby(["corpus", dim]).agg(questions=("question_id", "size"), in_ragas=("in_ragas", "sum"))
        rows.append(counts.reset_index().rename(columns={dim: "value"}).assign(dimension=dim))
    table = pd.concat(rows, ignore_index=True)[["corpus", "dimension", "value", "questions", "in_ragas"]]
    summaries = {}
    for name in ("gen_qa_summary", "filter_qa_summary"):
        path = run_path(cfg, f"{name}.json")
        if path.exists():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    return table, summaries


def detectable_differences(sources: dict[str, pd.DataFrame], inputs: Inputs, cfg: Config) -> pd.DataFrame:
    """How small a difference between a setup and the baseline the data can show, per corpus and
    metric: from the spread of the per-question differences (sd, pooled over the setups) and the
    number of questions, at alpha / (number of setups − 1), the level Holm requires of the smallest
    p-value of a family, with power stats.POWER. Also at n = ragas.sample_per_corpus, and the number
    of questions a difference of 0.05 would need. Use it to size the final run before paying for it.
    A sd of 0 means no question changed between the setups: nothing can be estimated from it."""
    base = label(cfg.analysis.baseline)
    others = [s for s in inputs.setups if s != base]
    alpha = (1 - cfg.analysis.ci) / max(len(others), 1)
    planned = cfg.ragas.sample_per_corpus
    rows = []
    for side, df in sources.items():
        metrics = HEADLINE_SPAN if side == "retrieval" else GEN_METRICS
        for corpus in inputs.corpora:
            dc = df[df["corpus"] == corpus]
            if dc.empty:
                continue
            for m in metrics:
                wide = dc.pivot(index="question_id", columns="setup", values=m)
                if base not in wide:
                    continue
                diffs = [(wide[s] - wide[base]).dropna() for s in others if s in wide]
                if not diffs:
                    continue
                sd = float(np.sqrt(np.mean([d.var(ddof=1) if len(d) > 1 else 0.0 for d in diffs])))
                n = int(np.median([len(d) for d in diffs]))
                changed = float(np.mean([(d.abs() > TIE).mean() for d in diffs]))
                rows.append({"corpus": corpus, "side": side, "metric": m, "n": n, "sd_diff": sd,
                             "changed_share": changed, "alpha": alpha,
                             "detectable_now": detectable_difference(sd, n, alpha),
                             "n_planned": planned, "detectable_planned": detectable_difference(sd, planned, alpha),
                             "n_for_0.05": questions_needed(sd, 0.05, alpha)})
    return pd.DataFrame(rows)


def chunk_sizes(cfg: Config, corpora: list[str]) -> pd.DataFrame:
    """data/chunks/<corpus>/_size_stats.json as one table (sizes in bge-m3 tokens)."""
    rows = []
    for corpus in corpora:
        path = cfg.paths.data_dir / "chunks" / corpus / "_size_stats.json"
        if path.exists():
            for setup, s in json.loads(path.read_text(encoding="utf-8")).items():
                rows.append({"corpus": corpus, "setup": label(setup), **s})
    return pd.DataFrame(rows)


def costs(cfg: Config) -> dict[str, Any]:
    """The judge's cost (ragas/cost.json: "table", all the calls the Ragas table rests on) and the
    generator's mean token counts per setup (answer_summary.json)."""
    out: dict[str, Any] = {}
    path = run_path(cfg, "ragas", "cost.json")
    if path.exists():
        cost = json.loads(path.read_text(encoding="utf-8"))
        out["judge"] = cost
        out["judge_by_prompt"] = pd.DataFrame([{"prompt": k, **v} for k, v in cost["table"]["by_prompt"].items()])
    path = run_path(cfg, "answer_summary.json")
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        out["generator"] = pd.DataFrame([{"corpus": c, "setup": label(s), **v} for c, by in summary.items()
                                         for s, v in by.items()])
    return out


# --- secondary views -----------------------------------------------------------------------------------

def context_vs_recall(r_means: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Secondary: per setup and k, the mean context tokens handed over against the span recall and hit
    reached (primary embedder, with the oracle as the floor of tokens needed)."""
    df = r_means[r_means["embedder"].isin([cfg.embedding.primary, "oracle"])
                 & r_means["metric"].isin(["context_tokens", "span_recall", "span_hit", "k"])]
    return df.pivot_table(index=["corpus", "embedder", "setup", "k_label"], columns="metric", values="mean").reset_index()


def evidence_rank(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Secondary: does the generator use evidence at rank 4–5 as well as at rank 1? Per corpus and
    rank of the first chunk overlapping the evidence (0 = none in the top k), over the retrieved setups."""
    r = g[g["setup"] != GOLD_PASSAGE]
    out = summary_table(r, ["abstention", "fc_recall", "faithfulness"], ["corpus", "evidence_rank"], **boot(cfg))
    return out.sort_values(["corpus", "evidence_rank", "metric"])


CORRELATED = ["span_hit", "span_recall", "mrr", "context_precision", "context_recall", "faithfulness",
              "answer_relevancy", "factual_correctness", "fc_recall", "fc_precision", "rouge_l", "token_f1",
              "answer_similarity", "response_words"]


def metric_correlations(g: pd.DataFrame) -> pd.DataFrame:
    """Secondary: Spearman's correlation between every pair of metrics over the answered rows of the
    retrieved setups (both corpora pooled): which metrics carry the same information."""
    r = g[(g["setup"] != GOLD_PASSAGE) & g["answered"]]
    rows = [{"metric_a": a, "metric_b": b, "rho": spearman(r[a], r[b]), "n": int((r[a].notna() & r[b].notna()).sum())}
            for a in CORRELATED for b in CORRELATED]
    return pd.DataFrame(rows)


LENGTH_METRICS = ["fc_precision", "factual_correctness", "fc_recall", "faithfulness"]


def length_vs_scores(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Secondary: are longer answers penalised? Per corpus, the answered rows of the retrieved setups
    in response-length terciles, with the claim-precision metrics (which count extra true detail as
    wrong) next to fc_recall; `rho` is Spearman's correlation of response_words with each metric."""
    r = g[(g["setup"] != GOLD_PASSAGE) & g["answered"]].copy()
    r["length"] = pd.concat([size_bins(grp["response_words"], BIN_NAMES, "words")
                             for _, grp in r.groupby("corpus", sort=False)])
    out = summary_table(r, LENGTH_METRICS, ["corpus", "length"], **boot(cfg))
    out["order"] = out["length"].map(bin_order)
    rho = {(c, m): spearman(grp["response_words"], grp[m]) for c, grp in r.groupby("corpus") for m in LENGTH_METRICS}
    out["rho"] = [rho[(c, m)] for c, m in zip(out["corpus"], out["metric"])]
    return out.sort_values(["corpus", "metric", "order"]).drop(columns="order")

