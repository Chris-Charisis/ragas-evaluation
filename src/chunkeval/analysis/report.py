"""report.md: the tables of views.py as Markdown, with the figures, in reading order; and the run
record (what exactly was measured). The CSVs hold every number; the Markdown shows the headline ones."""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from chunkeval.analysis.inputs import METHOD_LABELS, Inputs, label, run_path
from chunkeval.analysis.stats import LOW_N, MIN_CLUSTER_DOCS, TIE
from chunkeval.analysis.views import HEADLINE_SPAN, OUTCOMES, bin_order
from chunkeval.core.schema import REPO_ROOT, Config
from chunkeval.evaluation.answering import GOLD_PASSAGE, answer_path
from chunkeval.evaluation.retrieval import stored_provenance
from chunkeval.qa.filtering import read_filter_provenance
from chunkeval.qa.generation import qa_paths, read_provenance

# Below this many questions in a corpus, the report opens with a warning: read it as a test.
SMALL_SET = 100
ANSWER_METRICS = ["fc_recall", "factual_correctness", "fc_precision", "fc_f1", "token_f1", "answer_similarity"]
GROUNDING_METRICS = ["faithfulness", "faithfulness_abst1", "abstention", "answer_relevancy", "context_precision",
                     "context_recall"]
COMPARED_GEN = ["fc_recall", "factual_correctness", "faithfulness", "abstention", "answer_relevancy", "context_precision"]


# --- formatting ------------------------------------------------------------------------------------

def num(x: float, digits: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def ci_cell(r: pd.Series | dict, digits: int = 2) -> str:
    """"0.85 [0.67, 1.00]", with † when the cell has fewer than LOW_N questions."""
    if r is None or math.isnan(r["mean"]):
        return "–"
    mark = "†" if r.get("low_n", False) else ""
    return f"{num(r['mean'], digits)} [{num(r['low'], digits)}, {num(r['high'], digits)}]{mark}"


def diff_cell(r: pd.Series | dict, digits: int = 3, low: str = "low", high: str = "high") -> str:
    if math.isnan(r["diff"]):
        return "–"
    return f"{r['diff']:+.{digits}f} [{r[low]:+.{digits}f}, {r[high]:+.{digits}f}]"


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True) if len(df) else "_(nothing to show)_"


def rows_of(cells: pd.DataFrame) -> int:
    """The rows behind a group of summary cells (one per metric): the values used plus the NaNs dropped."""
    return int(cells["n"].iloc[0] + cells["nan"].iloc[0])


def digits_for(metric: str) -> int:
    return 0 if metric in ("context_tokens",) else 1 if metric == "k" else 2


def means_wide(means: pd.DataFrame, rows: list[str], metrics: list[str], row_name: str = "setup") -> pd.DataFrame:
    """One row per `rows` (values of `setup`), one "mean [low, high]" column per metric."""
    lookup = {(r["setup"], r["metric"]): r for _, r in means.iterrows()}
    return pd.DataFrame([{row_name: s, **{m: ci_cell(lookup.get((s, m)), digits_for(m)) for m in metrics}}
                         for s in rows if any((s, m) in lookup for m in metrics)])


def comparison_rows(comp: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Paired comparisons as Markdown rows: difference [CI], document-level CI, better/equal/worse,
    Holm p and the verdict (with the few-documents flag)."""
    rows = []
    for m in metrics:
        for _, r in comp[comp["metric"] == m].iterrows():
            flag = " (few documents)" if r["few_docs"] else ""
            rows.append({"metric": m, "setup": r["setup"], "difference [CI]": diff_cell(r),
                         "by document": diff_cell(r, low="doc_low", high="doc_high") + ("" if r["doc_ci_reliable"] else "*"),
                         "+ / = / −": f"{r['better']} / {r['equal']} / {r['worse']}",
                         "n": f"{r['n']}" + (f" ({r['dropped']} dropped)" if r["dropped"] else ""),
                         "p (Holm)": num(r["p_holm"], 3), "verdict": (r["verdict"] or "–") + flag})
    return pd.DataFrame(rows)


def verdict_summary(comp: pd.DataFrame, what: str) -> str:
    """One sentence: which comparisons pass the ranking rule."""
    shown = comp[comp["verdict"] != ""]
    if shown.empty:
        return f"**No {what} passes the ranking rule** (CI excluding 0 and Holm p < alpha), on any metric."
    items = [f"{r['corpus']}: {r['setup']} {r['verdict']} on {r['metric']}" + (f" at k={r['k_label']}" if "k_label" in r else "")
             + (" (few documents)" if r["few_docs"] else "") for _, r in shown.iterrows()]
    return f"**Passing the ranking rule ({len(items)}):** " + "; ".join(items) + "."


class Doc:
    """A Markdown document built line by line."""

    def __init__(self):
        self.lines: list[str] = []

    def add(self, *blocks: str) -> None:
        """Paragraphs, tables or list items, each followed by a blank line."""
        for block in blocks:
            self.lines += [block, ""]

    def figure(self, path: Path | None, out: Path, alt: str) -> None:
        if path is not None:
            self.add(f"![{alt}]({path.relative_to(out).as_posix()})")

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


# --- the run record ------------------------------------------------------------------------------------

def git_state() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def differing(settings: dict[str, dict[str, Any]], corpora: list[str]) -> list[str]:
    """The settings keys whose value differs between the `corpora` that have a record."""
    present = [settings[c] for c in corpora if c in settings]
    keys = dict.fromkeys(key for s in present for key in s)
    return [key for key in keys if len({json.dumps(s.get(key), sort_keys=True) for s in present}) > 1]


def question_settings(cfg: Config, corpora: list[str]) -> dict[str, Any]:
    """How each corpus's questions were generated (raw.provenance.json) and filtered
    (filtered.provenance.json; None for a corpus filtered before 2026-10-28), and which settings differ
    between the corpora: `gen-qa` and `filter-qa` can run per corpus, at different times."""
    gen, filt = read_provenance(qa_paths(cfg)["provenance"])[0], read_filter_provenance(cfg)
    by_corpus = {}
    for c in corpora:
        g, f = gen.get(c), filt.get(c)
        by_corpus[c] = {
            "generation": {"generator": g["generator"]["model"], "digest": (g["generator"].get("digest") or "")[:12],
                           "prompt_sha256": g["prompt_sha256"][:12]} if g else None,
            "filter": {"critic": f["critic"]["model"], "digest": (f["critic"].get("digest") or "")[:12],
                       "prompt_sha256": f["critic_prompt_sha256"][:12], "critic_min_score": f["critic_min_score"],
                       "dedup_cosine_threshold": f["dedup_cosine_threshold"]} if f else None}
    return {"by_corpus": by_corpus, "differ": {"gen-qa": differing(gen, corpora), "filter-qa": differing(filt, corpora)}}


def run_record(cfg: Config, inputs: Inputs, questions: pd.DataFrame, costs: dict[str, Any], generated: datetime,
               folder: str) -> dict[str, Any]:
    """What this report measured: data, models with their pins, settings, cost. Also written as JSON.
    `folder`: the report's folder name in runs/<run_id>/reports/."""
    record: dict[str, Any] = {"run_id": cfg.run_id, "report": folder,
                              "generated": generated.isoformat(timespec="seconds"),
                              "code": git_state(), "config": str(run_path(cfg, "config.yaml").relative_to(REPO_ROOT))
                              if run_path(cfg, "config.yaml").is_relative_to(REPO_ROOT) else str(run_path(cfg, "config.yaml"))}
    record["corpora"] = {}
    for corpus in inputs.corpora:
        q = questions[questions["corpus"] == corpus]
        parsed = cfg.paths.data_dir / "parsed" / corpus
        record["corpora"][corpus] = {
            "documents_parsed": len([p for p in parsed.glob("*.json") if not p.name.startswith("_")]) if parsed.exists() else None,
            "documents_with_questions": int(q["doc_id"].nunique()), "questions": len(q),
            "ragas_questions": int(q["in_ragas"].sum())}
    record["questions_made"] = question_settings(cfg, inputs.corpora)
    record["ragas_sample"] = (f"eval --pilot {inputs.pilot}" if inputs.pilot
                              else f"ragas.sample_per_corpus {cfg.ragas.sample_per_corpus}")
    c = cfg.chunking
    record["chunking"] = {"methods": c.methods, "overlap": c.overlap, "target_tokens": c.target_tokens,
                          "max_tokens": c.max_tokens, "overlap_ratio": c.overlap_ratio,
                          "size_tokenizer": f"{c.size_tokenizer}@{c.size_tokenizer_revision}"}
    record["embedders"] = {name: f"{m.model_id}@{m.revision}" for name, m in cfg.embedding.models.items()}
    r = cfg.retrieval
    record["retrieval"] = {"primary": cfg.embedding.primary, "mode": r.mode, "top_k": r.top_k, "k_values": r.k_values,
                           "budget_tokens": r.budget_tokens, "k_max": r.k_max}
    gen = stored_provenance(answer_path(cfg, inputs.corpora[0], GOLD_PASSAGE)) if inputs.corpora else None
    llm = cfg.llm
    record["models"] = {
        "qa_generator": llm.qa_generator.model, "qa_critic": llm.qa_critic.model,
        "rag_generator": llm.rag_generator.model + (f" (digest {gen['generator']['digest'][:12]})"
                                                    if gen and gen["generator"].get("digest") else ""),
        "rag_prompt_sha256": gen["prompt_sha256"][:12] if gen else None,
        "judge": llm.judge.model if inputs.ragas is not None else None,
        "ragas": package_version("ragas") if inputs.ragas is not None else None}
    a = cfg.analysis
    record["analysis"] = {"baseline": a.baseline, "bootstrap_iterations": a.bootstrap_iterations, "ci": a.ci,
                          "alpha": round(1 - a.ci, 4), "seed": cfg.seed}
    if "judge" in costs:
        t = costs["judge"]["table"]
        record["judge_cost"] = {"usd": t["cost_usd"], "calls": t["calls"], "per_question_usd": t["cost_per_question_usd"]}
    return record


# --- sections ------------------------------------------------------------------------------------------

def how_to_read(doc: Doc, cfg: Config) -> None:
    a = cfg.analysis
    doc.add("## How to read this report")
    doc.add(
        "- **Methods:** **fixed-size** cuts windows of a fixed number of characters at the nearest space, "
        "**sentence** packs whole sentences, and **section** packs whole sentences without crossing a section "
        "boundary. The config and the data files name them by their keys: "
        + ", ".join(f"{name} is `{key}`" for key, name in METHOD_LABELS.items()) + ".",
        f"- **Setups** are the chunkers: method × overlap. Retrieval uses `{cfg.embedding.primary}`, and the "
        f"generator is shown its top {cfg.retrieval.top_k} chunks. Two references bound them: the **gold passage** "
        "(each question answered from its own passage, i.e. perfect retrieval: the ceiling of the generation metrics) "
        "and the **oracle** (the fewest chunks of a chunker that cover the evidence: the ceiling a chunker allows). "
        "**Random** retrieval is chance.",
        f"- **Means** come with a {a.ci:.0%} bootstrap interval over questions ({a.bootstrap_iterations} draws): "
        f"`mean [low, high]`. † marks a cell with fewer than {LOW_N} questions.",
        "- **Comparisons are paired.** Every setup is scored on the same questions, so a comparison takes the "
        "difference per question, then its mean and interval. `+ / = / −` counts the questions where the setup is "
        "better, equal or worse: with metrics like span_hit, only the questions that change carry the difference.",
        f"- **The ranking rule.** A setup is called higher or lower only when its interval excludes 0 **and** its "
        f"Wilcoxon p-value, Holm-adjusted within its family, is below {1 - a.ci:.2f}. A family is the comparisons "
        "of one corpus, metric and k (5 setups against the baseline). Why both: the report makes well over 100 "
        "comparisons, and a 95% interval excludes 0 by chance about 1 time in 20, so with nothing to find ~5 "
        "\"winners\" would still appear. Holm multiplies the smallest p of a family by 5, the next by 4, and so "
        "on, which keeps the chance of even one false winner in the family under alpha.",
        f"- **By document.** The question-level interval treats questions as independent, but questions of one "
        f"document fail or succeed together (garbled text, a near-duplicate document). The `by document` "
        f"interval redraws whole documents. \"(few documents)\" flags a difference that holds by question but "
        f"not by document: it rests on a few documents. Below {MIN_CLUSTER_DOCS} documents this interval is "
        f"unreliable (marked *).",
        "- **Abstentions** (\"The context does not contain the answer.\") score 0 on answer relevancy, factual "
        "correctness and the text metrics, as Ragas would score them, and are never dropped, so the pairs stay "
        "complete. Faithfulness is reported over the answered rows, next to the abstention rate, and in a view "
        "where an abstention counts as faithful (`faithfulness_abst1`).",
        "- **Factual correctness, two ways.** `factual_correctness` is Ragas' standard f1, comparable with other "
        "Ragas reports; it counts as wrong any true claim the gold answer doesn't state (the question's framing, "
        "the source's name, an extra correct detail). `fc_recall` asks \"is the right answer in it?\": the share "
        "of the gold answer's claims the answer contains. `fc_precision` asks \"does it say only that?\".",
        "- **Secondary views** (end of the report) are diagnostics, not results.")


def run_record_section(doc: Doc, record: dict[str, Any]) -> None:
    doc.add("## 1. Run record")
    rows = [("run", record["run_id"]), ("report", f"reports/{record['report']}"), ("generated", record["generated"]),
            ("code", record["code"]),
            ("config", record["config"])]
    for corpus, c in record["corpora"].items():
        rows.append((f"{corpus}", f"{c['questions']} questions from {c['documents_with_questions']} documents "
                                  f"({c['documents_parsed']} parsed); {c['ragas_questions']} in the Ragas table"))
    for corpus, made in record["questions_made"]["by_corpus"].items():
        g, f = made["generation"], made["filter"]
        gen = f"gen-qa {g['generator']} (digest {g['digest'] or '–'}, prompt {g['prompt_sha256']})" if g else "gen-qa not recorded"
        filt = (f"filter-qa {f['critic']} (digest {f['digest'] or '–'}, prompt {f['prompt_sha256']}), min score "
                f"{f['critic_min_score']}, dedup {f['dedup_cosine_threshold']}" if f else "filter-qa not recorded")
        rows.append((f"{corpus} questions", f"{gen}; {filt}"))
    rows.append(("Ragas sample", record["ragas_sample"]))
    ch = record["chunking"]
    rows.append(("chunking", f"{', '.join(map(label, ch['methods']))} × overlap {ch['overlap']}; target {ch['target_tokens']}, "
                             f"max {ch['max_tokens']} tokens, overlap ratio {ch['overlap_ratio']}; sizes by {ch['size_tokenizer']}"))
    rows += [(f"embedder {name}", pin) for name, pin in record["embedders"].items()]
    rt = record["retrieval"]
    rows.append(("retrieval", f"{rt['primary']}, {rt['mode']}; top_k {rt['top_k']}; k {rt['k_values']}; "
                              f"budget {rt['budget_tokens']} tokens; k_max {rt['k_max']}"))
    rows += [(k, str(v)) for k, v in record["models"].items() if v]
    an = record["analysis"]
    rows.append(("analysis", f"baseline {label(an['baseline'])}; {an['bootstrap_iterations']} bootstrap draws; "
                             f"{an['ci']:.0%} intervals; alpha {an['alpha']}; seed {an['seed']}"))
    if "judge_cost" in record:
        jc = record["judge_cost"]
        rows.append(("judge cost", f"${jc['usd']:.2f} for {jc['calls']} calls (${jc['per_question_usd']:.4f} per "
                                   "question, all setups)"))
    doc.add(md(pd.DataFrame(rows, columns=["item", "value"])))


def question_set_section(doc: Doc, t: dict[str, pd.DataFrame], summaries: dict[str, Any], corpora: list[str]) -> None:
    doc.add("## 2. The question set")
    qs = t["question_set"]
    rows = []
    for corpus in corpora:
        q = qs[qs["corpus"] == corpus]
        total = int(q[q["dimension"] == "doc_id"]["questions"].sum())
        docs = q[q["dimension"] == "doc_id"].sort_values("questions", ascending=False)
        top = docs.iloc[0] if len(docs) else None

        def counts(dim: str) -> str:
            d = q[q["dimension"] == dim]
            if dim == "evidence_length":
                d = d.iloc[sorted(range(len(d)), key=lambda i: bin_order(d["value"].iloc[i]))]
            return ", ".join(f"{v} {n}" for v, n in zip(d["value"], d["questions"]))

        rows.append({"corpus": corpus, "questions": total, "documents": len(docs),
                     "largest document": f"{top['value']} ({top['questions']}, {top['questions'] / total:.0%})" if top is not None else "–",
                     "types": counts("question_type"), "evidence": counts("evidence_quotes"),
                     "evidence length": counts("evidence_length")})
    doc.add(md(pd.DataFrame(rows)))
    gen, flt = summaries.get("gen_qa_summary"), summaries.get("filter_qa_summary")
    if gen and flt:
        rows = []
        for corpus in corpora:
            g, f = gen.get(corpus, {}), flt.get(corpus, {})
            reasons = sorted(f.get("reasons", {}).items(), key=lambda kv: -kv[1])[:4]
            rows.append({"corpus": corpus, "units": g.get("units"), "pairs": g.get("pairs"),
                         "kept": f.get("kept"), "kept of units": f"{f.get('kept', 0) / g['units']:.0%}" if g.get("units") else "–",
                         "top rejection reasons": ", ".join(f"{k} {v}" for k, v in reasons)})
        doc.add("How the questions were made (gen-qa → filter-qa):", md(pd.DataFrame(rows)))
    doc.add("Every question with its attributes: `tables/question_attributes.csv`; the counts: `tables/question_set.csv`.")


def retrieval_section(doc: Doc, cfg: Config, inputs: Inputs, t: dict[str, pd.DataFrame], figs: dict[str, Path], out: Path) -> None:
    primary, top = cfg.embedding.primary, str(cfg.retrieval.top_k)
    rm = t["retrieval_means"]
    shown = ["span_hit", "span_recall", "all_evidence", "mrr", "precision", "context_tokens"]
    doc.add(f"## 3. Retrieval ({primary})")
    doc.add(f"### 3.1 At k = {top}", "What the generator is shown. Definitions of the span metrics: docs/evaluation.md, "
            "\"Retrieval and span metrics\".")
    for corpus in inputs.corpora:
        doc.add(f"**{corpus}**", md(means_wide(rm[(rm["corpus"] == corpus) & (rm["embedder"] == primary)
                                              & (rm["k_label"] == top)], inputs.setups, shown)))
        ks = sorted({k for k in rm["k_label"] if k != "budget"}, key=int)
        o = rm[(rm["corpus"] == corpus) & (rm["embedder"] == "oracle") & (rm["k_label"] == ks[-1])]
        rnd = rm[(rm["corpus"] == corpus) & (rm["embedder"] == "random") & (rm["k_label"] == top)]
        rows = []
        for s in inputs.setups:
            get = lambda d, m: d[(d["setup"] == s) & (d["metric"] == m)]
            rows.append({"setup": s, "oracle chunks": num(get(o, "k")["mean"].mean(), 2),
                         "oracle tokens": num(get(o, "context_tokens")["mean"].mean(), 0),
                         "oracle span_recall": num(get(o, "span_recall")["mean"].mean()),
                         "random span_hit": num(get(rnd, "span_hit")["mean"].mean())})
        doc.add(f"References for {corpus}: the oracle (at k ≤ {ks[-1]}: how many chunks and tokens each chunker "
                "needs to hand over all the evidence) and random retrieval (chance).", md(pd.DataFrame(rows)))
    doc.figure(figs.get("retrieval_top_k"), out, "span metrics per setup at top_k")

    doc.add(f"### 3.2 At equal context ({cfg.retrieval.budget_tokens} tokens)",
            "At a fixed k, setups with bigger or overlapping chunks hand over more text. Here each question gets "
            "the largest k whose chunks fit the token budget (`k` is its mean).")
    for corpus in inputs.corpora:
        doc.add(f"**{corpus}**", md(means_wide(rm[(rm["corpus"] == corpus) & (rm["embedder"] == primary)
                                              & (rm["k_label"] == "budget")], inputs.setups, [*shown, "k"])))
    doc.figure(figs.get("retrieval_budget"), out, "span metrics per setup at the token budget")

    vb = t["retrieval_vs_baseline"]
    doc.add(f"### 3.3 Setups against the baseline ({label(cfg.analysis.baseline)})",
            verdict_summary(vb, "setup") + f" Shown: k = {top}; every span metric at k = {top} and at the budget "
            "is in `tables/retrieval_vs_baseline.csv`.",
            "Precision and IoU count text that overlapping chunks repeat once, while `context_tokens` (what the "
            "generator reads) counts it every time: setups with overlap gain on precision and IoU at a fixed k "
            "without handing over less text. Read those two next to context_tokens.")
    for corpus in inputs.corpora:
        doc.add(f"**{corpus}**", md(comparison_rows(vb[(vb["corpus"] == corpus) & (vb["k_label"] == top)], HEADLINE_SPAN)))
    doc.figure(figs.get("retrieval_vs_baseline"), out, "paired differences against the baseline")

    doc.add("### 3.4 As k grows")
    doc.figure(figs.get("k_curves"), out, "span metrics as k grows")

    d = t["questions_by_setup"]
    doc.add("### 3.5 The questions that separate the setups",
            "Differences between chunkers come only from the questions whose result changes between setups. "
            "Per question: `tables/questions_by_setup.csv` (also fc_recall and abstention).")
    rows = []
    for corpus in inputs.corpora:
        for m in ("span_hit", "span_recall"):
            dm = d[(d["corpus"] == corpus) & (d["metric"] == m)]
            rows.append({"corpus": corpus, f"metric (k={top})": m, "questions": len(dm),
                         "setups differ": int((dm["class"] == "setups differ").sum()),
                         "all 1": int((dm["class"] == "all 1.0").sum()), "all 0": int((dm["class"] == "all 0.0").sum())})
    doc.add(md(pd.DataFrame(rows)))
    differ = d[(d["metric"] == "span_hit") & (d["class"] == "setups differ")]
    if len(differ):
        show = differ.head(30).copy()
        for s in inputs.setups:
            show[s] = show[s].map(lambda v: num(v, 0))
        doc.add(f"Questions whose span_hit@{top} differs between setups"
                + (f" (first 30 of {len(differ)})" if len(differ) > 30 else "") + ":",
                md(show[["question_id", "question_type", *inputs.setups]]))

    pd_ = t["per_document"]
    doc.add("### 3.6 Documents and distractors",
            "Per document (means over the setups at top_k); `chunk share` is its share of the corpus's chunks. "
            "A document with a large share supplies most distractors; a document whose questions fail everywhere "
            "may be broken (garbled text) or have a near-duplicate. All documents: `tables/per_document.csv`.")
    cols = ["corpus", "doc_id", "questions", "chunk_share", "span_hit", "span_recall"]
    cols += [c for c in ("fc_recall", "abstention", "gold_fc_recall") if c in pd_]
    show = pd_.groupby("corpus", sort=False).head(15)[cols].copy()
    for c in cols[3:]:
        show[c] = show[c].map(lambda v: num(v, 2))
    doc.add(md(show))
    ds = t["distractors"]
    if len(ds):
        pooled = (ds.groupby(["corpus", "question_doc", "distractor_doc"])["slots"].sum().reset_index()
                  .sort_values(["corpus", "slots"], ascending=[True, False]).groupby("corpus").head(8))
        doc.add(f"The documents whose chunks take most of the top-{top} slots of other documents' questions "
                "(summed over the setups; per setup: `tables/distractors.csv`):", md(pooled))

    levers, per_setup = t.get("embedder_levers"), t.get("embedder_robustness")
    if levers is not None and len(levers):
        doc.add("### 3.7 Does the ranking hold under another embedder?",
                f"The span metrics at k = {top} under each secondary embedder, on the same questions. "
                "`chunker gain`: the best setup − the baseline under the primary embedder; `embedder gain`: the "
                "secondary − the primary embedder, averaged over the setups. `spread`: best − worst setup under "
                "each embedder (a smaller spread under the better embedder means it makes up for chunk boundaries). "
                "With 6 setups, one swapped pair of near-tied setups is not a contradiction: read the rankings.")
        rows = []
        for _, r in levers.iterrows():
            rows.append({"corpus": r["corpus"], "embedder": r["embedder"], "metric": r["metric"],
                         "Kendall tau": num(r["kendall_tau"]),
                         "chunker gain": f"{r['best_setup']} {num(r['chunker_gain'], 3)} [{num(r['chunker_low'], 3)}, "
                                         f"{num(r['chunker_high'], 3)}] {r['chunker_verdict']}".strip(),
                         "embedder gain": f"{r['embedder_gain_mean']:+.3f} ({r['embedder_gain_min']:+.3f} to "
                                          f"{r['embedder_gain_max']:+.3f})",
                         "spread": f"{num(r['spread_primary'], 3)} → {num(r['spread_secondary'], 3)}"})
        doc.add(md(pd.DataFrame(rows)))
        for (corpus, emb), grp in per_setup[per_setup["metric"] == "span_recall"].groupby(["corpus", "embedder"], sort=False):
            show = pd.DataFrame([{"setup": r["setup"],
                                  "primary (rank)": f"{num(r['mean_primary'])} ({r['rank_primary']:g})",
                                  f"{emb} (rank)": f"{num(r['mean_secondary'])} ({r['rank_secondary']:g})",
                                  "difference [CI]": diff_cell(r), "p (Holm)": num(r["p_holm"], 3),
                                  "verdict": r["verdict"] or "–"} for _, r in grp.iterrows()])
            doc.add(f"**{corpus}, span_recall**: rankings side by side (all metrics: `tables/embedder_robustness.csv`)",
                    md(show))


def generation_section(doc: Doc, cfg: Config, inputs: Inputs, t: dict[str, pd.DataFrame], figs: dict[str, Path],
                       out: Path) -> None:
    rows = [*inputs.setups, GOLD_PASSAGE]
    doc.add(f"## 4. Generation (judge {cfg.llm.judge.model})",
            "On the Ragas sample of each corpus" + (f" (a pilot: `eval --pilot {inputs.pilot}`)" if inputs.pilot else "")
            + ". Every row with its derived columns and outcome: `tables/generation_rows.csv`.")

    f = t["failure_funnel"]
    doc.add("### 4.1 Where the answers end up",
            "Each answer, by whether retrieval showed the evidence (span_hit at top_k) and what the generator did "
            "with it. \"Every gold claim\" is fc_recall 1. With the evidence shown, the losses are the generator's "
            "(or the judge's); without it, an abstention is the right behaviour and an answer without a gold claim "
            "is a hallucination risk. `answered_unfaithful`: answered rows with a statement the contexts don't support.")
    doc.figure(figs.get("failure_funnel"), out, "outcomes per setup")
    for corpus in inputs.corpora:
        fc = f[f["corpus"] == corpus].set_index("setup").reindex(rows).reset_index()
        present = [o for o in OUTCOMES if fc[o].sum() > 0]
        doc.add(f"**{corpus}**", md(fc[["setup", "rows", *present, "answered_unfaithful"]].astype(str)))
    doc.add("Outcomes: " + "; ".join(f"`{k}` {v}" for k, v in OUTCOMES.items()) + ".")

    gm = t["generation_means"]
    doc.add("### 4.2 Scores per setup, with the gold passage as the ceiling")
    for corpus in inputs.corpora:
        sub = gm[gm["corpus"] == corpus]
        doc.add(f"**{corpus}**: is the answer right?", md(means_wide(sub, rows, ANSWER_METRICS)),
                f"**{corpus}**: grounding, abstention and the judge's view of the contexts",
                md(means_wide(sub, rows, GROUNDING_METRICS)))
    doc.figure(figs.get("generation_means"), out, "generation metrics per setup")

    vb = t["generation_vs_baseline"]
    doc.add(f"### 4.3 Setups against the baseline ({label(cfg.analysis.baseline)})",
            verdict_summary(vb, "setup") + " Every metric: `tables/generation_vs_baseline.csv`.")
    for corpus in inputs.corpora:
        doc.add(f"**{corpus}**", md(comparison_rows(vb[vb["corpus"] == corpus], COMPARED_GEN)))
    doc.figure(figs.get("generation_vs_baseline"), out, "generation: paired differences against the baseline")

    mr = t["metric_response"]
    doc.add("### 4.4 The gold passage: which metrics respond to retrieval",
            "Per question, the gold passage's score − the mean of its setups. A metric whose gap isn't shown "
            "doesn't respond to retrieval and can't rank chunkers (the previous round's answer relevancy was "
            "~0.54 everywhere). Setup − gold passage per setup: `tables/generation_vs_gold.csv`.")
    for corpus in inputs.corpora:
        sub = mr[mr["corpus"] == corpus]
        doc.add(f"**{corpus}**", md(pd.DataFrame([{
            "metric": r["metric"], "gold passage": num(r["gold_mean"]), "setups": num(r["setups_mean"]),
            "gap [CI]": diff_cell(r), "by document": diff_cell(r, low="doc_low", high="doc_high"),
            "p (Holm)": num(r["p_holm"], 3), "responds": "yes" if r["responds"] else "not shown"} for _, r in sub.iterrows()])))
    doc.figure(figs.get("generation_vs_gold"), out, "setups against the gold passage")
    gold = t["generation_rows"][t["generation_rows"]["setup"] == GOLD_PASSAGE]
    checks = []
    for corpus in inputs.corpora:
        gc = gold[gold["corpus"] == corpus]
        checks.append({"corpus": corpus, "gold rows": len(gc),
                       "context recall 1": f"{(gc['context_recall'] >= 1 - TIE).mean():.0%}",
                       "context precision 1": f"{(gc['context_precision'] >= 1 - TIE).mean():.0%}",
                       "abstained": int(gc["abstained"].sum())})
    doc.add("The judge checked on the gold passage: its one context holds all the evidence, so context recall and "
            "precision should be 1; the shortfall estimates the judge's error rate (the rows: `cases.md`).",
            md(pd.DataFrame(checks)))

    ab = t["abstentions"]
    doc.add("### 4.5 Abstentions and faithfulness",
            "`with evidence`: abstained although a chunk held the evidence (an over-cautious generator, or "
            "evidence split so it looked incomplete). `without evidence`: answered although no chunk held it. "
            "Faithfulness over the answered rows, and with abstentions counted as faithful.")
    for corpus in inputs.corpora:
        sub = ab[ab["corpus"] == corpus].set_index("setup").reindex(rows).reset_index()
        doc.add(f"**{corpus}**", md(pd.DataFrame([{
            "setup": r["setup"], "abstained": f"{int(r['abstained'])} of {int(r['questions'])}",
            "rate [CI]": ci_cell({"mean": r["abstention_mean"], "low": r["abstention_low"], "high": r["abstention_high"]}),
            "with evidence": int(r["abstained_with_evidence"]), "answered without evidence": int(r["answered_without_evidence"]),
            "faithfulness (answered)": ci_cell({"mean": r["faithfulness_mean"], "low": r["faithfulness_low"],
                                                "high": r["faithfulness_high"]}),
            "faithfulness (abstained = 1)": ci_cell({"mean": r["faithfulness_abst1_mean"], "low": r["faithfulness_abst1_low"],
                                                     "high": r["faithfulness_abst1_high"]})}
            for _, r in sub.iterrows() if not pd.isna(r["questions"])])))

    jv, dis, tv = t["judge_vs_spans"], t["judge_span_disagreements"], t["text_vs_judge"]
    doc.add("### 4.6 The judge checked against the spans and the text metrics",
            "Context precision says whether a chunk is useful; span_hit whether a chunk overlaps the gold evidence. "
            "Where they disagree the judge is often right: a chunk can hold the fact outside the gold evidence (same "
            "document) or in another document, which the span metrics count as a miss. Every disagreement, with where "
            "the useful chunks come from: `tables/judge_span_disagreements.csv`.")
    doc.add("Cohen's kappa corrects the agreement for chance; it is low or negative when nearly every row is a hit "
            "for both, even at high agreement, so read the counts with it.")
    show = jv.copy()
    show["agreement"] = show["agreement"].map(lambda v: f"{v:.0%}")
    show["kappa"] = show["kappa"].map(num)
    doc.add(md(show.astype(str)))
    if len(dis):
        kinds = dis.assign(origin=dis["useful_chunks"].str.findall(r"\d+: (evidence|same document|other: [^;]+)"))
        kinds = kinds.explode("origin").fillna({"origin": "(no chunk useful)"})
        counts = kinds.groupby(["corpus", "kind", "origin"]).size().reset_index(name="chunks")
        doc.add("Where the judge's useful chunks come from, in the rows where it disagrees with the spans:",
                md(counts.astype(str)))
    doc.add("Spearman's correlation of fc_recall with the judge-free text metrics and with Ragas' factual "
            "correctness (answered rows): a setup where they part ways points at the judge, not the chunker.",
            md(tv.assign(**{c: tv[c].map(num) for c in ("token_f1", "answer_similarity", "rouge_l", "factual_correctness")}).astype(str)))

    be = t["by_evidence"]
    doc.add("### 4.7 With and without the evidence shown",
            "With the evidence, a setup should come near the gold passage; without it, gold claims in the answer "
            "were found elsewhere (another chunk or document holds the same fact) or guessed.")
    for corpus in inputs.corpora:
        sub = be[(be["corpus"] == corpus) & be["metric"].isin(["fc_recall", "abstention", "faithfulness"])]
        rows_ = []
        for (setup, shown), grp in sub.groupby(["setup", "evidence_shown"], sort=False):
            rows_.append({"setup": setup, "evidence shown": "yes" if shown else "no", "rows": rows_of(grp),
                          **{r["metric"]: ci_cell(r) for _, r in grp.iterrows()}})
        doc.add(f"**{corpus}**", md(pd.DataFrame(rows_)))

    sd = t["score_distributions"]
    doc.add("### 4.8 Score distributions",
            "Means hide lumps: context recall on a 1–2 sentence gold answer is 0, 0.5 or 1, and Ragas' factual "
            "correctness is 0 for many correct answers. Shares per setup: `tables/score_distributions.csv`.")
    doc.figure(figs.get("score_distributions"), out, "distributions of the per-question scores")
    pooled = sd[sd["setup"] != GOLD_PASSAGE].assign(z=lambda d: d["zero"] * d["n"], o=lambda d: d["one"] * d["n"])
    pooled = pooled.groupby(["corpus", "metric"], sort=False)[["n", "z", "o"]].sum().reset_index()
    pooled = pooled[pooled["metric"].isin(["span_recall", *COMPARED_GEN, "context_recall", "fc_precision"])]
    doc.add(md(pd.DataFrame([{"corpus": r["corpus"], "metric": r["metric"], "rows": int(r["n"]),
                              "at 0": f"{r['z'] / r['n']:.0%}", "between": f"{1 - (r['z'] + r['o']) / r['n']:.0%}",
                              "at 1": f"{r['o'] / r['n']:.0%}"} for _, r in pooled.iterrows()])))


def breakdowns_section(doc: Doc, inputs: Inputs, t: dict[str, pd.DataFrame], top: str) -> None:
    b = t["breakdowns"]
    doc.add("## 5. Breakdowns",
            f"Means per setup by question type, number of evidence quotes (1 vs 2+: where chunk boundaries should "
            f"matter most), evidence length (terciles per corpus) and kind (when two-span pairs exist). Shown: "
            f"span_hit@{top} and fc_recall (gold passage last); every metric with its interval: "
            f"`tables/breakdowns.csv`. † fewer than {LOW_N} questions.")
    for dim in dict.fromkeys(b["dimension"]):
        doc.add(f"### By {dim.replace('_', ' ')}")
        for side, metric in (("retrieval", "span_hit"), ("generation", "fc_recall")):
            sub = b[(b["dimension"] == dim) & (b["side"] == side) & (b["metric"] == metric)]
            if sub.empty:
                continue
            for corpus in inputs.corpora:
                sc = sub[sub["corpus"] == corpus]
                values = sorted(dict.fromkeys(sc["value"]), key=lambda v: (bin_order(str(v)), str(v)))
                rows = []
                for v in values:
                    sv = sc[sc["value"] == v].set_index("setup")
                    n = rows_of(sv)
                    rows.append({dim: f"{v} (n={n})", **{s: num(sv.loc[s, "mean"]) + ("†" if sv.loc[s, "low_n"] else "")
                                                         for s in [*inputs.setups, GOLD_PASSAGE] if s in sv.index}})
                doc.add(f"**{corpus}, {metric}**", md(pd.DataFrame(rows)))


def tail_sections(doc: Doc, cfg: Config, t: dict[str, pd.DataFrame], costs: dict[str, Any], record: dict[str, Any]) -> None:
    doc.add("## 6. Chunk sizes",
            "In bge-m3 tokens (`data/chunks/<corpus>/_size_stats.json`). Overlap repeats `overlap_tokens_mean` "
            "tokens per chunk on average.")
    cs = t["chunk_sizes"]
    if len(cs):
        cols = [c for c in ("corpus", "setup", "count", "mean", "median", "p5", "p95", "min", "max", "words_mean",
                            "overlap_tokens_mean", "window_chars") if c in cs]
        doc.add(md(cs[cols].round(1).astype(str).replace("nan", "–")))

    doc.add("## 7. Cost")
    if "judge" in costs:
        jt = costs["judge"]["table"]
        doc.add(f"The judge ({costs['judge']['judge']}): **${jt['cost_usd']:.2f}** for the {jt['calls']} calls this "
                f"table rests on, ${jt['cost_per_question_usd']:.4f} per question across its setups "
                f"({jt['input_tokens']:,} input and {jt['output_tokens']:,} output tokens).",
                md(costs["judge_by_prompt"][["prompt", "calls", "input_tokens", "output_tokens", "cost_usd"]].astype(str)))
    if "generator" in costs:
        g = costs["generator"]
        doc.add(f"The generator ({cfg.llm.rag_generator.model}, local): mean tokens per answer.",
                md(g[["corpus", "setup", "prompt_tokens_mean", "completion_tokens_mean", "response_words_mean"]].astype(str)))

    dd = t["detectable_differences"]
    doc.add("## 8. Planning: the smallest difference the data can show",
            "From the spread of the per-question differences between a setup and the baseline (pooled over the "
            f"setups): the smallest true difference found with {80}% probability at alpha / (setups − 1), the level "
            "Holm asks of a family's smallest p-value. `changed`: the share of questions whose score differs from "
            "the baseline's. Use it to size the final run before paying for the judge. A spread of 0 (no question "
            "changed) says nothing.")
    if len(dd):
        doc.add(md(pd.DataFrame([{
            "corpus": r["corpus"], "side": r["side"], "metric": r["metric"], "n": r["n"], "sd": num(r["sd_diff"], 3),
            "changed": f"{r['changed_share']:.0%}", "detectable now": num(r["detectable_now"], 3),
            f"at n={r['n_planned']}": num(r["detectable_planned"], 3),
            "n for 0.05": "–" if math.isnan(r["n_for_0.05"]) else f"{r['n_for_0.05']:.0f}"}
            for _, r in dd[dd["metric"].isin([*HEADLINE_SPAN, *COMPARED_GEN])].iterrows()])))

    au = t.get("audit")
    doc.add("## 9. Questions to read by hand",
            "Before trusting the generation results, read the cards in [cases.md](cases.md): every question the gold "
            "passage abstains on or answers incompletely, the judge's shortfalls on the gold passage, correct answers "
            "Ragas' factual correctness scores 0, the questions the setups differ on, the questions no setup retrieves, "
            "and the judge/span disagreements.")
    if au is not None and len(au):
        doc.add(md(au.groupby(["corpus", "reason"], sort=False)["question_id"].nunique().reset_index(name="questions")))


def secondary_section(doc: Doc, t: dict[str, pd.DataFrame], figs: dict[str, Path], out: Path) -> None:
    doc.add("## Secondary views",
            "Diagnostics for a reviewer, not results: read them to understand the main numbers, not to rank setups. "
            "Their tables are in `secondary/`.")
    doc.add("### S1. Context handed over against span recall",
            "Each setup across k, with each chunker's oracle (the fewest tokens holding the evidence): what a "
            "setup pays in context for what it finds.")
    doc.figure(figs.get("context_vs_recall"), out, "context tokens against span recall")
    if "evidence_rank" in t:
        er = t["evidence_rank"]
        doc.add("### S2. Is evidence at a low rank used?",
                "By the rank of the first chunk overlapping the evidence (none: no top-k chunk does).")
        doc.figure(figs.get("evidence_rank"), out, "scores by evidence rank")
        rows = []
        for (corpus, rank), grp in er.groupby(["corpus", "evidence_rank"], sort=True):
            rows.append({"corpus": corpus, "rank": "none" if rank == 0 else f"{rank:.0f}",
                         "rows": rows_of(grp), **{r["metric"]: ci_cell(r) for _, r in grp.iterrows()}})
        doc.add(md(pd.DataFrame(rows)))
    if "metric_correlations" in t:
        doc.add("### S3. Which metrics carry the same information",
                "Spearman's correlation over the answered rows of the retrieved setups, both corpora pooled.")
        doc.figure(figs.get("metric_correlations"), out, "metric correlations")
    if "length_vs_scores" in t:
        lv = t["length_vs_scores"]
        doc.add("### S4. Are longer answers penalised?",
                "Claim precision and Ragas' factual correctness count extra true detail as wrong; fc_recall doesn't. "
                "`rho`: Spearman's correlation of response words with the metric.")
        doc.figure(figs.get("length_vs_scores"), out, "scores by answer length")
        rho = lv.drop_duplicates(["corpus", "metric"])[["corpus", "metric", "rho"]]
        doc.add(md(rho.assign(rho=rho["rho"].map(num)).astype(str)))


def files_section(doc: Doc, out: Path) -> None:
    doc.add("## Files")
    lines = []
    for folder in ("tables", "figures", "secondary"):
        for p in sorted((out / folder).glob("*")):
            lines.append(f"- `{folder}/{p.name}`")
    doc.add(*lines, "- `cases.md`, `run_record.json`")


def render(cfg: Config, inputs: Inputs, t: dict[str, pd.DataFrame], figs: dict[str, Path], record: dict[str, Any],
           costs: dict[str, Any], summaries: dict[str, Any], out: Path) -> str:
    doc = Doc()
    doc.add(f"# Evaluation report: {cfg.run_id}",
            f"Generated {record['generated']} from `runs/{cfg.run_id}/` by `chunkeval report` · "
            f"`reports/{record['report']}` · code {record['code']}")
    small = {c: v for c, v in record["corpora"].items()
             if v["questions"] < SMALL_SET or v["documents_with_questions"] < MIN_CLUSTER_DOCS}
    if small:
        sizes = ", ".join(f"{c} {v['questions']} questions from {v['documents_with_questions']} documents"
                          for c, v in small.items())
        doc.add(f"> **Small question set** ({sizes}). Every interval is wide and the by-document intervals are "
                "unreliable: read this report as a test of the analysis, not as results about the chunkers.")
    differ = {stage: keys for stage, keys in record["questions_made"]["differ"].items() if keys}
    if differ:
        which = "; ".join(f"{stage}: {', '.join(keys)}" for stage, keys in differ.items())
        doc.add(f"> **The corpora's questions were made with different settings** ({which}; see the run record). "
                "Comparisons within a corpus are unaffected, but a difference between corpora may come from these "
                "settings rather than from the documents.")
    how_to_read(doc, cfg)
    run_record_section(doc, record)
    question_set_section(doc, t, summaries, inputs.corpora)
    retrieval_section(doc, cfg, inputs, t, figs, out)
    if inputs.ragas is not None:
        generation_section(doc, cfg, inputs, t, figs, out)
    else:
        doc.add("## 4. Generation", "No Ragas table yet (`chunkeval eval`): the generation sections are left out.")
    breakdowns_section(doc, inputs, t, str(cfg.retrieval.top_k))
    tail_sections(doc, cfg, t, costs, record)
    secondary_section(doc, t, figs, out)
    files_section(doc, out)
    return doc.text()


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

