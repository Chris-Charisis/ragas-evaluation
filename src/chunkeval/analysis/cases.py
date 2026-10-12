"""cases.md: one readable card per question worth reading by hand, so a reviewer sees examples, not
only means. A question gets a card when (reasons in this order, every reason it has is listed):
  - an audit reason (views.audit_lists): the gold passage abstains or misses gold claims, the judge
    scores the gold passage's context below 1, or Ragas' factual correctness scores a correct answer 0;
  - the setups differ on span_hit at top_k or on fc_recall: the questions every chunker comparison
    rests on;
  - no setup retrieves its evidence at top_k;
  - the judge and the spans disagree on whether a retrieved chunk helps (views.judge_vs_spans).
Each card: the question, the gold answer and its evidence quotes; per setup, the top-k chunks (✓ =
overlaps the evidence, · = same document elsewhere, else the other document's id), the judge's
verdicts by rank, the scores and the outcome; then the answers, identical ones grouped.
"""
from __future__ import annotations

import math

import pandas as pd

from chunkeval.analysis.inputs import Inputs
from chunkeval.analysis.views import OUTCOMES
from chunkeval.core.schema import Config
from chunkeval.evaluation.answering import GOLD_PASSAGE
from chunkeval.evaluation.retrieval_metrics import evidence_intervals, overlap_len

REASONS = ["gold passage abstains", "gold passage: answer misses gold claims", "gold passage: context recall < 1",
           "gold passage: context precision < 1", "correct answer, Ragas factual correctness 0",
           "setups differ on span_hit", "setups differ on fc_recall", "no setup retrieves the evidence",
           "judge and spans disagree"]


def card_reasons(inputs: Inputs, audit: pd.DataFrame, discriminating: pd.DataFrame,
                 disagreements: pd.DataFrame | None) -> dict[str, list[str]]:
    """question_id -> its reasons for a card, in REASONS order."""
    reasons: dict[str, set[str]] = {}

    def add(ids, reason):
        for qid in ids:
            reasons.setdefault(qid, set()).add(reason)

    for reason, grp in audit.groupby("reason"):
        add(grp["question_id"], reason)
    d = discriminating
    add(d[(d["metric"] == "span_hit") & (d["class"] == "setups differ")]["question_id"], "setups differ on span_hit")
    add(d[(d["metric"] == "fc_recall") & (d["class"] == "setups differ")]["question_id"], "setups differ on fc_recall")
    add(d[(d["metric"] == "span_hit") & (d["class"] == "all 0.0")]["question_id"], "no setup retrieves the evidence")
    if disagreements is not None and len(disagreements):
        add(disagreements["question_id"], "judge and spans disagree")
    return {qid: [r for r in REASONS if r in rs] for qid, rs in reasons.items()}


def chunk_marks(inputs: Inputs, qid: str, setup: str, k: int) -> str:
    """The top-k chunks of a setup as "1✓ 2· 3 report_16 …"."""
    qa = inputs.questions[qid]
    evidence = evidence_intervals(qa)
    marks = []
    for h in inputs.hits[(qa.corpus, setup)][qid][:k]:
        if h.doc_id != qa.doc_id:
            marks.append(f"{h.rank} {h.doc_id}")
        else:
            marks.append(f"{h.rank}{'✓' if overlap_len(evidence, [(h.start, h.end)]) else '·'}")
    return " ".join(marks)


def fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, (bool,)) or str(value) in ("True", "False"):
        return "yes" if value else "no"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def one_line(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def card(inputs: Inputs, cfg: Config, qid: str, reasons: list[str], g: pd.DataFrame | None) -> list[str]:
    qa = inputs.questions[qid]
    k = cfg.retrieval.top_k
    lines = [f"### {qid}", "",
             f"{qa.doc_id} · {qa.question_type} · {len(evidence_intervals(qa))} evidence quote(s) · "
             f"listed for: {'; '.join(reasons)}", "",
             f"**Question:** {one_line(qa.question)}", "", f"**Gold answer:** {one_line(qa.answer)}", "",
             "**Evidence:**", ""]
    lines += [f"> {one_line(e.quote)}" for e in qa.evidence] + [""]

    rows = None if g is None else g[g["question_id"] == qid].set_index("setup")
    scored = rows is not None and len(rows) > 0
    columns = ["setup", f"top-{k} chunks"]
    if scored:
        columns += ["judge useful", "span hit", "outcome", "fc_recall", "Ragas FC", "faithfulness", "relevancy"]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for setup in [*inputs.setups, *([GOLD_PASSAGE] if scored else [])]:
        cells = [setup, "its own passage" if setup == GOLD_PASSAGE else chunk_marks(inputs, qid, setup, k)]
        if scored:
            r = rows.loc[setup]
            cells += [r["cp_verdicts"], fmt(r["span_hit"], 0), OUTCOMES[r["outcome"]].split(": ", 1)[-1],
                      fmt(r["fc_recall"]), fmt(r["factual_correctness"]), fmt(r["faithfulness"]),
                      fmt(r["answer_relevancy"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if g is not None and not scored:
        lines += ["Not in the Ragas sample: retrieval only.", ""]

    if scored:
        answers: dict[str, list[str]] = {}
        for setup in [*inputs.setups, GOLD_PASSAGE]:
            if setup in rows.index:
                answers.setdefault(one_line(inputs.details[(qid, setup)]["response"]) or "(no answer)", []).append(setup)
        lines += ["**Answers:**", ""]
        lines += [f"- *{', '.join(setups)}:* {text}" for text, setups in answers.items()]
        lines.append("")
    return lines


def casebook(inputs: Inputs, cfg: Config, reasons: dict[str, list[str]], g: pd.DataFrame | None) -> str:
    """The whole cases.md: a legend, an index by reason, then the cards by corpus and reason."""
    order = sorted(reasons, key=lambda q: (inputs.corpora.index(inputs.questions[q].corpus),
                                           REASONS.index(reasons[q][0]), q))
    lines = ["# Cases to read", "",
             f"Run `{cfg.run_id}`: {len(order)} question(s) worth reading by hand (the reasons are in "
             f"src/chunkeval/analysis/cases.py). Chunks: the top {cfg.retrieval.top_k} of "
             f"{cfg.embedding.primary} in rank order; ✓ overlaps the gold evidence, · is the same document "
             "elsewhere, otherwise the other document's id. Judge useful: the context-precision verdict per "
             "rank (1 = useful). Outcome and fc_recall: see report.md, \"Where the answers end up\".", ""]
    for reason in REASONS:
        ids = [q for q in order if reason in reasons[q]]
        if ids:
            lines.append(f"- **{reason}** ({len(ids)}): " + ", ".join(f"[{q}](#{anchor(q)})" for q in ids))
    lines.append("")
    for corpus in inputs.corpora:
        ids = [q for q in order if inputs.questions[q].corpus == corpus]
        if ids:
            lines += [f"## {corpus}", ""]
            for q in ids:
                lines += card(inputs, cfg, q, reasons[q], g)
    return "\n".join(lines) + "\n"


def anchor(heading: str) -> str:
    """GitHub-style anchor of a "### <question_id>" heading."""
    return "".join(ch for ch in heading.lower() if ch.isalnum() or ch in "-_ ").replace(" ", "-")
