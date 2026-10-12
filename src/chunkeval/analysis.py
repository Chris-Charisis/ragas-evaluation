"""Stage 9 — aggregate per-question scores into tables with uncertainty.

Why: in the previous report several differences were <= 0.008 and there was no way to tell them
from noise (docx, last recommendation). With per-question scores on the SAME questions for every
config we can do paired statistics, which are much more sensitive than comparing two means.

Reads:  runs/<run_id>/metrics/retrieval_per_question.parquet   (stage 6) one row per (question_id, embedder,
            corpus, method, overlap, k_label); the baselines are embedder "oracle" / "random"
        runs/<run_id>/ragas/per_question.parquet   (stage 8) one row per (question_id, corpus, setup)
        runs/<run_id>/ragas/details.jsonl, runs/<run_id>/ragas/cost.json, runs/<run_id>/answer_summary.json,
        data/chunks/<corpus>/_size_stats.json
Writes: runs/<run_id>/report/*.csv|md|png

The generation table (stage 8; ragas_eval.py's docstring has the details). Read this before using it:
  * Rows: `setup` is one of the 6 variants or "gold_passage" (each question answered from its own
    passage); `method` / `overlap` are null for gold_passage. Within a corpus every setup has the same
    questions, so every comparison between setups is paired.
  * Score columns (ragas_eval.METRICS): context_precision, context_recall, faithfulness, answer_relevancy,
    factual correctness, and the judge-free rouge_l / token_f1 / answer_similarity. Extra columns from the
    same judge calls: cp_verdicts (by rank, "10010"), cr_sentences/cr_attributed, faith_statements/
    faith_supported, rr_cosine/rr_noncommittal, the fc_* claim counts; flags abstained, answer_error,
    evidence_shown (= span_hit at retrieval.top_k).
  * Factual correctness: report BOTH versions, each with one line on what it measures (decided 2026-10-12):
      - factual_correctness: Ragas' standard f1, comparable with other Ragas reports. The judge checks the
        answer's claims against the gold answer alone, so a true claim the gold answer doesn't state
        (restating the question's framing, naming the source, an extra correct detail) counts as wrong:
        the dev pilot scored complete, correct answers 0 this way.
      - fc_recall: the share of the gold answer's claims the answer contains, i.e. "is the right answer
        in it?". fc_precision: the share of the answer's claims the gold answer supports ("does it say
        only that?", which penalises extra detail); fc_f1 is their harmonic mean.
      - fc_recall_ragas is a diagnostic only (Ragas' recall mixes the two claim sets): leave it out of
        the main tables.
  * Abstentions (decided 2026-10-11): an abstained row has answer_relevancy, every factual-correctness
    score and the text metrics at 0 (set in code, as Ragas would score it), and faithfulness NaN. So:
      - never drop abstained rows from the other metrics (the pairs must stay complete);
      - report faithfulness over the answered rows WITH the abstention rate next to it, plus a sensitivity
        view where abstentions count as faithfulness 1 (an abstention makes no unsupported claim);
      - an answer_error row has every response metric NaN (count them; 0 on the dev docs).
  * NaN otherwise means the judge's output was unusable, or faithfulness found no statements. The
    stage-8 summary counts them; report the count wherever a mean drops them.
  * Small numbers: context_recall on a 1–2 sentence gold answer takes few values (0, 0.5, 1), and
    factual_correctness is rounded to 2 decimals by Ragas: prefer paired differences and the share of
    questions better / equal / worse over raw means when they are close.
"""
from __future__ import annotations

from typing import Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config


def bootstrap_ci(values, n_iter: int, ci: float, rng) -> tuple[float, float, float]:
    """(mean, low, high) by percentile bootstrap over questions.

    TODO(you): `rng = numpy.random.default_rng(seed)`; resample indices with replacement
    (`rng.integers(0, n, size=(n_iter, n))`), take means, `numpy.percentile` at (1-ci)/2 and 1-(1-ci)/2.
    Drop NaNs first — and report how many you dropped.
    """
    raise NotImplementedError


def paired_diff(a, b, n_iter: int, ci: float, rng) -> dict[str, float]:
    """Mean difference (a - b) over the SAME questions with a paired bootstrap CI.

    TODO(you): align a and b by question_id first (inner join), then resample the per-question
    DIFFERENCES. Also return the share of questions where a > b, a == b, a < b — for binary-ish
    metrics like span_hit that is more readable than a mean difference. A Wilcoxon signed-rank test
    (`scipy.stats.wilcoxon`) is an optional complement.
    Multiple comparisons: 5 methods vs baseline × many metrics — be modest when reading p-values.
    NaN (a faithfulness of an abstention, an unusable judge output): drop the question from that pair
    only, and return how many were dropped, so the n behind each difference is visible.
    """
    raise NotImplementedError


def summary_table(df, metrics: list[str], group_by: list[str], cfg: Config):
    """Mean ± CI per (corpus, method, overlap) for each metric — the main results table.

    Hint: the main tables filter `embedder == cfg.embedding.primary` (bge_m3); the same
    function with `embedder` in `group_by` produces the side-by-side retrieval table.
    Generation table: group by (corpus, setup), with gold_passage as the last row of each corpus (the
    ceiling, see `generation_views`); put the abstention rate in a column of its own, next to
    faithfulness, and both factual-correctness versions side by side (module docstring).
    """
    raise NotImplementedError


def embedder_robustness(df, metric: str, cfg: Config) -> dict[str, Any]:
    """Do the chunker conclusions hold under the secondary embedder (Qwen3) too?

    TODO(you) — on the span metrics (the only ones computed for both embedders), per corpus:
      - Ranking agreement: rank the 6 configs by mean `metric` under each embedder and compare
        (`scipy.stats.kendalltau`; with 6 items also just show both rankings side by side —
        a single swapped pair of near-tied configs is not a contradiction).
      - Per config: paired diff qwen3 − bge_m3 on the same questions (`paired_diff`) → "how much
        would changing the embedder buy?"
      - Compare that with the best chunker's gain over the baseline under the primary embedder →
        "is the embedder or the chunker the bigger lever?" — the question this check exists for.
      - Interaction: does the gap between methods shrink or grow with the stronger embedder?
        (A better embedder often compensates for poor chunk boundaries.)
    """
    raise NotImplementedError


def breakdowns(df, cfg: Config) -> dict[str, Any]:
    """Same tables split by question_type, kind (single vs two_span) and section.

    Hint: the two_span slice is where overlap and recursive boundaries should matter most — if
    they don't show a difference there, they won't anywhere.
    Generation metrics: also split by `evidence_shown` (did retrieval hand over the evidence?). With
    it, the setups' generation scores should approach the gold passage's; without it, a high
    factual_correctness / fc_recall is a hallucination or another document holding the same fact (the
    dev pilot's report_15/report_16 case), and an abstention is the right behaviour.
    """
    raise NotImplementedError


def generation_views(ragas_df, cfg: Config) -> dict[str, Any]:
    """What the gold passage and the abstentions add to the generation table.

    TODO(you), per corpus and metric, paired on the same questions:
      - Ceiling: setup − gold_passage (`paired_diff`) is what retrieval with that chunker costs;
        1 − gold_passage is the generator's and the judge's own error. A metric on which gold_passage
        doesn't beat the setups doesn't respond to retrieval: flag it (the previous round's answer
        relevancy was ~0.54 everywhere).
      - Judge error: gold_passage's context_recall should be ~1 by construction (its one context holds
        all the evidence), and so should its context_precision (one chunk, which is the passage): the
        shortfall estimates the judge's error rate. Read some of those rows in details.jsonl.
      - Abstentions: the abstention rate per setup; faithfulness over answered rows and the view with
        abstentions = 1; the share of abstentions with `evidence_shown` (an over-cautious generator)
        vs answers without it (hallucination risk) — answer_summary.json has the counts.
      - Text metrics vs the judge: the correlation of token_f1 / answer_similarity with fc_recall per
        setup; a large disagreement points at the judge, not the chunker.
    """
    raise NotImplementedError


def audit_lists(ragas_df, details_path: str) -> dict[str, Any]:
    """Questions to read by hand before trusting the generation results (for the README audit).

    TODO(you): from the gold_passage rows, list with the question, gold answer and response (from
    details.jsonl):
      - every question the gold passage abstains on: mostly weak Q-A pairs (7 of 35 on the dev docs);
      - gold rows with fc_recall < 1 or context_recall < 1: a judge error or a gold answer the passage
        doesn't support;
      - rows where factual_correctness is 0 but fc_recall is 1 (correct answers Ragas' f1 rejects).
    """
    raise NotImplementedError


def plots(tables: dict[str, Any], out_dir: str) -> None:
    """Optional figures: dot plot with CI whiskers per metric (methods on y, score on x),
    one panel per corpus. Hint: matplotlib `errorbar`; keep colours consistent per method."""
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """Load both long tables, build summary + paired diffs vs cfg.analysis.baseline +
    breakdowns + embedder robustness + chunk size stats, write CSV/Markdown so the docx can be
    rewritten from them.

    TODO(you), for the generation side:
      - Joining with retrieval: the retrieval rows with embedder == cfg.embedding.primary and
        k_label == str(cfg.retrieval.top_k) match the ragas rows on (question_id, corpus, method,
        overlap) — what the generator was shown. gold_passage rows have no retrieval row.
      - Context precision/recall (judge) vs span_hit/span_recall (deterministic, stage 6) measure the
        same thing two ways: report their agreement per setup, it validates the judge for free. Where
        they disagree, the judge is often right: on the dev docs "some chunk useful" agreed with span_hit
        in 91% of the rows, and 15 of the 18 disagreements were chunks holding the fact OUTSIDE the
        gold evidence (elsewhere in the document, or in report_16, the near-duplicate of report_15),
        which the span metrics count as misses. List those questions rather than call them errors.
      - Which metrics respond to retrieval (dev docs, answered rows, gold passage vs setups):
        fc_recall 0.90 vs 0.82 (0.86 with the evidence shown, 0.06 without) does; faithfulness (0.96 vs
        0.94, near its ceiling) and answer_relevancy (0.79 vs 0.81) don't. Check this on the full corpus
        before reading chunker differences into them.
      - Cost: report the judge's cost (ragas/cost.json: "table") next to the results.
      - Don't rank chunkers on a metric whose paired CIs all include 0 (the dev docs: every one).
    """
    raise NotImplementedError
