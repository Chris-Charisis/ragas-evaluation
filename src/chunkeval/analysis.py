"""Stage 9 — aggregate per-question scores into tables with uncertainty.

Why: in the previous report several differences were <= 0.008 and there was no way to tell them
from noise (docx, last recommendation). With per-question scores on the SAME questions for every
config we can do paired statistics, which are much more sensitive than comparing two means.

Reads:  runs/<run_id>/metrics/retrieval_per_question.parquet, runs/<run_id>/ragas/per_question.parquet,
        data/chunks stats
Writes: runs/<run_id>/report/*.csv|md|png
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
    """
    raise NotImplementedError


def summary_table(df, metrics: list[str], group_by: list[str], cfg: Config):
    """Mean ± CI per (corpus, method, overlap) for each metric — the main results table.

    Hint: the main tables filter `embedder == cfg.embedding.primary` (bge_m3); the same
    function with `embedder` in `group_by` produces the side-by-side retrieval table.
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
    """
    raise NotImplementedError


def plots(tables: dict[str, Any], out_dir: str) -> None:
    """Optional figures: dot plot with CI whiskers per metric (methods on y, score on x),
    one panel per corpus. Hint: matplotlib `errorbar`; keep colours consistent per method."""
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """Load both long tables, build summary + paired diffs vs cfg.analysis.baseline +
    breakdowns + embedder robustness + chunk size stats, write CSV/Markdown so the docx can be
    rewritten from them."""
    raise NotImplementedError
