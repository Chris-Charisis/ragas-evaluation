"""Stage 6b — deterministic, LLM-free retrieval metrics from character spans.

Why: every question carries evidence char spans (qa/generation.py) and every retrieved chunk
carries (doc_id, start, end). So "did retrieval find the evidence?" is a set-overlap computation —
exact, free, and computable for ALL questions × 12 configs (+ size sweep), unlike Ragas which
needs judge calls. These are the primary signal for choosing a chunker; Ragas then explains it.

Inspired by Chroma's chunking-evaluation methodology (token-level IoU / precision / recall between
retrieved text and the relevant excerpts) — read their technical report for the definitions.

Computed for BOTH embedders (bge_m3 primary, qwen3_06b secondary) — this is the only stage where
the secondary embedder is evaluated, because it is free (no LLM calls).

Reads:  data/qa/filtered.jsonl, runs/<run_id>/retrieval/<embedder>/*.jsonl
Writes: runs/<run_id>/metrics/retrieval_per_question.parquet (long format, with an `embedder` column)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config

if TYPE_CHECKING:
    from chunkeval.qa.generation import QAPair
    from chunkeval.evaluation.retrieval import Hit

Interval = tuple[int, int]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Union of half-open intervals, sorted and merged.

    Why: with overlap, retrieved chunks overlap each other; counting their chars twice would
    inflate "retrieved length" and deflate precision. Always merge before measuring.
    """
    raise NotImplementedError


def overlap_len(a: list[Interval], b: list[Interval]) -> int:
    """Total length of the intersection of two merged interval lists (two-pointer sweep)."""
    raise NotImplementedError


def question_metrics(qa: "QAPair", hits: list["Hit"], k: int) -> dict[str, float]:
    """All span metrics for one question at cutoff k.

    TODO(you) — define precisely (and write the definitions in the report):
      - doc_hit@k: any of the top-k hits is from qa.doc_id.
      - span_hit@k: any top-k hit overlaps any evidence span (only hits from the same doc count!).
      - span_recall@k: |evidence ∩ union(top-k)| / |evidence|  (chars).
      - all_evidence@k (two-span only): every evidence span is touched by some hit — the metric
        that most directly shows the effect of overlap and section boundaries.
      - mrr: 1 / rank of the first overlapping hit (0 if none in k_max).
      - precision@k: |evidence ∩ union(top-k)| / |union(top-k)| — chars of useful text over all
        text handed to the generator.
      - iou@k: |∩| / |evidence ∪ union(top-k)|.
      - context_tokens@k: sum of n_tokens of the top-k (cost / budget).
      Chars vs tokens: chars are simpler and fine for comparisons between methods; Chroma uses
      tokens. Pick one and be consistent.
    """
    raise NotImplementedError


def budget_k(hits: list["Hit"], budget_tokens: int) -> int:
    """Largest k whose cumulative n_tokens stays <= budget_tokens (at least 1).

    Why: at fixed k=5, methods with smaller chunks hand the generator less text, so they are
    handicapped on recall (or advantaged on precision). The budget-matched view compares methods
    at (approximately) equal context size — report both views.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """Compute metrics at k in {1, 3, 5, 10, budget} for every (question, config); write long table.

    Hint: one row per (question_id, embedder, corpus, method, overlap, k_label) with all metrics as
    columns + question attributes (question_type, kind, section) for the breakdowns in analysis.py.
    Oracle/random baselines get embedder = "none".
    Budget-matched k uses n_tokens from the size tokenizer (bge-m3) for both embedders — the
    budget is about how much text reaches the generator, not about the embedder.
    """
    raise NotImplementedError
