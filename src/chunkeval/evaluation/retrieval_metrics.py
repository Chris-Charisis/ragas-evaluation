"""Stage 6b — deterministic, LLM-free retrieval metrics from character spans.

Why: every question carries evidence char spans (qa/generation.py) and every retrieved chunk
carries (doc_id, start, end). So "did retrieval find the evidence?" is a set-overlap computation —
exact, free, and computable for ALL questions × 12 configs (+ size sweep), unlike Ragas which
needs judge calls. These are the primary signal for choosing a chunker; Ragas then explains it.

Inspired by Chroma's chunking-evaluation methodology (token-level IoU / precision / recall between
retrieved text and the relevant excerpts). Here the unit is the CHARACTER, not the token: the
evidence spans and chunk offsets are character offsets into the same canonical text, so the
overlaps are exact and need no tokenizer, and comparisons between methods are unaffected. Only
context_tokens counts bge-m3 tokens (the chunks' n_tokens), because it measures what the
generator is handed.

Computed for BOTH embedders (bge_m3 primary, qwen3_06b secondary) — this is the only stage where
the secondary embedder is evaluated, because it is free (no LLM calls) — and for the two
baselines of retrieval.py (oracle, random), which go in the same `embedder` column.

Reads:  data/qa/filtered.jsonl, runs/<run_id>/retrieval/<retriever>/*.jsonl (+ their provenance)
Writes: runs/<run_id>/metrics/retrieval_per_question.parquet (long format, see `run`)
        runs/<run_id>/retrieval_summary.json (means per corpus, retriever, variant and k)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config
from chunkeval.core.store import run_dir, write_json, write_table

if TYPE_CHECKING:
    from chunkeval.qa.generation import QAPair
    from chunkeval.evaluation.retrieval import Hit

log = logging.getLogger(__name__)

Interval = tuple[int, int]

# The metric columns of every row (see `question_metrics` for the definitions).
METRICS = ["doc_hit", "span_hit", "all_evidence", "span_recall", "precision", "iou", "mrr", "context_tokens"]
# What identifies a row of the long table; a repeat would count a question twice in analysis.py.
ROW_KEY = ["question_id", "embedder", "corpus", "method", "overlap", "k_label"]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Union of half-open intervals, sorted and merged; touching intervals (end == start) merge
    too, and empty ones are dropped.

    Why: with overlap, retrieved chunks overlap each other; counting their chars twice would
    inflate "retrieved length" and deflate precision. Always merge before measuring.
    """
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_len(intervals: list[Interval]) -> int:
    """Total length of a merged interval list."""
    return sum(end - start for start, end in intervals)


def overlap_len(a: list[Interval], b: list[Interval]) -> int:
    """Total length of the intersection of two merged interval lists (two-pointer sweep)."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        # advance whichever ends first: it can't overlap anything further in the other list
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def evidence_intervals(qa: "QAPair") -> list[Interval]:
    """The pair's evidence quotes as merged char intervals (quotes that touch become one)."""
    evidence = merge_intervals([(e.start, e.end) for e in qa.evidence])
    if not evidence:
        raise ValueError(f"{qa.id}: no evidence span, so no span metric is defined")
    return evidence


def question_metrics(qa: "QAPair", hits: list["Hit"], k: int) -> dict[str, float]:
    """All span metrics for one question on its top-k hits (fewer if the list is shorter: the
    oracle stops once the evidence is covered). E = the merged evidence intervals, R = the merged
    union of the top-k chunks (per document: offsets of different documents never overlap).

      - k:              the number of hits actually used.
      - doc_hit:        some top-k hit is from qa.doc_id.
      - span_hit:       some top-k hit overlaps the evidence (only hits from the same doc count).
      - all_evidence:   every interval of E is touched by some top-k hit. Equal to span_hit when
                        the evidence is one interval; with several quotes, it shows whether the
                        chunks split the evidence apart (overlap and section boundaries).
      - span_recall:    |E ∩ R| / |E|                  share of the evidence handed over.
      - precision:      |E ∩ R| / |R|                  useful chars over all chars handed over,
                        hits from other documents included in |R|.
      - iou:            |E ∩ R| / |E ∪ R|.
      - mrr:            1 / rank of the first hit that overlaps the evidence, 0 if none in the
                        top k (the mean over questions is MRR@k).
      - context_tokens: sum of n_tokens over the top-k (the generator's cost / budget). Chunks that
                        overlap each other count fully: the generator reads the repeated text twice.
    """
    evidence = evidence_intervals(qa)
    top = hits[:k]
    by_doc: dict[str, list[Interval]] = defaultdict(list)
    for h in top:
        by_doc[h.doc_id].append((h.start, h.end))
    retrieved = {doc: merge_intervals(spans) for doc, spans in by_doc.items()}
    retrieved_chars = sum(total_len(spans) for spans in retrieved.values())
    same_doc = retrieved.get(qa.doc_id, [])
    shared = overlap_len(evidence, same_doc)
    evidence_chars = total_len(evidence)
    first = next((rank for rank, h in enumerate(top, start=1)
                  if h.doc_id == qa.doc_id and overlap_len(evidence, [(h.start, h.end)])), None)
    return {
        "k": len(top),
        "doc_hit": float(qa.doc_id in retrieved),
        "span_hit": float(shared > 0),
        "all_evidence": float(all(overlap_len([span], same_doc) for span in evidence)),
        "span_recall": shared / evidence_chars,
        "precision": shared / retrieved_chars if retrieved_chars else 0.0,
        "iou": shared / (evidence_chars + retrieved_chars - shared),
        "mrr": 1 / first if first else 0.0,
        "context_tokens": sum(h.n_tokens for h in top),
    }


def budget_k(hits: list["Hit"], budget_tokens: int) -> int:
    """Largest k whose cumulative n_tokens stays <= budget_tokens (at least 1).

    Why: at fixed k=5, methods with smaller chunks hand the generator less text, so they are
    handicapped on recall (or advantaged on precision). The budget-matched view compares methods
    at (approximately) equal context size — report both views.
    """
    total = 0
    for k, h in enumerate(hits):
        total += h.n_tokens
        if total > budget_tokens:
            return max(k, 1)
    return max(len(hits), 1)


def question_attributes(qa: "QAPair") -> dict[str, Any]:
    """The question's own columns, for the breakdowns in analysis.py."""
    evidence = evidence_intervals(qa)
    return {"doc_id": qa.doc_id, "question_type": qa.question_type, "kind": qa.kind,
            "section": " > ".join(qa.meta.get("section_path", [])),
            "n_evidence": len(evidence), "evidence_chars": total_len(evidence)}


def file_rows(qa_pairs: list["QAPair"], hits: dict[str, list["Hit"]], retriever: str, corpus: str,
              method: str, overlap: bool, k_values: list[int], budget_tokens: int) -> list[dict[str, Any]]:
    """The rows of one retrieval file: per question, one row per k in `k_values` and one for the
    budget-matched k (k_label "budget")."""
    rows = []
    for qa in qa_pairs:
        ranked = hits[qa.id]
        cuts = [(str(k), k) for k in k_values] + [("budget", budget_k(ranked, budget_tokens))]
        for label, k in cuts:
            rows.append({"question_id": qa.id, "embedder": retriever, "corpus": corpus, "method": method,
                         "overlap": overlap, "k_label": label, **question_metrics(qa, ranked, k),
                         **question_attributes(qa)})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means per corpus → retriever → variant → k_label (with the question count), for the log
    and a first look; analysis.py adds the confidence intervals."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        variant = f"{r['method']}_{'overlap' if r['overlap'] else 'no_overlap'}"
        groups[(r["corpus"], r["embedder"], variant, r["k_label"])].append(r)
    summary: dict[str, Any] = {}
    for (corpus, retriever, variant, label), group in groups.items():
        means = {m: round(sum(r[m] for r in group) / len(group), 4) for m in ["k", *METRICS]}
        summary.setdefault(corpus, {}).setdefault(retriever, {}).setdefault(variant, {})[label] = \
            {"questions": len(group), **means}
    return summary


def run(cfg: Config, secrets: Secrets, corpora: list[str], embedders: list[str]) -> None:
    """CLI entry (second half of `chunkeval retrieve`): score every retrieval file into one long
    table, one row per (question_id, embedder, corpus, method, overlap, k_label), k_label in
    retrieval.k_values plus "budget"; the metrics as columns, then the question's attributes
    (doc_id, question_type, kind, section, n_evidence, evidence_chars). The baselines are rows
    with embedder "oracle" / "random".

    The table covers the whole configured grid (every corpus × retrieval.embedders + baselines),
    not only what this invocation retrieved, so `retrieve --embedder qwen3_06b` doesn't drop the
    bge_m3 rows. A file of `corpora` × (`embedders` + baselines) must exist and be up to date
    (retrieval.run just made it); any other file that is missing or stale (made before the
    questions, chunks or query settings changed) is left out with a warning naming the command.
    Budget-matched k uses n_tokens from the size tokenizer (bge-m3) for every retriever — the
    budget is about how much text reaches the generator, not about the embedder.
    """
    # Imported here: retrieval.py imports this module's interval helpers for its oracle.
    from chunkeval.evaluation.retrieval import (BASELINES, file_provenance, load_hits, load_questions,
                                                retrieval_path, stored_provenance, variant_chunks, variants)

    questions = load_questions(cfg)
    in_scope = {(c, r) for c in corpora for r in [*embedders, *BASELINES]}
    r = cfg.retrieval
    retrievers = list(dict.fromkeys([*r.embedders, *embedders, *BASELINES]))
    rows: list[dict[str, Any]] = []
    left_out: list[str] = []
    truncated = 0
    for corpus in cfg.paths.corpora:
        pairs = questions.get(corpus, [])
        if not pairs:
            continue
        for method, overlap, variant in variants(cfg):
            sha = variant_chunks(cfg, corpus, variant)[1]["chunks_sha256"]
            for retriever in retrievers:
                path = retrieval_path(cfg, retriever, corpus, variant)
                if stored_provenance(path) != file_provenance(cfg, retriever, corpus, variant, sha, pairs):
                    if (corpus, retriever) in in_scope:
                        raise RuntimeError(f"{path}: missing or stale right after retrieval; re-run `chunkeval retrieve`")
                    left_out.append(f"{retriever} {corpus}/{variant}")
                    continue
                hits = load_hits(path)
                rows += file_rows(pairs, hits, retriever, corpus, method, overlap, r.k_values, r.budget_tokens)
                # a budget k that took every hit may have been cut short by k_max
                truncated += sum(len(hits[qa.id]) == r.k_max and budget_k(hits[qa.id], r.budget_tokens) == r.k_max
                                 for qa in pairs)
    if left_out:
        log.warning("left out of the table, missing or stale (run `chunkeval retrieve` for them): %s",
                    ", ".join(left_out))
    if truncated:
        log.warning("%d question(s) fit all %d hits into budget_tokens %d: their budget view is capped "
                    "by k_max, raise retrieval.k_max", truncated, r.k_max, r.budget_tokens)

    path = write_table(run_dir(cfg, "metrics") / "retrieval_per_question.parquet", rows, unique_by=ROW_KEY)
    summary = summarize(rows)
    write_json(run_dir(cfg) / "retrieval_summary.json", summary)
    log.info("%d rows written to %s", len(rows), path)
    top = str(r.top_k)
    for corpus, by_retriever in summary.items():
        for retriever, by_variant in by_retriever.items():
            for variant, by_k in by_variant.items():
                at, budget = by_k[top], by_k["budget"]
                log.info("%-7s %-9s %-21s @%s: hit %.2f recall %.2f precision %.2f mrr %.2f %4.0f tokens | "
                         "budget: k %.1f recall %.2f precision %.2f", corpus, retriever, variant, top,
                         at["span_hit"], at["span_recall"], at["precision"], at["mrr"], at["context_tokens"],
                         budget["k"], budget["span_recall"], budget["precision"])
