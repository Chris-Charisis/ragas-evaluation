"""Chunker invariants — the evaluation is only valid if these hold for all 3 methods x overlap.

Hint: `@pytest.mark.parametrize("method", ["character", "sentence", "recursive"])` and
`@pytest.mark.parametrize("overlap", [False, True])` run each test on all 6 configurations.
"""
import pytest


def test_chunk_text_matches_offsets():
    """For every chunk: chunk.text == doc.text[chunk.start:chunk.end]."""
    pytest.skip("TODO(you)")


def test_chunks_cover_document():
    """The union of chunk spans covers all non-whitespace characters of doc.text."""
    pytest.skip("TODO(you)")


def test_no_chunk_exceeds_max_tokens():
    """n_tokens (bge-m3 tokenizer) <= cfg chunking.max_tokens — the experimental cap that keeps
    size distributions comparable (bge-m3 itself would accept up to 8192)."""
    pytest.skip("TODO(you)")


def test_no_overlap_means_disjoint():
    """With overlap=False, consecutive spans do not overlap."""
    pytest.skip("TODO(you)")


def test_overlap_within_budget():
    """With overlap=True, each overlap is > 0 and <= overlap_ratio * target (+ one sentence slack
    for the sentence-based methods)."""
    pytest.skip("TODO(you)")


def test_sentence_boundaries_respected():
    """Sentence/recursive chunks start and end on sentence boundaries — test with the tricky
    abbreviations ("e.g.", "et al.", "Fig. 3", "3.5 t/ha") that broke the old regex splitter."""
    pytest.skip("TODO(you)")


def test_recursive_never_crosses_sections():
    """Every recursive chunk lies inside one section group (merged tiny siblings allowed)."""
    pytest.skip("TODO(you)")


def test_size_stats_reported():
    """size_stats returns count/mean/p95/max and mean is close to target for the character method."""
    pytest.skip("TODO(you)")
