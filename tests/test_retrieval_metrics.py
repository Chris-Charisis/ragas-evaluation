"""Span metrics on hand-built toy cases — compute the expected numbers by hand first."""
import pytest


def test_merge_intervals():
    """[(0,10),(5,20),(30,40)] -> [(0,20),(30,40)]; touching intervals (end == start) merge too."""
    pytest.skip("TODO(you)")


def test_overlap_len():
    """Intersection length of two merged interval lists, including the empty case."""
    pytest.skip("TODO(you)")


def test_oracle_retriever_scores_one():
    """A hit exactly equal to the evidence span: span_recall = precision = iou = 1, mrr = 1."""
    pytest.skip("TODO(you)")


def test_wrong_document_scores_zero():
    """Hits with the same offsets but a DIFFERENT doc_id must not count as overlap."""
    pytest.skip("TODO(you)")


def test_overlapping_hits_not_double_counted():
    """Two retrieved chunks overlapping each other: precision uses the merged union length."""
    pytest.skip("TODO(you)")


def test_two_span_all_evidence():
    """all_evidence@k is 1 only when BOTH evidence spans are touched."""
    pytest.skip("TODO(you)")


def test_budget_k():
    """budget_k picks the largest k within the token budget and never returns 0."""
    pytest.skip("TODO(you)")
