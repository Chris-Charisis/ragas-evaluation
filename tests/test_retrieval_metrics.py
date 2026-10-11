"""Span metrics on hand-built toy cases — the expected numbers are computed by hand in each test."""
import pytest

from chunkeval.evaluation.retrieval import Hit
from chunkeval.evaluation.retrieval_metrics import budget_k, merge_intervals, overlap_len, question_metrics
from chunkeval.qa.generation import EvidenceSpan, QAPair


def pair(*spans, doc="doc_a"):
    """A question on `doc` whose evidence quotes sit at `spans`."""
    return QAPair(id="q1", doc_id=doc, corpus="papers", unit_id="u1", question="What?", answer="That.",
                  question_type="factoid", evidence=[EvidenceSpan(s, e, "quote", "exact") for s, e in spans])


def hit(start, end, doc="doc_a", n_tokens=100):
    return Hit(rank=0, chunk_id=f"{doc}:{start}", doc_id=doc, start=start, end=end, n_tokens=n_tokens, score=0.0)


def test_merge_intervals():
    """[(0,10),(5,20),(30,40)] -> [(0,20),(30,40)]; touching intervals (end == start) merge too."""
    assert merge_intervals([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert merge_intervals([(5, 8), (0, 5)]) == [(0, 8)]                    # unsorted, touching
    assert merge_intervals([(0, 30), (5, 10)]) == [(0, 30)]                 # contained
    assert merge_intervals([(3, 3), (7, 9)]) == [(7, 9)]                    # empty interval dropped
    assert merge_intervals([]) == []


def test_overlap_len():
    """Intersection length of two merged interval lists, including the empty case."""
    assert overlap_len([(0, 10), (20, 30)], [(5, 25)]) == 10                # 5 + 5
    assert overlap_len([(5, 25)], [(0, 10), (20, 30)]) == 10                # symmetric
    assert overlap_len([(0, 10)], [(10, 20)]) == 0                          # touching shares nothing
    assert overlap_len([(0, 100)], [(10, 20), (30, 40), (90, 120)]) == 30
    assert overlap_len([], [(0, 10)]) == overlap_len([(0, 10)], []) == 0


def test_oracle_retriever_scores_one():
    """A hit exactly equal to the evidence span: span_recall = precision = iou = 1, mrr = 1."""
    m = question_metrics(pair((100, 200)), [hit(100, 200)], k=5)
    assert m == {"k": 1, "doc_hit": 1.0, "span_hit": 1.0, "all_evidence": 1.0, "span_recall": 1.0,
                 "precision": 1.0, "iou": 1.0, "mrr": 1.0, "context_tokens": 100}


def test_wrong_document_scores_zero():
    """Hits with the same offsets but a DIFFERENT doc_id must not count as overlap."""
    m = question_metrics(pair((100, 200)), [hit(100, 200, doc="doc_b")], k=5)
    assert m["doc_hit"] == m["span_hit"] == m["all_evidence"] == 0
    assert m["span_recall"] == m["precision"] == m["iou"] == m["mrr"] == 0
    assert m["context_tokens"] == 100                                       # still handed to the generator


def test_overlapping_hits_not_double_counted():
    """Two retrieved chunks overlapping each other: precision uses the merged union length."""
    m = question_metrics(pair((100, 150)), [hit(100, 200), hit(150, 250)], k=5)
    assert m["span_recall"] == 1.0
    assert m["precision"] == pytest.approx(50 / 150)                       # union (100, 250), not 200 chars
    assert m["iou"] == pytest.approx(50 / 150)
    assert m["context_tokens"] == 200                                       # the generator reads both chunks


def test_two_span_all_evidence():
    """all_evidence@k is 1 only when BOTH evidence spans are touched."""
    qa = pair((0, 50), (500, 550))
    one = question_metrics(qa, [hit(0, 100)], k=5)
    assert (one["span_hit"], one["all_evidence"], one["span_recall"]) == (1.0, 0.0, 0.5)
    both = question_metrics(qa, [hit(0, 100), hit(480, 600)], k=5)
    assert (both["all_evidence"], both["span_recall"]) == (1.0, 1.0)
    # quotes that touch are one interval: a single hit covering it touches "all" evidence
    assert question_metrics(pair((0, 50), (50, 80)), [hit(60, 70)], k=5)["all_evidence"] == 1.0


def test_precision_and_iou_count_other_documents():
    """Text from another document reaches the generator too: it is in |R| but never overlaps."""
    m = question_metrics(pair((0, 100)), [hit(0, 100), hit(0, 100, doc="doc_b")], k=5)
    assert (m["span_recall"], m["precision"], m["iou"]) == (1.0, 0.5, 0.5)


def test_mrr_and_hit_only_see_the_top_k():
    hits = [hit(0, 50, doc="doc_b"), hit(300, 400), hit(100, 200)]          # the match is at rank 3
    qa = pair((120, 180))
    at1, at3 = question_metrics(qa, hits, k=1), question_metrics(qa, hits, k=3)
    assert (at1["span_hit"], at1["mrr"], at1["doc_hit"]) == (0.0, 0.0, 0.0)
    assert (at3["span_hit"], at3["mrr"], at3["doc_hit"]) == (1.0, pytest.approx(1 / 3), 1.0)
    assert question_metrics(qa, hits, k=10)["k"] == 3                        # fewer hits than k: all of them


def test_question_without_evidence_raises():
    with pytest.raises(ValueError, match="no evidence"):
        question_metrics(pair(), [hit(0, 10)], k=1)


def test_budget_k():
    """budget_k picks the largest k within the token budget and never returns 0."""
    hits = [hit(i * 10, i * 10 + 5, n_tokens=n) for i, n in enumerate([500, 500, 400, 100])]
    assert budget_k(hits, 1300) == 2                                         # 1000 fits, 1400 doesn't
    assert budget_k(hits, 1000) == 2                                         # exactly at the budget fits
    assert budget_k(hits, 100) == 1                                          # even the first is too big
    assert budget_k(hits, 5000) == 4                                         # everything fits
    assert budget_k([], 1300) == 1
