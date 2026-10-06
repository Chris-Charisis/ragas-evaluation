"""Embedder conventions and named-vector integrity — mistakes here fail SILENTLY otherwise
(both models are 1024-dim, so a mix-up never raises an error, it just degrades retrieval)."""
import pytest


def test_output_shape_and_unit_norm():
    """Both embedders return float32 arrays of shape (n, 1024) with unit-norm rows."""
    pytest.skip("TODO(you)")


def test_qwen3_query_prompt_applied_to_queries_only():
    """For Qwen3, encode_queries(q) != encode_documents(q) (the instruction changes the vector);
    for bge-m3 the two are identical (no instruction)."""
    pytest.skip("TODO(you)")


def test_query_prompt_is_not_placeholder():
    """cfg embedding.models.qwen3_06b.query_prompt was replaced (not "TODO") and follows the
    "Instruct: ...\\nQuery:" format."""
    pytest.skip("TODO(you)")


def test_embedders_produce_different_spaces():
    """The same text embedded by bge-m3 and Qwen3 gives clearly different vectors — guards against
    loading the same model twice under two names."""
    pytest.skip("TODO(you)")


def test_retrieval_sanity_per_embedder():
    """For a tiny corpus (3 passages) each embedder ranks the obviously relevant passage first for a
    matching question — catches wrong pooling/padding (e.g. Qwen3 last-token pooling with right padding)."""
    pytest.skip("TODO(you)")


def test_cached_vectors_aligned_with_chunk_ids():
    """The cache loader refuses a matrix whose saved chunk_ids differ (order included) from the chunk file."""
    pytest.skip("TODO(you)")


def test_chunks_fit_secondary_model():
    """Every chunk's token count in the Qwen3 tokenizer is <= its max_seq_length (no truncation)."""
    pytest.skip("TODO(you)")
