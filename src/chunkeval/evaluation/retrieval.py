"""Stage 6a — retrieve top-k chunks for every question, collection and EMBEDDER.

Reuse: `WeaviateRagasRetriever` (ragas_evaluation.py:621-681) is a good base — keep its
context-manager/close() handling. Changes needed:
  * return chunk metadata (doc_id, start, end, n_tokens, chunk_id) and the score, not just text —
    span metrics need the offsets;
  * query a NAMED vector (`target_vector=`) — each collection holds a bge_m3 and a qwen3_06b vector;
  * the old dimension check is useless now: both embedders are 1024-dim, so a query encoded with
    the wrong model passes it. Bind each retriever to (embedder, target_vector) and assert they
    are the same name instead;
  * support `mode="hybrid"` (BM25 + dense) for the follow-up experiment;
  * retrieve a generous k_max once (e.g. 20) and derive k=5 and the budget-matched view from it.

Query vectors are computed ONCE per (embedder, corpus) and cached — encoding with each model's
query convention lives in embedding.Embedder.encode_queries; this module only sends vectors.
That also means retrieval itself needs no GPU (handy while an Ollama model is loaded).

Reads:  data/qa/filtered.jsonl, Weaviate v2 collections, data/embeddings/queries/<embedder>/...
Writes: runs/<run_id>/retrieval/<embedder>/<corpus>_<method>_<ovl>.jsonl  (question_id -> ranked hits)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config


@dataclass
class Hit:
    rank: int
    chunk_id: str
    doc_id: str
    start: int
    end: int
    n_tokens: int
    score: float
    text: str


def encode_questions(cfg: Config, embedder_name: str, corpus: str,
                     questions: list[dict[str, Any]]) -> Any:
    """Encode (or load from cache) all question vectors of a corpus with one embedder.

    TODO(you):
      - Use `embedding.Embedder.encode_queries` — it applies the query convention (no instruction
        for bge-m3, the "Instruct: ...\\nQuery:" prompt for Qwen3). Never call `encode_documents`
        on questions: for Qwen3 that silently drops the instruction and costs retrieval quality.
      - Cache the matrix + question ids (same alignment check as for chunk vectors), keyed by the
        embedder name AND a hash of the query prompt — changing the instruction must invalidate it.
      - Unload the model afterwards (Embedder.unload) before loading the next one.
    """
    raise NotImplementedError


class WeaviateRetriever:
    """Generalised version of the old WeaviateRagasRetriever, bound to one named vector."""

    def __init__(self, client, collection_name: str, target_vector: str, cfg: Config) -> None:
        """TODO(you): get the collection; check that `target_vector` is one of the collection's
        named vectors (`collection.config.get().vector_config` keys) and fail early if not."""
        raise NotImplementedError

    def retrieve(self, question: str, query_vector: Any, k: int) -> list[Hit]:
        """TODO(you):
          - dense: `collection.query.near_vector(near_vector=..., target_vector=self.target_vector,
            limit=k, return_metadata=MetadataQuery(distance=True))` → score = 1 - distance (cosine).
          - hybrid: `collection.query.hybrid(query=question, vector=..., target_vector=...,
            alpha=cfg alpha, limit=k, return_metadata=MetadataQuery(score=True))`. alpha=1 is pure
            vector, 0 pure BM25. The BM25 side is the same for both embedders — only the dense
            side changes.
          - Fill `Hit` from `obj.properties` + metadata. Keep the rank order.
        """
        raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """For each embedder × corpus × method × overlap: retrieve k_max hits for each question.

    TODO(you):
      - Loop order: embedders (outer, from cfg.retrieval.embedders) → corpus → configs.
      - Questions of corpus X are only asked against corpus-X collections (each corpus is a
        separate index, as in the previous round).
      - Write one JSONL per (embedder, collection); resumable.
      - Oracle + random baselines are embedder-independent: write them ONCE, in the same format,
        so evaluation/retrieval_metrics.py can score them like any other "method".
    """
    raise NotImplementedError
