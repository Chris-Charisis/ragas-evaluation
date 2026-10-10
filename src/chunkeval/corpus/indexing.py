"""Stage 3 — load chunks + their cached vectors into Weaviate (one collection per config).

Why new collections: the existing ones (e.g. Papers_sentences_with_overlap) store only `text`, so
we can't compute span metrics. v2 collections also store doc_id + char offsets.
Collections: cfg.weaviate.collection_template -> e.g. papers_sentence_overlap_v2 (12 total).

Two embedders, ONE collection per config: every object carries two NAMED VECTORS —
  "bge_m3"    (primary)   and   "qwen3_06b" (secondary)
(names = keys of cfg.embedding.models). Same text, same ids, same BM25 index; retrieval
chooses the vector space with `target_vector`. That keeps 12 collections instead of 24 and
guarantees both embedders are evaluated on exactly the same chunks.

Vectors are computed beforehand by embedding.py (`chunkeval embed`) and only READ here — so this
stage needs no GPU and can run while an Ollama model occupies it.

Reads:  data/chunks/<corpus>/<method>_<ovl>.jsonl, data/embeddings/<embedder>/...
Writes: Weaviate collections
Reuse:  `list_collections` / `count_entries` (ragas_evaluation.py:289-300) and the connection
        settings (ragas_evaluation.py:283-286, but API key from .env).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config

if TYPE_CHECKING:
    from chunkeval.corpus.chunking.base import Chunk


def connect(cfg: Config, api_key: str | None):
    """Return a connected Weaviate v4 client.

    TODO(you): `weaviate.connect_to_local(host=..., port=..., grpc_port=...,
    auth_credentials=weaviate.classes.init.Auth.api_key(key))`. The v4 client keeps a gRPC
    channel open — always close it (context manager or try/finally), as the old code did.
    """
    raise NotImplementedError


def create_collection(client, name: str, vector_names: list[str], recreate: bool = False) -> None:
    """Create a collection with one self-provided NAMED vector per embedder.

    TODO(you):
      - Properties: text (TEXT), chunk_id, doc_id, corpus, method (TEXT, skip vectorization),
        overlap (BOOL), start, end, n_tokens (INT), section_path (TEXT_ARRAY).
      - Named vectors in weaviate-client 4.22: `vector_config=[...]` with one
        `Configure.Vectors.self_provided(name=<vector name>, vector_index_config=...)` per entry of
        `vector_names` (signature checked against the installed client). The older
        `Configure.NamedVectors.none(name=...)` still exists but is the legacy API.
      - Distance: both models produce normalised vectors → cosine. Set it explicitly in the vector
        index config (`Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)`) rather
        than relying on the default, so the choice is visible in the code.
      - Put the embedder model_ids and revisions (and the size tokenizer + its revision) in the collection `description` —
        a cheap provenance record you can read back with `collection.config.get()`.
      - All named vectors must exist at creation: decide whether to require BOTH embeddings to be
        cached before indexing (simplest), or to add the secondary later (newer servers support
        adding a named vector to an existing collection — check the docs for 1.39).
      - Keep the BM25 inverted index enabled on `text` (default) — needed for the hybrid follow-up.
      - `recreate=False` by default: refuse to overwrite an existing collection unless asked.
    """
    raise NotImplementedError


def load_vectors(cfg: Config, embedder: str, corpus: str, method: str,
                 overlap: bool, chunk_ids: list[str]) -> Any:
    """Load the cached (n, dim) matrix written by embedding.py for one chunk file.

    TODO(you): check that the cached chunk_ids equal `chunk_ids` IN ORDER before returning —
    otherwise every object would get its neighbour's vector and nothing would fail loudly.
    """
    raise NotImplementedError


def index_chunks(client, name: str, chunks: list["Chunk"], vectors: dict[str, Any]) -> int:
    """Batch-insert chunks with all their named vectors; return the number inserted.

    `vectors` maps vector name -> (n, dim) matrix aligned with `chunks`.

    TODO(you):
      - `with collection.batch.dynamic() as batch: batch.add_object(properties=..., uuid=...,
        vector={"bge_m3": [...], "qwen3_06b": [...]})` — with named vectors, `vector` is a DICT
        keyed by vector name (a bare list would go to an unnamed default vector, which this
        collection doesn't have).
      - Convert numpy rows with `.tolist()`; float32 is enough.
      - Inspect `collection.batch.failed_objects` afterwards — batch errors do NOT raise by default.
      - Deterministic uuid (`weaviate.util.generate_uuid5(chunk_id)`) makes re-runs idempotent.
    """
    raise NotImplementedError


def verify_index(client, expected: dict[str, int], vector_names: list[str]) -> dict[str, Any]:
    """Compare object counts with chunk counts, and spot-check the named vectors.

    TODO(you):
      - Counts: reuse `count_entries` logic — `collection.aggregate.over_all(total_count=True)`.
      - Vectors: fetch a few objects with `include_vector=True` and check that every name in
        `vector_names` is present with dim 1024 and unit norm. Also check the two vectors of the
        same object DIFFER — identical vectors would mean the same model was written twice.
      - Return the mismatches; the cli should fail if any.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None) -> None:
    """CLI entry (`chunkeval index`): load every (method, overlap) chunk file of each corpus in
    `corpora` into its collection, with both named vectors. `limit`: only the chunks of the first
    N documents per corpus (smoke test against a real Weaviate).

    TODO(you): `connect` with `secrets.weaviate_api_key` → for each config: `create_collection`
    (name from `config.collection_name`) → `index_chunks` → `verify_index`; raise if it reports
    mismatches, so a half-built index never reaches `retrieve`.
    """
    raise NotImplementedError
