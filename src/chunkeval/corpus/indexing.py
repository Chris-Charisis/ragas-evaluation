"""Stage 3 — load chunks + their cached vectors into Weaviate (one collection per config).

Why new collections: the existing ones (e.g. Papers_sentences_with_overlap) store only `text`, so
we can't compute span metrics. v2 collections also store doc_id + char offsets.
Collections: cfg.weaviate.collection_template -> e.g. Papers_sentence_overlap_v2 (12 total).

Two embedders, ONE collection per config: every object carries two NAMED VECTORS —
  "bge_m3"    (primary)   and   "qwen3_06b" (secondary)
(names = embedding.primary + embedding.secondary). Same text, same ids, same BM25 index; retrieval
chooses the vector space with `target_vector`. That keeps 12 collections instead of 24 and
guarantees both embedders are evaluated on exactly the same chunks. Both vectors must be cached
before a collection is built: Weaviate fixes the named vectors when it creates the collection.

Vectors are computed beforehand by embedding.py (`chunkeval embed`) and only READ here — so this
stage needs no GPU and can run while an Ollama model occupies it.

Re-runs: each collection's `description` holds its provenance (`provenance`: a hash of the chunks,
and the settings that made the vectors). A collection built from the same chunks and vectors is
verified and skipped. One built from anything else (re-chunked, re-embedded, another --limit) is
refused until `--recreate`, which rebuilds every selected collection. So `retrieve` never runs
against a stale index by accident.

Reads:  data/chunks/<corpus>/<method>_<ovl>.jsonl, data/embeddings/<embedder>/... (via embedding.load_embeddings)
Writes: Weaviate collections
Next:   retrieval.py (reuses `connect`; queries one named vector with `target_vector`)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.init import Auth
from weaviate.collections import Collection
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateStartUpError
from weaviate.util import generate_uuid5

from chunkeval.core.config import Secrets, collection_name
from chunkeval.core.schema import Config
from chunkeval.corpus.chunking.base import Chunk, load_chunks
from chunkeval.corpus.embedding import load_embeddings, load_specs

log = logging.getLogger(__name__)


def _id_property(name: str, data_type: DataType = DataType.TEXT) -> Property:
    """Filterable, not searchable, matched whole (FIELD: "paper_4" is one token, not "paper" + "4")."""
    return Property(name=name, data_type=data_type, index_searchable=False, tokenization=Tokenization.FIELD)


# One property per Chunk field except `meta`, with the field's name (see `object_properties`). Only
# `text` is searchable, so BM25 (the hybrid follow-up) scores the text the generator reads, never
# ids or section titles (section_path is metadata only, see Chunker.chunk).
PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    *(_id_property(name) for name in ("chunk_id", "doc_id", "corpus", "method")),
    Property(name="overlap", data_type=DataType.BOOL),
    *(Property(name=name, data_type=DataType.INT) for name in ("start", "end", "n_tokens")),
    _id_property("section_path", DataType.TEXT_ARRAY),
]

# Objects per collection whose properties and vectors `verify_collection` compares with the source.
VERIFY_SAMPLE = 8


def vector_names(cfg: Config) -> list[str]:
    """The named vectors of every collection: the primary embedder, then the secondary ones."""
    return [cfg.embedding.primary, *cfg.embedding.secondary]


def connect(cfg: Config, api_key: str) -> weaviate.WeaviateClient:
    """A connected Weaviate v4 client. Use it as a context manager, so the gRPC channel is always
    closed: `with connect(cfg, secrets.weaviate_api_key.get_secret_value()) as client: ...`"""
    w = cfg.weaviate
    try:
        client = weaviate.connect_to_local(host=w.host, port=w.port, grpc_port=w.grpc_port,
                                           auth_credentials=Auth.api_key(api_key))
    except WeaviateStartUpError as e:
        raise RuntimeError(f"Weaviate not reachable at {w.host}:{w.port} (gRPC {w.grpc_port}); "
                           f"start it with `docker compose up -d`") from e
    except UnexpectedStatusCodeError as e:
        if e.status_code == 401:
            raise RuntimeError("Weaviate rejected WEAVIATE_API_KEY: the container started with another key; "
                               "after changing .env, restart it with `docker compose up -d`") from e
        raise
    log.info("Weaviate %s at %s:%d", client.get_meta()["version"], w.host, w.port)
    return client


def load_variant(cfg: Config, corpus: str, variant: str,
                 limit: int | None = None) -> tuple[list[Chunk], dict[str, np.ndarray]]:
    """The chunks of one chunk file, and their cached vectors by embedder name (row i = chunk i).

    `load_embeddings` refuses a cache whose chunk_ids differ from the file's (order included) or
    that was made with other settings than the config's. `limit`: only the chunks of the first N
    documents, sorted by doc_id, so a smoke test always picks the same ones.
    """
    path = cfg.paths.data_dir / "chunks" / corpus / f"{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `chunkeval chunk` first")
    chunks = load_chunks(path)
    ids = [c.chunk_id for c in chunks]
    vectors = {name: load_embeddings(cfg, name, corpus, variant, ids) for name in vector_names(cfg)}
    if limit:
        docs = set(sorted({c.doc_id for c in chunks})[:limit])
        rows = [i for i, c in enumerate(chunks) if c.doc_id in docs]
        chunks = [chunks[i] for i in rows]
        vectors = {name: matrix[rows] for name, matrix in vectors.items()}
    return chunks, vectors


def provenance(cfg: Config, corpus: str, variant: str, chunks: list[Chunk], limit: int | None) -> dict[str, Any]:
    """What a collection is built from. Stored as JSON in its description, so `collection.config.get()`
    reads it back, and compared on every re-run: any difference means the collection is stale.

    The chunks enter by a hash of every field (text, offsets, sizes, sections), the vectors by the
    settings that made them (model id, Hub revision, prompt, ...), which `load_embeddings` has
    checked against the cache.
    """
    specs = load_specs(cfg)
    records = json.dumps([asdict(c) for c in chunks], sort_keys=True, ensure_ascii=False)
    return {
        "chunks": f"data/chunks/{corpus}/{variant}.jsonl",
        "limit": limit,
        "n": len(chunks),
        "chunks_sha256": hashlib.sha256(records.encode("utf-8")).hexdigest(),
        "size_tokenizer": f"{cfg.chunking.size_tokenizer}@{cfg.chunking.size_tokenizer_revision}",
        "vectors": {name: specs[name].settings for name in vector_names(cfg)},
    }


def stored_provenance(collection: Collection) -> dict[str, Any] | None:
    """The provenance in an existing collection's description; None if it has none (not built here)."""
    try:
        stored = json.loads(collection.config.get().description or "")
    except json.JSONDecodeError:
        return None
    return stored if isinstance(stored, dict) else None


def object_properties(chunk: Chunk) -> dict[str, Any]:
    return {p.name: getattr(chunk, p.name) for p in PROPERTIES}


def create_collection(client: weaviate.WeaviateClient, name: str, vectors: list[str], description: str) -> Collection:
    """Create a collection with one self-provided named vector per embedder in `vectors`.

    Exact (flat, brute-force) search, not HNSW: an approximate index can miss a true neighbour, and
    which one depends on the insertion order, so rebuilding a collection could move the retrieval
    metrics. A collection holds a few thousand chunks (~55 per document), where brute force takes
    milliseconds per query. Cosine is set explicitly: both models output unit-norm vectors. The
    BM25 inverted index on `text` stays on (the default), for the hybrid follow-up.
    """
    index = Configure.VectorIndex.flat(distance_metric=VectorDistances.COSINE)
    return client.collections.create(
        name, description=description, properties=PROPERTIES,
        vector_config=[Configure.Vectors.self_provided(name=v, vector_index_config=index) for v in vectors])


def index_chunks(collection: Collection, chunks: list[Chunk], vectors: dict[str, np.ndarray]) -> None:
    """Insert every chunk with all its named vectors; raise if any object failed.

    Each uuid derives from the chunk_id, so a chunk can be fetched by id (`verify_collection`).
    With named vectors, `vector` is a dict by name: a bare list would go to an unnamed default
    vector, which these collections don't have.
    """
    rows = {name: matrix.tolist() for name, matrix in vectors.items()}
    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            batch.add_object(properties=object_properties(chunk), uuid=generate_uuid5(chunk.chunk_id),
                             vector={name: r[i] for name, r in rows.items()})
    failed = collection.batch.failed_objects       # batch errors don't raise by themselves
    if failed:
        raise RuntimeError(f"{collection.name}: {len(failed)} of {len(chunks)} objects failed to insert, "
                           f"the first with: {failed[0].message}")


def verify_collection(collection: Collection, chunks: list[Chunk], vectors: dict[str, np.ndarray],
                      sample: int = VERIFY_SAMPLE) -> list[str]:
    """Compare a collection with the chunks and vectors it should hold; return the problems found.

    The object count must equal the chunk count. For `sample` chunks spread over the list, first and
    last included, the object must carry the chunk's properties and, under each name, exactly that
    embedder's cached vector. Both models output 1024-dim unit vectors, so this is the check that
    catches rows shifted by one or the two names swapped; a dimension or norm check can't.
    """
    problems = []
    count = collection.aggregate.over_all(total_count=True).total_count
    if count != len(chunks):
        problems.append(f"{count} objects for {len(chunks)} chunks")
    rows = sorted({round(x) for x in np.linspace(0, len(chunks) - 1, min(sample, len(chunks)))})
    for i in rows:
        chunk = chunks[i]
        obj = collection.query.fetch_object_by_id(generate_uuid5(chunk.chunk_id), include_vector=True)
        if obj is None:
            problems.append(f"{chunk.chunk_id}: no object")
            continue
        if obj.properties != object_properties(chunk):
            problems.append(f"{chunk.chunk_id}: properties differ from the chunk file")
        if set(obj.vector) != set(vectors):
            problems.append(f"{chunk.chunk_id}: named vectors {sorted(obj.vector)}, expected {sorted(vectors)}")
        for name, matrix in vectors.items():
            if name in obj.vector and not np.allclose(obj.vector[name], matrix[i], atol=1e-5):
                problems.append(f"{chunk.chunk_id}: vector {name!r} differs from the cached one")
    return problems


def index_collection(client: weaviate.WeaviateClient, name: str, chunks: list[Chunk], vectors: dict[str, np.ndarray],
                     source: dict[str, Any], recreate: bool = False) -> bool:
    """Build one collection from `chunks` and `vectors`, unless it is already up to date; return
    True if it was (re)built. `source`: its `provenance`.

    An existing collection is skipped when its stored provenance equals `source` and it passes
    `verify_collection`; otherwise it is refused (RuntimeError naming --recreate). With `recreate`
    it is deleted and built again. A new collection is verified after the insert, and a failure
    raises, so a half-built index never reaches `retrieve`.
    """
    def fail(reason: str) -> RuntimeError:
        return RuntimeError(f"{name}: {reason}; rebuild it with `chunkeval index --recreate`")

    if client.collections.exists(name):
        collection = client.collections.get(name)
        if not recreate:
            stored = stored_provenance(collection)
            if stored is None:
                raise fail("exists, but its description holds no provenance (not built by this stage)")
            if stored != source:
                differs = [key for key in source if stored.get(key) != source[key]]
                raise fail(f"exists, built from other inputs ({', '.join(differs)} differ)")
            problems = verify_collection(collection, chunks, vectors)
            if problems:
                raise fail(f"exists with the right provenance but fails verification: {'; '.join(problems[:5])}")
            log.info("%s: up to date (%d objects), skipped", name, len(chunks))
            return False
        client.collections.delete(name)
        log.info("%s: deleted (--recreate)", name)

    start = time.monotonic()
    collection = create_collection(client, name, list(vectors), json.dumps(source, sort_keys=True))
    index_chunks(collection, chunks, vectors)
    problems = verify_collection(collection, chunks, vectors)
    if problems:
        raise RuntimeError(f"{name}: verification failed after indexing: {'; '.join(problems[:5])}")
    log.info("%s: %d objects with vectors %s indexed and verified in %.1fs",
             name, len(chunks), list(vectors), time.monotonic() - start)
    return True


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None, recreate: bool = False) -> None:
    """CLI entry (`chunkeval index`): load every (method, overlap) chunk file of each corpus in
    `corpora` into its collection, with all named vectors. `limit`: only the chunks of the first N
    documents per corpus (smoke test against a real Weaviate). `recreate`: rebuild collections that
    already exist, even up-to-date ones.
    """
    built = skipped = 0
    with connect(cfg, secrets.weaviate_api_key.get_secret_value()) as client:
        for corpus in corpora:
            for method in cfg.chunking.methods:
                for overlap in cfg.chunking.overlap:
                    variant = f"{method}_{'overlap' if overlap else 'no_overlap'}"
                    chunks, vectors = load_variant(cfg, corpus, variant, limit)
                    source = provenance(cfg, corpus, variant, chunks, limit)
                    name = collection_name(cfg, corpus, method, overlap)
                    if index_collection(client, name, chunks, vectors, source, recreate):
                        built += 1
                    else:
                        skipped += 1
    log.info("%d collections built, %d up to date", built, skipped)
