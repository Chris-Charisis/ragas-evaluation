"""The `index` stage: chunks + cached vectors -> Weaviate collections with two named vectors.

Offline tests cover what needs no server: the property schema, --limit, provenance. The `weaviate`
tests run against the local Weaviate (`docker compose up -d`) on tiny fake data (vectors from a
hash, no GPU), and only create and delete collections named *_pytest, never the real *_v2 ones.
"""
import dataclasses
import json
import zlib

import numpy as np
import pytest
from weaviate.classes.config import DataType, VectorDistances
from weaviate.classes.query import Filter

from chunkeval.core.config import collection_name, load_config, load_secrets
from chunkeval.core.schema import REPO_ROOT
from chunkeval.core.store import write_jsonl
from chunkeval.corpus import indexing
from chunkeval.corpus.chunking.base import Chunk
from chunkeval.corpus.embedding import cache_path, embed_chunk_file, load_specs
from chunkeval.corpus.indexing import (PROPERTIES, connect, create_collection, index_chunks, load_variant,
                                       provenance, run, stored_provenance, verify_collection)

# Out of order on purpose: --limit must sort by doc_id. Under word tokenization "paper_4" would
# also match "paper_4_suppl" ({paper, 4} is a subset of {paper, 4, suppl}).
DOCS = ["paper_4_suppl", "paper_4"]
VARIANT = "sentence_overlap"


class FakeEmbedder:
    """What embed_chunk_file needs from an Embedder: unit vectors from a hash of (model, text), no GPU."""

    def __init__(self, spec):
        self.spec = spec

    def encode_documents(self, texts, batch_size):
        rows = [np.random.default_rng(zlib.crc32(f"{self.spec.name}|{t}".encode())).standard_normal(self.spec.dim)
                for t in texts]
        vectors = np.array(rows, dtype=np.float32).reshape(len(texts), self.spec.dim)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_chunks(variant: str = VARIANT, n: int = 3) -> list[Chunk]:
    method, _, overlap = variant.partition("_")
    return [Chunk(chunk_id=f"{doc}:{variant}:{i}", doc_id=doc, corpus="papers", method=method,
                  overlap=overlap == "overlap", start=100 * i, end=100 * i + 60,
                  text=f"Irrigated rice yielded {i + 2} t/ha in plot {i} of {doc}.", n_tokens=15,
                  section_path=["2. Methods", f"2.{i + 1} Sites"], meta={"overlap_tokens": 0})
            for doc in DOCS for i in range(n)]


def write_variant(cfg, chunks: list[Chunk], variant: str = VARIANT, embed: bool = True) -> None:
    """A chunk file, and (with `embed`) its vectors cached by both fake models, as `embed` writes them."""
    path = cfg.paths.data_dir / "chunks" / "papers" / f"{variant}.jsonl"
    write_jsonl(path, chunks)
    if embed:
        for name, spec in load_specs(cfg).items():
            embed_chunk_file(path, FakeEmbedder(spec), cache_path(cfg, name, "papers", variant), 8)


@pytest.fixture
def stage(cfg, tmp_path):
    """cfg with data/ and the vector cache in tmp_path, one variant (one collection per corpus,
    named *_pytest) and its 6 chunks cached by both models."""
    cfg.paths.data_dir = tmp_path / "data"
    cfg.embedding.cache_dir = tmp_path / "embeddings"
    cfg.chunking.methods, cfg.chunking.overlap = ["sentence"], [True]
    cfg.weaviate.collection_template = "{corpus}_{method}_{overlap}_pytest"
    write_variant(cfg, make_chunks())
    return cfg


# --- offline ---------------------------------------------------------------------------------------

def test_properties_are_the_chunk_fields():
    """Every Chunk field but `meta` is stored under its own name, and only `text` is searchable,
    so BM25 never scores ids or section titles."""
    assert sorted(p.name for p in PROPERTIES) == sorted(f.name for f in dataclasses.fields(Chunk) if f.name != "meta")
    searchable = [p.name for p in PROPERTIES
                  if p.dataType in (DataType.TEXT, DataType.TEXT_ARRAY) and p.indexSearchable is not False]
    assert searchable == ["text"]


def test_limit_takes_the_first_docs_with_their_own_vectors(stage):
    chunks, vectors = load_variant(stage, "papers", VARIANT)
    first, first_vectors = load_variant(stage, "papers", VARIANT, limit=1)
    assert {c.doc_id for c in first} == {"paper_4"}             # sorted by doc_id, not file order
    rows = [i for i, c in enumerate(chunks) if c.doc_id == "paper_4"]
    assert [c.chunk_id for c in first] == [chunks[i].chunk_id for i in rows]
    for name in vectors:
        np.testing.assert_array_equal(first_vectors[name], vectors[name][rows])


def test_load_variant_names_the_missing_stage(stage):
    with pytest.raises(FileNotFoundError, match="chunkeval chunk"):
        load_variant(stage, "papers", "character_overlap")
    write_variant(stage, make_chunks("character_overlap"), "character_overlap", embed=False)
    with pytest.raises(FileNotFoundError, match="chunkeval embed"):
        load_variant(stage, "papers", "character_overlap")


def test_provenance_tracks_chunks_vectors_and_limit(stage):
    chunks, _ = load_variant(stage, "papers", VARIANT)
    base = provenance(stage, "papers", VARIANT, chunks, None)
    assert provenance(stage, "papers", VARIANT, load_variant(stage, "papers", VARIANT)[0], None) == base
    assert json.loads(json.dumps(base)) == base                  # survives the trip through the description

    edited = [dataclasses.replace(chunks[0], end=chunks[0].end + 1), *chunks[1:]]
    assert provenance(stage, "papers", VARIANT, edited, None)["chunks_sha256"] != base["chunks_sha256"]
    limited = provenance(stage, "papers", VARIANT, chunks[:3], 1)
    assert (limited["limit"], limited["n"]) == (1, 3) and limited["chunks_sha256"] != base["chunks_sha256"]

    stage.embedding.models["qwen3_06b"].revision = "0" * 40      # a newer pinned commit of the model
    assert provenance(stage, "papers", VARIANT, chunks, None)["vectors"] != base["vectors"]


# --- against the local Weaviate --------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """One connection for the module; skips when WEAVIATE_API_KEY is unset or Weaviate isn't running."""
    secrets = load_secrets()
    if not secrets.weaviate_api_key:
        pytest.skip("WEAVIATE_API_KEY not set")
    try:
        c = connect(load_config(REPO_ROOT / "configs" / "default.yaml"), secrets.weaviate_api_key.get_secret_value())
    except RuntimeError as e:
        pytest.skip(str(e))
    yield c
    c.close()


@pytest.fixture
def live(stage, client):
    """`stage` against the real Weaviate, with every *_pytest collection deleted before and after."""
    def clean():
        for name in client.collections.list_all(simple=True):
            if name.endswith("_pytest"):
                client.collections.delete(name)
    clean()
    yield stage
    clean()


NAME = "Papers_sentence_overlap_pytest"


def run_stage(cfg, **options):
    run(cfg, load_secrets(), corpora=["papers"], **options)


@pytest.mark.weaviate
def test_index_builds_a_verified_collection(live, client):
    run_stage(live)
    assert collection_name(live, "papers", "sentence", True) == NAME
    collection = client.collections.get(NAME)
    config = collection.config.get()
    assert set(config.vector_config) == {"bge_m3", "qwen3_06b"}
    for vector in config.vector_config.values():
        assert vector.vector_index_config.vector_index_type() == "flat"      # exact search
        assert vector.vector_index_config.distance_metric == VectorDistances.COSINE

    chunks, vectors = load_variant(live, "papers", VARIANT)
    assert stored_provenance(collection) == provenance(live, "papers", VARIANT, chunks, None)
    assert verify_collection(collection, chunks, vectors, sample=len(chunks)) == []   # every object
    # each named vector is searched on its own: a chunk's vector finds that chunk first
    for name, matrix in vectors.items():
        for i, chunk in enumerate(chunks):
            hit = collection.query.near_vector(matrix[i].tolist(), target_vector=name, limit=1).objects[0]
            assert hit.properties["chunk_id"] == chunk.chunk_id, name


@pytest.mark.weaviate
def test_ids_match_whole_and_bm25_scores_text_only(live, client):
    run_stage(live)
    collection = client.collections.get(NAME)
    hits = collection.query.fetch_objects(filters=Filter.by_property("doc_id").equal("paper_4"), limit=20).objects
    assert sorted(h.properties["chunk_id"] for h in hits) == [f"paper_4:{VARIANT}:{i}" for i in range(3)]
    assert collection.query.bm25("Methods", limit=5).objects == []        # only in section_path
    assert collection.query.bm25("irrigated", limit=5).objects


@pytest.mark.weaviate
def test_rerun_skips_up_to_date_and_refuses_stale(live, client, monkeypatch):
    run_stage(live)
    calls = []
    real = indexing.index_chunks
    monkeypatch.setattr(indexing, "index_chunks", lambda *args: calls.append(args) or real(*args))
    run_stage(live)
    assert calls == []                                              # up to date: nothing inserted

    write_variant(live, make_chunks(n=4))                           # re-chunked (and re-embedded)
    with pytest.raises(RuntimeError, match=r"chunks_sha256.*--recreate"):
        run_stage(live)
    assert calls == [] and client.collections.get(NAME).aggregate.over_all(total_count=True).total_count == 6

    run_stage(live, recreate=True)
    assert len(calls) == 1
    assert client.collections.get(NAME).aggregate.over_all(total_count=True).total_count == 8


@pytest.mark.weaviate
def test_limit_indexes_only_the_first_docs(live, client):
    run_stage(live, limit=1)
    collection = client.collections.get(NAME)
    assert {o.properties["doc_id"] for o in collection.iterator()} == {"paper_4"}
    with pytest.raises(RuntimeError, match="limit"):                # a full run needs --recreate
        run_stage(live)


@pytest.mark.weaviate
def test_incomplete_or_foreign_collection_is_refused(live, client):
    """A crash mid-insert leaves the right provenance but too few objects; a collection made by
    hand has none. Neither is skipped as up to date."""
    chunks, vectors = load_variant(live, "papers", VARIANT)
    source = provenance(live, "papers", VARIANT, chunks, None)
    collection = create_collection(client, NAME, list(vectors), json.dumps(source))
    index_chunks(collection, chunks[:2], {name: m[:2] for name, m in vectors.items()})
    with pytest.raises(RuntimeError, match="2 objects for 6 chunks"):
        run_stage(live)

    client.collections.delete(NAME)
    create_collection(client, NAME, list(vectors), "made by hand")
    with pytest.raises(RuntimeError, match="no provenance"):
        run_stage(live)


@pytest.mark.weaviate
@pytest.mark.parametrize("mistake", ["swapped names", "rows shifted by one"])
def test_verify_catches_misassigned_vectors(live, client, mistake):
    """Both models give 1024-dim unit vectors, so these mistakes insert without any error."""
    chunks, vectors = load_variant(live, "papers", VARIANT)
    wrong = ({"bge_m3": vectors["qwen3_06b"], "qwen3_06b": vectors["bge_m3"]} if mistake == "swapped names"
             else {name: np.roll(m, 1, axis=0) for name, m in vectors.items()})
    collection = create_collection(client, NAME, list(vectors), "{}")
    index_chunks(collection, chunks, wrong)
    problems = verify_collection(collection, chunks, vectors)
    assert len(problems) == 2 * len(chunks)                         # every sampled object, both names
    assert all("differs from the cached one" in p for p in problems)


@pytest.mark.weaviate
def test_failed_inserts_raise(live, client):
    """Batch errors don't raise by themselves; index_chunks must."""
    chunks, vectors = load_variant(live, "papers", VARIANT)
    collection = create_collection(client, NAME, list(vectors), "{}")
    with pytest.raises(RuntimeError, match="2 of 2 objects failed"):
        index_chunks(collection, chunks[:2], {"bge_m3": vectors["bge_m3"][:2], "other": vectors["qwen3_06b"][:2]})
