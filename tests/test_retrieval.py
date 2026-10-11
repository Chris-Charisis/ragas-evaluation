"""The `retrieve` stage: query vectors, the baselines, the input checks, re-runs, and the metrics table.

Offline tests run retrieval.run and retrieval_metrics.run on tiny_doc with hand-cut chunks, a fake
embedder (hash vectors, no GPU) and a fake retriever (ranks chunks by words shared with the
question), so no Weaviate. The `weaviate` tests check WeaviateRetriever itself against the local
Weaviate, on a *_pytest collection only.
"""
import contextlib
import logging
import re
import zlib

import numpy as np
import pandas as pd
import pytest

from chunkeval.core.config import Secrets, load_config, load_secrets
from chunkeval.core.schema import REPO_ROOT
from chunkeval.core.store import write_json, write_jsonl
from chunkeval.corpus import indexing
from chunkeval.corpus.chunking.base import Chunk
from chunkeval.corpus.embedding import cache_path, embed_chunk_file, load_embeddings, load_specs
from chunkeval.evaluation import retrieval, retrieval_metrics
from chunkeval.evaluation.retrieval import (BASELINES, WeaviateRetriever, encode_questions, hit_from_chunk,
                                            load_hits, load_questions, oracle_hits, random_hits, retrieval_path,
                                            variant_chunks)
from chunkeval.evaluation.retrieval_metrics import evidence_intervals, question_metrics
from chunkeval.qa.filtering import filter_paths
from chunkeval.qa.generation import EvidenceSpan, QAPair, qa_paths, text_sha256

VARIANT = "sentence_overlap"
EMBEDDERS = ["bge_m3", "qwen3_06b"]
QUESTIONS = {   # question -> evidence quotes in tiny_doc
    "How many replicates of every treatment did each Sahel trial site have?":
        ["Each site had four replicates of every treatment."],
    "How much did irrigation change the harvest in the Sahel?":
        ["The largest gain was in the Sahel, where irrigation doubled the harvest."],
    "Which constraints limited the rainfed plots and the remote villages?":
        ["Weeds were the main constraint in rainfed plots.", "Seed supply remained a bottleneck in remote villages."],
}


def make_pairs(doc) -> list[QAPair]:
    pairs = []
    for i, (question, quotes) in enumerate(QUESTIONS.items()):
        evidence = [EvidenceSpan(doc.text.index(q), doc.text.index(q) + len(q), q, "exact") for q in quotes]
        pairs.append(QAPair(id=f"{doc.doc_id}:q{i}", doc_id=doc.doc_id, corpus="papers", unit_id=f"q{i}",
                            question=question, answer="-", question_type="factoid", evidence=evidence,
                            meta={"section_path": ["3. Results"]}))
    return pairs


def windows(doc, size=200, step=150, doc_id=None) -> list[Chunk]:
    """Overlapping character windows of `size` chars (`size // 4` "tokens"), each `doc.text[start:end]`."""
    doc_id = doc_id or doc.doc_id
    return [Chunk(chunk_id=f"{doc_id}:{VARIANT}:{i}", doc_id=doc_id, corpus="papers", method="sentence",
                  overlap=True, start=s, end=s + size, text=doc.text[s:s + size], n_tokens=size // 4)
            for i, s in enumerate(range(0, len(doc.text) - size + 1, step))]


def hash_vectors(name: str, texts: list[str], dim: int) -> np.ndarray:
    rows = [np.random.default_rng(zlib.crc32(f"{name}|{t}".encode())).standard_normal(dim) for t in texts]
    vectors = np.array(rows, dtype=np.float32).reshape(len(texts), dim)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeEmbedder:
    """Embedder without a GPU: unit vectors from a hash of (model, prompt + text). `loads` records
    every model load."""
    loads: list[str] = []

    def __init__(self, spec):
        self.spec = spec
        FakeEmbedder.loads.append(spec.name)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def encode_documents(self, texts, batch_size):
        return hash_vectors(self.spec.name, [self.spec.document_prompt + t for t in texts], self.spec.dim)

    def encode_queries(self, texts, batch_size):
        return hash_vectors(self.spec.name, [self.spec.query_prompt + t for t in texts], self.spec.dim)


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class FakeRetriever:
    """Stands in for WeaviateRetriever: ranks the chunk file by words shared with the question.
    `built` records the (collection, target vector) of every instance."""
    built: list[tuple[str, str]] = []

    def __init__(self, client, collection_name, target_vector, cfg, index_provenance=None):
        FakeRetriever.built.append((collection_name, target_vector))
        self.chunks = variant_chunks(cfg, "papers", VARIANT)[0]

    def retrieve(self, question, query_vector, k):
        ranked = sorted(self.chunks, key=lambda c: (-len(words(question) & words(c.text)), c.start))
        return [hit_from_chunk(c, rank, 0.0) for rank, c in enumerate(ranked[:k], start=1)]


def write_chunks(cfg, chunks: list[Chunk]) -> None:
    """The chunk file, and its vectors cached by both fake models (as `embed` writes them)."""
    path = cfg.paths.data_dir / "chunks" / "papers" / f"{VARIANT}.jsonl"
    write_jsonl(path, chunks)
    for name, spec in load_specs(cfg).items():
        embed_chunk_file(path, FakeEmbedder(spec), cache_path(cfg, name, "papers", VARIANT), 8)


@pytest.fixture
def stage(cfg, tiny_doc, tmp_path, monkeypatch):
    """cfg with data/ in tmp_path: tiny_doc parsed, its 3 questions in filtered.jsonl (one with two
    quotes), one variant of overlapping windows with cached vectors; k_max 5, k at 1 and 3, a
    budget of 2 chunks. Query encoding uses FakeEmbedder (no GPU check)."""
    cfg.paths.data_dir = tmp_path / "data"
    cfg.embedding.cache_dir = tmp_path / "embeddings"
    cfg.chunking.methods, cfg.chunking.overlap = ["sentence"], [True]
    cfg.weaviate.collection_template = "{corpus}_{method}_{overlap}_pytest"
    cfg.retrieval.k_max, cfg.retrieval.k_values, cfg.retrieval.top_k = 5, [1, 3], 3
    cfg.retrieval.budget_tokens = 100                                   # 2 windows of 50 tokens
    write_json(cfg.paths.data_dir / "parsed" / "papers" / "tiny.json", tiny_doc)
    write_json(qa_paths(cfg)["provenance"], {"settings": {}, "docs": {"papers/tiny": text_sha256(tiny_doc)}})
    write_jsonl(filter_paths(cfg)["filtered"], make_pairs(tiny_doc))
    monkeypatch.setattr(FakeEmbedder, "loads", [])
    monkeypatch.setattr(FakeRetriever, "built", [])
    write_chunks(cfg, windows(tiny_doc))
    FakeEmbedder.loads.clear()                                          # count query encodings only
    monkeypatch.setattr(retrieval, "Embedder", FakeEmbedder)
    monkeypatch.setattr(retrieval, "check_gpu_free", lambda secrets: None)
    return cfg


@pytest.fixture
def offline(stage, monkeypatch):
    """`stage` without Weaviate: no connection, FakeRetriever instead of WeaviateRetriever."""
    monkeypatch.setattr(retrieval, "WeaviateRetriever", FakeRetriever)
    monkeypatch.setattr(indexing, "connect", lambda cfg, key: contextlib.nullcontext(object()))
    return stage


SECRETS = Secrets(weaviate_api_key="key")


def retrieve(cfg, embedders=EMBEDDERS):
    retrieval.run(cfg, SECRETS, corpora=["papers"], embedders=embedders)


def score(cfg, embedders=EMBEDDERS) -> pd.DataFrame:
    retrieval_metrics.run(cfg, SECRETS, corpora=["papers"], embedders=embedders)
    return pd.read_parquet(cfg.paths.runs_dir / cfg.run_id / "metrics" / "retrieval_per_question.parquet")


# --- the baselines ------------------------------------------------------------------------------

def test_oracle_covers_the_evidence_greedily():
    """Evidence (90, 200) over chunks a (0, 100), b (80, 180), c (150, 300), d (300, 400):
    b adds 90 chars, then c the remaining 20; a adds nothing new and d never touches it."""
    spans = {"a": (0, 100), "b": (80, 180), "c": (150, 300), "d": (300, 400)}
    chunks = [Chunk(chunk_id=k, doc_id="doc", corpus="papers", method="sentence", overlap=True,
                    start=s, end=e, text="", n_tokens=(e - s) // 4) for k, (s, e) in spans.items()]
    qa = QAPair(id="q", doc_id="doc", corpus="papers", unit_id="u", question="?", answer="", question_type="factoid",
                evidence=[EvidenceSpan(90, 200, "", "exact")])
    hits = oracle_hits(qa, chunks, k_max=20)
    assert [(h.chunk_id, h.rank, h.score) for h in hits] == [("b", 1, 90.0), ("c", 2, 20.0)]
    assert question_metrics(qa, hits, 2)["span_recall"] == 1.0
    assert oracle_hits(qa, chunks, k_max=1) == hits[:1]                  # capped by k_max


def test_oracle_covers_every_quote(tiny_doc):
    qa = make_pairs(tiny_doc)[2]                                         # two quotes
    hits = oracle_hits(qa, windows(tiny_doc, size=60, step=50), k_max=20)
    assert question_metrics(qa, hits, len(hits))["span_recall"] == 1.0
    assert sum(h.score for h in hits) == sum(e - s for s, e in evidence_intervals(qa))


def test_oracle_needs_one_chunk_when_the_evidence_fits(tiny_doc):
    """With windows of 200 chars every 150, a quote under 50 chars always fits in one window."""
    qa = make_pairs(tiny_doc)[0]
    hits = oracle_hits(qa, windows(tiny_doc), k_max=20)
    assert len(hits) == 1 and question_metrics(qa, hits, 1)["span_recall"] == 1.0


def test_random_is_reproducible_and_never_repeats(tiny_doc):
    chunks = windows(tiny_doc)
    qa = make_pairs(tiny_doc)[0]
    hits = random_hits(qa, chunks, VARIANT, k_max=5, seed=42)
    assert hits == random_hits(qa, chunks, VARIANT, k_max=5, seed=42)
    assert len({h.chunk_id for h in hits}) == 5
    assert hits != random_hits(qa, chunks, VARIANT, k_max=5, seed=43)
    assert len(random_hits(qa, chunks[:3], VARIANT, k_max=5, seed=42)) == 3   # fewer chunks than k_max


# --- query vectors --------------------------------------------------------------------------------

def test_query_vectors_cached_per_questions_and_prompt(stage):
    questions = load_questions(stage)
    first = encode_questions(stage, SECRETS, "qwen3_06b", questions)["papers"]
    assert FakeEmbedder.loads == ["qwen3_06b"]
    spec = load_specs(stage)["qwen3_06b"]
    expected = hash_vectors("qwen3_06b", [spec.query_prompt + qa.question for qa in questions["papers"]], spec.dim)
    np.testing.assert_allclose(first, expected)                         # the QUERY prompt, not the document one

    np.testing.assert_array_equal(encode_questions(stage, SECRETS, "qwen3_06b", questions)["papers"], first)
    assert FakeEmbedder.loads == ["qwen3_06b"]                          # cached: model not loaded again

    stage.embedding.models["qwen3_06b"].query_prompt = "Instruct: something else\nQuery:"
    encode_questions(stage, SECRETS, "qwen3_06b", questions)
    questions["papers"][0].question = "How many replicates did each site have?"
    encode_questions(stage, SECRETS, "qwen3_06b", questions)
    assert FakeEmbedder.loads == ["qwen3_06b"] * 3                      # new prompt, then new question


# --- the stage --------------------------------------------------------------------------------------

def test_run_writes_every_retriever_and_scores_it(offline, tiny_doc):
    retrieve(offline)
    pairs = load_questions(offline)["papers"]
    for retriever in [*EMBEDDERS, *BASELINES]:
        hits = load_hits(retrieval_path(offline, retriever, "papers", VARIANT))
        assert list(hits) == [qa.id for qa in pairs]
        for ranked in hits.values():
            assert [h.rank for h in ranked] == list(range(1, len(ranked) + 1))
            assert 1 <= len(ranked) <= offline.retrieval.k_max
            assert all(h.text == "" and tiny_doc.text[h.start:h.end] for h in ranked)   # offsets, no text
    assert sorted(FakeRetriever.built) == [("Papers_sentence_overlap_pytest", e) for e in EMBEDDERS]
    assert FakeEmbedder.loads == EMBEDDERS                              # each model loaded once

    df = score(offline)
    assert len(df) == len(pairs) * 4 * 3                                # questions × retrievers × (k 1, 3, budget)
    assert set(df["embedder"]) == {*EMBEDDERS, *BASELINES}
    assert set(df["k_label"]) == {"1", "3", "budget"}
    assert not df.duplicated(retrieval_metrics.ROW_KEY).any()
    budget = df[df["k_label"] == "budget"]
    assert (budget.loc[budget["embedder"] != "oracle", "k"] == 2).all()  # 2 × 50 tokens = the budget
    oracle = df[(df["embedder"] == "oracle") & (df["k_label"] == "3")]
    assert (oracle["span_recall"] == 1.0).all() and (oracle["mrr"] == 1.0).all()
    # the fake retriever ranks by shared words: the replicates question finds its window first
    first = df[(df["embedder"] == "bge_m3") & (df["k_label"] == "1") & (df["question_id"] == pairs[0].id)]
    assert first["span_hit"].item() == 1.0
    assert {"question_type", "kind", "section", "n_evidence", "evidence_chars", "doc_id"} <= set(df.columns)
    assert df.loc[df["question_id"] == pairs[2].id, "n_evidence"].eq(2).all()


def test_rerun_retrieves_only_what_changed(offline):
    retrieve(offline)
    paths = {r: retrieval_path(offline, r, "papers", VARIANT) for r in [*EMBEDDERS, *BASELINES]}
    mtimes = {r: p.stat().st_mtime_ns for r, p in paths.items()}
    retrieve(offline)
    assert len(FakeRetriever.built) == 2 and FakeEmbedder.loads == EMBEDDERS   # nothing asked again
    assert {r: p.stat().st_mtime_ns for r, p in paths.items()} == mtimes

    offline.embedding.models["qwen3_06b"].query_prompt = "Instruct: find the passage\nQuery:"
    retrieve(offline)
    assert FakeRetriever.built[2:] == [("Papers_sentence_overlap_pytest", "qwen3_06b")]
    assert FakeEmbedder.loads == [*EMBEDDERS, "qwen3_06b"]
    assert {r for r, p in paths.items() if p.stat().st_mtime_ns != mtimes[r]} == {"qwen3_06b"}

    pairs = load_questions(offline)["papers"]                             # filter-qa changed the question set
    write_jsonl(filter_paths(offline)["filtered"], pairs[:2])
    retrieve(offline)
    assert all(len(load_hits(p)) == 2 for p in paths.values())          # every retriever, baselines included


def test_mode_switch_is_refused(offline):
    retrieve(offline)
    offline.retrieval.mode = "hybrid"
    with pytest.raises(RuntimeError, match="bump run_id"):
        retrieve(offline)


def test_unindexed_document_is_refused(offline, tiny_doc):
    write_chunks(offline, windows(tiny_doc, doc_id="other_doc"))         # chunks of another document only
    with pytest.raises(RuntimeError, match="no chunks of tiny.*chunkeval chunk"):
        retrieve(offline)


def test_chunks_of_another_parse_are_refused(offline, tiny_doc):
    chunks = windows(tiny_doc)
    chunks[3].start += 1                                                # offsets no longer match the text
    chunks[3].end += 1
    write_chunks(offline, chunks)
    with pytest.raises(RuntimeError, match=r"1 chunk\(s\) don't match.*tiny:sentence_overlap:3"):
        retrieve(offline)


def test_metrics_leave_out_stale_files_outside_the_run(offline, caplog):
    retrieve(offline)
    offline.embedding.models["qwen3_06b"].query_prompt = "Instruct: find the passage\nQuery:"
    with caplog.at_level(logging.WARNING):
        df = score(offline, embedders=["bge_m3"])                         # `retrieve --embedder bge_m3`
    assert "qwen3_06b" not in set(df["embedder"]) and "bge_m3" in set(df["embedder"])
    assert "qwen3_06b papers/sentence_overlap" in caplog.text
    with pytest.raises(RuntimeError, match="stale right after retrieval"):
        score(offline, embedders=["qwen3_06b"])                           # in this run's scope: an error


# --- against the local Weaviate ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """One connection for the module; skips when WEAVIATE_API_KEY is unset or Weaviate isn't running."""
    secrets = load_secrets()
    if not secrets.weaviate_api_key:
        pytest.skip("WEAVIATE_API_KEY not set")
    try:
        c = indexing.connect(load_config(REPO_ROOT / "configs" / "default.yaml"),
                             secrets.weaviate_api_key.get_secret_value())
    except RuntimeError as e:
        pytest.skip(str(e))
    yield c
    c.close()


@pytest.fixture
def live(stage, client):
    """`stage` with its chunks indexed in Papers_sentence_overlap_pytest (deleted before and after)."""
    def clean():
        for name in client.collections.list_all(simple=True):
            if name.endswith("_pytest"):
                client.collections.delete(name)
    clean()
    indexing.run(stage, load_secrets(), corpora=["papers"])
    yield stage
    clean()


NAME = "Papers_sentence_overlap_pytest"


@pytest.mark.weaviate
def test_dense_search_in_one_named_vector(live, client):
    chunks, index = variant_chunks(live, "papers", VARIANT)
    vectors = load_embeddings(live, "qwen3_06b", "papers", VARIANT, [c.chunk_id for c in chunks])
    retriever = WeaviateRetriever(client, NAME, "qwen3_06b", live, index)
    for i in (0, len(chunks) // 2, len(chunks) - 1):
        hits = retriever.retrieve("unused in dense mode", vectors[i], k=3)
        assert hits[0] == hit_from_chunk(chunks[i], 1, hits[0].score)   # the chunk's own vector finds it
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)            # cosine similarity
        assert [h.rank for h in hits] == [1, 2, 3] and hits[0].score >= hits[1].score >= hits[2].score
    with pytest.raises(ValueError, match="no named vector 'bge-m3'"):
        WeaviateRetriever(client, NAME, "bge-m3", live)


@pytest.mark.weaviate
def test_hybrid_with_alpha_zero_is_bm25(live, client):
    live.retrieval.mode, live.retrieval.hybrid_alpha = "hybrid", 0.0
    chunks, index = variant_chunks(live, "papers", VARIANT)
    retriever = WeaviateRetriever(client, NAME, "bge_m3", live, index)
    hits = retriever.retrieve("replicates", np.zeros(1024, dtype=np.float32) + 1 / 32, k=3)
    assert "replicates" in next(c.text for c in chunks if c.chunk_id == hits[0].chunk_id)


@pytest.mark.weaviate
def test_collection_of_other_chunks_is_refused(live, client, tiny_doc):
    chunks = windows(tiny_doc, size=180)                                # re-chunked, not re-indexed
    write_chunks(live, chunks)
    with pytest.raises(RuntimeError, match=r"chunks_sha256.*index --recreate"):
        WeaviateRetriever(client, NAME, "bge_m3", live, variant_chunks(live, "papers", VARIANT)[1])
    with pytest.raises(RuntimeError, match="no such collection"):
        WeaviateRetriever(client, "Papers_character_overlap_pytest", "bge_m3", live)
