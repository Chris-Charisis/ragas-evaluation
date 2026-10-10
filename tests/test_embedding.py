"""Embedder conventions, the vector cache and the `embed` stage — mistakes here fail SILENTLY otherwise
(both models are 1024-dim, so a mix-up never raises an error, it just degrades retrieval).

Offline tests run the cache and the stage with `FakeEmbedder` (a vector per text from its hash, no
GPU). The `gpu` tests load both real models once (`embedders` fixture in conftest, ~10 s) and check
what only the real models show: prompts, pooling and padding, the two vector spaces.
"""
import json
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from chunkeval.core import llm
from chunkeval.core.config import Secrets
from chunkeval.core.schema import REPO_ROOT
from chunkeval.core.store import write_jsonl
from chunkeval.corpus import embedding
from chunkeval.corpus.chunking.base import Chunk, load_chunks
from chunkeval.corpus.embedding import (DEFAULT_OLLAMA_URL, cache_path, check_gpu_free, check_vectors,
                                        is_cached, load_embeddings, load_specs, run, variants)

EMBEDDERS = ["bge_m3", "qwen3_06b"]


def make_chunks(corpus: str, variant: str, n: int = 3) -> list[Chunk]:
    method, _, overlap = variant.partition("_")
    return [Chunk(chunk_id=f"doc:{variant}:{i}", doc_id="doc", corpus=corpus, method=method,
                  overlap=overlap == "overlap", start=10 * i, end=10 * i + 9,
                  text=f"{corpus} {variant} chunk {i}", n_tokens=5) for i in range(n)]


def chunks_file(cfg, corpus: str, variant: str):
    return cfg.paths.data_dir / "chunks" / corpus / f"{variant}.jsonl"


def chunk_ids(cfg, corpus: str, variant: str) -> list[str]:
    return [c.chunk_id for c in load_chunks(chunks_file(cfg, corpus, variant))]


@pytest.fixture
def stage(cfg, tmp_path):
    """cfg with data/ and the vector cache in tmp_path, and a 3-chunk file for every (corpus, variant)."""
    cfg.paths.data_dir = tmp_path / "data"
    cfg.embedding.cache_dir = tmp_path / "embeddings"
    for corpus in cfg.paths.corpora:
        for variant in variants(cfg):
            write_jsonl(chunks_file(cfg, corpus, variant), make_chunks(corpus, variant))
    return cfg


@pytest.fixture
def fake(monkeypatch):
    """Replace `Embedder` with a fake and the Ollama check with a stub; record what happens.

    events.loads: embedder names in load order; events.encoded: (embedder, text) pairs;
    events.gpu_checks: calls of check_gpu_free. Set events.gpu_busy to make the check fail, or
    events.fail_on to a text prefix to make encoding fail on it.
    """
    events = SimpleNamespace(loads=[], encoded=[], gpu_checks=0, gpu_busy=False, fail_on=None)

    class FakeEmbedder:
        def __init__(self, spec, device="cuda"):
            self.spec = spec
            events.loads.append(spec.name)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            pass

        def encode_documents(self, texts, batch_size):
            if events.fail_on and any(t.startswith(events.fail_on) for t in texts):
                raise RuntimeError("CUDA error: the fake GPU fell over")
            events.encoded += [(self.spec.name, t) for t in texts]
            rows = [np.random.default_rng(zlib.crc32(f"{self.spec.name}|{t}".encode())).standard_normal(self.spec.dim)
                    for t in texts]
            vectors = np.array(rows, dtype=np.float32).reshape(len(texts), self.spec.dim)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def fake_check(secrets):
        events.gpu_checks += 1
        if events.gpu_busy:
            raise RuntimeError("Ollama holds qwen3.5:9b-q8_0 on the GPU")

    monkeypatch.setattr(embedding, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embedding, "check_gpu_free", fake_check)
    return events


def run_all(cfg):
    run(cfg, Secrets(), corpora=list(cfg.paths.corpora), embedders=EMBEDDERS)


def reset(events):
    events.loads.clear()
    events.encoded.clear()
    events.gpu_checks = 0


# --- config ---------------------------------------------------------------------------------------

def test_load_specs(cfg):
    specs = load_specs(cfg)
    assert list(specs) == list(cfg.embedding.models)
    assert [name for name, s in specs.items() if s.primary] == [cfg.embedding.primary]
    for name, spec in specs.items():
        assert spec.name == name
        assert spec.model_id == cfg.embedding.models[name].model_id


def test_query_prompt_is_not_placeholder(cfg):
    """Qwen3's query prompt is a task-specific instruction in the "Instruct: ...\\nQuery:" format, not
    the repo's generic web-search one; bge-m3 takes none, and documents never get a prompt."""
    qwen = cfg.embedding.models["qwen3_06b"].query_prompt
    assert qwen.startswith("Instruct: ") and qwen.endswith("\nQuery:")
    assert "TODO" not in qwen and "web search" not in qwen
    assert cfg.embedding.models["bge_m3"].query_prompt == ""
    assert all(m.document_prompt == "" for m in cfg.embedding.models.values())


# --- the vector cache and the stage, offline ------------------------------------------------------

def test_run_embeds_every_file(stage, fake):
    """Each model is loaded once (embedders are the outer loop), after the GPU check, and every
    chunk file gets a matrix plus its metadata."""
    run_all(stage)
    assert fake.loads == EMBEDDERS and fake.gpu_checks == 2
    for name in EMBEDDERS:
        for corpus in stage.paths.corpora:
            for variant in variants(stage):
                vectors = load_embeddings(stage, name, corpus, variant, chunk_ids(stage, corpus, variant))
                assert vectors.shape == (3, stage.embedding.models[name].dim)
                meta = json.loads(cache_path(stage, name, corpus, variant).with_suffix(".json").read_text(encoding="utf-8"))
                assert (meta["embedder"], meta["corpus"], meta["variant"], meta["n"]) == (name, corpus, variant, 3)
                assert meta["settings"]["revision"] == stage.embedding.models[name].revision
                assert meta["settings"]["dtype"] == "float16"


def test_rerun_embeds_only_what_changed(stage, fake):
    run_all(stage)
    reset(fake)
    run_all(stage)
    assert fake.loads == [] and fake.gpu_checks == 0             # all cached: the GPU is never touched

    # `chunk` rewrites every file on each run; identical content must stay cached (sha, not mtime)
    path = chunks_file(stage, "papers", "sentence_overlap")
    write_jsonl(path, load_chunks(path))
    run_all(stage)
    assert fake.loads == []

    write_jsonl(path, make_chunks("papers", "sentence_overlap", n=4))
    run_all(stage)
    assert fake.loads == EMBEDDERS
    assert len(fake.encoded) == 2 * 4                             # only the changed file, by both models


def test_settings_change_reembeds_that_embedder_only(stage, fake):
    run_all(stage)
    reset(fake)
    stage.embedding.models["qwen3_06b"].query_prompt = "Instruct: something else\nQuery:"
    run_all(stage)
    assert fake.loads == []                                       # query prompts don't touch document vectors
    stage.embedding.models["qwen3_06b"].revision = "0" * 40      # a newer pinned commit of the model
    run_all(stage)
    assert fake.loads == ["qwen3_06b"]
    assert len(fake.encoded) == 3 * len(variants(stage)) * len(stage.paths.corpora)


def test_failed_run_resumes_where_it_stopped(stage, fake):
    """A crash mid-run keeps every finished file; the next run embeds only the rest."""
    fake.fail_on = "papers recursive_overlap"                     # the last papers file
    with pytest.raises(RuntimeError, match="fell over"):
        run_all(stage)
    assert not cache_path(stage, "bge_m3", "papers", "recursive_overlap").with_suffix(".json").exists()
    fake.fail_on = None
    reset(fake)
    run_all(stage)
    redone = {text.rsplit(" chunk", 1)[0] for name, text in fake.encoded if name == "bge_m3"}
    assert redone == {"papers recursive_overlap", *(f"reports {v}" for v in variants(stage))}
    for name in EMBEDDERS:
        for corpus in stage.paths.corpora:
            for variant in variants(stage):
                assert is_cached(cache_path(stage, name, corpus, variant), chunks_file(stage, corpus, variant),
                                 load_specs(stage)[name])


def test_cached_vectors_aligned_with_chunk_ids(stage, fake):
    """The cache loader refuses a matrix whose saved chunk_ids differ (order included) from the chunk file."""
    run(stage, Secrets(), corpora=["papers"], embedders=["bge_m3"])
    ids = chunk_ids(stage, "papers", "sentence_overlap")
    load_embeddings(stage, "bge_m3", "papers", "sentence_overlap", ids)
    for wrong in (ids[::-1], ids[:-1], [*ids, "doc:sentence_overlap:3"]):
        with pytest.raises(ValueError, match="chunk_ids"):
            load_embeddings(stage, "bge_m3", "papers", "sentence_overlap", wrong)


def test_load_refuses_missing_or_stale_cache(stage, fake):
    ids = chunk_ids(stage, "papers", "sentence_overlap")
    with pytest.raises(FileNotFoundError, match="chunkeval embed"):
        load_embeddings(stage, "bge_m3", "papers", "sentence_overlap", ids)
    run(stage, Secrets(), corpora=["papers"], embedders=["bge_m3"])
    stage.embedding.models["bge_m3"].document_prompt = "passage: "
    with pytest.raises(ValueError, match="chunkeval embed"):
        load_embeddings(stage, "bge_m3", "papers", "sentence_overlap", ids)


def test_run_without_chunk_files(cfg, tmp_path, fake):
    cfg.paths.data_dir = tmp_path / "data"
    with pytest.raises(FileNotFoundError, match="chunkeval chunk"):
        run_all(cfg)
    assert fake.loads == []


def test_run_stops_while_gpu_busy(stage, fake):
    """No model is loaded and nothing is written while Ollama holds the GPU."""
    fake.gpu_busy = True
    with pytest.raises(RuntimeError, match="Ollama holds"):
        run_all(stage)
    assert fake.loads == [] and not stage.embedding.cache_dir.exists()


@pytest.mark.parametrize("break_it", ["dtype", "shape", "nan", "norm"])
def test_check_vectors_fails_loudly(break_it):
    vectors = np.eye(3, 4, dtype=np.float32)
    check_vectors(vectors, 3, 4, "ok")
    broken = {"dtype": vectors.astype(np.float64), "shape": vectors[:2], "nan": np.where(vectors == 1, np.nan, 0).astype(np.float32),
              "norm": 2 * vectors}[break_it]
    with pytest.raises(ValueError):
        check_vectors(broken, 3, 4, break_it)


# --- the Ollama guard -----------------------------------------------------------------------------

@pytest.fixture
def ollama_ps(monkeypatch):
    """Stub `llm.loaded_ollama_models`: set state.loaded to a model list or an exception."""
    state = SimpleNamespace(loaded=[], urls=[])

    def loaded_ollama_models(base_url):
        state.urls.append(base_url)
        if isinstance(state.loaded, Exception):
            raise state.loaded
        return state.loaded

    monkeypatch.setattr(llm, "loaded_ollama_models", loaded_ollama_models)
    return state


def test_gpu_check_names_loaded_models(ollama_ps):
    ollama_ps.loaded = ["qwen3.5:9b-q8_0", "gpt-oss:20b"]
    with pytest.raises(RuntimeError, match="ollama stop qwen3.5:9b-q8_0 && ollama stop gpt-oss:20b"):
        check_gpu_free(Secrets(ollama_base_url="http://localhost:11435/v1"))
    assert ollama_ps.urls == ["http://localhost:11435/v1"]


def test_gpu_check_passes_when_nothing_loaded_or_unreachable(ollama_ps):
    check_gpu_free(Secrets())
    ollama_ps.loaded = requests.ConnectionError("connection refused")
    check_gpu_free(Secrets())
    assert ollama_ps.urls == [DEFAULT_OLLAMA_URL] * 2              # no OLLAMA_BASE_URL: Ollama's default port


# --- the real models (GPU) ------------------------------------------------------------------------

PASSAGES = [
    "Irrigated lowland rice in the Sahel yielded 6.2 t/ha, against 2.1 t/ha for rainfed upland rice.",
    "Soil samples were air-dried, sieved to 2 mm, and total nitrogen was measured by the Kjeldahl method.",
    "The training course on seed production gathered 35 extension agents from 12 countries in Cotonou.",
]
QUESTIONS = [
    "How much more did irrigated rice yield than rainfed rice?",
    "How was the nitrogen content of the soil determined?",
    "How many extension agents attended the seed production course?",
]


@pytest.mark.gpu
def test_output_shape_and_unit_norm(embedders):
    """Both embedders return float32 arrays of shape (n, 1024) with unit-norm rows, for documents
    and queries, and an empty (0, 1024) array for no input."""
    for name, e in embedders.items():
        check_vectors(e.encode_documents(PASSAGES, 2), len(PASSAGES), e.spec.dim, name)
        check_vectors(e.encode_queries(QUESTIONS, 2), len(QUESTIONS), e.spec.dim, name)
        check_vectors(e.encode_documents([], 2), 0, e.spec.dim, name)


@pytest.mark.gpu
def test_loaded_at_pinned_revision(embedders):
    """Every file of the model comes from the config's commit, not from whatever "main" is today."""
    for name, e in embedders.items():
        assert e.model[0].auto_model.config._commit_hash == e.spec.revision, name


@pytest.mark.gpu
def test_query_prompt_applied_to_queries_only(embedders):
    """For Qwen3, encode_queries(q) != encode_documents(q) (the instruction changes the vector);
    for bge-m3 the two are identical (no instruction)."""
    for name, e in embedders.items():
        cos = (e.encode_queries(QUESTIONS, 8) * e.encode_documents(QUESTIONS, 8)).sum(axis=1)
        if e.spec.query_prompt:
            assert cos.max() < 0.99, name                         # measured 0.91-0.94; 1.0 without it
        else:
            assert cos.min() > 0.99999, name


@pytest.mark.gpu
def test_embedders_produce_different_spaces(embedders):
    """Two different models are loaded, and the same text gets unrelated vectors from them —
    guards against loading the same model twice under two names."""
    a, b = embedders["bge_m3"], embedders["qwen3_06b"]
    assert a.model[0].auto_model.config.name_or_path != b.model[0].auto_model.config.name_or_path
    cos = (a.encode_documents(PASSAGES, 8) * b.encode_documents(PASSAGES, 8)).sum(axis=1)
    assert np.abs(cos).max() < 0.5


@pytest.mark.gpu
def test_retrieval_sanity_per_embedder(embedders):
    """For a tiny corpus (3 passages) each embedder ranks the matching passage first for every
    question — catches a broken query convention, pooling or padding."""
    for name, e in embedders.items():
        scores = e.encode_queries(QUESTIONS, 8) @ e.encode_documents(PASSAGES, 8).T
        assert list(scores.argmax(axis=1)) == [0, 1, 2], (name, scores.round(3))


@pytest.mark.gpu
def test_padding_does_not_change_vectors(embedders):
    """A short text gets the same vector alone and padded next to a long one, as document and as
    query — the direct check of pooling + padding side (Qwen3: last-token pooling, right padding)."""
    long = " ".join(PASSAGES * 15)
    for name, e in embedders.items():
        for encode, text in ((e.encode_documents, PASSAGES[1]), (e.encode_queries, QUESTIONS[1])):
            alone = encode([text], 1)[0]
            padded = encode([long, text], 2)[1]
            assert alone @ padded > 0.999, (name, encode.__name__)


@pytest.mark.gpu
def test_too_long_text_raises_instead_of_truncating(embedders):
    e = embedders["bge_m3"]
    with pytest.raises(ValueError, match="truncated silently"):
        e.encode_documents(["word " * (e.spec.max_seq_length + 10)], 1)


def chunk_files():
    return sorted((REPO_ROOT / "data" / "chunks").glob("*/*.jsonl"))


@pytest.mark.gpu
@pytest.mark.skipif(not chunk_files(), reason="no chunk files in data/chunks: run `chunkeval chunk` first")
def test_chunks_fit_every_model(embedders):
    """Every real chunk fits each model's max_seq_length in that model's own tokenizer (special
    tokens included). `encode_documents` would refuse a longer one anyway; this finds it before a run."""
    for name, e in embedders.items():
        for path in chunk_files():
            longest = max(e.count_tokens([c.text for c in load_chunks(path)]))
            assert longest <= e.spec.max_seq_length, (name, path.name, longest)
