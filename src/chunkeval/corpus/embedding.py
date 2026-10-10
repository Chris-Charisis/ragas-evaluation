"""Embedding models: bge-m3 (primary) and Qwen3-Embedding-0.6B (secondary robustness check).

Why two embedders:
  * PRIMARY = BAAI/bge-m3 — the model the main 12-config grid is evaluated with (span metrics +
    RAG answers + Ragas). 8192-token context (no more 512 cap → larger chunk sizes can be tested),
    1024-dim dense vectors, no query instruction needed.
  * SECONDARY = Qwen/Qwen3-Embedding-0.6B — only the deterministic span metrics are recomputed with
    it. If the chunker ranking (e.g. recursive > sentence > character) is the same under both
    embedders, the conclusion is robust; the primary-vs-secondary gap also tells you how much an
    embedder upgrade is worth compared with a chunker change.

Both models are stored side by side as Weaviate NAMED VECTORS in the same collection (indexing.py),
so it is still 12 collections, and retrieval picks the vector space with `target_vector`.

!! Pitfall: both models output 1024-dim vectors. The old dimension check
   (`_assert_matching_dimensions`, ragas_evaluation.py:646) can NOT detect a query embedded with
   the wrong model — it would silently return garbage neighbours. Always go through named vectors
   and the embedder name; never pass a "bare" vector around without its embedder name.

Pinned models: each model loads at the Hub commit in its config `revision`, and that revision is
part of the cached vectors' settings. So documents (cached here) and queries (retrieval.py) always
come from the same weights, even if the authors push a new version mid-experiment.

GPU: the 16 GB card is shared with Ollama. `run` calls `check_gpu_free` before loading each model
and stops while Ollama holds one (`ollama ps` must be empty); peak use is ~1.5 GB for bge-m3 and
~3.4 GB for Qwen3 at batch 32 (dev chunks).

Reads:  data/chunks/<corpus>/<method>_<ovl>.jsonl
Writes: data/embeddings/<embedder>/<corpus>/<method>_<ovl>.npy   (n, dim) float32, rows in chunk-file order
        data/embeddings/<embedder>/<corpus>/<method>_<ovl>.json  chunk_ids in row order + what made the vectors
Next:   indexing.py (vectors, via `load_embeddings`), retrieval.py (query encoding)
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config
from chunkeval.core.store import write_json
from chunkeval.corpus.chunking.base import load_chunks

log = logging.getLogger(__name__)

# Measured on 271 dev chunks against fp32: fp16 gives cosine >= 0.99996 and identical top-5
# neighbours for both models; bf16 (Qwen3's native dtype) only >= 0.9993, and 1-1.5% of the
# top-5 neighbours change. So fp16 for both: half the memory and ~3x the speed of fp32 at no
# measurable cost.
DTYPE = "float16"

# Where `check_gpu_free` looks when OLLAMA_BASE_URL is not set: Ollama's default port.
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


@dataclass
class EmbedderSpec:
    """One entry of cfg.embedding.models."""

    name: str                  # short id used as the Weaviate named vector, e.g. "bge_m3"
    model_id: str              # Hugging Face id
    revision: str              # Hub commit hash the model is loaded at (pinned in the config)
    query_prompt: str          # prepended to QUERIES only ("" for bge-m3)
    document_prompt: str       # prepended to documents ("" for both models)
    max_seq_length: int        # longest input in tokens; longer texts raise instead of being truncated
    dim: int                   # expected output dimension (1024 for both)
    primary: bool = False

    @property
    def settings(self) -> dict[str, str | int]:
        """Everything that changes a DOCUMENT vector. Cached vectors made with other settings are
        stale. The query prompt isn't here: it only changes query vectors (retrieval.py's cache)."""
        return {"model_id": self.model_id, "revision": self.revision, "document_prompt": self.document_prompt,
                "max_seq_length": self.max_seq_length, "dtype": DTYPE}


def load_specs(cfg: Config) -> dict[str, EmbedderSpec]:
    """Every embedder of cfg.embedding.models by name, the primary one marked. Names and the
    primary/secondary lists are validated when the config loads (schema.Embedding)."""
    return {name: EmbedderSpec(name=name, primary=name == cfg.embedding.primary, **m.model_dump())
            for name, m in cfg.embedding.models.items()}


def check_vectors(vectors: np.ndarray, n: int, dim: int, where: str) -> None:
    """Raise ValueError unless `vectors` is a finite float32 (n, dim) matrix with unit-norm rows."""
    if vectors.dtype != np.float32 or vectors.shape != (n, dim):
        raise ValueError(f"{where}: expected float32 ({n}, {dim}), got {vectors.dtype} {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{where}: NaN or inf in the vectors")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        raise ValueError(f"{where}: rows are not unit-norm (norms {norms.min():.4f}..{norms.max():.4f})")


class Embedder:
    """One sentence-transformers model on the GPU, with that model's query/document conventions.

    Use it as a context manager, so the GPU is freed even when encoding fails:
        with Embedder(spec) as embedder:
            vectors = embedder.encode_documents(texts, cfg.embedding.batch_size)

    Conventions (tests/test_embedding.py checks each one on the real models):
      - fp16 weights (DTYPE); vectors come back float32 and unit-norm, so cosine == dot product,
        matching the Weaviate distance.
      - Pooling is defined in each model's repo (bge-m3: CLS, Qwen3: last token), not here. Qwen3's
        tokenizer pads on the right, and sentence-transformers' last-token pooling takes the last
        NON-padding token, so the model card's left-padding advice (meant for code that takes
        position -1) isn't needed.
      - sentence-transformers sorts each `encode` call by text length before batching and restores
        the order, so batches pad to similar lengths without any code here.
      - A text longer than max_seq_length raises instead of being truncated silently.
    """

    def __init__(self, spec: EmbedderSpec, device: str = "cuda") -> None:
        import torch                                     # imported here: slow, and only the GPU path needs it
        from sentence_transformers import SentenceTransformer

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"{spec.name}: device {device!r}, but torch sees no CUDA GPU")
            free, total = torch.cuda.mem_get_info()      # includes other processes, unlike torch's own stats
            log.info("GPU: %.1f of %.1f GB free", free / 1e9, total / 1e9)
            torch.cuda.reset_peak_memory_stats()

        start = time.monotonic()
        self.spec = spec
        # `revision` applies to every file: modules.json, pooling config, tokenizer and weights.
        # bge-m3 has only pytorch_model.bin, so transformers also starts a background download of a
        # bot-made model.safetensors from a Hub PR (~2.2 GB, once). It isn't loaded: the weights
        # are the .bin of the pinned commit.
        self.model = SentenceTransformer(spec.model_id, revision=spec.revision, device=device,
                                         model_kwargs={"torch_dtype": getattr(torch, DTYPE)})
        # The repos default to 8192 (bge-m3) and 32k (Qwen3) tokens; a lower cap keeps the memory
        # of a batch predictable, and `_encode` refuses longer texts.
        self.model.max_seq_length = spec.max_seq_length
        dim = self.model.get_sentence_embedding_dimension()
        if dim != spec.dim:
            raise ValueError(f"{spec.name}: {spec.model_id} outputs {dim}-dim vectors, config says {spec.dim}")
        loaded = getattr(self.model[0].auto_model.config, "_commit_hash", None)
        if loaded != spec.revision:
            raise RuntimeError(f"{spec.name}: asked for revision {spec.revision}, the loaded model "
                               f"reports {loaded}")
        log.info("%s: loaded %s at %s (%s) in %.1fs",
                 spec.name, spec.model_id, spec.revision[:12], DTYPE, time.monotonic() - start)

    def __enter__(self) -> Embedder:
        return self

    def __exit__(self, *exc_info) -> None:
        self.unload()

    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode chunk texts → float32 (n, dim), with the document prompt ("" for both models;
        kept so the config, not the code, decides)."""
        return self._encode(texts, self.spec.document_prompt, batch_size)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode questions → float32 (n, dim), with the model's QUERY prompt: none for bge-m3,
        "Instruct: <task>\\nQuery:" for Qwen3 (set in the config; the question follows directly).
        Never encode questions with `encode_documents`: for Qwen3 that silently drops the instruction."""
        return self._encode(texts, self.spec.query_prompt, batch_size)

    def _encode(self, texts: list[str], prompt: str, batch_size: int) -> np.ndarray:
        import torch

        if not texts:
            return np.empty((0, self.spec.dim), dtype=np.float32)
        where = f"{self.spec.name} ({len(texts)} texts)"
        longest = max(self.count_tokens([prompt + t for t in texts]))
        if longest > self.spec.max_seq_length:
            raise ValueError(f"{where}: a text of {longest} tokens exceeds max_seq_length "
                             f"({self.spec.max_seq_length}); it would be truncated silently")
        # prompt is passed even when "": a non-None prompt overrides any default prompt the model
        # repo defines, so the config alone decides.
        vectors = self.model.encode(texts, prompt=prompt, batch_size=batch_size,
                                    convert_to_tensor=True, show_progress_bar=False)
        # The model's Normalize module ran in fp16: normalise again in float32, so rows are unit-norm
        # to ~1e-7 and dot products are exact cosines.
        vectors = torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy()
        check_vectors(vectors, len(texts), self.spec.dim, where)
        return vectors

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Tokens per text in this model's tokenizer, special tokens included — what the model
        actually reads, to compare with max_seq_length."""
        ids = self.model.tokenizer(texts, add_special_tokens=True, verbose=False)["input_ids"]
        return [len(x) for x in ids]

    def unload(self) -> None:
        """Free the GPU memory (a second call is a no-op).

        Why: the 16 GB GPU is shared with Ollama (the generator alone is ~10 GB, gpt-oss ~14 GB).
        Unload the embedders before the next LLM stage — otherwise Ollama silently offloads layers
        to the CPU and the LLM stage gets much slower.
        """
        if self.model is None:
            return
        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("%s: unloaded (peak GPU memory %.2f GB)", self.spec.name, torch.cuda.max_memory_allocated() / 1e9)


def check_gpu_free(secrets: Secrets) -> None:
    """Stop before an embedder is loaded while Ollama holds a model on the GPU.

    Sharing the 16 GB card doesn't fail loudly: the embedders may still fit next to a small model,
    but a model Ollama loads during the run gets pushed onto the CPU without an error. So `ollama
    ps` must be empty. The model is named, not unloaded: it may belong to another project's run.
    An unreachable server holds nothing, so that passes.
    """
    from chunkeval.core.llm import loaded_ollama_models   # imported here: pulls in the openai SDK

    base_url = secrets.ollama_base_url or DEFAULT_OLLAMA_URL
    try:
        loaded = loaded_ollama_models(base_url)
    except requests.RequestException:
        log.info("Ollama not reachable at %s: no local LLM holds the GPU", base_url)
        return
    if loaded:
        raise RuntimeError(f"Ollama holds {', '.join(loaded)} on the GPU, which embedding shares. "
                           f"Free it first: {' && '.join(f'ollama stop {m}' for m in loaded)}")
    log.info("Ollama at %s: no model loaded", base_url)


# ---------------------------------------------------------------------------------------------
# The vector cache
# ---------------------------------------------------------------------------------------------

def variants(cfg: Config) -> list[str]:
    """The chunk-file names of every (method, overlap) in the config, as `chunk` writes them."""
    return [f"{m}_{'overlap' if o else 'no_overlap'}" for m in cfg.chunking.methods for o in cfg.chunking.overlap]


def cache_path(cfg: Config, embedder: str, corpus: str, variant: str) -> Path:
    """data/embeddings/<embedder>/<corpus>/<variant>.npy; its metadata is the .json next to it."""
    return cfg.embedding.cache_dir / embedder / corpus / f"{variant}.npy"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_cached(npy_path: Path, chunks_path: Path, spec: EmbedderSpec) -> bool:
    """True if `npy_path` holds the vectors of exactly this chunk file, made with these settings.

    Compares the chunk file's sha256, not its mtime: `chunk` rewrites every file on each run, and
    an unchanged output must not cost a re-embedding.
    """
    meta_path = npy_path.with_suffix(".json")
    if not (npy_path.exists() and meta_path.exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("chunks_sha256") == _sha256(chunks_path) and meta.get("settings") == spec.settings


def _save_npy(path: Path, vectors: np.ndarray) -> None:
    """np.save, atomically: write a temp file in the target folder, then swap it in (as store.write_jsonl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        try:
            np.save(tmp, vectors)
        except BaseException:
            tmp.close()
            os.unlink(tmp.name)
            raise
    os.replace(tmp.name, path)


def embed_chunk_file(chunks_path: Path, embedder: Embedder, npy_path: Path, batch_size: int) -> int:
    """Embed one chunk file and cache the vectors; return the number of chunks.

    Writes the (n, dim) float32 matrix to `npy_path` and, next to it, a .json with the chunk_ids in
    row order and what made the vectors: the settings (model id, Hub revision, prompt, ...), library
    versions, and the chunk file's sha256 (for `is_cached`). The .json is removed first and written
    last, so it only ever sits next to the complete matrix it describes.
    """
    sha = _sha256(chunks_path)
    chunks = load_chunks(chunks_path)
    corpus, variant = chunks_path.parent.name, chunks_path.stem
    start = time.monotonic()
    vectors = embedder.encode_documents([c.text for c in chunks], batch_size)
    seconds = time.monotonic() - start

    meta_path = npy_path.with_suffix(".json")
    meta_path.unlink(missing_ok=True)
    _save_npy(npy_path, vectors)
    spec = embedder.spec
    write_json(meta_path, {
        "embedder": spec.name,
        "model_id": spec.model_id,
        "settings": spec.settings,
        "dim": spec.dim,
        "corpus": corpus,
        "variant": variant,
        "chunks_sha256": sha,
        "n": len(chunks),
        "versions": {p: version(p) for p in ("sentence-transformers", "transformers", "torch")},
        "created": datetime.now().isoformat(timespec="seconds"),
        "chunk_ids": [c.chunk_id for c in chunks],
    })
    log.info("%s %s/%s: %d chunks in %.1fs", spec.name, corpus, variant, len(chunks), seconds)
    return len(chunks)


def load_embeddings(cfg: Config, embedder: str, corpus: str, variant: str, chunk_ids: list[str]) -> np.ndarray:
    """The cached (n, dim) vectors of one chunk file, row i belonging to chunk_ids[i].

    Raises instead of returning vectors that would silently describe the wrong thing:
      - FileNotFoundError: nothing cached for this file;
      - ValueError: the cached chunk_ids differ from `chunk_ids` (order included) — every object
        would get its neighbour's vector; the cache was made with other settings than the
        config's; or the matrix is malformed.
    The fix is always `chunkeval embed`, which re-embeds exactly the stale files.
    """
    spec = load_specs(cfg)[embedder]
    npy_path = cache_path(cfg, embedder, corpus, variant)
    meta_path = npy_path.with_suffix(".json")
    where = f"{embedder} {corpus}/{variant}"
    if not meta_path.exists():
        raise FileNotFoundError(f"{where}: no cached vectors in {npy_path.parent}, run `chunkeval embed`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["settings"] != spec.settings:
        raise ValueError(f"{where}: cached with {meta['settings']}, the config says {spec.settings}; "
                         f"run `chunkeval embed`")
    cached = meta["chunk_ids"]
    if cached != chunk_ids:
        row = next((i for i, (a, b) in enumerate(zip(cached, chunk_ids)) if a != b), min(len(cached), len(chunk_ids)))
        raise ValueError(f"{where}: cached chunk_ids ({len(cached)}) differ from the chunk file's "
                         f"({len(chunk_ids)}) from row {row} on; run `chunkeval embed`")
    vectors = np.load(npy_path)
    check_vectors(vectors, len(chunk_ids), spec.dim, where)
    return vectors


def run(cfg: Config, secrets: Secrets, corpora: list[str], embedders: list[str]) -> None:
    """CLI entry (`chunkeval embed`): embed every chunk file of `corpora` with each of `embedders`
    (default: the primary, then the secondary ones), skipping files whose cache is up to date.

    Embedders are the OUTER loop (load a model, embed all its files, unload) because loading takes
    far longer than embedding one file. A model is loaded only when one of its files needs
    embedding, and only after `check_gpu_free`: a re-run with everything cached never touches the GPU.
    """
    specs = load_specs(cfg)
    files = [(corpus, variant, cfg.paths.data_dir / "chunks" / corpus / f"{variant}.jsonl")
             for corpus in corpora for variant in variants(cfg)]
    missing = [str(path) for _, _, path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} chunk files missing, run `chunkeval chunk` first: {missing}")

    for name in embedders:
        spec = specs[name]
        todo = [(path, cache_path(cfg, name, corpus, variant)) for corpus, variant, path in files
                if not is_cached(cache_path(cfg, name, corpus, variant), path, spec)]
        if not todo:
            log.info("%s: all %d chunk files already embedded, model not loaded", name, len(files))
            continue
        log.info("%s: %d of %d chunk files to embed", name, len(todo), len(files))
        check_gpu_free(secrets)                          # before EACH model: Ollama may have loaded one meanwhile
        with Embedder(spec) as embedder:
            for chunks_path, npy_path in todo:
                embed_chunk_file(chunks_path, embedder, npy_path, cfg.embedding.batch_size)


# ---------------------------------------------------------------------------------------------
# Optional follow-up — bge-m3 sparse ("lexical weights") retrieval.
# bge-m3 can also output a learned sparse vector, a stronger lexical signal than BM25 for rare
# terms and acronyms (the docx's hybrid-retrieval argument). Weaviate (client 4.22) has no
# user-provided sparse vectors, so this would run client-side: take Weaviate's dense top-N,
# re-score with sparse similarity, fuse. Needs the FlagEmbedding package (not installed):
#   `FlagEmbedding.BGEM3FlagModel(...).encode(texts, return_dense=..., return_sparse=True)`
#   → "lexical_weights"; `model.compute_lexical_matching_score(q, d)`.
# Keep it out of the core grid; compare against Weaviate's BM25 hybrid on the best chunker.
# ---------------------------------------------------------------------------------------------
