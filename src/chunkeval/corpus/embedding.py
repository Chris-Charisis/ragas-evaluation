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

Reads:  data/chunks/<corpus>/<method>_<ovl>.jsonl
Writes: data/embeddings/<embedder>/<corpus>/<method>_<ovl>.npy (+ ids file, same order)
Next:   indexing.py (vectors), retrieval.py (query encoding)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config


@dataclass
class EmbedderSpec:
    """One entry of cfg.embedding.models."""

    name: str                  # short id used as the Weaviate named vector, e.g. "bge_m3"
    model_id: str              # Hugging Face id
    query_prompt: str          # prepended to QUERIES only ("" for bge-m3)
    document_prompt: str       # prepended to documents ("" for both models)
    max_seq_length: int        # truncation length used at encode time
    dim: int                   # expected output dimension (1024 for both)
    primary: bool = False


class Embedder:
    """Wraps one sentence-transformers model with the query/document conventions of that model.

    TODO(you) — general:
      - `sentence_transformers.SentenceTransformer(model_id, device="cuda", model_kwargs=...,
        tokenizer_kwargs=...)`. Both models load with plain sentence-transformers (5.0 is installed);
        pooling is defined in each model's repo (bge-m3 = CLS, Qwen3 = last token), so you don't
        implement pooling yourself.
      - fp16 halves memory: `model_kwargs={"torch_dtype": torch.float16}`.
      - Set `model.max_seq_length = spec.max_seq_length` explicitly. Qwen3 supports a 32k context;
        your chunks are far smaller, and a smaller cap keeps the memory of a batch predictable.
      - Always `normalize_embeddings=True` → cosine == dot product, matching the Weaviate distance.
    """

    def __init__(self, spec: EmbedderSpec, device: str = "cuda") -> None:
        raise NotImplementedError

    def encode_documents(self, texts: list[str], batch_size: int) -> Any:
        """Encode chunk texts → float32 array (n, dim).

        TODO(you):
          - Neither model uses a document prompt. Keep the `document_prompt` hook anyway so the
            config, not the code, decides.
          - Sort texts by length before batching (and restore the order afterwards): padding a
            batch to its longest member wastes most of the compute on chunk lists with a long tail.
          - Assert the shape (n, spec.dim) and that the rows are unit-norm.
        """
        raise NotImplementedError

    def encode_queries(self, texts: list[str], batch_size: int) -> Any:
        """Encode questions → float32 array (n, dim), applying the model's QUERY convention.

        TODO(you):
          - bge-m3: no instruction — encode the raw question.
          - Qwen3-Embedding: queries need an instruction in the form
            "Instruct: <task description>\\nQuery:" (the model ships a generic "web search" one as
            prompt_name="query"). Qwen reports that a task-specific instruction helps by a few
            points — write one that describes YOUR task (scientific/agricultural questions →
            passages from papers and reports). Put it in the config, and pass it with
            `encode(..., prompt=spec.query_prompt)`. Documents get NO instruction.
          - Qwen3 uses last-token pooling: the model card recommends left padding
            (`tokenizer_kwargs={"padding_side": "left"}`) — check the current model card.
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Tokens in this model's tokenizer — used to verify no chunk exceeds max_seq_length."""
        raise NotImplementedError

    def unload(self) -> None:
        """Free the GPU memory.

        Why: the 16 GB GPU is shared with Ollama (the generator alone is ~10 GB, gpt-oss ~14 GB).
        Embed while no Ollama model is loaded (`ollama ps` empty; `llm.unload_ollama_model` or
        `ollama stop <model>`), and unload the embedders before the next LLM stage — otherwise
        Ollama silently offloads layers to the CPU and the LLM stage gets much slower.
        Hint: `del self.model`, `gc.collect()`, `torch.cuda.empty_cache()`.
        """
        raise NotImplementedError


def load_specs(cfg: Config) -> dict[str, EmbedderSpec]:
    """Build EmbedderSpecs from cfg.embedding.models, marking cfg.embedding.primary.

    TODO(you): validate that exactly one embedder is primary and that every `name` is a valid
    Weaviate named-vector name (letters, digits, underscores).
    """
    raise NotImplementedError


def embed_chunk_file(chunks_path: Path, embedder: Embedder, out_dir: Path, batch_size: int) -> Path:
    """Embed one chunk file and cache the vectors (one .npy + the chunk_ids in the same order).

    TODO(you):
      - `numpy.save` the (n, dim) float32 matrix; save the chunk_ids alongside (json/txt) and
        check on load that they match the chunk file — a silent row misalignment would attach
        the wrong vector to every chunk.
      - Skip if the cache exists and the chunk file has not changed (compare mtime or a hash).
      - Record embedder name + model_id + sentence-transformers version in a small metadata
        file next to the matrix.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """Embed every chunk file with every configured embedder, one model at a time.

    Hint: loop embedders in the OUTER loop (load model → embed all 12 chunk files → unload), not
    the inner one — loading a model takes far longer than embedding one file.
    """
    raise NotImplementedError


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
