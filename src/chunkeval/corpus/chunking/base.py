"""Shared types and utilities for all chunkers.

Why token-based sizes: the old chunkers counted words (`split(" ")`), but the embedding model
counts tokens — and scientific text (numbers, units, species names, acronyms) has a very uneven
words-to-tokens ratio. Sizes are measured in the PRIMARY embedder's tokenizer (bge-m3, set in
cfg.chunking.size_tokenizer) so "256 tokens" means the same thing for every method.

Note: with bge-m3 (8192-token context) the max chunk size is no longer imposed by the model —
cfg.chunking.max_tokens is now an experimental choice that keeps the size distribution
comparable across methods (e.g. a huge table becoming one 3000-token chunk would distort results).
The same chunks are embedded by the secondary model (Qwen3, 32k context), so one ruler suffices;
just verify the chunks also fit its max_seq_length (embedding.Embedder.count_tokens).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc


@dataclass
class Chunk:
    """One retrievable unit. `text == doc.text[start:end]` must ALWAYS hold."""

    chunk_id: str
    doc_id: str
    corpus: str
    method: str              # "character" | "sentence" | "recursive"
    overlap: bool
    start: int
    end: int
    text: str
    n_tokens: int
    section_path: list[str] = field(default_factory=list)   # filled by every chunker if known
    meta: dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """Wraps the size tokenizer (bge-m3 — an XLM-RoBERTa SentencePiece tokenizer).

    TODO(you):
      - `transformers.AutoTokenizer.from_pretrained(cfg.chunking.size_tokenizer)` — only the
        tokenizer is loaded here, not the 2 GB model.
      - Count WITHOUT special tokens (`add_special_tokens=False`); bge-m3 adds <s> and </s> at
        encode time, which is irrelevant at 8192 but keep it in mind if you ever cap near a limit.
      - XLM-R tokens are not bge-large WordPiece tokens: "256 tokens" is a slightly different amount
        of text than in the previous round. Report mean words per chunk too, so the docx numbers
        stay comparable.
      - Fast tokenizers support `return_offsets_mapping=True`, which maps each token to its char
        range. That is the key tool for the character chunker calibration and for cutting a text
        at "the char position of token N".
      - Tokenising every candidate string repeatedly is slow; cache or tokenise once per sentence.
    """

    def __init__(self, model_name: str) -> None:
        raise NotImplementedError

    def count(self, text: str) -> int:
        raise NotImplementedError

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        """Char span of every token in `text` (no special tokens)."""
        raise NotImplementedError


class Chunker(ABC):
    """Base class. Subclasses only decide WHERE to cut; offsets/ids/validation are shared."""

    method: str = "base"

    def __init__(self, counter: TokenCounter, target_tokens: int, overlap: bool,
                 overlap_ratio: float, max_tokens: int = 512) -> None:
        self.counter = counter
        self.target_tokens = target_tokens
        self.overlap = overlap
        self.overlap_ratio = overlap_ratio
        self.max_tokens = max_tokens

    @abstractmethod
    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        """Return (start, end) char spans of the chunks, in document order."""

    def chunk(self, doc: "CanonicalDoc") -> list[Chunk]:
        """Turn spans into validated Chunk objects.

        TODO(you):
          - Build `Chunk`s from `self.spans(doc)`; stable chunk_id e.g. f"{doc_id}:{method}:{ovl}:{i}".
          - Fill `section_path` for EVERY method (look up the section containing the chunk start)
            so analyses by section are possible — but don't feed it into the embedding text,
            or recursive would get an unfair advantage (see prompts/rag_answer.md).
          - Call `validate_chunks` before returning — fail loudly during development.
        """
        raise NotImplementedError


def validate_chunks(doc: "CanonicalDoc", chunks: list[Chunk], max_tokens: int) -> None:
    """Assert the invariants the evaluation relies on.

    TODO(you): check (1) text == doc.text[start:end], (2) n_tokens <= max_tokens,
    (3) the union of spans covers all non-whitespace of doc.text (no lost content),
    (4) no-overlap chunkers produce disjoint spans. The same checks live in tests/.
    """
    raise NotImplementedError


def size_stats(chunks: list[Chunk]) -> dict[str, float]:
    """count / mean / median / p5 / p95 / max of n_tokens — reported per (corpus, method, overlap).

    Why: sentence and recursive chunkers produce chunks SMALLER than the target on average, so at
    fixed top-k they hand the generator less text. Reporting this (and the budget-matched view in
    evaluation/retrieval_metrics.py) keeps the comparison honest. Hint: numpy.percentile.
    """
    raise NotImplementedError
