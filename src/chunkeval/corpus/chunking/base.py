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

Size rule of every method: target_tokens is a SOFT target (the mean chunk lands near it), max_tokens
the hard cap. The character chunker calibrates its window so the mean is the target; sentence and
recursive pack whole units with `pack`, which adds a unit while that brings the chunk closer to the
target. So all methods hand the generator about the same amount of text per chunk, and the
comparison is about WHERE each method cuts, not how much it keeps.
"""
from __future__ import annotations

import bisect
import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from chunkeval.core.store import read_jsonl

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
    meta: dict[str, Any] = field(default_factory=dict)      # overlap_tokens: shared with the previous chunk


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read a chunk file written by the `chunk` stage (data/chunks/<corpus>/<variant>.jsonl) back
    into Chunks, in file order (relative paths resolve against the repo root). `embed` and `index`
    rely on that order: vectors are cached and checked per chunk_id, in order."""
    return [Chunk(**row) for row in read_jsonl(path)]


class TokenCounter:
    """Wraps the size tokenizer (bge-m3 — an XLM-RoBERTa SentencePiece tokenizer).

    - Only the tokenizer is loaded (~20 MB), not the 2 GB model.
    - Counts WITHOUT special tokens: bge-m3 adds <s> and </s> at encode time, which is irrelevant
      at 8192 but keep it in mind if you ever cap near a limit.
    - XLM-R tokens are not bge-large WordPiece tokens: "256 tokens" is a slightly different amount
      of text than in the previous round, so `size_stats` reports mean words per chunk too.
    - Its pre-tokenizer splits at whitespace first, so text cut at whitespace tokenises exactly as
      it does inside the whole document: the counts of consecutive sentences add up to the count
      of their concatenation, separators included (tested). `pack` relies on that and sums unit
      counts instead of re-tokenising every candidate chunk.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer      # imported here: slow, and only `chunk` needs it

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise ValueError(f"{model_name}: no fast tokenizer, so no offset mapping (char_offsets)")

    def count(self, text: str) -> int:
        return self.count_many([text])[0]

    def count_many(self, texts: list[str]) -> list[int]:
        """Token counts of several texts in one batched call (much faster than looping `count`)."""
        if not texts:
            return []
        # verbose=False: whole documents exceed model_max_length (8192), which would log a warning
        # meant for model inputs; here they are only counted.
        ids = self.tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
        return [len(x) for x in ids]

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        """Char span of every token in `text` (no special tokens). The span of a word's first
        token includes the whitespace before the word: "▁world" in "Hello world" is (5, 11)."""
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return [(start, end) for start, end in enc["offset_mapping"]]


@functools.cache
def load_counter(model_name: str) -> TokenCounter:
    """One TokenCounter per tokenizer and process: the `chunk` stage and the tests share it."""
    return TokenCounter(model_name)


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

    @property
    def variant(self) -> str:
        """"sentence_overlap", "character_no_overlap", ...: the chunk file name and id part."""
        return f"{self.method}_{'overlap' if self.overlap else 'no_overlap'}"

    @property
    def overlap_budget(self) -> float:
        """Tokens a chunk should repeat from the previous one (0 without overlap)."""
        return self.overlap_ratio * self.target_tokens if self.overlap else 0.0

    @abstractmethod
    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        """Return (start, end) char spans of the chunks, in document order."""

    def chunk(self, doc: "CanonicalDoc") -> list[Chunk]:
        """Turn spans into validated Chunk objects.

        - chunk_id = "<doc_id>:<variant>:<i>", stable as long as the doc and the chunker are.
        - section_path: the path of the section containing the chunk start, for EVERY method, so
          analyses by section are possible. Metadata only: never feed it into the embedding text,
          or recursive would get an unfair advantage (see prompts/rag_answer.md).
        - meta["overlap_tokens"]: tokens shared with the previous chunk of the doc; size_stats
          reports the mean, the overlap each method really ends up using.
        - `validate_chunks` runs before returning: a broken invariant fails loudly here, not as
          a silently wrong metric later.
        """
        spans = self.spans(doc)
        texts = [doc.text[start:end] for start, end in spans]
        # doc.text[start:prev_end] is "" when the chunk does not overlap the previous one
        repeated = [doc.text[start:prev_end] for (_, prev_end), (start, _) in zip([(0, 0), *spans], spans)]
        counts = self.counter.count_many(texts + repeated)

        block_starts = [b.start for b in doc.blocks]
        sections = {s.id: s for s in doc.sections}
        chunks = []
        for i, ((start, end), text, n_tokens, overlap_tokens) in enumerate(
                zip(spans, texts, counts[:len(spans)], counts[len(spans):])):
            # A chunk starts on a non-space character, so inside a block, never in a separator.
            block = doc.blocks[bisect.bisect_right(block_starts, start) - 1]
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}:{self.variant}:{i}", doc_id=doc.doc_id, corpus=doc.corpus,
                method=self.method, overlap=self.overlap, start=start, end=end, text=text,
                n_tokens=n_tokens, section_path=list(sections[block.section_id].path),
                meta={"overlap_tokens": overlap_tokens}))
        validate_chunks(doc, chunks, self.max_tokens)
        return chunks

    def pack_spans(self, text: str, units: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Chunk spans from consecutive unit spans (sentences, paragraphs) by `pack`; a unit longer
        than max_tokens is first split into `token_windows` of at most target_tokens."""
        sizes = self.counter.count_many([text[start:end] for start, end in units])
        if any(n > self.max_tokens for n in sizes):
            fitted: list[tuple[int, int]] = []
            for (start, end), n in zip(units, sizes):
                fitted += (token_windows(self.counter, text, start, end, self.target_tokens)
                           if n > self.max_tokens else [(start, end)])
            units = fitted
            sizes = self.counter.count_many([text[start:end] for start, end in units])
        ranges = pack(sizes, self.target_tokens, self.max_tokens, self.overlap_budget)
        return [(units[i][0], units[j - 1][1]) for i, j in ranges]


def pack(sizes: list[int], target: int, max_tokens: int, overlap_budget: float = 0.0) -> list[tuple[int, int]]:
    """Group consecutive units into chunks: [i, j) index ranges into `sizes` (token counts).

    The packing loop of the old `create_overlapping_chunks` (ragas_evaluation.py), on tokens, shared
    by SentenceChunker and RecursiveChunker so both follow the same size rules:
      * nearest to target: a unit joins the chunk while that brings the chunk closer to `target`
        (size + unit/2 <= target) and keeps it <= max_tokens. A chunk always takes at least one
        new unit, so a unit bigger than the target becomes a chunk of its own.
      * overlap: a chunk starts with trailing units of the previous one (`overlap_start`); they
        count towards its size, as the shared text does in the character chunker's windows.
      * tail: a last chunk with < target/2 new tokens joins the previous one if that stays
        <= max_tokens, instead of becoming a tiny chunk.
    Every unit must be <= max_tokens: split longer ones first (`token_windows`).
    """
    if any(s > max_tokens for s in sizes):
        raise ValueError(f"a unit of {max(sizes)} tokens exceeds max_tokens ({max_tokens}): split it first")
    ranges: list[tuple[int, int]] = []
    start = fresh = size = 0     # the chunk is units[start:j]; units[fresh:j] were not in the previous one
    for j, s in enumerate(sizes):
        if j > fresh and (size + s > max_tokens or size + s / 2 > target):
            ranges.append((start, j))
            # Never the whole previous chunk (start + 1), and room for unit j next to the overlap.
            start = overlap_start(sizes, start + 1, j, overlap_budget, min(target / 2, max_tokens - s))
            fresh, size = j, sum(sizes[start:j])
        size += s
    if sizes:
        ranges.append((start, len(sizes)))
    if len(ranges) > 1:
        (first, prev_end), (_, end) = ranges[-2], ranges[-1]
        if sum(sizes[prev_end:end]) < target / 2 and sum(sizes[first:end]) <= max_tokens:
            ranges[-2:] = [(first, end)]
    return ranges


def overlap_start(sizes: list[int], lo: int, end: int, budget: float, cap: float) -> int:
    """Index of the first unit of sizes[lo:end] to repeat at the start of the next chunk.

    Walks back from `end`, rounding the overlap to the nearest whole unit ("sentence methods round
    to whole sentences"), but always takes at least one unit so every boundary gets some context.
    Never more than `cap` tokens: the guard against an overlap that fills most of a chunk (a long
    last sentence then gets no overlap). budget <= 0 (no overlap) -> `end`.
    """
    if budget <= 0:
        return end
    k, total = end, 0
    while k > lo:
        s = sizes[k - 1]
        if total + s > cap or (k < end and total + s / 2 > budget):
            break
        total += s
        k -= 1
    return k


def token_windows(counter: TokenCounter, text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Split text[start:end] into near-equal pieces of at most ~`size` tokens.

    The fallback for a unit too long to be a chunk: a sentence > max_tokens (a long list or a
    run-on sentence with no full stop), or a character window of very dense text. Equal pieces,
    so no tiny last piece. Each cut moves to the nearest word start within a tenth of a window,
    so no word is split and the pieces tokenise as they did in place; only a "word" longer than
    that (a URL, a DNA sequence) is cut between subword tokens.
    """
    sub = text[start:end]
    offsets = counter.char_offsets(sub)
    n = len(offsets)
    k = math.ceil(n / (0.8 * size))      # pieces of <= 0.8*size, so the cut shifts below keep them <= size
    if k <= 1:
        return [(start, end)]
    radius = max(1, size // 10)

    def is_word_start(t: int) -> bool:
        c = offsets[t][0]
        return c == 0 or sub[c].isspace() or sub[c - 1].isspace()

    cuts, last = [0], 0                  # char cuts in `sub`; token index of the last cut
    for i in range(1, k):
        ideal = round(i * n / k)
        candidates = [t for t in range(max(last + 1, ideal - radius), min(n, ideal + radius + 1)) if is_word_start(t)]
        last = min(candidates, key=lambda t: abs(t - ideal)) if candidates else max(ideal, last + 1)
        cuts.append(offsets[last][0])
    cuts.append(len(sub))
    pieces = [trim(text, start + a, start + b) for a, b in zip(cuts, cuts[1:])]
    return [(a, b) for a, b in pieces if a < b]


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """The span moved inwards past leading/trailing whitespace (start == end if all whitespace)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def validate_chunks(doc: "CanonicalDoc", chunks: list[Chunk], max_tokens: int) -> None:
    """Raise ValueError if an invariant the evaluation relies on is broken:
    (1) text == doc.text[start:end], (2) n_tokens <= max_tokens, (3) the spans cover all
    non-whitespace of doc.text (no lost content), (4) spans in document order, and disjoint for
    no-overlap chunkers. tests/test_chunking.py checks the same for every method.
    """
    n = len(doc.text)
    covered = 0                          # doc.text[:covered] is inside some chunk
    for i, c in enumerate(chunks):
        where = f"{doc.doc_id}: chunk {c.chunk_id}"
        if not 0 <= c.start < c.end <= n:
            raise ValueError(f"{where}: span {c.start}..{c.end} empty or outside 0..{n}")
        if c.text != doc.text[c.start:c.end]:
            raise ValueError(f"{where}: text differs from doc.text[{c.start}:{c.end}]")
        if c.n_tokens > max_tokens:
            raise ValueError(f"{where}: {c.n_tokens} tokens > max_tokens ({max_tokens})")
        if i:
            prev = chunks[i - 1]
            if not (prev.start < c.start and prev.end < c.end):
                raise ValueError(f"{where}: out of document order after {prev.chunk_id}")
            if not c.overlap and c.start < prev.end:
                raise ValueError(f"{where}: overlaps {prev.chunk_id} in a no-overlap chunker")
        if gap := doc.text[covered:c.start].strip():
            raise ValueError(f"{where}: text before it is in no chunk: {gap[:80]!r}")
        covered = max(covered, c.end)
    if gap := doc.text[covered:].strip():
        raise ValueError(f"{doc.doc_id}: text after the last chunk is in no chunk: {gap[:80]!r}")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError(f"{doc.doc_id}: duplicate chunk ids")


def size_stats(chunks: list[Chunk]) -> dict[str, float]:
    """count / mean / median / p5 / p95 / min / max of n_tokens, mean words per chunk, and the mean
    overlap with the previous chunk of the same doc — reported per (corpus, method, overlap).

    Why: chunks of different sizes hand the generator different amounts of text at fixed top-k;
    reporting this (and the budget-matched view in evaluation/retrieval_metrics.py) keeps the
    comparison honest. Words per chunk keep the numbers comparable with the docx (word-based).
    """
    if not chunks:
        return {"count": 0}
    tokens = np.array([c.n_tokens for c in chunks])
    p5, median, p95 = np.percentile(tokens, [5, 50, 95])
    overlaps = [c.meta["overlap_tokens"] for prev, c in zip(chunks, chunks[1:]) if prev.doc_id == c.doc_id]
    return {
        "count": len(chunks),
        "mean": round(float(tokens.mean()), 1),
        "median": float(median),
        "p5": float(p5),
        "p95": float(p95),
        "min": int(tokens.min()),
        "max": int(tokens.max()),
        "words_mean": round(float(np.mean([len(c.text.split()) for c in chunks])), 1),
        "overlap_tokens_mean": round(float(np.mean(overlaps)), 1) if overlaps else 0.0,
    }
