"""Sentence chunker — packs whole sentences up to the token target.

Rule: segment `doc.text` into sentences, pack them greedily towards target_tokens (`base.pack`:
a sentence joins while that brings the chunk closer to the target); overlap = repeat the trailing
sentence(s) of the previous chunk, rounded to whole sentences (`base.overlap_start`).
Why: preserves complete semantic units; in the previous round it beat word-based chunking on
recall and faithfulness.

Ported from `create_overlapping_chunks` (ragas_evaluation.py:218-261), but:
  * sizes in tokens (TokenCounter), not `len(sentence.split(" "))`
  * sentence boundaries from pysbd, not the regex `(?<=\\.)\\s+`
  * char offsets (start of first sentence, end of last) instead of strings joined with " ",
    which lost the original whitespace/newlines and broke `text == doc.text[start:end]`
  * the overlap counts towards the chunk size (the old loop reset the count without it, so
    chunks with overlap ran over the maximum)
"""
from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import pysbd

from chunkeval.corpus.chunking.base import Chunker, trim

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import Block, CanonicalDoc

# Block types that are one unit, never segmented: a heading is one line, and a table split into
# rows loses its header (tables are not in CanonicalDoc.text yet, see parsing.table_text).
ATOMIC_BLOCKS = {"heading", "table"}

# Abbreviations pysbd 0.3.4 wrongly ends a sentence after ("ca. 40 countries", "Eq. (1)",
# "approx. 3 t/ha"); "e.g.", "et al.", "Fig. 3", "Z. mays", "vs.", "cf." it already handles.
_NO_BREAK_AFTER = re.compile(r"\b(?:ca|approx|eqs?)\.$", re.IGNORECASE)


@functools.cache
def _segmenter() -> pysbd.Segmenter:
    # clean=False is essential: cleaning rewrites the text and the offsets stop matching.
    return pysbd.Segmenter(language="en", clean=False, char_span=True)


@functools.lru_cache(maxsize=65536)
def _sentence_bounds(text: str) -> tuple[tuple[int, int], ...]:
    """Sentence spans of `text`, cached: `chunk` segments every block four times (sentence and
    recursive, each with and without overlap), and pysbd is the slow part."""
    # Only the START of each pysbd sentence is used, and each sentence runs to the next start:
    # pysbd silently drops a sentence it cannot find again in the text, and this way the
    # previous sentence absorbs it instead of the text going missing.
    starts = sorted({0, *(span.start for span in _segmenter().segment(text))})
    spans: list[tuple[int, int]] = []
    for a, b in zip(starts, [*starts[1:], len(text)]):
        a, b = trim(text, a, b)
        if a == b:
            continue
        if spans and _NO_BREAK_AFTER.search(text[spans[-1][0]:spans[-1][1]]):
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return tuple(spans)


def sentence_spans(text: str, base_offset: int = 0) -> list[tuple[int, int]]:
    """Return (start, end) char spans of the sentences in `text`, shifted by base_offset.

    pysbd handles "e.g.", "et al.", "Fig. 3", "Z. mays", decimals — the cases the old regex broke
    on — plus the fixes in _NO_BREAK_AFTER. Spans are trimmed and together cover all non-space
    text. Call it per block (`block_sentences`): faster than the whole document, and a sentence
    never spans two blocks (e.g. heading + paragraph).
    """
    return [(base_offset + a, base_offset + b) for a, b in _sentence_bounds(text)]


def block_sentences(doc: "CanonicalDoc", block: "Block") -> list[tuple[int, int]]:
    """The units of one block in doc.text offsets: its sentences, or the whole block for a
    heading or table (ATOMIC_BLOCKS). A unit longer than max_tokens is split later, in
    `Chunker.pack_spans`."""
    if block.type in ATOMIC_BLOCKS:
        return [(block.start, block.end)]
    return sentence_spans(doc.text[block.start:block.end], block.start)


class SentenceChunker(Chunker):
    method = "sentence"

    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        """Sentences of the WHOLE document in order, packed by `pack_spans`. This method ignores
        sections — that is the difference to RecursiveChunker: a chunk MAY cross a heading."""
        units = [span for block in doc.blocks for span in block_sentences(doc, block)]
        return self.pack_spans(doc.text, units)
