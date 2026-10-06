"""Sentence chunker — packs whole sentences up to the token target.

Rule: segment `doc.text` into sentences, greedily pack them while the chunk stays <= target_tokens;
overlap = repeat the trailing 1–2 sentences of the previous chunk (capped by overlap_ratio).
Why: preserves complete semantic units; in the previous round it beat word-based chunking on
recall and faithfulness.

Reuse: the packing + backward-overlap loop of `create_overlapping_chunks`
(ragas_evaluation.py:218-261) is the right algorithm — port it, but:
  * sizes in tokens (TokenCounter), not `len(sentence.split(" "))`
  * sentence boundaries from a real segmenter, not the regex `(?<=\\.)\\s+`
  * keep char offsets (start of first sentence, end of last) instead of joining strings with " "
    — joining loses the original whitespace/newlines and breaks `text == doc.text[start:end]`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chunkeval.corpus.chunking.base import Chunker

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc


def sentence_spans(text: str, base_offset: int = 0) -> list[tuple[int, int]]:
    """Return (start, end) char spans of the sentences in `text`, shifted by base_offset.

    TODO(you):
      - `pysbd.Segmenter(language="en", clean=False, char_span=True).segment(text)` returns
        TextSpan objects with .start/.end. `clean=False` is essential: cleaning rewrites the text
        and the offsets stop matching.
      - pysbd handles "e.g.", "et al.", "Fig. 3", decimals — the cases the old regex broke on.
        spaCy (`en_core_web_sm` + `doc.sents`) is the heavier alternative; pick one and test it
        on a few nasty paragraphs from your corpus (citations, species names "Z. mays", units).
      - Segment block-by-block (paragraphs from CanonicalDoc.blocks), not the whole document:
        it is faster and a sentence should never span two blocks (e.g. heading + paragraph).
      - Tables: treat a whole table block as one "sentence" (atomic) unless it is > max_tokens.
    """
    raise NotImplementedError


class SentenceChunker(Chunker):
    method = "sentence"

    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        """TODO(you):
          - Sentences of the WHOLE document in order (this method ignores sections — that is the
            difference to RecursiveChunker; a chunk MAY cross a heading).
          - Greedy pack up to target_tokens. Decide: is the target a max (old behaviour, chunks end
            up smaller on average) or a soft target? Keep it consistent with RecursiveChunker.
          - A single sentence longer than max_tokens (long tables, reference lists) must be split
            — fall back to token windows for that sentence only.
          - Overlap: walk back over the previous chunk's sentences while the overlap budget
            (overlap_ratio * target_tokens) allows — this is exactly the old `reversed(chunks[-1])`
            loop. Guard against the infinite loop where the overlap alone fills a whole chunk.
        """
        raise NotImplementedError
