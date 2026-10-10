"""Character chunker — the structure-agnostic baseline.

Rule: fixed-size windows over `doc.text`, ignoring sentences and sections.
Why keep it: it is the baseline the other methods must beat (previously "words"); it is also what
many production systems use by default.

Choices (mention them in the report, they define the baseline):
  * Character windows, not token windows: the research question is about *character* splitting.
    The window size in chars is calibrated per corpus (`calibrate`) so the MEAN chunk is
    target_tokens; individual chunks vary with the local text density (numbers, units).
  * Each cut moves to the nearest whitespace (at most SNAP_CHARS away), so no word is cut in half:
    the old word-based baseline never split words either, and a split word gives two broken
    tokens at every boundary. Beyond SNAP_CHARS (a URL) the cut stays mid-word.
  * A last window with less than half a window of new text joins the previous one.
  * A window over max_tokens (very dense text) is split into token windows; does not happen on
    the dev docs (max ~1.5x the mean), logged when it does.
"""
from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from chunkeval.corpus.chunking.base import Chunker, token_windows, trim

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc

log = logging.getLogger(__name__)

SNAP_CHARS = 30


def snap_to_space(text: str, pos: int, radius: int = SNAP_CHARS) -> int:
    """The whitespace position nearest to `pos` (the earlier one on a tie), or `pos` itself if
    there is none within `radius`."""
    for d in range(radius + 1):
        for p in (pos - d, pos + d):
            if 0 < p < len(text) and text[p].isspace():
                return p
    return pos


def windows(text: str, window: int, overlap_chars: int = 0) -> list[tuple[int, int]]:
    """Windows of ~`window` chars over `text`, cut at whitespace; consecutive windows share
    ~`overlap_chars` (step = window - overlap_chars). Spans are trimmed of whitespace."""
    radius = min(SNAP_CHARS, window // 4)    # a cut never moves back to (or before) the window start
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = trim(text, 0, n)[0]
    while start < n:
        end = n if start + window >= n else snap_to_space(text, start + window, radius)
        spans.append(trim(text, start, end))
        if end >= n:
            break
        nxt = snap_to_space(text, end - overlap_chars, radius) if overlap_chars else end
        if nxt <= start:                     # overlap nearly a whole window: still move forward
            nxt = end
        start = trim(text, nxt, n)[0]
    if len(spans) > 1 and spans[-1][1] - max(spans[-1][0], spans[-2][1]) < window / 2:
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    return [(a, b) for a, b in spans if a < b]


class CharacterChunker(Chunker):
    method = "character"

    def __init__(self, *args, window_chars: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_chars = window_chars     # set by `calibrate` (or given directly)

    def calibrate(self, sample_docs: list["CanonicalDoc"]) -> int:
        """Pick the window size in characters so the MEAN chunk is ~target_tokens, store and return it.

        Chars per token differ between papers and reports (numbers/units tokenise into many
        pieces), so `chunk` calibrates per corpus, on all its docs. First guess from the corpus
        density, then one correction on the actual windows (snapping, trimming and the merged
        last window shift the mean a little). Always measured without overlap, so both overlap
        variants of a corpus get the same window.
        """
        texts = [doc.text for doc in sample_docs]
        window = round(self.target_tokens * sum(map(len, texts)) / sum(self.counter.count_many(texts)))
        sizes = self.counter.count_many([text[a:b] for text in texts for a, b in windows(text, window)])
        self.window_chars = round(window * self.target_tokens / statistics.mean(sizes))
        return self.window_chars

    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        if self.window_chars is None:
            raise RuntimeError("CharacterChunker: call calibrate() first (or pass window_chars)")
        overlap_chars = round(self.window_chars * self.overlap_ratio) if self.overlap else 0
        spans = windows(doc.text, self.window_chars, overlap_chars)
        sizes = self.counter.count_many([doc.text[a:b] for a, b in spans])
        if all(n <= self.max_tokens for n in sizes):
            return spans
        fitted: list[tuple[int, int]] = []
        for (a, b), n in zip(spans, sizes):
            if n > self.max_tokens:
                log.warning("%s: a %d-char window has %d tokens > max_tokens (%d), split into token windows",
                            doc.doc_id, b - a, n, self.max_tokens)
                fitted += token_windows(self.counter, doc.text, a, b, self.target_tokens)
            else:
                fitted.append((a, b))
        return fitted
