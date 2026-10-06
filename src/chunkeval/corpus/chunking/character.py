"""Character chunker — the structure-agnostic baseline.

Rule: fixed-size windows over `doc.text`, ignoring sentences and sections.
Why keep it: it is the baseline the other methods must beat (previously "words"); it is also what
many production systems use by default.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chunkeval.corpus.chunking.base import Chunker

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc


class CharacterChunker(Chunker):
    method = "character"

    def calibrate(self, sample_docs: list["CanonicalDoc"]) -> int:
        """Pick the window size in characters so the MEAN chunk is ~target_tokens.

        TODO(you):
          - Estimate chars-per-token on a sample of the corpus (it differs between papers and
            reports — numbers/units tokenise into many pieces). Calibrate per corpus.
          - Alternative worth considering: cut directly at token boundaries using
            `TokenCounter.char_offsets` (a "token window" chunker). Then sizes are exact and no
            calibration is needed. Either is fine — document which one you chose, since the user
            question was about *character* splitting.
          - Whatever you choose, a window must never exceed max_tokens (check, and shrink if needed).
        """
        raise NotImplementedError

    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        """TODO(you):
          - Step = window - overlap_chars, where overlap_chars = round(window * overlap_ratio) if
            self.overlap else 0.
          - Decide on boundary handling: pure fixed windows cut words in half. A tiny tweak
            (snap to the nearest whitespace) keeps it a "character" method but avoids broken
            tokens — mention it in the report if you do it, because it changes the baseline.
          - The last window may be tiny; decide whether to merge it into the previous one.
        """
        raise NotImplementedError
