"""Recursive-chapter chunker — sentence packing that never crosses a section boundary.

Rule: section -> paragraph -> sentence. A chunk only contains text of ONE section (leaf section, or
a merged group of tiny siblings under the same parent). Overlap is sentence-based and stays inside
the section. Tables (+ caption) are atomic chunks.
Why: best method on papers in the previous round (faithfulness + recall); reports could not use it
because they had no detectable structure. Docling's section tree (parsing.py) now provides it for
both corpora — and reports are where the docx expects the biggest gain.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chunkeval.corpus.chunking.base import Chunker

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc, Section


class RecursiveChunker(Chunker):
    method = "recursive"

    def __init__(self, *args, min_section_tokens: int = 40, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_section_tokens = min_section_tokens

    def section_groups(self, doc: "CanonicalDoc") -> list[list["Section"]]:
        """Group leaf sections into the units that chunks may not cross.

        TODO(you):
          - Start from leaf sections (no children). A heading-only section ("2. Methods" directly
            followed by "2.1 ...") has no body — attach its heading text to the first child.
          - Merge adjacent tiny siblings (< min_section_tokens) that share the same parent; never
            merge across parents (that would recreate the "Methods glued to Discussion" problem).
          - Decide where the heading line goes: include it at the start of the first chunk of the
            section (it is part of doc.text, so offsets stay valid). Don't PREPEND headings to
            every chunk — that is a different technique ("contextual headers") and would confound
            the comparison; it's a good follow-up experiment though.
        """
        raise NotImplementedError

    def split_group(self, doc: "CanonicalDoc", group: list["Section"]) -> list[tuple[int, int]]:
        """Chunk one section group.

        TODO(you):
          - If the whole group fits in target_tokens → one chunk.
          - Else split at PARAGRAPH (block) boundaries first, and only fall back to sentence
            packing inside a paragraph that is itself too long — that's what makes it "recursive".
            Reuse `sentence.sentence_spans` + the same packing helper as SentenceChunker
            (factor the packing loop out so both chunkers share it).
          - Table blocks: their own chunk. If > max_tokens, split by rows and repeat the header
            row in each piece — careful: a repeated header is NOT contiguous in doc.text, so
            either (a) allow such chunks to carry `meta["synthetic"]=True` and exclude them from
            the text==doc.text[start:end] check, or (b) skip header repetition. Pick one and test it.
          - Overlap only between consecutive chunks of the SAME group.
        """
        raise NotImplementedError

    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        """TODO(you): concatenate `split_group` over `section_groups`, in document order."""
        raise NotImplementedError
