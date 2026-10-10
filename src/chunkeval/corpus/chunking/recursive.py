"""Recursive-chapter chunker — sentence packing that never crosses a section boundary.

Rule: section -> paragraph -> sentence. A chunk only contains text of ONE section group (a leaf
section, or a merged run of tiny siblings under the same parent). Overlap is sentence-based and
stays inside the group. Tables (+ caption) will be atomic units once parsing emits them.
Why: best method on papers in the previous round (faithfulness + recall); reports could not use it
because they had no detectable structure. Docling's section tree (parsing.py) now provides it for
both corpora — and reports are where the docx expects the biggest gain.

Choices:
  * The heading line stays where it is in doc.text: at the start of the first chunk of its
    section. It is NOT prepended to every chunk — that is a different technique ("contextual
    headers") and would confound the comparison; a good follow-up experiment though.
  * Same size rules as SentenceChunker (`base.pack`, soft target), so the two differ only in
    where they may cut.
  * Tables (future): an atomic unit; one over max_tokens is split into token windows WITHOUT
    repeating the header row, so `text == doc.text[start:end]` holds for every chunk.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chunkeval.corpus.chunking.base import Chunker
from chunkeval.corpus.chunking.sentence import block_sentences

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc, Section


class RecursiveChunker(Chunker):
    method = "recursive"

    def __init__(self, *args, min_section_tokens: int = 40, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_section_tokens = min_section_tokens

    def section_groups(self, doc: "CanonicalDoc") -> list[list["Section"]]:
        """Group sections into the units that chunks may not cross; each group's text is the
        blocks of its sections (Block.section_id), contiguous in doc.text.

        - A leaf section brings all its text. A parent brings its OWN text only (heading + intro
          before its first subsection) and counts as a sibling of its children, so it can merge
          with them: a heading-only parent ("2. Methods" right before "2.1 ...") is a tiny unit
          and joins its first child.
        - Adjacent tiny siblings (< min_section_tokens) under the same parent merge until the group
          reaches min_section_tokens; a tiny group left at the end of a sibling run joins the
          previous group of that run. Never across parents: that would recreate the "Methods glued
          to Discussion" problem.
        """
        sections = {s.id: s for s in doc.sections}
        parents = {s.parent_id for s in doc.sections}
        own: dict[str, tuple[int, int]] = {}     # section id -> its blocks' span, in document order
        for b in doc.blocks:
            own[b.section_id] = (own.get(b.section_id, (b.start, b.end))[0], b.end)
        sizes = self.counter.count_many([doc.text[a:b] for a, b in own.values()])

        groups: list[tuple[str | None, list[Section], int]] = []    # (sibling key, sections, tokens)
        for sid, n in zip(own, sizes):
            key = sid if sid in parents else sections[sid].parent_id
            if groups and groups[-1][0] == key and groups[-1][2] < self.min_section_tokens:
                groups[-1][1].append(sections[sid])
                groups[-1] = (key, groups[-1][1], groups[-1][2] + n)
            else:
                groups.append((key, [sections[sid]], n))

        merged: list[tuple[str | None, list[Section], int]] = []
        for key, group, n in groups:
            # A group that is still tiny after merging forward had no same-parent sibling after it.
            if merged and merged[-1][0] == key and n < self.min_section_tokens:
                merged[-1] = (key, merged[-1][1] + group, merged[-1][2] + n)
            else:
                merged.append((key, group, n))
        return [group for _, group, _ in merged]

    def split_group(self, doc: "CanonicalDoc", group: list["Section"]) -> list[tuple[int, int]]:
        """Chunk one section group: paragraphs (blocks) are the units; only a block longer than
        target_tokens falls back to its sentences — that is what makes it "recursive". The units
        are packed like SentenceChunker's (`pack_spans`), so a group that fits the target is one
        chunk, and overlap only links consecutive chunks of the SAME group."""
        ids = {s.id for s in group}
        blocks = [b for b in doc.blocks if b.section_id in ids]
        sizes = self.counter.count_many([doc.text[b.start:b.end] for b in blocks])
        units: list[tuple[int, int]] = []
        for block, n in zip(blocks, sizes):
            units += [(block.start, block.end)] if n <= self.target_tokens else block_sentences(doc, block)
        return self.pack_spans(doc.text, units)

    def spans(self, doc: "CanonicalDoc") -> list[tuple[int, int]]:
        return [span for group in self.section_groups(doc) for span in self.split_group(doc, group)]
