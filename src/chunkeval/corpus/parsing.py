"""Stage 1 — layout-aware parsing of PDFs into a CanonicalDoc.

Why:
  * Reports (free-format) had no recursive-chapter chunker because their section structure was
    not detectable with plain extracted text. Docling detects headings, tables, captions, lists
    and page furniture from the PDF layout, so ALL corpora get a section tree.
  * Page headers/footers/page numbers glued into the text were one reason report chunks were
    incoherent (docx, "Papers vs Reports"). Dropping them here helps every chunker equally.
  * Most importantly: every downstream component (all chunkers AND the Q-A generator) works on
    the SAME `CanonicalDoc.text`. That is what makes character offsets comparable across
    chunkers and enables the deterministic span metrics in evaluation/retrieval_metrics.py.

Reads:  PDFs from cfg.paths.corpora[<corpus>]
Writes: data/parsed/<corpus>/<doc_id>.json
Next:   chunking/* and qa/generation.py
"""
from __future__ import annotations

import bisect
import functools
import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from chunkeval.core.config import Secrets
from chunkeval.core.schema import REPO_ROOT, Config
from chunkeval.core.store import write_json, write_jsonl

if TYPE_CHECKING:
    # Docling is imported inside the functions that use it, so stages that only read parsed JSON
    # (chunk, gen-qa, ...) never pay its import time.
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DocItem, DoclingDocument, TableItem

log = logging.getLogger(__name__)

BlockType = Literal["heading", "paragraph", "list_item", "table", "caption", "other"]

# Docling label -> Block type, for every label parse_pdf can turn into text. Which of them are
# kept is cfg.parsing.include_labels (v1: headings + running text). To support a new label, add
# it here AND to schema.DoclingLabel.
LABEL_BLOCK_TYPES: dict[str, BlockType] = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    # Plain-text items, ready for future versions: they only need adding to include_labels.
    # A caption comes right after its table/picture in reading order, so it stays adjacent.
    "caption": "caption",
    "footnote": "other",
    # Not supported yet:
    #   "table":             needs its own renderer, see `table_text`.
    #   "picture", "chart":  no text of their own; would need Docling's picture description
    #                        model (PdfPipelineOptions.do_picture_description).
    #   "formula":           text is empty unless PdfPipelineOptions.do_formula_enrichment=True.
}

SEPARATOR = "\n\n"               # between two blocks; never part of a block
FRONT_MATTER_ID = "s0"
FRONT_MATTER_TITLE = "(front matter)"

# Bump when a change in this module changes the parsed output: `parse_corpus` then parses the
# PDFs again instead of serving the stale cached JSONs.
PARSER_VERSION = 2

# The simple case of a duplicated text layer: a paragraph contained in the paragraph right before
# it (or containing it) is a copy. The minimum length keeps short legitimate repeats.
DEDUP_MIN_CHARS = 50

# heading_levels="font_size" (see `font_size_levels`): heading sizes within 5% of each other are
# one level (19.8 vs 19.4 pt); at most 3 levels, deeper sizes are folded into level 3.
FONT_SIZE_TOLERANCE = 0.05
FONT_SIZE_MAX_LEVEL = 3


@dataclass
class Block:
    """A layout element with its position in `CanonicalDoc.text` (half-open [start, end))."""

    type: BlockType
    start: int
    end: int
    section_id: str          # id of the innermost Section that contains this block


@dataclass
class Section:
    """A node of the heading hierarchy, with the char range it covers (heading + body)."""

    id: str
    title: str
    level: int               # 1 = chapter, 2 = section, 3 = subsection, ...
    start: int
    end: int
    parent_id: str | None = None
    path: list[str] = field(default_factory=list)   # titles from root to this section


@dataclass
class CanonicalDoc:
    """The single text representation every stage shares.

    Invariants (checked by `validate_doc` on every parse; test them in tests/test_parsing.py):
      * every Block/Section satisfies 0 <= start < end <= len(text)
      * blocks are sorted and do not overlap
      * a section's range contains the ranges of all its children
    """

    doc_id: str
    corpus: str              # "papers" | "reports"
    title: str
    text: str
    sections: list[Section]
    blocks: list[Block]
    meta: dict[str, Any] = field(default_factory=dict)   # source path, page count, parser version...


@dataclass
class Piece:
    """One cleaned layout element before the text is assembled (see `assemble_doc`)."""

    type: BlockType
    text: str
    level: int | None = None     # headings only, set by `infer_heading_levels`
    source: Any = field(default=None, repr=False)   # the Docling item (page + bbox), if any


def parse_pdf(pdf_path: Path, corpus: str, cfg: Config) -> CanonicalDoc:
    """Convert one PDF to a CanonicalDoc (doc_id = file stem).

    1. Docling conversion + `collect_pieces`: body items in reading order, filtered by
       cfg.parsing.include_labels, each cleaned BEFORE it gets offsets (a later strip/collapse
       would shift every offset).
    2. `text_layer_problems`: if the PDF's text layer is broken (unreadable characters or a
       duplicated layer), step 1 is redone with full-page OCR.
    3. `infer_heading_levels`: Docling reports level 1 for every PDF heading, so levels come from
       cfg.parsing.heading_levels: numbering ("2." -> 1, "2.1" -> 2, "(a)" -> one below), plus the
       font size measured in the PDF for unnumbered headings, or simply 1 everywhere ("flat").
    4. `assemble_doc`: append the pieces with SEPARATOR, recording each block's offsets as it
       goes, then build the section tree and validate the invariants.

    Docling's own chunkers (HybridChunker) are deliberately not used: they would be a 4th
    chunking method with its own rules and break the "one canonical text" rule.
    """
    pdf_path = Path(pdf_path)
    include = set(cfg.parsing.include_labels)

    ocr = "auto"                 # Docling's default: OCR only for bitmap regions
    result, pieces, skipped, duplicates = _convert(pdf_path, include, full_page_ocr=False)
    # The first pass's ratios are kept for EVERY doc: they are the data to calibrate the thresholds on.
    text_layer = {"auto": text_layer_ratios(result.document, pieces)}
    if problems := text_layer_problems(text_layer["auto"], cfg):
        log.warning("%s: %s; converting again with full-page OCR", pdf_path.name, ", ".join(problems))
        ocr = "full_page"
        result, pieces, skipped, duplicates = _convert(pdf_path, include, full_page_ocr=True)
        text_layer["full_page"] = text_layer_ratios(result.document, pieces)
        if problems := text_layer_problems(text_layer["full_page"], cfg):
            log.warning("%s: still %s after OCR", pdf_path.name, ", ".join(problems))
    if result.status.value != "success":
        log.warning("%s: Docling status %s, some pages may be missing", pdf_path.name, result.status.value)

    ddoc = result.document
    if not pieces:
        raise ValueError(f"{pdf_path.name}: no text extracted")
    headings = [p for p in pieces if p.type == "heading"]
    mode = cfg.parsing.heading_levels
    sizes = heading_font_sizes(pdf_path, ddoc, [p.source for p in headings]) if mode == "font_size" else None
    for piece, level in zip(headings, infer_heading_levels([p.text for p in headings], mode, sizes)):
        piece.level = level

    try:
        source = str(pdf_path.resolve().relative_to(REPO_ROOT))
    except ValueError:           # a PDF outside the repo: keep the path as given
        source = str(pdf_path)
    meta = {
        "source": source,
        "pages": len(ddoc.pages),
        "parser": _parser_name(),
        "parser_version": PARSER_VERSION,
        "conversion_status": result.status.value,
        "ocr": ocr,              # "auto" | "full_page"
        "text_layer": text_layer,   # {"auto": ratios, "full_page": ratios (only if OCR ran)}
        "include_labels": sorted(include),
        "heading_levels": mode,
        # font_size mode: measured size per heading section ("s1", "s2"... = headings in order)
        **({"heading_font_sizes": {f"s{i}": s for i, s in enumerate(sizes, start=1)}} if sizes is not None else {}),
        "skipped_labels": dict(sorted(skipped.items())),   # body items not kept, by label
        "duplicates_dropped": duplicates,
    }
    doc = assemble_doc(pdf_path.stem, corpus, doc_title(ddoc, fallback=pdf_path.stem), pieces, meta)
    log.info("%s: %d pages, %d blocks, %d sections, %d chars (ocr=%s)",
             pdf_path.name, meta["pages"], len(doc.blocks), len(doc.sections), len(doc.text), ocr)
    return doc


@functools.cache
def _converter(full_page_ocr: bool = False) -> DocumentConverter:
    """One DocumentConverter per OCR mode, reused for every PDF: building one loads the layout,
    table and OCR models (the slow part); each conversion after that takes seconds on the GPU."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    if full_page_ocr:
        options.ocr_options.mode = OcrMode.FULL_PAGE
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def _convert(pdf_path: Path, include: set[str], full_page_ocr: bool):
    """Docling conversion followed by `collect_pieces`: (result, pieces, skipped, duplicates)."""
    result = _converter(full_page_ocr).convert(pdf_path)
    return result, *collect_pieces(result.document, include)


def _parser_name() -> str:
    return f"docling {version('docling')}"


def text_layer_ratios(ddoc: DoclingDocument, pieces: list[Piece]) -> dict[str, float]:
    """Health ratios of a conversion's text layer (stored in meta.text_layer)."""
    return {"replacement_char_ratio": round(replacement_char_ratio(ddoc), 5),
            "repeated_paragraph_ratio": round(repeated_paragraph_ratio(pieces), 5)}


def text_layer_problems(ratios: dict[str, float], cfg: Config) -> list[str]:
    """Why the text layer counts as broken (empty list = healthy, no OCR needed).

    Full-page OCR reads the rendered page instead of the PDF's text layer, which fixes:
      - unreadable characters (U+FFFD, "�"): fonts without a Unicode mapping. Lost ligatures alone
        stay far below the threshold ("a�ecting" in paper_4: 0.05%);
      - a duplicated hidden text layer (MDPI papers), which Docling turns into overlapping copies
        of paragraphs no dedup rule can untangle; it also glues page headers into paragraphs and
        swallows headings.
    The price is OCR noise (e.g. "3,1," for "3.1."), so it stays a fallback.
    """
    problems = []
    if ratios["replacement_char_ratio"] > cfg.parsing.ocr_max_replacement_char_ratio:
        problems.append(f"{ratios['replacement_char_ratio']:.1%} of the characters unreadable (U+FFFD)")
    if ratios["repeated_paragraph_ratio"] > cfg.parsing.ocr_max_repeated_paragraph_ratio:
        problems.append(f"{ratios['repeated_paragraph_ratio']:.1%} of the paragraphs repeated (duplicated text layer)")
    return problems


def replacement_char_ratio(ddoc: DoclingDocument) -> float:
    """Share of U+FFFD among the characters of all body text items."""
    texts = [item.text for item, _ in ddoc.iterate_items() if getattr(item, "text", "")]
    total = sum(map(len, texts))
    return sum(t.count("�") for t in texts) / total if total else 0.0


def repeated_paragraph_ratio(pieces: list[Piece]) -> float:
    """Share of long paragraphs (>= 100 chars) whose first 80 chars also occur in another one."""
    paragraphs = [p.text for p in pieces if p.type == "paragraph" and len(p.text) >= 100]
    joined = "\n".join(paragraphs)
    return sum(joined.count(p[:80]) > 1 for p in paragraphs) / len(paragraphs) if paragraphs else 0.0


def collect_pieces(ddoc: DoclingDocument, include: set[str]) -> tuple[list[Piece], Counter[str], int]:
    """Cleaned pieces in reading order, plus the labels skipped and the number of duplicates dropped.

    `iterate_items()` yields Docling's BODY layer only, so page headers, footers and page numbers
    (its FURNITURE layer) never show up here. Children of pictures (text inside figures) are not
    traversed either.
    """
    pieces: list[Piece] = []
    skipped: Counter[str] = Counter()
    duplicates = 0
    for item, _ in ddoc.iterate_items():
        label = item.label.value
        if label not in include:
            skipped[label] += 1
            continue
        text = clean_text(item_text(item, ddoc))
        if not text:                 # e.g. a text box that held only a bullet glyph
            skipped["(empty)"] += 1
            continue
        block_type = LABEL_BLOCK_TYPES[label]

        prev = pieces[-1] if pieces else None
        if (block_type == "paragraph" and prev is not None and prev.type == "paragraph"
                and min(len(text), len(prev.text)) >= DEDUP_MIN_CHARS):
            if text in prev.text:
                duplicates += 1
                continue
            if prev.text in text:    # the longer copy wins
                pieces[-1] = Piece(block_type, text, source=item)
                duplicates += 1
                continue

        pieces.append(Piece(block_type, text, source=item))
    return pieces, skipped, duplicates


def item_text(item: DocItem, ddoc: DoclingDocument) -> str:
    """Raw text of one Docling item; `clean_text` normalises it afterwards."""
    if item.label.value == "table":
        return table_text(item, ddoc)
    text = item.text
    marker = getattr(item, "marker", "")   # list items: "1.", "·" — Docling keeps them apart from the text
    if marker and not text.startswith(marker):
        text = f"{marker} {text}"
    return text


def table_text(item: TableItem, ddoc: DoclingDocument) -> str:
    """Linearise a table for CanonicalDoc.text — placeholder for a future version.

    TODO(future), notes from Docling 2.133 / docling-core 2.99:
      - `item.export_to_markdown(ddoc)` gives a GitHub-style markdown table, but PREPENDS the
        caption text, while `iterate_items` also yields the caption as its own item right after
        the table. Either skip "caption" items whose parent is a table, or render the cell grid
        (`item.data.grid`) without the caption.
      - Tables split across pages arrive as several TableItems ("Table 1. Cont."); consider
        merging them.
      - Then add "table": "table" to LABEL_BLOCK_TYPES and "table" to schema.DoclingLabel.
    """
    raise NotImplementedError("tables are not part of CanonicalDoc.text yet, see parsing.table_text")


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean_text(text: str) -> str:
    """Normalise one item's text, BEFORE it gets offsets.

    - control characters become spaces (paper_8 extracts its bullet glyph as "\\x0f"), whitespace
      runs collapse to one space, the ends are stripped;
    - doubled text is collapsed: PDFs with a duplicated text layer give "3. Findings 3. Findings".
    """
    text = " ".join(_CONTROL_CHARS.sub(" ", text).split())
    words = text.split(" ")
    half = len(words) // 2
    if len(words) % 2 == 0 and words[:half] == words[half:]:
        text = " ".join(words[:half])
    return text


# "2 ", "2. ", "2.1 ", "2.1.3. " followed by a letter; also the OCR variants "3.Findings", "3,1, ".
_NUMBERED = re.compile(r"(\d{1,2}(?:[.,]\d{1,2}){0,4})[.,]?\s*(?=[^\W\d_])")
_APPENDIX_NUMBERED = re.compile(r"[A-Z]((?:\.\d{1,2}){1,4})\.?\s")   # "A.1 ", "B.2.3. "
_LETTERED = re.compile(r"\(?[a-z]\)\s")                         # "(a) ", "a) "


def infer_heading_levels(titles: list[str], mode: str = "numbering",
                         sizes: list[float | None] | None = None) -> list[int]:
    """Levels of the headings given in reading order, by cfg.parsing.heading_levels (`mode`).

    Docling gives level 1 to every heading of a PDF, so the level has to come from elsewhere:
      "flat":      1 for every heading.
      "numbering": "2. Methods" -> 1, "2.1 Data" -> 2, "2.1.3 Soils" -> 3, "A.1 Survey" -> 2,
                   "(a) All Centers" -> one below the last numbered heading ("2030 Vision" is
                   not numbering). Unnumbered headings ("Abstract", "References", every heading
                   of most reports) get 1.
      "font_size": numbering where present (the most reliable signal: "1." and "3.1." headings
                   can share one font size), `font_size_levels(sizes)` for the other headings.
    TODO: the PDF's bookmark outline, when present, would be another signal.
    """
    if mode == "flat":
        return [1] * len(titles)
    unnumbered = font_size_levels(sizes) if mode == "font_size" and sizes is not None else [1] * len(titles)
    levels: list[int] = []
    last_numbered: int | None = None
    for title, fallback in zip(titles, unnumbered):
        if m := _NUMBERED.match(title):
            level = last_numbered = len(re.split(r"[.,]", m.group(1)))
        elif m := _APPENDIX_NUMBERED.match(title):
            level = last_numbered = m.group(1).count(".") + 1
        elif last_numbered is not None and _LETTERED.match(title):
            level = last_numbered + 1
        else:
            level = fallback
        levels.append(level)
    return levels


def font_size_levels(sizes: list[float | None]) -> list[int]:
    """Heading levels from font sizes (bigger = higher), one per size given (None = unknown).

    - Clusters: distinct sizes sorted from big to small; a size within FONT_SIZE_TOLERANCE of
      the current cluster's centre (mean size, weighted by number of headings) joins it. The
      weighted centre stops a chain of close sizes from drifting (paper_4: 9.7 -> 9.3 -> 8.9
      would merge two levels if each size were only compared with its neighbour), and stops a
      one-off size at the top of a cluster (a title) from pulling it.
    - Only clusters shared by >= 2 headings define levels: the largest is level 1, the next 2,
      ..., capped at FONT_SIZE_MAX_LEVEL.
    - A one-off size (the document title, a journal banner like "ORIGINAL ARTICLE") gets the
      level of the first level-defining cluster at or below it, so a title bigger than every
      section heading is level 1 next to them, not a root above the whole document.
    - Unknown sizes (nothing measurable in the bbox) get the most common level.
    No level-defining cluster at all (e.g. every heading the same size) -> all 1.
    """
    counts = Counter(s for s in sizes if s is not None)
    clusters: list[list[float]] = []
    for size in sorted(counts, reverse=True):
        if clusters:
            members = clusters[-1]
            centre = sum(s * counts[s] for s in members) / sum(counts[s] for s in members)
            if centre - size <= FONT_SIZE_TOLERANCE * centre:
                members.append(size)
                continue
        clusters.append([size])
    defining = [i for i, cluster in enumerate(clusters) if sum(counts[s] for s in cluster) >= 2]
    if not defining:
        return [1] * len(sizes)

    level_of: dict[float, int] = {}
    for i, cluster in enumerate(clusters):
        # Clusters are sorted big -> small: the first defining cluster at or below cluster i,
        # or the smallest defining one if cluster i is smaller than all of them.
        rank = next((k for k, d in enumerate(defining) if d >= i), len(defining) - 1) + 1
        level_of.update(dict.fromkeys(cluster, min(rank, FONT_SIZE_MAX_LEVEL)))
    known = [level_of[s] for s in sizes if s is not None]
    most_common = Counter(known).most_common(1)[0][0]
    return [level_of[s] if s is not None else most_common for s in sizes]


def heading_font_sizes(pdf_path: Path, ddoc: DoclingDocument, items: list[DocItem]) -> list[float | None]:
    """Visual font size of each heading item, read from the PDF with pypdfium2 (None = unknown).

    Docling's document keeps no font information, but gives each item's page and bbox. The size
    is the median height of the "loose" character boxes (font ascent to descent, as rendered)
    whose centre lies in that bbox. pdfium's FPDFText_GetFontSize is useless here: most of these
    PDFs set size 1 and scale through the text matrix. Works for OCR'd docs too: a broken text
    layer (mojibake, duplicated) still has correctly sized glyphs; a scanned page has none -> None.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page_chars: dict[int, list[tuple[float, float, float]]] = {}
        sizes: list[float | None] = []
        for item in items:
            heights: list[float] = []
            for prov in getattr(item, "prov", None) or []:
                if prov.page_no not in page_chars:
                    page_chars[prov.page_no] = _char_boxes(pdf[prov.page_no - 1])
                box = prov.bbox.to_bottom_left_origin(ddoc.pages[prov.page_no].size.height)
                heights += [h for x, y, h in page_chars[prov.page_no]
                            if box.l <= x <= box.r and box.b <= y <= box.t]
            sizes.append(round(statistics.median(heights), 1) if heights else None)
        return sizes
    finally:
        pdf.close()


def _char_boxes(page: Any) -> list[tuple[float, float, float]]:
    """(x centre, y centre, height) of the loose box of every visible character on a pdfium page."""
    textpage = page.get_textpage()
    try:
        boxes = []
        for i in range(textpage.count_chars()):
            if textpage.get_text_range(i, 1).isspace():
                continue
            left, bottom, right, top = textpage.get_charbox(i, loose=True)
            boxes.append(((left + right) / 2, (bottom + top) / 2, top - bottom))
        return boxes
    finally:
        textpage.close()


def doc_title(ddoc: DoclingDocument, fallback: str) -> str:
    """Docling's TITLE item if there is one, else the longest heading on page 1 (a journal banner
    like "ORIGINAL ARTICLE" often precedes the real title), else `fallback`."""
    first_page: list[str] = []
    for item, _ in ddoc.iterate_items():
        label = item.label.value
        if label == "title" and (title := clean_text(item.text)):
            return title
        if label == "section_header" and item.prov and item.prov[0].page_no == 1:
            first_page.append(clean_text(item.text))
    return max(first_page, key=len, default="") or fallback


def assemble_doc(doc_id: str, corpus: str, title: str, pieces: list[Piece],
                 meta: dict[str, Any] | None = None) -> CanonicalDoc:
    """Join the pieces into the canonical text, recording each block's offsets while appending.

    No Docling here, so tests can build a CanonicalDoc from hand-written pieces. A heading piece
    without a level counts as level 1.
    """
    parts: list[str] = []
    spans: list[tuple[BlockType, int, int]] = []
    headings: list[tuple[str, int, int]] = []
    pos = 0
    for piece in pieces:
        if parts:
            parts.append(SEPARATOR)
            pos += len(SEPARATOR)
        start, pos = pos, pos + len(piece.text)
        parts.append(piece.text)
        spans.append((piece.type, start, pos))
        if piece.type == "heading":
            headings.append((piece.text, piece.level or 1, start))
    text = "".join(parts)

    sections = build_section_tree(headings, text)
    # Sections are sorted by start, and a block belongs to the last section that starts at or
    # before it: a section only closes when the next heading opens a new one.
    starts = [s.start for s in sections]
    blocks = [Block(block_type, start, end, sections[bisect.bisect_right(starts, start) - 1].id)
              for block_type, start, end in spans]

    doc = CanonicalDoc(doc_id, corpus, title, text, sections, blocks, meta or {})
    validate_doc(doc)
    return doc


def build_section_tree(headings: list[tuple[str, int, int]], text: str) -> list[Section]:
    """Turn (title, level, start_offset) headings into Sections with end offsets and parents.

    A section runs from its heading to the next heading of the same or a higher level (smaller
    number), with trailing whitespace trimmed, so `text[start:end]` is heading + body.
    Text before the first heading gets a synthetic FRONT_MATTER section at level 1: a sibling of
    the top-level sections rather than a root above them, so it is a leaf like any other.
    Ids: "s0" = front matter, "s1", "s2", ... = the headings in reading order.
    """
    def trimmed(end: int) -> int:
        while end > 0 and text[end - 1].isspace():
            end -= 1
        return end

    sections: list[Section] = []
    front_end = trimmed(headings[0][2] if headings else len(text))
    if front_end > 0:
        sections.append(Section(FRONT_MATTER_ID, FRONT_MATTER_TITLE, 1, 0, front_end,
                                path=[FRONT_MATTER_TITLE]))

    open_sections: list[Section] = []        # the current heading path; levels strictly increase
    for i, (title, level, start) in enumerate(headings, start=1):
        while open_sections and open_sections[-1].level >= level:
            open_sections.pop().end = trimmed(start)
        parent = open_sections[-1] if open_sections else None
        section = Section(f"s{i}", title, level, start, end=len(text),
                          parent_id=parent.id if parent else None,
                          path=[*(parent.path if parent else []), title])
        sections.append(section)
        open_sections.append(section)
    for section in open_sections:
        section.end = trimmed(len(text))
    return sections


def validate_doc(doc: CanonicalDoc) -> None:
    """Raise ValueError if a CanonicalDoc invariant is broken (cheap, so run on every parse)."""
    n = len(doc.text)
    for x in [*doc.sections, *doc.blocks]:
        if not 0 <= x.start < x.end <= n:
            raise ValueError(f"{doc.doc_id}: empty or outside 0..{n}: {x}")
    for a, b in zip(doc.blocks, doc.blocks[1:]):
        if a.end > b.start:
            raise ValueError(f"{doc.doc_id}: blocks out of order or overlapping: {a} / {b}")
    sections = {s.id: s for s in doc.sections}
    for s in doc.sections:
        parent = sections.get(s.parent_id) if s.parent_id else None
        if s.parent_id and (parent is None or not parent.start <= s.start < s.end <= parent.end):
            raise ValueError(f"{doc.doc_id}: section {s.id} not inside its parent {s.parent_id}")
    for b in doc.blocks:
        s = sections.get(b.section_id)
        if s is None or not s.start <= b.start < b.end <= s.end:
            raise ValueError(f"{doc.doc_id}: block {b} not inside its section")


# Numbering in front of a heading, for title matching: "5. ", "3.1 ", "A.2 ", "(a) ".
_TITLE_NUMBERING = re.compile(r"^(?:\d{1,2}(?:[.,]\d{1,2})*|[A-Z](?:\.\d{1,2})+|\(?[a-z]\))[.,]?\s+")


def normalise_title(title: str) -> str:
    """"5. References:" -> "references": numbering, case and punctuation removed."""
    title = _TITLE_NUMBERING.sub("", title.strip(), count=1)
    return " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())


def is_skippable_section(section: Section, cfg: Config) -> bool:
    """True for References / Acknowledgements / TOC etc. (used by Q-A unit sampling, NOT chunking).

    Chunkers still index these sections (production indexes them too); we only avoid generating
    questions from them. Matches whole titles after `normalise_title`, not prefixes ("contents"
    must not match "Contents of soil nitrogen"), and checks the whole path, so the subsections
    of a skipped section are skipped too.
    """
    skip = {normalise_title(t) for t in cfg.parsing.skip_section_titles}
    return any(normalise_title(title) in skip for title in section.path)


def parse_corpus(corpus: str, cfg: Config, limit: int | None = None) -> list[Path]:
    """Parse every PDF of a corpus, write data/parsed/<corpus>/<doc_id>.json, return the paths.

    `limit`: only the first N PDFs, sorted by file name, so a smoke test always picks the same ones.

    - doc_id = file stem: stable across runs (unlike the old loop index `research_Paper_{i}`,
      which shifted when a download failed) and readable in every later artefact.
    - Cache: a JSON is reused while `_is_up_to_date` holds; otherwise the PDF is parsed again.
    - One failing PDF does not stop the corpus: its traceback goes to the stage log and it is
      listed in data/parsed/<corpus>/_failures.jsonl (rewritten every run, so it lists the
      failures of the LAST run; `.jsonl` keeps it out of the `*.json` glob of later stages).
    """
    pdf_dir = cfg.paths.corpora[corpus]
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"corpus '{corpus}': folder {pdf_dir} not found")
    pdfs = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")[:limit]
    if not pdfs:
        raise FileNotFoundError(f"corpus '{corpus}': no PDFs in {pdf_dir}")
    out_dir = cfg.paths.data_dir / "parsed" / corpus

    paths: list[Path] = []
    failures: list[dict[str, str]] = []
    cached = 0
    for i, pdf in enumerate(pdfs, start=1):
        out = out_dir / f"{pdf.stem}.json"
        if _is_up_to_date(out, cfg):
            paths.append(out)
            cached += 1
            continue
        log.info("[%d/%d] parsing %s", i, len(pdfs), pdf.name)
        try:
            write_json(out, parse_pdf(pdf, corpus, cfg))
        except Exception as e:       # Ctrl+C (KeyboardInterrupt) still stops the loop
            log.exception("%s: parsing failed", pdf.name)
            failures.append({"pdf": pdf.name, "error": f"{type(e).__name__}: {e}"})
            continue
        paths.append(out)

    write_jsonl(out_dir / "_failures.jsonl", failures)
    log.info("%s: %d parsed, %d up to date, %d failed", corpus, len(paths) - cached, cached, len(failures))
    if failures:
        log.warning("%s: %d PDF(s) failed, see %s", corpus, len(failures), out_dir / "_failures.jsonl")
    return paths


def _is_up_to_date(path: Path, cfg: Config) -> bool:
    """Whether a cached JSON is what `parse_pdf` would produce now: same parser code
    (PARSER_VERSION), Docling version, include_labels and heading_levels, and the same OCR
    decision under the current thresholds. The first pass's ratios are stored, so a threshold
    change only re-parses the PDFs whose decision flips. A missing or unreadable file is never
    up to date."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))["meta"]
        wants_ocr = bool(text_layer_problems(meta["text_layer"]["auto"], cfg))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return (meta.get("parser_version") == PARSER_VERSION
            and meta.get("parser") == _parser_name()
            and meta.get("include_labels") == sorted(cfg.parsing.include_labels)
            and meta.get("heading_levels") == cfg.parsing.heading_levels
            and meta.get("ocr") == ("full_page" if wants_ocr else "auto"))


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None) -> None:
    """CLI entry (`chunkeval parse`): parse each corpus in `corpora`; `limit` PDFs per corpus."""
    for corpus in corpora:
        paths = parse_corpus(corpus, cfg, limit=limit)
        log.info("%s: %d documents parsed into %s", corpus, len(paths), cfg.paths.data_dir / "parsed" / corpus)


def load_doc(path: str | Path) -> CanonicalDoc:
    """Read a parsed JSON back into dataclasses (relative paths resolve against the repo root).

    Validated on load: every later stage trusts the offsets, so a hand-edited or corrupt file
    should fail here, not as a silently wrong chunk.
    """
    data = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    doc = CanonicalDoc(**{**data,
                          "sections": [Section(**s) for s in data["sections"]],
                          "blocks": [Block(**b) for b in data["blocks"]]})
    validate_doc(doc)
    return doc
