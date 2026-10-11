"""Stage 4 — chunker-independent Q-A generation with located evidence spans.

Why this design (revisits "idea 2" — the rationale of the ground-truth context):
  * Old approach: Q-A pairs were generated from `create_overlapping_chunks(MAX_CHUNK_SIZE=400)`
    output — i.e. from one of the candidate chunkers. Any question whose evidence aligns with that
    chunker's boundaries favours it (and its size). The docx even credits precision to "chunk size
    matching the Q-A passage size" — that is the bias showing.
  * New approach: generation units come from the document STRUCTURE (paragraph groups from
    CanonicalDoc.blocks), which no chunker owns. The model must return VERBATIM evidence quotes;
    we locate them in CanonicalDoc.text → character spans. This
      - gives ground truth that every chunker can be scored against (evaluation/retrieval_metrics.py),
      - filters hallucinated Q-A by construction (quote not found → drop),
      - removes the recall ceiling from invented claims in reference answers (docx, Context Recall (c)).
  * ~20% two-span questions (evidence in two adjacent paragraphs) directly test boundary handling
    and overlap — the effect the docx found but could only infer indirectly.

Reads:  data/parsed/<corpus>/*.json
Writes: data/qa/raw.jsonl (one QAPair per line), data/qa/gen_dropped.jsonl (one DroppedUnit per
        line), data/qa/raw.provenance.json (what both were made with; resume is refused when it
        differs), runs/<run_id>/gen_qa_summary.json
Prompt: prompts/qa_generation.md
"""
from __future__ import annotations

import bisect
import functools
import hashlib
import itertools
import json
import logging
import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Callable, Literal, get_args

import numpy as np
from rapidfuzz import fuzz

from chunkeval.core.config import Secrets
from chunkeval.core.llm import ChatLLM, LLMOutputError, LLMResult, ollama_model_digest
from chunkeval.core.prompts import load_sections
from chunkeval.core.schema import REPO_ROOT, Config, QuestionType
from chunkeval.core.store import append_jsonl, done_ids, read_jsonl, run_dir, write_json
from chunkeval.corpus.chunking.base import load_counter, pack
from chunkeval.corpus.chunking.sentence import block_sentences
from chunkeval.corpus.parsing import is_skippable_section, load_doc

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc

log = logging.getLogger(__name__)


@dataclass
class GenerationUnit:
    """The passage shown to the generator: one span of running text (single), or two adjacent
    paragraphs (two_span). See `sample_units`."""

    unit_id: str
    doc_id: str
    corpus: str
    spans: list[tuple[int, int]]          # 1 span (single) or 2 spans (two-span)
    section_path: list[str]               # of the first span
    kind: Literal["single", "two_span"] = "single"
    # n_tokens, tokens_per_word (a garbled-text signal for the audit), split_paragraph (single:
    # cut from a paragraph over unit_max_tokens), crosses_section (two-span).
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSpan:
    start: int
    end: int
    quote: str
    match: Literal["exact", "normalized", "fuzzy"]
    score: float = 100.0


@dataclass
class QAPair:
    id: str
    doc_id: str
    corpus: str
    unit_id: str
    question: str
    answer: str                            # gold/reference answer (Ragas `reference`)
    question_type: str                     # as the generator labelled it; meta["requested_types"]: as asked
    evidence: list[EvidenceSpan] = field(default_factory=list)
    kind: str = "single"
    generator_model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DroppedUnit:
    """A unit that gave no Q-A pair, and why. Written to gen_dropped.jsonl (except "api_error"),
    so a re-run doesn't pay for it again and the drop rate per corpus and reason can be reported."""

    id: str
    doc_id: str
    corpus: str
    unit_id: str
    kind: str
    # skip: the model declined; bad_output: LLMOutputError (invalid JSON, cut off...);
    # quote_not_found: a quote `locate_quote` can't find; two_span_one_side: a two-span unit with
    # located quotes in only one span; api_error: no answer (timeout, server down), retried next run.
    reason: Literal["skip", "bad_output", "quote_not_found", "two_span_one_side", "api_error"]
    detail: str
    generator_model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


QUESTION_TYPES: tuple[str, ...] = get_args(QuestionType)

# What the generator must return: a Q-A pair, or {"skip": true, "reason": ...} when the passage has
# nothing worth asking (the escape hatch of the prompt, which replaces the old "N/A" / "I don't
# know"). `ChatLLM.chat` validates every answer against it; with thinking off, Qwen3.5 on Ollama
# ignores it as a decoding constraint, so the prompt spells out the same shapes.
# - The pair may carry "skip": false: a model shown both shapes sometimes writes it, and rejecting
#   that would lose a good pair over a formality.
# - The skip "reason" is optional, so a bare {"skip": true} still counts as a skip, not an error.
# - A root `anyOf` is fine for Ollama and jsonschema, but OpenAI's strict mode requires an object at
#   the root: switching qa_generator to OpenAI with strict mode would need a flat object instead.
QA_JSON_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 1},
                "answer": {"type": "string", "minLength": 1},
                "evidence_quotes": {"type": "array", "items": {"type": "string", "minLength": 1},
                                    "minItems": 1, "maxItems": 3},
                "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
                "skip": {"const": False},
            },
            "required": ["question", "answer", "evidence_quotes", "question_type"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"skip": {"const": True}, "reason": {"type": "string"}},
            "required": ["skip"],
            "additionalProperties": False,
        },
    ],
}

PROMPT_PATH = REPO_ROOT / "prompts" / "qa_generation.md"
# The sections of the prompt file and the placeholders each must use ("system" is sent as is).
PROMPT_SECTIONS: dict[str, set[str] | None] = {
    "system": None,
    "user": {"doc_title", "passages", "question_type", "two_span"},
    "passage": {"label", "section_path", "text"},
    "two_span": set(),
}


# Block types a unit is made of: running text. Headings are context (the section path in the
# prompt), never evidence. Tables would need parsing.table_text first (not in CanonicalDoc yet).
UNIT_BLOCK_TYPES = {"paragraph", "list_item"}


@dataclass
class _Part:
    """A piece of running text a single unit is packed from: a whole block, or one sentence of a
    block longer than unit_max_tokens."""
    start: int
    end: int
    tokens: int
    section_id: str
    whole_block: bool


def _overlap(a: GenerationUnit, b: GenerationUnit) -> bool:
    return any(s1 < e2 and s2 < e1 for s1, e1 in a.spans for s2, e2 in b.spans)


def _spread(rng: np.random.Generator, units: list[GenerationUnit], k: int,
            taken: list[GenerationUnit]) -> list[GenerationUnit]:
    """Up to `k` of `units` (in document order), spread over the whole document instead of the
    first k: one at random from each of k equal consecutive strata. Never a unit that overlaps one
    in `taken` or one picked before (a stratum with no such unit left gives none)."""
    if k <= 0 or not units:
        return []
    picked: list[GenerationUnit] = []
    for stratum in np.array_split(np.arange(len(units)), min(k, len(units))):
        free = [i for i in stratum if not any(_overlap(units[i], u) for u in [*taken, *picked])]
        if free:
            picked.append(units[rng.choice(free)])
    return picked


def sample_units(doc: "CanonicalDoc", cfg: Config, rng: np.random.Generator) -> list[GenerationUnit]:
    """Pick the generation units of one document, in document order.

    Units come from the document structure, which no chunker owns:
      * single: consecutive paragraphs and list items of one section, packed with the chunkers'
        own `pack` towards the middle of qa.unit_min_tokens..unit_max_tokens. A group under
        unit_min_tokens is not a unit, so a section with less text than that gets none (on the
        dev reports these are office addresses, ISBN and credits). A paragraph over
        unit_max_tokens is cut into its sentences, which are packed like whole paragraphs:
        skipping it would leave a quarter of paper_4 without questions. A single "sentence" over
        unit_max_tokens breaks the run instead: in practice garbled OCR without full stops
        (on the dev docs, the only two are paper_4's, 529 and 401 tokens at 5–9 tokens per word).
      * two_span: two adjacent whole paragraphs (the next non-heading block, possibly in the
        next section), each >= unit_min_tokens / 2 and together <= unit_max_tokens. The question
        must need both, which tests chunk boundaries and overlap directly.
    Blocks in a section where `parsing.is_skippable_section` holds are never used.

    Choice: round(max_questions_per_doc * two_span_ratio) two-span units first, then single ones
    up to max_questions_per_doc, each kind spread over the document by `_spread`, and no two units
    overlapping (two questions from the same text would be near-duplicates). `rng` makes it
    reproducible: same document, config and seed, same units.

    Ids are "s<start>" / "t<start>" (char offset of the first span), so they stay stable across
    runs and don't depend on the order of sampling.
    """
    qa = cfg.qa
    counter = load_counter(cfg.chunking.size_tokenizer, cfg.chunking.size_tokenizer_revision)
    sections = {s.id: s for s in doc.sections}
    body = [b for b in doc.blocks if b.type != "heading"]
    usable = [b.type in UNIT_BLOCK_TYPES and not is_skippable_section(sections[b.section_id], cfg)
              for b in body]
    sizes = counter.count_many([doc.text[b.start:b.end] for b in body])

    def make(kind: str, spans: list[tuple[int, int]], section_id: str, **meta: Any) -> GenerationUnit:
        texts = [doc.text[s:e] for s, e in spans]
        n_tokens = sum(counter.count_many(texts))
        return GenerationUnit(
            unit_id=f"{kind[0]}{spans[0][0]}", doc_id=doc.doc_id, corpus=doc.corpus, spans=spans,
            section_path=list(sections[section_id].path), kind=kind,  # type: ignore[arg-type]
            meta={"n_tokens": n_tokens,
                  "tokens_per_word": round(n_tokens / sum(len(t.split()) for t in texts), 2), **meta})

    # Single units. `parts` holds None where a run must break: an unusable block, or a sentence
    # too long for any unit.
    parts: list[_Part | None] = []
    too_long = 0
    for block, ok, n in zip(body, usable, sizes):
        if not ok:
            parts.append(None)
        elif n <= qa.unit_max_tokens:
            parts.append(_Part(block.start, block.end, n, block.section_id, whole_block=True))
        else:
            sentences = block_sentences(doc, block)
            counts = counter.count_many([doc.text[s:e] for s, e in sentences])
            too_long += sum(k > qa.unit_max_tokens for k in counts)
            parts += [_Part(s, e, k, block.section_id, whole_block=False) if k <= qa.unit_max_tokens else None
                      for (s, e), k in zip(sentences, counts)]
    target = (qa.unit_min_tokens + qa.unit_max_tokens) // 2
    singles: list[GenerationUnit] = []
    # A run: consecutive parts of one section with no break between (None groups apart).
    for section_id, run_iter in itertools.groupby(parts, key=lambda p: p and p.section_id):
        if section_id is None:
            continue
        run = list(run_iter)
        for i, j in pack([p.tokens for p in run], target, qa.unit_max_tokens):
            group = run[i:j]
            if sum(p.tokens for p in group) >= qa.unit_min_tokens:
                singles.append(make("single", [(group[0].start, group[-1].end)], section_id,
                                    split_paragraph=not all(p.whole_block for p in group)))

    # Two-span units.
    pairs = [make("two_span", [(a.start, a.end), (b.start, b.end)], a.section_id,
                  crosses_section=a.section_id != b.section_id)
             for (a, ok_a, n_a), (b, ok_b, n_b) in itertools.pairwise(zip(body, usable, sizes))
             if ok_a and ok_b and min(n_a, n_b) >= qa.unit_min_tokens / 2
             and n_a + n_b <= qa.unit_max_tokens]

    two_span = _spread(rng, pairs, round(qa.max_questions_per_doc * qa.two_span_ratio), [])
    chosen = two_span + _spread(rng, singles, qa.max_questions_per_doc - len(two_span), two_span)
    log.debug("%s: %d single and %d two-span candidates (%d sentences over unit_max_tokens skipped), %d picked",
              doc.doc_id, len(singles), len(pairs), too_long, len(chosen))
    return sorted(chosen, key=lambda u: u.spans[0][0])


@functools.cache
def load_prompt(path: Path = PROMPT_PATH) -> dict[str, str]:
    """The sections of the generation prompt file, by name (read once per process); exactly those
    of PROMPT_SECTIONS, each using exactly its placeholders (see `prompts.load_sections`)."""
    return load_sections(path, PROMPT_SECTIONS)


def section_path_at(doc: "CanonicalDoc", offset: int) -> list[str]:
    """Heading path of the section holding the block that starts at or before `offset`."""
    block = doc.blocks[bisect.bisect_right([b.start for b in doc.blocks], offset) - 1]
    return next(s.path for s in doc.sections if s.id == block.section_id)


def render_passages(doc: "CanonicalDoc", spans: list[tuple[int, int]], template: str) -> str:
    """The spans of a unit, each rendered with a prompt's `passage` template ($label, $section_path,
    $text): labelled A, B, ... when there are several, each with its own section path."""
    labels = [""] if len(spans) == 1 else [f" {chr(ord('A') + i)}" for i in range(len(spans))]
    return "\n".join(
        Template(template).substitute(label=label, section_path=" > ".join(section_path_at(doc, start)),
                                      text=doc.text[start:end])
        for label, (start, end) in zip(labels, spans))


def build_messages(doc: "CanonicalDoc", unit: GenerationUnit, question_type: str) -> list[dict[str, str]]:
    """System + user chat messages for one unit, from prompts/qa_generation.md.

    The system message holds every rule, so it is the same for every call; the user message holds
    the document title, the unit's spans (labelled A and B in a two-span unit, each with its own
    section path) and the question type asked for. The section path is context only: the spans
    hold paragraphs, so a quoted heading is never located.
    """
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {question_type!r}; known: {QUESTION_TYPES}")
    prompt = load_prompt()
    user = Template(prompt["user"]).substitute(
        doc_title=doc.title, passages=render_passages(doc, unit.spans, prompt["passage"]), question_type=question_type,
        two_span=Template(prompt["two_span"]).substitute() if unit.kind == "two_span" else "")
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user.strip()}]


# Shorter quotes are ambiguous (they can match in several places) and pin down almost nothing
# as evidence: `locate_quote` rejects them.
MIN_QUOTE_WORDS = 5

# Typographic variants a model may or may not copy faithfully, folded to ASCII by `_normalize`
# (paper_4 alone has 33 en/em dashes).
_FOLD = str.maketrans({**dict.fromkeys("‐‑‒–—―−", "-"),
                       **dict.fromkeys("‘’‚‛′", "'"),
                       **dict.fromkeys("“”„‟", '"')})
# Invisible characters a model never copies: soft hyphen (report_15 has one), zero-width ones.
_INVISIBLE = frozenset("­​‌‍⁠﻿")
# A word broken at a line end, "nitro-\ngen". Docling already joins these, but other converters
# and OCR may not. A hyphen before a plain space is a real one: "drought- and heat-tolerant".
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])[-‐‑][^\S\n]*\n\s*(?=[^\W\d_])")
# Stripped from both ends of a quote before the normalized search: models add or drop a final
# full stop, and sometimes wrap the quote in quotation marks.
_QUOTE_EDGES = " .,;:!?\"'"


def _normalize(text: str, offset: int = 0) -> tuple[str, list[int]]:
    """Fold `text` for matching, and map every char of the result back to the original.

    Lower case, NFKC ("ﬁ" -> "fi", "…" -> "...", "m²" -> "m2"), the dashes and quotes of _FOLD,
    invisible characters dropped, line-break hyphenation joined, and every whitespace run
    collapsed to one space (a paragraph break "\\n\\n" included). `index[i]` is the offset
    (plus `offset`) of the original char that produced `normalized[i]`, so a match in the
    normalized text maps back to an exact span of the original.
    """
    joined = {i for m in _LINE_BREAK_HYPHEN.finditer(text) for i in range(m.start(), m.end())}
    chars: list[str] = []
    index: list[int] = []
    after_space = True                   # True at the start, so leading whitespace is dropped
    for i, ch in enumerate(text):
        if i in joined or ch in _INVISIBLE:
            continue
        if ch.isspace():
            if not after_space:
                chars.append(" ")
                index.append(offset + i)
            after_space = True
            continue
        # NFKC and lower() can turn one char into several; each of them maps to this offset.
        for c in unicodedata.normalize("NFKC", ch).translate(_FOLD).lower():
            chars.append(c)
            index.append(offset + i)
        after_space = False
    return "".join(chars), index


def _word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """A fuzzy alignment can start or end inside a word ("ch site had ..."): widen it to whole
    words. Then trim _QUOTE_EDGES at both ends, as they were trimmed from the quote: an
    alignment as long as the quote can otherwise pick up the next space or full stop."""
    while 0 < start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while 0 < end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    while start < end and text[start] in _QUOTE_EDGES:
        start += 1
    while end > start and text[end - 1] in _QUOTE_EDGES:
        end -= 1
    return start, end


def locate_quote(quote: str, doc_text: str, search_spans: list[tuple[int, int]],
                 fuzzy_threshold: float) -> EvidenceSpan | None:
    """Find a model quote in the canonical text and return its char span, or None.

    Only `search_spans` (the unit's own spans) are searched: that is faster, and an identical
    sentence elsewhere in the document is not the passage the question was written from.
    Three steps in increasing cost; the first that matches wins:
      1. exact: `str.find` of the stripped quote, inside each span;
      2. normalized: the same search after `_normalize` on both sides (case, dashes and quotes,
         NFKC, whitespace runs and paragraph breaks, line-break hyphenation), with the
         punctuation around the quote stripped. Offsets map back through `_normalize`'s index;
      3. fuzzy: `rapidfuzz.fuzz.partial_ratio_alignment` on the normalized texts, accepted at
         score >= `fuzzy_threshold` (0–100); the best-scoring span wins, widened to whole words.
         It catches a typo the model fixed or a word it changed (in a 45-char quote, "four" ->
         "4" scored 91); a paraphrase scores far lower (~63).
    Each span is searched on its own, so a quote crossing from one span of a two-span unit into
    the other is not found. Quotes under MIN_QUOTE_WORDS words are rejected.
    `EvidenceSpan.quote` keeps the model's quote; the located text is `doc_text[start:end]`.
    """
    quote = quote.strip()
    if len(quote.split()) < MIN_QUOTE_WORDS:
        return None

    # 1. Exact. str.find with start/end only matches a quote that lies wholly inside the span.
    for start, end in search_spans:
        pos = doc_text.find(quote, start, end)
        if pos >= 0:
            return EvidenceSpan(pos, pos + len(quote), quote, "exact")

    # 2. Normalized.
    needle = _normalize(quote)[0].strip(_QUOTE_EDGES)
    if not needle:                       # a "quote" made of punctuation only
        return None
    windows = [_normalize(doc_text[start:end], start) for start, end in search_spans]
    for text, index in windows:
        pos = text.find(needle)
        if pos >= 0:
            return EvidenceSpan(index[pos], index[pos + len(needle) - 1] + 1, quote, "normalized")

    # 3. Fuzzy.
    best = None
    for text, index in windows:
        # partial_ratio aligns the SHORTER string inside the longer one: against a span shorter
        # than the quote it would match the whole span inside the quote, with score 100.
        if len(text) < len(needle):
            continue
        aligned = fuzz.partial_ratio_alignment(needle, text, score_cutoff=fuzzy_threshold)
        if aligned is not None and (best is None or aligned.score > best[0].score):
            best = (aligned, text, index)
    if best is None:
        return None
    aligned, text, index = best
    start, end = _word_bounds(text, aligned.dest_start, aligned.dest_end)
    return EvidenceSpan(index[start], index[end - 1] + 1, quote, "fuzzy", round(aligned.score, 1))


# A single unit's skip of any other type is retried once with this one, when it is in
# qa.question_types: in the live checks on the dev docs most single-unit skips were "nothing of the
# type asked for" (mostly numeric), not "nothing worth asking", and units are the scarce resource.
# Two-span skips are final: they were "A and B aren't linked" or garbled text, which no type fixes
# (0 of 2 retries gave a pair in the smoke run).
FALLBACK_TYPE = "factoid"

# Bump when a change in this module changes what `gen-qa` writes for the same inputs: the
# provenance check then refuses to resume records made by the old code.
GENERATION_VERSION = 2


def assign_types(doc: "CanonicalDoc", units: list[GenerationUnit], cfg: Config,
                 rng: np.random.Generator) -> list[str]:
    """The question type asked for each unit: qa.question_types in turn from a random start per
    document, so the mix is balanced over the corpus. "numeric" passes to the next type for a unit
    without a single digit (the model can only skip it)."""
    types = cfg.qa.question_types
    offset = int(rng.integers(len(types)))
    asked = []
    for i, unit in enumerate(units):
        k = offset + i
        if types[k % len(types)] == "numeric" and len(types) > 1 \
                and not any(re.search(r"\d", doc.text[s:e]) for s, e in unit.spans):
            k += 1
        asked.append(types[k % len(types)])
    return asked


def _outcome(doc: "CanonicalDoc", unit: GenerationUnit, asked: list[str],
             results: list[LLMResult | Exception], cfg: Config) -> QAPair | DroppedUnit:
    """Turn the answers for one unit (one per attempt; the last decides) into a pair or a drop."""
    last = results[-1]
    ok = [r for r in results if isinstance(r, LLMResult)]
    common = {"id": f"{doc.doc_id}:{unit.unit_id}", "doc_id": doc.doc_id, "corpus": doc.corpus,
              "unit_id": unit.unit_id, "kind": unit.kind,
              "generator_model": ok[-1].model if ok else ""}
    meta: dict[str, Any] = {
        "requested_types": asked, "section_path": unit.section_path, "unit_spans": unit.spans, **unit.meta,
        "prompt_tokens": sum(r.prompt_tokens for r in ok),
        "completion_tokens": sum(r.completion_tokens for r in ok)}
    if len(results) > 1 and isinstance(results[0], LLMResult):
        meta["first_skip_reason"] = results[0].parsed.get("reason", "")

    if isinstance(last, LLMOutputError):
        return DroppedUnit(**common, reason="bad_output", detail=str(last), meta=meta)
    if isinstance(last, Exception):
        return DroppedUnit(**common, reason="api_error", detail=f"{type(last).__name__}: {last}", meta=meta)
    output = last.parsed
    meta["output"] = output
    if output.get("skip"):
        return DroppedUnit(**common, reason="skip", detail=output.get("reason", ""), meta=meta)

    located = [locate_quote(q, doc.text, unit.spans, cfg.qa.fuzzy_match_threshold) for q in output["evidence_quotes"]]
    missing = [q for q, ev in zip(output["evidence_quotes"], located) if ev is None]
    if missing:
        return DroppedUnit(**common, reason="quote_not_found", detail=" | ".join(missing), meta=meta)
    evidence = [ev for ev in located if ev is not None]
    if unit.kind == "two_span":
        sides = [any(s <= ev.start < e for ev in evidence) for s, e in unit.spans]
        if not all(sides):
            return DroppedUnit(**common, reason="two_span_one_side",
                               detail=f"quotes only in passage {'AB'[sides.index(True)]}", meta=meta)
    del meta["output"]            # a pair keeps its fields, not the raw output a second time
    return QAPair(**common, question=output["question"], answer=output["answer"],
                  question_type=output["question_type"], evidence=evidence, meta=meta)


def generate_for_doc(doc: "CanonicalDoc", llm: ChatLLM, cfg: Config, rng: np.random.Generator,
                     done: set[str] | frozenset[str] = frozenset(),
                     on_outcome: Callable[[QAPair | DroppedUnit], None] | None = None,
                     ) -> list[QAPair | DroppedUnit]:
    """Units -> LLM calls -> parsed JSON -> located evidence -> one pair or drop per unit.

    - Units from `sample_units`, the type asked for each from `assign_types` (both from `rng`, so
      the same per run); units whose id ("<doc_id>:<unit_id>") is in `done` are skipped.
    - One question per call, concurrently (`llm.chat_many`). A single unit's skip of another type
      is asked again once as FALLBACK_TYPE; a two-span skip is final.
    - A pair needs EVERY quote located, and for a two-span unit at least one in each span.
      Everything else becomes a DroppedUnit with its reason (see there).
    - `on_outcome(outcome)` runs as soon as a unit's outcome is final, so the caller can save it
      at once and a crash loses at most the units in flight.
    Returns the outcomes in completion order.
    """
    units = sample_units(doc, cfg, rng)
    asked = assign_types(doc, units, cfg, rng)
    todo = [(u, t) for u, t in zip(units, asked) if f"{doc.doc_id}:{u.unit_id}" not in done]
    outcomes: list[QAPair | DroppedUnit] = []

    def finish(outcome: QAPair | DroppedUnit) -> None:
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    retry = FALLBACK_TYPE in cfg.qa.question_types
    retries: list[tuple[GenerationUnit, str, LLMResult]] = []

    def first(i: int, result: LLMResult | Exception) -> None:
        unit, qtype = todo[i]
        if (retry and unit.kind == "single" and qtype != FALLBACK_TYPE
                and isinstance(result, LLMResult) and result.parsed.get("skip")):
            retries.append((unit, qtype, result))
        else:
            finish(_outcome(doc, unit, [qtype], [result], cfg))

    def second(i: int, result: LLMResult | Exception) -> None:
        unit, qtype, skipped = retries[i]
        finish(_outcome(doc, unit, [qtype, FALLBACK_TYPE], [skipped, result], cfg))

    if todo:
        llm.chat_many([build_messages(doc, u, t) for u, t in todo], json_schema=QA_JSON_SCHEMA, on_result=first)
    if retries:
        llm.chat_many([build_messages(doc, u, FALLBACK_TYPE) for u, _, _ in retries],
                      json_schema=QA_JSON_SCHEMA, on_result=second)
    return outcomes


def qa_paths(cfg: Config) -> dict[str, Path]:
    """The files of this stage: pairs, drops, and the provenance both were made with."""
    qa_dir = cfg.paths.data_dir / "qa"
    return {"raw": qa_dir / "raw.jsonl", "dropped": qa_dir / "gen_dropped.jsonl",
            "provenance": qa_dir / "raw.provenance.json"}


def pair_from_record(record: dict[str, Any]) -> QAPair:
    """A QAPair back from its JSONL record (raw.jsonl, filtered.jsonl), evidence included. JSON has
    no tuples, so meta["unit_spans"] comes back as lists."""
    return QAPair(**{**record, "evidence": [EvidenceSpan(**ev) for ev in record["evidence"]]})


def load_pairs(path: Path) -> list[QAPair]:
    """Every QAPair of a JSONL file, in file order."""
    return [pair_from_record(r) for r in read_jsonl(path)]


def provenance(cfg: Config, digest: str | None) -> dict[str, Any]:
    """Everything that decides what a unit gives: compared on every re-run, so a resumed run never
    mixes two generators, prompts or unit rules in one file. `digest`: the generator's Ollama
    digest (None for OpenAI, where the dated model name is the pin).

    Not included: qa.docs_per_corpus and --limit (they choose WHICH documents, not what a document
    gives) and the qa settings of filter-qa."""
    qa = cfg.qa
    return {
        "version": GENERATION_VERSION,
        "generator": {**cfg.llm.qa_generator.model_dump(), "digest": digest},
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
        "seed": cfg.seed,
        "size_tokenizer": f"{cfg.chunking.size_tokenizer}@{cfg.chunking.size_tokenizer_revision}",
        "skip_section_titles": cfg.parsing.skip_section_titles,
        "qa": {key: getattr(qa, key) for key in ("max_questions_per_doc", "unit_min_tokens", "unit_max_tokens",
                                                 "two_span_ratio", "question_types", "fuzzy_match_threshold")},
    }


def select_docs(cfg: Config, corpus: str, limit: int | None) -> list["CanonicalDoc"]:
    """The parsed documents of a corpus to generate from: the first `limit` by name (smoke test),
    else all of them, or a seeded sample of qa.docs_per_corpus if there are more."""
    parsed_dir = cfg.paths.data_dir / "parsed" / corpus
    paths = sorted(parsed_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"corpus '{corpus}': no parsed docs in {parsed_dir}, run `chunkeval parse` first")
    if limit:
        paths = paths[:limit]
    elif len(paths) > cfg.qa.docs_per_corpus:
        rng = np.random.default_rng([cfg.seed, zlib.crc32(corpus.encode())])
        paths = [paths[i] for i in sorted(rng.choice(len(paths), cfg.qa.docs_per_corpus, replace=False))]
    return [load_doc(p) for p in paths]


def text_sha256(doc: "CanonicalDoc") -> str:
    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()


def check_provenance(paths: dict[str, Path], settings: dict[str, Any], docs: list["CanonicalDoc"],
                     recreate: bool) -> dict[str, str]:
    """Decide whether existing records can be resumed; return the stored doc hashes to extend.

    Records are resumed when they were made with the same `settings` and every document already
    in them still has the same text (offsets of the evidence spans point into it). Otherwise
    RuntimeError, until `recreate`, which deletes the records of EVERY corpus: the settings are
    shared, so no part of the files is still valid then.
    """
    def fail(reason: str) -> RuntimeError:
        return RuntimeError(f"{paths['raw'].parent}: {reason}; start over with `chunkeval gen-qa --recreate`")

    if recreate:
        for path in paths.values():
            if path.exists():
                path.unlink()
                log.info("deleted %s (--recreate)", path)
        return {}
    if not paths["provenance"].exists():
        if paths["raw"].exists() or paths["dropped"].exists():
            raise fail(f"Q-A records without {paths['provenance'].name} (not made by this stage)")
        return {}
    stored = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    if stored["settings"] != settings:
        differs = [key for key in settings if stored["settings"].get(key) != settings[key]]
        raise fail(f"records made with other settings ({', '.join(differs)} differ)")
    hashes: dict[str, str] = stored["docs"]
    changed = [f"{d.corpus}/{d.doc_id}" for d in docs if hashes.get(f"{d.corpus}/{d.doc_id}", text_sha256(d)) != text_sha256(d)]
    if changed:
        raise fail(f"documents re-parsed since their Q-A pairs were made: {', '.join(changed)}")
    return hashes


def summarize(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Per corpus, over everything in the files (all runs): units, pairs, the pair rate, drops by
    reason, how the evidence was located, the question types, and the two-span units within one
    section vs across a section boundary (on the dev docs the model skipped every cross-section
    pair it was given: re-check on the full corpus, see README "Before the QA stage")."""
    summary: dict[str, dict[str, Any]] = {}

    def entry(corpus: str) -> dict[str, Any]:
        return summary.setdefault(corpus, {
            "units": 0, "pairs": 0, "dropped": Counter(), "evidence": Counter(), "question_types": Counter(),
            "retried": 0, "two_span": {side: {"units": 0, "pairs": 0} for side in ("same_section", "cross_section")}})

    for path, is_pair in ((paths["raw"], True), (paths["dropped"], False)):
        if not path.exists():
            continue
        for r in read_jsonl(path):
            e = entry(r["corpus"])
            e["units"] += 1
            e["retried"] += len(r["meta"]["requested_types"]) > 1
            if r["kind"] == "two_span":
                side = e["two_span"]["cross_section" if r["meta"]["crosses_section"] else "same_section"]
                side["units"] += 1
                side["pairs"] += is_pair
            if is_pair:
                e["pairs"] += 1
                e["question_types"][r["question_type"]] += 1
                e["evidence"].update(ev["match"] for ev in r["evidence"])
            else:
                e["dropped"][r["reason"]] += 1
    for e in summary.values():
        e["pair_rate"] = round(e["pairs"] / e["units"], 3)
        for key in ("dropped", "evidence", "question_types"):
            e[key] = dict(e[key].most_common())
    return summary


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None,
        recreate: bool = False) -> None:
    """CLI entry (`chunkeval gen-qa`): generate for the parsed documents of each corpus in
    `corpora` (`select_docs`; `limit`: the first N per corpus).

    Writes data/qa/raw.jsonl (one QAPair per line), data/qa/gen_dropped.jsonl (one DroppedUnit
    per line, not api_error) and data/qa/raw.provenance.json, appending one line per unit as soon
    as it is final, so a crash or Ctrl+C loses only the units in flight. A re-run resumes: a unit
    already in either file is not asked again, as long as `check_provenance` passes; `recreate`
    starts over. Units without an answer (API errors) are not saved: the stage fails at the end,
    and a re-run asks them again.

    Generates more than the final set needs (filtering drops 40–60%); filter-qa samples down.
    The CLI already ran `check_ollama` before this and unloads the generator afterwards.
    Ends with `summarize`, logged and written to runs/<run_id>/gen_qa_summary.json.
    """
    paths = qa_paths(cfg)
    docs = {corpus: select_docs(cfg, corpus, limit) for corpus in corpora}
    llm = ChatLLM("qa_generator", cfg, secrets)
    digest = ollama_model_digest(llm.base_url, llm.model) if llm.provider == "ollama" else None
    settings = provenance(cfg, digest)
    hashes = check_provenance(paths, settings, [d for ds in docs.values() for d in ds], recreate)
    done = done_ids(paths["raw"]) | done_ids(paths["dropped"])

    api_errors = 0

    def save(outcome: QAPair | DroppedUnit) -> None:
        nonlocal api_errors
        if isinstance(outcome, QAPair):
            append_jsonl(paths["raw"], outcome)
        elif outcome.reason == "api_error":
            api_errors += 1
        else:
            append_jsonl(paths["dropped"], outcome)

    for corpus, corpus_docs in docs.items():
        for i, doc in enumerate(corpus_docs, start=1):
            # Recorded before the first record of the doc, so a crash mid-doc can't leave records
            # whose doc text is unknown to the next run's check.
            hashes[f"{corpus}/{doc.doc_id}"] = text_sha256(doc)
            write_json(paths["provenance"], {"settings": settings, "docs": hashes})
            rng = np.random.default_rng([cfg.seed, zlib.crc32(doc.doc_id.encode())])
            outcomes = generate_for_doc(doc, llm, cfg, rng, done, on_outcome=save)
            pairs = sum(isinstance(o, QAPair) for o in outcomes)
            log.info("%s [%d/%d] %s: %d units asked, %d pairs", corpus, i, len(corpus_docs), doc.doc_id,
                     len(outcomes), pairs)

    summary = summarize(paths)
    for corpus, s in summary.items():
        two = s["two_span"]
        log.info("%s: %d units -> %d pairs (%.0f%%), %d retried as %s; dropped %s; evidence %s; types %s; "
                 "two-span pairs/units: same section %d/%d, across sections %d/%d",
                 corpus, s["units"], s["pairs"], 100 * s["pair_rate"], s["retried"], FALLBACK_TYPE,
                 s["dropped"], s["evidence"], s["question_types"],
                 two["same_section"]["pairs"], two["same_section"]["units"],
                 two["cross_section"]["pairs"], two["cross_section"]["units"])
    write_json(run_dir(cfg) / "gen_qa_summary.json", summary)
    if api_errors:
        raise RuntimeError(f"{api_errors} unit(s) got no answer (API errors, see the warnings above) and were "
                           f"not saved: re-run `chunkeval gen-qa` to ask them again")
