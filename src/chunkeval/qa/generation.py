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
Writes: data/qa/raw.jsonl (one QAPair per line, resumable)
Prompt: prompts/qa_generation.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config

if TYPE_CHECKING:
    from chunkeval.core.llm import ChatLLM
    from chunkeval.corpus.parsing import CanonicalDoc


@dataclass
class GenerationUnit:
    """The passage shown to the generator. One or two contiguous block groups."""

    unit_id: str
    doc_id: str
    corpus: str
    spans: list[tuple[int, int]]          # 1 span (single) or 2 spans (two-span)
    section_path: list[str]
    kind: Literal["single", "two_span"] = "single"


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
    question_type: str
    evidence: list[EvidenceSpan] = field(default_factory=list)
    kind: str = "single"
    generator_model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# TODO(you): define the JSON schema the generator must follow (used for constrained decoding).
# Fields: question (str), answer (str), evidence_quotes (array of str, 1–3), question_type (enum),
# skip (bool). Hint: keep `additionalProperties: false`; with `strict` JSON schema mode on OpenAI
# every property must be listed in `required`.
QA_JSON_SCHEMA: dict[str, Any] = {}


def sample_units(doc: "CanonicalDoc", cfg: Config, rng) -> list[GenerationUnit]:
    """Pick generation units from one document.

    TODO(you):
      - Walk the blocks in order; group consecutive paragraph blocks of the same section until
        unit_min_tokens..unit_max_tokens. Skip sections where `parsing.is_skippable_section`.
      - Two-span units: two adjacent groups (possibly across a paragraph or section boundary —
        crossing a section boundary is a deliberately hard case for the recursive chunker).
      - Stratify: spread units across the document (intro/methods/results...) instead of taking the
        first N; cap at max_questions_per_doc. Use `numpy.random.default_rng(seed)` passed in, so
        sampling is reproducible.
      - Decide: tables as units? They produce numeric questions (good for reports) but quotes from
        linearised markdown tables are hard to locate — maybe a separate, small quota.
    """
    raise NotImplementedError


def build_messages(doc: "CanonicalDoc", unit: GenerationUnit, question_type: str | None) -> list[dict[str, str]]:
    """Build system+user chat messages from prompts/qa_generation.md.

    Hint: load the prompt text from the file (so you can iterate on it without touching code) and
    fill placeholders with `str.format` or `string.Template`. Carry over the rules of the old
    `build_prompt` (ragas_evaluation.py:186).
    """
    raise NotImplementedError


def locate_quote(quote: str, doc_text: str, search_spans: list[tuple[int, int]],
                 fuzzy_threshold: float) -> EvidenceSpan | None:
    """Find a model quote in the canonical text and return its char span.

    TODO(you), in increasing cost:
      1. exact `str.find` inside each search span (the unit's own spans — searching the unit, not
         the whole document, is faster and avoids matching an identical sentence elsewhere);
      2. normalized match: collapse whitespace, unify quotes/dashes, de-hyphenate line breaks
         ("nitro-\\ngen") — but you then need a map from normalized positions back to original
         offsets (build it while normalizing: keep a list `orig_index[i]` for each kept char);
      3. fuzzy: `rapidfuzz.fuzz.partial_ratio_alignment(quote, window)` returns score and
         dest_start/dest_end — accept only if score >= fuzzy_threshold.
      Return None if nothing matches. Very short quotes (< ~5 words) are ambiguous: reject them.
    """
    raise NotImplementedError


def generate_for_doc(doc: "CanonicalDoc", llm: "ChatLLM", cfg: Config, rng) -> list[QAPair]:
    """Units → LLM calls → parsed JSON → located evidence → QAPair list.

    TODO(you):
      - Assign question types so the final mix is balanced (e.g. round-robin over cfg types).
      - Drop `skip: true` outputs and outputs where ANY quote can't be located — but count and log
        the reasons; the drop rate per corpus is itself an interesting number to report.
      - For two-span units, require at least one located quote in EACH span.
      - Stable ids, e.g. f"{doc_id}:{unit_id}" — needed to resume and to pair results across chunkers.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets, corpora: list[str], limit: int | None = None) -> None:
    """CLI entry (`chunkeval gen-qa`): generate for the parsed documents of each corpus in
    `corpora`, appending to data/qa/raw.jsonl. `limit`: only the first N documents per corpus.

    Hint: resumable via `store.done_ids`. Generate more than you need (filtering will drop a lot —
    plan for 40–60% loss) and sample down at the end. The CLI already ran `check_ollama` before
    this and unloads the generator afterwards.
    """
    raise NotImplementedError
