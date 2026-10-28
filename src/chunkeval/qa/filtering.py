"""Stage 5 — layered Q-A filtering: deterministic checks -> LLM critic -> de-duplication.

Why: the old judge (`build_evaluation_prompt`, ragas_evaluation.py:424) only saw the question and
answer, so it could check format but not truth. Here:
  1. cheap deterministic checks remove obvious junk before paying for any LLM call;
  2. a critic from a DIFFERENT model family (gpt-oss-20b) sees the evidence passage and scores
     every pair on 5 criteria (6 for a two-span pair); the threshold is applied in code (`decide`);
  3. near-duplicate questions are removed so no topic is over-weighted.
Every rejected pair is written with its stage and reasons → you can audit the filter.

Every pair that passes is kept, with no down-sampling (decided 2026-10-11): filtered.jsonl is the
final question set, and every later stage evaluates exactly its question ids (the paired
statistics in analysis.py depend on it). Ragas draws its own subset from it.

Re-runs: the critic's answers are cached in critic_scores.jsonl, keyed by a hash of everything that
decides them (`critic_key`). A re-run asks the critic only about pairs whose key is new: new pairs,
or all of them after a change of prompt or critic. Changing a threshold asks nothing. The question
vectors of the de-duplication are cached too, by question text (`dedup_vectors`): only new questions
are encoded, so a re-run never moves a cosine. The checks, the decision and the de-duplication run
every time and rewrite filtered.jsonl and rejected.jsonl.

Per corpus (`--corpus`): every step works within one corpus (the de-duplication compares questions
of the same corpus only), so the stage filters only the corpora asked for and carries the other
corpora's lines of filtered.jsonl and rejected.jsonl over as they are: what a full run would give
them. What decided each corpus's lines is recorded in filtered.provenance.json (`filter_settings`),
and the report warns when the corpora of a run were filtered differently.

Reads:  data/qa/raw.jsonl (+ raw.provenance.json), data/parsed/<corpus>/<doc_id>.json
Writes: data/qa/filtered.jsonl   the QAPairs that passed, the critic's scores in meta["filter"]
        data/qa/rejected.jsonl   every other QAPair, meta["filter"]: stage, reasons, scores
        data/qa/filtered.provenance.json   the filter settings by corpus
        data/qa/critic_scores.jsonl   the critic's answers (append-only cache, see above)
        data/embeddings/dedup/<primary embedder>/<corpus>.npy (+ .json)   the question vectors
        runs/<run_id>/filter_qa_summary.json, runs/<run_id>/filter_qa_labelling.csv
Prompt: prompts.qa_critic (prompts/qa_critic.md)
"""
from __future__ import annotations

import csv
import functools
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from chunkeval.core.config import Secrets
from chunkeval.core.llm import ChatLLM, LLMOutputError, LLMResult, ollama_model_digest, unload_ollama_model
from chunkeval.core.prompts import load_sections
from chunkeval.core.schema import REPO_ROOT, Config
from chunkeval.core.store import append_jsonl, filter_jsonl, read_jsonl, run_dir, write_json, write_jsonl
from chunkeval.corpus.embedding import Embedder, check_gpu_free, load_specs, save_npy
from chunkeval.corpus.parsing import load_doc
from chunkeval.qa.generation import QAPair, load_pairs, qa_paths, render_passages, text_sha256

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc

log = logging.getLogger(__name__)


# --- 1. deterministic checks ----------------------------------------------------------------------

# Words a question uses to point at its source instead of naming its subject.
_SOURCE = (r"(?:stud(?:y|ies)|papers?|articles?|reports?|documents?|passages?|texts?|excerpts?|works?|"
           r"research|surveys?|reviews?|analys[ie]s|sections?|chapters?|abstract|introduction|discussion|"
           r"conclusions?|contexts?|areas?|projects?|manuscripts?|publications?)")

# The old `phrase_blacklist` (ragas_evaluation.py:337) as case-insensitive regexes: its `phrase in q`
# was case-sensitive, so "In this study ..." slipped through. Widened to the source references of
# the dev pairs ("described in the abstract", "for the study", "according to the study's
# conclusions"). "in the context of <topic>" is left alone: good questions use it to name their
# topic; "in this context" is caught.
BLACKLIST_PATTERNS: list[str] = [
    rf"\b(?:this|these|that|those)\s+{_SOURCE}\b",
    r"\b(?:in|of|from|by|for|during)\s+the\s+(?:stud(?:y|ies)|paper|article|report|document|passage|text|"
    r"excerpt|abstract|introduction|discussion|conclusions?|survey|project)\b",
    r"\bthe\s+(?:authors?|researchers?|passage|text|excerpt|abstract)\b",
    r"\bthe\s+(?:study|paper|article|report|document)'s\b",
    r"\baccording\s+to\s+(?:the|this|these)\s+(?:stud(?:y|ies)|papers?|articles?|reports?|documents?|passages?|"
    r"texts?|excerpts?|authors?|researchers?|sections?|analys[ie]s|surveys?|data|findings|results|abstract|"
    r"conclusions?)\b",
]
# A question opening with a pronoun has lost its subject ("What did they find ..." is fine).
PRONOUN_OPENING = r"^(?:it|its|they|their|them|this|these|those|he|she|his|her)\b"
# Figure, table or section numbers mean nothing to someone searching a collection.
FIGURE_REFERENCE = r"\b(?:fig(?:ure)?s?|tables?|appendix|annex|box|eq(?:uation)?s?|section|chapter)\.?\s*\d"
# Two questions in one ("What is X, and what other name does it have?"): a second wh-word after
# "and"/"or", or a second question mark.
COMPOUND = r"\b(?:and|or)\s+(?:what|which|how|why|when|where|who|whom|whose)\b"
# Bounds in words. Shorter is a bare term ("What is FIGS?"); the longest dev question has 37 words.
QUESTION_MIN_WORDS, QUESTION_MAX_WORDS = 5, 40
# Answers that say the passage doesn't answer the question: replaces the old "N/A" / "I don't know"
# filtering. Narrow on purpose: "farmers had no information on prices" is a real answer.
ABSTENTION_PATTERNS: list[str] = [
    r"\bnot\s+(?:explicitly\s+|directly\s+)?(?:mentioned|stated|specified|provided|given|described|discussed|"
    r"reported)\s+in\s+the\s+(?:passage|text|document|excerpt|study|paper|report)\b",
    r"\bthe\s+(?:passage|text|document|excerpt)\s+(?:does\s+not|doesn't|did\s+not)\b",
    r"\b(?:cannot|can't|can\s+not)\s+be\s+(?:determined|answered|found|inferred)\b",
    r"\bi\s+(?:don't|do\s+not|cannot|can't)\s+(?:know|answer|tell)\b",
    r"^\W*(?:n/?a|none|unknown|not\s+applicable)\W*$",
]

_FLAGS = re.IGNORECASE
_BLACKLIST = [re.compile(p, _FLAGS) for p in BLACKLIST_PATTERNS]
_ABSTENTION = [re.compile(p, _FLAGS) for p in ABSTENTION_PATTERNS]

_NUMBER = re.compile(r"\d+(?:[.,·]\d+)*")
# A thousands separator Docling keeps from the PDF: no-break, narrow no-break or thin space.
_THIN_SEPARATOR = re.compile(r"(?<=\d)[   ](?=\d{3}\b)")
_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
         "sixteen seventeen eighteen nineteen".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUMBER_WORDS = {**{w: i for i, w in enumerate(_UNITS)}, **_TENS, "hundred": 100, "thousand": 1000, "dozen": 12}
_SPELT = re.compile(rf"\b({'|'.join(_TENS)})[- ]({'|'.join(_UNITS[1:10])})\b|\b({'|'.join(_NUMBER_WORDS)})\b", _FLAGS)


def _values(digits: str) -> list[float]:
    """The value of one digit string: "116,921" -> 116921 (thousands separators), "3,4" -> 3.4
    (decimal comma), "1·5" -> 1.5. Section-like "3.1.2" gives each part."""
    s = digits.replace("·", ".")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d+,\d+", s):
        s = s.replace(",", ".")
    try:
        return [float(s)]
    except ValueError:
        return [float(part) for part in re.split(r"[.,]", s) if part]


def numbers(text: str, spelt: bool = False) -> set[float]:
    """The numbers written in digits in `text`, plus, with `spelt`, those written as words ("ten",
    "twenty-five", not "half" or "a third")."""
    text = _THIN_SEPARATOR.sub("", text)
    found = {v for m in _NUMBER.finditer(text) for v in _values(m.group())}
    if spelt:
        for m in _SPELT.finditer(text):
            tens, unit, word = m.groups()
            found.add(_TENS[tens.lower()] + _UNITS.index(unit.lower()) if tens else _NUMBER_WORDS[word.lower()])
    return found


def deterministic_checks(qa: QAPair, doc: "CanonicalDoc") -> list[str]:
    """Return a list of failure reasons ([] = pass), each "<code>" or "<code>: <detail>".

    On the question: a reference to its source (BLACKLIST_PATTERNS), a pronoun opening, a figure or
    table number, no final "?", two questions in one, fewer than QUESTION_MIN_WORDS or more than
    QUESTION_MAX_WORDS words. On the answer: an abstention, and every number in it must appear in
    the located evidence or the question (written in digits or words there): a strong
    hallucination check for numeric answers. An answer that repeats a quote is fine.
    """
    q, a = qa.question.strip(), qa.answer.strip()
    reasons = []
    sources = [repr(m.group()) for p in _BLACKLIST if (m := p.search(q))]
    if sources:
        reasons.append("source_reference: " + ", ".join(sources))
    if m := re.search(PRONOUN_OPENING, q, _FLAGS):
        reasons.append(f"pronoun_opening: {m.group()!r}")
    if m := re.search(FIGURE_REFERENCE, q, _FLAGS):
        reasons.append(f"figure_reference: {m.group()!r}")
    if not q.endswith("?"):
        reasons.append("not_a_question")
    compound = re.search(COMPOUND, q, _FLAGS)
    if compound or q.count("?") > 1:
        reasons.append(f"compound_question: {compound.group()!r}" if compound else "compound_question: two '?'")
    words = len(q.split())
    if words < QUESTION_MIN_WORDS:
        reasons.append(f"question_too_short: {words} words")
    elif words > QUESTION_MAX_WORDS:
        reasons.append(f"question_too_long: {words} words")
    if any(p.search(a) for p in _ABSTENTION):
        reasons.append("abstention")
    evidence = " ".join(doc.text[ev.start:ev.end] for ev in qa.evidence)
    missing = numbers(a) - numbers(evidence, spelt=True) - numbers(q, spelt=True)
    if missing:
        reasons.append("number_not_in_evidence: " + ", ".join(f"{v:g}" for v in sorted(missing)))
    return reasons


# --- 2. the critic --------------------------------------------------------------------------------

# The critic is asked twice per pair (prompts/qa_critic.md says why): the "passage" call sees the
# passage and judges the answer; the "question" call sees the question alone, as a user of the
# retrieval system asks it, and judges the question. Each answer starts with a note the scores rest
# on, then holds one {"reason", "score"} per criterion.
CALL_NOTES: dict[str, tuple[str, dict[str, Any]]] = {
    "passage": ("question_asks_for", {"type": "string", "minLength": 1}),
    "question": ("unresolved_references", {"type": "array", "items": {"type": "string"}}),
}
CALL_CRITERIA: dict[str, tuple[str, ...]] = {
    "passage": ("groundedness", "answerability"),
    "question": ("standalone", "specificity", "usefulness"),
}
# Passage call, two-span pairs only: such a pair must need both passages, or it doesn't test the
# chunk boundary between them.
TWO_SPAN_CRITERION = "needs_both"
CRITERIA: tuple[str, ...] = (*CALL_CRITERIA["passage"], TWO_SPAN_CRITERION, *CALL_CRITERIA["question"])

# The shipped prompt, the default of `load_critic_prompt`; the stage reads cfg.prompts.qa_critic.
CRITIC_PROMPT_PATH = REPO_ROOT / "prompts" / "qa_critic.md"
# The sections of the prompt file and the placeholders each must use (the system ones are sent as is).
CRITIC_PROMPT_SECTIONS: dict[str, set[str] | None] = {
    "system_passage": None,
    "system_question": None,
    "user_passage": {"doc_title", "passages", "question", "answer", "quotes", "two_span"},
    "user_question": {"question"},
    "passage": {"label", "section_path", "text"},
    "two_span": set(),
}

# Bump when a change in this module changes what the critic is asked or how its answer is read:
# every cached answer then misses its key and is asked again.
CRITIC_VERSION = 1


def criteria_for(call: str, kind: str) -> tuple[str, ...]:
    """The criteria one call scores for a pair of this kind."""
    names = CALL_CRITERIA[call]
    return (*names, TWO_SPAN_CRITERION) if call == "passage" and kind == "two_span" else names


def critic_json_schema(call: str, kind: str) -> dict[str, Any]:
    """The critic's output for one call: the call's note first, then per criterion an object with
    the reason BEFORE the score, an integer 1–5. `ChatLLM.chat` validates every answer against it;
    gpt-oss on Ollama also follows it as a decoding constraint."""
    criterion = {"type": "object",
                 "properties": {"reason": {"type": "string", "minLength": 1},
                                "score": {"type": "integer", "minimum": 1, "maximum": 5}},
                 "required": ["reason", "score"], "additionalProperties": False}
    note, note_schema = CALL_NOTES[call]
    names = criteria_for(call, kind)
    return {"type": "object", "properties": {note: note_schema, **{name: criterion for name in names}},
            "required": [note, *names], "additionalProperties": False}


@functools.cache
def load_critic_prompt(path: Path = CRITIC_PROMPT_PATH) -> dict[str, str]:
    """The sections of the critic prompt file, by name (read once per process); exactly those of
    CRITIC_PROMPT_SECTIONS, each using exactly its placeholders (see `prompts.load_sections`)."""
    return load_sections(path, CRITIC_PROMPT_SECTIONS)


def build_critic_messages(doc: "CanonicalDoc", qa: QAPair, call: str,
                          prompt_path: Path = CRITIC_PROMPT_PATH) -> list[dict[str, str]]:
    """System + user chat messages for one call about one pair, from the prompt file (cfg.prompts.qa_critic).

    Each system message is the same for every call of its kind. The "passage" user message holds
    the document title, the unit's passages as the generator saw them (A and B for a two-span
    pair), the question, the answer and the evidence as located in the passage (not the model's
    quote, which may differ slightly). The "question" user message holds the question only.
    """
    prompt = load_critic_prompt(prompt_path)
    if call == "question":
        user = Template(prompt["user_question"]).substitute(question=qa.question)
    else:
        spans = [(start, end) for start, end in qa.meta["unit_spans"]]
        user = Template(prompt["user_passage"]).substitute(
            doc_title=doc.title, passages=render_passages(doc, spans, prompt["passage"]),
            question=qa.question, answer=qa.answer,
            quotes="\n".join(f"- {doc.text[ev.start:ev.end]}" for ev in qa.evidence),
            two_span=Template(prompt["two_span"]).substitute() if qa.kind == "two_span" else "")
    return [{"role": "system", "content": prompt[f"system_{call}"]},
            {"role": "user", "content": user.strip()}]


def critic_key(messages: list[dict[str, str]], schema: dict[str, Any], critic: dict[str, Any]) -> str:
    """sha256 of everything that decides the critic's answer: the messages (prompt, passage, pair),
    the output schema and `critic` (model, settings, Ollama digest, seed, CRITIC_VERSION)."""
    blob = json.dumps({"messages": messages, "schema": schema, "critic": critic}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def critic_scores(pairs: list[QAPair], docs: dict[tuple[str, str], "CanonicalDoc"], llm: ChatLLM,
                  critic: dict[str, Any], cache: dict[str, dict[str, Any]],
                  on_record: Callable[[dict[str, Any]], None] | None = None,
                  prompt_path: Path = CRITIC_PROMPT_PATH) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    """Ask the critic to score each pair, in both calls; return its records by (pair id, call),
    from `cache` (records by key) or asked now, and the number of API errors.

    A record: {"id", "call", "key", "model", "output": the parsed JSON or None, "error": None or
    why the answer was unusable (LLMOutputError: the same request at temperature 0 gives the same
    answer, so it is kept, not asked again), "prompt_tokens", "completion_tokens"}. New records go
    to `on_record` as soon as they arrive, so a crash loses only the requests in flight. One batch
    per call and schema (a two-span pair's passage call adds needs_both). A request that failed
    (timeout, server down) gives no record: nothing to cache, the next run asks it again.
    Keeps the keep/drop decision OUT of this function (see `decide`).
    """
    records: dict[tuple[str, str], dict[str, Any]] = {}
    todo: dict[tuple[str, tuple[str, ...]], list[tuple[QAPair, list[dict[str, str]], str]]] = defaultdict(list)
    for qa in pairs:
        for call in CALL_CRITERIA:
            messages = build_critic_messages(docs[(qa.corpus, qa.doc_id)], qa, call, prompt_path)
            key = critic_key(messages, critic_json_schema(call, qa.kind), critic)
            if key in cache:
                records[(qa.id, call)] = cache[key]
            else:
                todo[(call, criteria_for(call, qa.kind))].append((qa, messages, key))
    log.info("critic: %d of %d answers cached, %d to ask", len(records), 2 * len(pairs),
             sum(map(len, todo.values())))

    api_errors = 0
    for (call, names), items in todo.items():
        def save(i: int, result: LLMResult | Exception, call: str = call, items=items) -> None:
            nonlocal api_errors
            qa, _, key = items[i]
            record = {"id": qa.id, "call": call, "key": key, "model": llm.model, "output": None, "error": None,
                      "prompt_tokens": 0, "completion_tokens": 0}
            if isinstance(result, LLMResult):
                record.update(model=result.model, output=result.parsed, prompt_tokens=result.prompt_tokens,
                              completion_tokens=result.completion_tokens)
            elif isinstance(result, LLMOutputError):
                record["error"] = str(result)
            else:
                api_errors += 1
                return
            records[(qa.id, call)] = record
            if on_record is not None:
                on_record(record)

        kind = "two_span" if TWO_SPAN_CRITERION in names else "single"
        llm.chat_many([messages for _, messages, _ in items], json_schema=critic_json_schema(call, kind),
                      on_result=save)
    return records, api_errors


def decide(scores: dict[str, Any], cfg: Config) -> list[str]:
    """Apply the threshold → list of failure reasons ([] = pass): every criterion scored (5, or 6 for
    a two-span pair) must reach cfg.qa.critic_min_score; each one below gives "critic_<name>: <score>".
    `scores`: the critic's merged output, so the notes are in it too (not criteria, ignored)."""
    return [f"critic_{name}: {scores[name]['score']}" for name in CRITERIA
            if name in scores and scores[name]["score"] < cfg.qa.critic_min_score]


# --- 3. de-duplication ----------------------------------------------------------------------------

def embed_questions(questions: list[str], cfg: Config, secrets: Secrets) -> np.ndarray:
    """The questions as unit-norm vectors of the primary embedder (bge-m3), on the GPU. Encoded
    with `encode_documents`: question-vs-question similarity is symmetric, so no query instruction.
    Stops while Ollama holds a model (`check_gpu_free`): `run` unloads the critic first."""
    check_gpu_free(secrets)
    with Embedder(load_specs(cfg)[cfg.embedding.primary]) as embedder:
        return embedder.encode_documents(questions, cfg.embedding.batch_size)


def dedup_cache_path(cfg: Config, corpus: str) -> Path:
    """data/embeddings/dedup/<primary embedder>/<corpus>.npy; its metadata is the .json next to it."""
    return cfg.embedding_cache_dir / "dedup" / cfg.embedding.primary / f"{corpus}.npy"


def question_key(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def load_dedup_cache(cfg: Config, corpus: str, settings: dict[str, Any]) -> dict[str, np.ndarray]:
    """question_key -> vector of every question of `corpus` encoded so far with `settings` (the
    primary embedder's); empty without a cache or when it was made with other settings."""
    path = dedup_cache_path(cfg, corpus)
    meta_path = path.with_suffix(".json")
    if not (path.exists() and meta_path.exists()):
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("settings") != settings:
        return {}
    return dict(zip(meta["keys"], np.load(path)))


def save_dedup_cache(cfg: Config, corpus: str, settings: dict[str, Any], vectors: dict[str, np.ndarray]) -> None:
    path = dedup_cache_path(cfg, corpus)
    meta_path = path.with_suffix(".json")
    meta_path.unlink(missing_ok=True)            # written last: only ever next to its complete matrix
    save_npy(path, np.stack(list(vectors.values())))
    write_json(meta_path, {"embedder": cfg.embedding.primary, "corpus": corpus, "settings": settings,
                           "n": len(vectors), "created": datetime.now().isoformat(timespec="seconds"),
                           "keys": list(vectors)})


def dedup_vectors(pairs: list[QAPair], cfg: Config, secrets: Secrets) -> np.ndarray:
    """The questions of `pairs` as unit-norm vectors (row i = pairs[i]), cached per corpus by question
    text: only the questions no run has encoded yet go to `embed_questions` (GPU). Why: fp16 batches
    pad differently with other questions in them, which moves a cosine in its last digits, so without
    the cache a re-run or another corpus could flip a pair sitting at the threshold."""
    settings = load_specs(cfg)[cfg.embedding.primary].settings
    known = {corpus: load_dedup_cache(cfg, corpus, settings) for corpus in dict.fromkeys(qa.corpus for qa in pairs)}
    todo = list(dict.fromkeys((qa.corpus, qa.question) for qa in pairs if question_key(qa.question) not in known[qa.corpus]))
    cached = sum(question_key(qa.question) in known[qa.corpus] for qa in pairs)
    log.info("dedup: %d of %d question vectors cached, %d to encode", cached, len(pairs), len(todo))
    if todo:
        for (corpus, question), vector in zip(todo, embed_questions([q for _, q in todo], cfg, secrets)):
            known[corpus][question_key(question)] = vector
        for corpus in dict.fromkeys(corpus for corpus, _ in todo):
            save_dedup_cache(cfg, corpus, settings, known[corpus])
    return np.stack([known[qa.corpus][question_key(qa.question)] for qa in pairs])


def deduplicate(pairs: list[QAPair], vectors: np.ndarray, threshold: float,
                ) -> tuple[list[QAPair], list[tuple[QAPair, str, float]]]:
    """Remove near-duplicate questions; return (kept, removed as (pair, id of the kept pair it
    duplicates, cosine)).

    Greedy: walk `pairs` in order and keep a question unless its cosine similarity to an already
    kept question of the same corpus is above `threshold` (each corpus is retrieved on its own).
    Earlier pairs win, so the caller orders them by preference. `vectors[i]` is the unit-norm
    vector of pairs[i] (cosine = dot product); a plain matrix is fine for a few thousand questions.
    """
    kept_rows: dict[str, list[int]] = defaultdict(list)
    kept: list[QAPair] = []
    removed: list[tuple[QAPair, str, float]] = []
    for i, qa in enumerate(pairs):
        rows = kept_rows[qa.corpus]
        if rows:
            sims = vectors[rows] @ vectors[i]
            j = int(np.argmax(sims))
            if sims[j] > threshold:
                removed.append((qa, pairs[rows[j]].id, round(float(sims[j]), 4)))
                continue
        rows.append(i)
        kept.append(qa)
    return kept, removed


# --- labelling, summary, run ----------------------------------------------------------------------

LABELLING_SIZE = 50
LABELLING_COLUMNS = ["id", "corpus", "kind", "question_type", "doc_title", "section_path", "passage",
                     "question", "answer", "evidence", "label", "notes"]


def export_for_labelling(kept: list[QAPair], rejected: list[QAPair], docs: dict[tuple[str, str], "CanonicalDoc"],
                         n: int, path: Path, rng: np.random.Generator) -> int:
    """Write a random sample of KEPT and REJECTED pairs to CSV for the manual check; return the
    number of rows (0 when the file exists already).

    Why: 50 hand labels give the filter's precision (are kept pairs good?) and a rough recall (are
    rejected pairs really bad?). Half kept, half rejected by the checks or the critic (pairs removed
    as duplicates are good pairs, not a filter verdict), shuffled, with the evidence passage so
    labelling is quick. The filter's decision is NOT in the file, so it can't bias the labels: fill
    in `label` (1 = good pair, 0 = bad), then join by id with filtered.jsonl / rejected.jsonl.
    Never overwrites the file, which may hold labels by then: delete it to draw a new sample.
    """
    if path.exists():
        log.info("%s exists, not overwritten (delete it for a new sample)", path)
        return 0
    judged = [qa for qa in rejected if qa.meta["filter"]["stage"] != "dedup"]
    sample: list[QAPair] = []
    for group, k in ((kept, n // 2), (judged, n - n // 2)):
        if group:
            sample += [group[i] for i in rng.choice(len(group), min(k, len(group)), replace=False)]
    rng.shuffle(sample)
    rows = []
    for qa in sample:
        doc = docs[(qa.corpus, qa.doc_id)]
        rows.append({
            "id": qa.id, "corpus": qa.corpus, "kind": qa.kind, "question_type": qa.question_type,
            "doc_title": doc.title, "section_path": " > ".join(qa.meta["section_path"]),
            "passage": "\n\n--- B ---\n\n".join(doc.text[s:e] for s, e in qa.meta["unit_spans"]),
            "question": qa.question, "answer": qa.answer,
            "evidence": "\n".join(doc.text[ev.start:ev.end] for ev in qa.evidence), "label": "", "notes": ""})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABELLING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize(kept: list[QAPair], rejected: list[QAPair], cfg: Config) -> dict[str, dict[str, Any]]:
    """Per corpus: pairs in, kept, the keep rate, rejections by stage and by reason code, the
    critic's score distribution per criterion (mean, how many below the minimum), and the kept
    pairs by question type and kind (two-span pairs are the scarce ones)."""
    summary: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for qa, is_kept in [(qa, True) for qa in kept] + [(qa, False) for qa in rejected]:
        e = summary.setdefault(qa.corpus, {"pairs": 0, "kept": 0, "rejected": Counter(), "reasons": Counter(),
                                           "kept_types": Counter(), "kept_kinds": Counter()})
        f = qa.meta["filter"]
        e["pairs"] += 1
        if is_kept:
            e["kept"] += 1
            e["kept_types"][qa.question_type] += 1
            e["kept_kinds"][qa.kind] += 1
        else:
            e["rejected"][f["stage"]] += 1
            e["reasons"].update(reason.split(":")[0] for reason in f["reasons"])
        critic = f.get("critic") or {}
        for name in CRITERIA:
            if name in critic:
                scores[qa.corpus][name].append(critic[name]["score"])
    for corpus, e in summary.items():
        e["keep_rate"] = round(e["kept"] / e["pairs"], 3)
        for key in ("rejected", "reasons", "kept_types", "kept_kinds"):
            e[key] = dict(e[key].most_common())
        e["critic_scores"] = {name: {"scored": len(v), "mean": round(float(np.mean(v)), 2),
                                     "below_min": sum(x < cfg.qa.critic_min_score for x in v)}
                              for name, v in scores[corpus].items()}
    return summary


def filter_paths(cfg: Config) -> dict[str, Path]:
    """The files this stage writes in data/qa."""
    qa_dir = cfg.paths.data_dir / "qa"
    return {"filtered": qa_dir / "filtered.jsonl", "rejected": qa_dir / "rejected.jsonl",
            "critic": qa_dir / "critic_scores.jsonl", "provenance": qa_dir / "filtered.provenance.json"}


def filter_settings(cfg: Config, critic: dict[str, Any]) -> dict[str, Any]:
    """What decides which of a corpus's pairs are kept: the critic (`critic`: model, settings, Ollama
    digest, seed, CRITIC_VERSION) and its prompt, the threshold, and the de-duplication's embedder
    and cosine. Recorded per corpus, since `--corpus` can filter corpora at different times."""
    spec = load_specs(cfg)[cfg.embedding.primary]
    return {"critic": critic, "critic_prompt_sha256": hashlib.sha256(cfg.prompts.qa_critic.read_bytes()).hexdigest(),
            "critic_min_score": cfg.qa.critic_min_score, "dedup_embedder": {"name": spec.name, **spec.settings},
            "dedup_cosine_threshold": cfg.qa.dedup_cosine_threshold}


def read_filter_provenance(cfg: Config) -> dict[str, dict[str, Any]]:
    """The filter settings by corpus (`filter_settings`); a corpus filtered before they were recorded
    (2026-10-28) is missing."""
    path = filter_paths(cfg)["provenance"]
    return json.loads(path.read_text(encoding="utf-8"))["settings_by_corpus"] if path.exists() else {}


def delete_records(paths: dict[str, Path], corpora: list[str]) -> None:
    """`remove-corpus`: every line of `corpora` out of filtered.jsonl and rejected.jsonl, and their
    filter settings out of filtered.provenance.json; a file left empty is deleted."""
    for name in ("filtered", "rejected"):
        removed = filter_jsonl(paths[name], lambda r: r["corpus"] not in corpora)
        if removed:
            log.info("deleted %d record(s) of %s from %s", removed, ", ".join(corpora), paths[name])
    if paths["provenance"].exists():
        settings = json.loads(paths["provenance"].read_text(encoding="utf-8"))["settings_by_corpus"]
        settings = {c: s for c, s in settings.items() if c not in corpora}
        if settings:
            write_json(paths["provenance"], {"settings_by_corpus": settings})
        else:
            paths["provenance"].unlink()


def carry_over(paths: dict[str, Path], raw: list[QAPair], corpora: list[str]) -> tuple[list[QAPair], list[QAPair]]:
    """The kept and the rejected pairs of every corpus outside `corpora`, as the last run wrote them.
    Warns for such a corpus whose pairs in raw.jsonl aren't the ones filtered: gen-qa ran for it since."""
    def outside(path: Path) -> list[QAPair]:
        return [qa for qa in load_pairs(path) if qa.corpus not in corpora] if path.exists() else []

    kept, rejected = outside(paths["filtered"]), outside(paths["rejected"])
    raw_ids, filtered_ids = defaultdict(set), defaultdict(set)
    for qa in raw:
        if qa.corpus not in corpora:
            raw_ids[qa.corpus].add(qa.id)
    for qa in kept + rejected:
        filtered_ids[qa.corpus].add(qa.id)
    for corpus in sorted(raw_ids.keys() | filtered_ids.keys()):
        new, gone = raw_ids[corpus] - filtered_ids[corpus], filtered_ids[corpus] - raw_ids[corpus]
        if new or gone:
            log.warning("%s: carried over as filtered last time, but raw.jsonl has %d pair(s) not filtered and lacks "
                        "%d filtered one(s): run `chunkeval filter-qa --corpus %s`", corpus, len(new), len(gone), corpus)
    return kept, rejected


def load_docs(cfg: Config, pairs: list[QAPair], provenance_path: Path) -> dict[tuple[str, str], "CanonicalDoc"]:
    """The parsed documents of `pairs`, by (corpus, doc_id), checked against the text hashes gen-qa
    recorded: evidence offsets point into that text, so a document re-parsed since is refused."""
    stored = json.loads(provenance_path.read_text(encoding="utf-8"))["docs"]
    docs = {}
    for corpus, doc_id in sorted({(qa.corpus, qa.doc_id) for qa in pairs}):
        doc = load_doc(cfg.paths.data_dir / "parsed" / corpus / f"{doc_id}.json")
        if stored.get(f"{corpus}/{doc_id}") != text_sha256(doc):
            raise RuntimeError(f"{corpus}/{doc_id} was re-parsed after its Q-A pairs were made, so their evidence "
                               f"offsets are stale: generate again with `chunkeval gen-qa --recreate`")
        docs[(corpus, doc_id)] = doc
    return docs


def _doc_order(qa: QAPair) -> tuple[str, str, int]:
    return qa.corpus, qa.doc_id, qa.meta["unit_spans"][0][0]


def _preference(qa: QAPair) -> tuple[bool, int, tuple[str, str, int]]:
    """Which of two duplicates to keep: a two-span pair (the scarce kind), then the higher total
    critic score, then document order (so the choice is reproducible)."""
    critic = qa.meta["filter"]["critic"]
    return qa.kind != "two_span", -sum(critic[name]["score"] for name in CRITERIA if name in critic), _doc_order(qa)


def run(cfg: Config, secrets: Secrets, corpora: list[str]) -> None:
    """CLI entry (`chunkeval filter-qa`): the pairs of `corpora` in raw.jsonl → checks → critic →
    decide → dedup → filtered.jsonl + rejected.jsonl, both in document order. Every other corpus's
    lines in both files are carried over as they are (`carry_over`).

    Every pair that passes is kept (no final down-sampling). The critic is asked only about pairs
    that passed the checks, and only when its answer isn't cached (see the module docstring); then
    it is unloaded, because bge-m3 needs the GPU for the questions the de-duplication hasn't encoded
    yet. Pairs without an answer (API errors) stop the stage before anything is written: a re-run
    asks exactly those. Ends with the filter settings of `corpora` (filtered.provenance.json),
    `summarize` (logged and written to runs/<run_id>/filter_qa_summary.json) and, the first time, a
    sample of this run's pairs for hand labelling (runs/<run_id>/filter_qa_labelling.csv).
    The CLI already ran `check_ollama` before this and unloads the critic afterwards.
    """
    gen, paths = qa_paths(cfg), filter_paths(cfg)
    if not gen["raw"].exists():
        raise FileNotFoundError(f"{gen['raw']} not found: run `chunkeval gen-qa` first")
    raw = load_pairs(gen["raw"])
    pairs = sorted((qa for qa in raw if qa.corpus in corpora), key=_doc_order)
    for corpus in corpora:
        if not any(qa.corpus == corpus for qa in pairs):
            log.warning("%s: no pairs in %s, nothing to filter", corpus, gen["raw"])
    carried_kept, carried_rejected = carry_over(paths, raw, corpora)
    docs = load_docs(cfg, pairs, gen["provenance"])
    rejected: list[QAPair] = []

    def reject(qa: QAPair, stage: str, reasons: list[str], **extra: Any) -> None:
        qa.meta["filter"] = {"stage": stage, "reasons": reasons, **extra}
        rejected.append(qa)

    # 1. Deterministic checks.
    candidates = []
    for qa in pairs:
        reasons = deterministic_checks(qa, docs[(qa.corpus, qa.doc_id)])
        if reasons:
            reject(qa, "checks", reasons)
        else:
            candidates.append(qa)
    log.info("checks: %d of %d pairs pass", len(candidates), len(pairs))

    # 2. The critic.
    llm = ChatLLM("qa_critic", cfg, secrets)
    critic = {**cfg.llm.qa_critic.model_dump(), "seed": cfg.seed, "version": CRITIC_VERSION,
              "digest": ollama_model_digest(llm.base_url, llm.model) if llm.provider == "ollama" else None}
    cache = {r["key"]: r for r in read_jsonl(paths["critic"])} if paths["critic"].exists() else {}
    records, api_errors = critic_scores(candidates, docs, llm, critic, cache,
                                        on_record=lambda record: append_jsonl(paths["critic"], record),
                                        prompt_path=cfg.prompts.qa_critic)
    if llm.provider == "ollama":
        unload_ollama_model(llm.base_url, llm.model)
    if api_errors:
        raise RuntimeError(f"{api_errors} pair(s) got no answer from the critic (API errors, see the warnings "
                           f"above); nothing written: re-run `chunkeval filter-qa` to ask them again")

    # 3. The decision, on the two calls' outputs merged (notes and scores).
    passed = []
    for qa in candidates:
        answers = {call: records[(qa.id, call)] for call in CALL_CRITERIA}
        merged = {name: value for r in answers.values() if r["output"] for name, value in r["output"].items()}
        errors = [f"critic_bad_output: {call} call: {r['error']}" for call, r in answers.items() if r["error"]]
        if errors:
            reject(qa, "critic", errors, critic=merged)
        elif reasons := decide(merged, cfg):
            reject(qa, "critic", reasons, critic=merged)
        else:
            qa.meta["filter"] = {"critic": merged}
            passed.append(qa)
    log.info("critic: %d of %d pairs pass (every criterion >= %d)", len(passed), len(candidates),
             cfg.qa.critic_min_score)

    # 4. De-duplication.
    passed.sort(key=_preference)
    vectors = (dedup_vectors(passed, cfg, secrets) if len(passed) > 1
               else np.ones((len(passed), 1), dtype=np.float32))
    kept, removed = deduplicate(passed, vectors, cfg.qa.dedup_cosine_threshold)
    for qa, duplicate_of, cosine in removed:
        reject(qa, "dedup", [f"duplicate_of: {duplicate_of} ({cosine})"], critic=qa.meta["filter"]["critic"],
               duplicate_of=duplicate_of, cosine=cosine)
    log.info("dedup: %d near-duplicate(s) removed (cosine > %s)", len(removed), cfg.qa.dedup_cosine_threshold)

    kept.sort(key=_doc_order)                   # the labelling sample draws from this order
    rejected.sort(key=_doc_order)
    write_jsonl(paths["filtered"], sorted(carried_kept + kept, key=_doc_order))
    write_jsonl(paths["rejected"], sorted(carried_rejected + rejected, key=_doc_order))
    by_corpus = read_filter_provenance(cfg)
    for corpus in corpora:
        by_corpus.pop(corpus, None)
    by_corpus |= {corpus: filter_settings(cfg, critic) for corpus in dict.fromkeys(qa.corpus for qa in pairs)}
    write_json(paths["provenance"], {"settings_by_corpus": by_corpus})

    summary = summarize(carried_kept + kept, carried_rejected + rejected, cfg)
    for corpus, s in summary.items():
        if corpus in corpora:
            log.info("%s: %d pairs -> %d kept (%.0f%%); rejected by stage %s; reasons %s; kept by type %s, by kind %s",
                     corpus, s["pairs"], s["kept"], 100 * s["keep_rate"], s["rejected"], s["reasons"],
                     s["kept_types"], s["kept_kinds"])
    write_json(run_dir(cfg) / "filter_qa_summary.json", summary)
    labelling = run_dir(cfg) / "filter_qa_labelling.csv"
    n = export_for_labelling(kept, rejected, docs, LABELLING_SIZE, labelling, np.random.default_rng(cfg.seed))
    if n:
        log.info("wrote %d pairs for hand labelling to %s", n, labelling)
