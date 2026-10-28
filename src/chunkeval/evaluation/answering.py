"""Stage 7 — the RAG answer (Ragas `response`) to every question, from what retrieval handed over.

Why this stage exists: in the old script the Ragas `answer` column was the Q-A generator's own
answer, so faithfulness and answer relevancy did not depend on the chunker at all. A real RAG
answer — from a FIXED generator and prompt, conditioned on each setup's top-k chunks — makes the
generation metrics depend on retrieval, and on nothing else that differs between setups.

What is answered, per corpus, for every question of filtered.jsonl:
  * each (method, overlap) setup: from the top `retrieval.top_k` chunks the PRIMARY embedder
    retrieved (cfg.embedding.primary = bge_m3), in rank order. The secondary embedder (Qwen3) is a
    retrieval-only robustness check; answering with it would double the LLM time. If the
    robustness table shows a large difference for one setup, answer that ONE setup with the
    secondary as a targeted follow-up. Fixed k only, no budget-matched variant: on the dev docs the
    budget fits 4.2–5.0 chunks on average, so a second run would nearly repeat this one.
  * the gold passage (setup "gold_passage"): the passage the question was generated from (its
    unit's spans, which hold all the evidence), i.e. perfect retrieval. One answer per question,
    the same for every setup: the ceiling of the generation metrics, which separates the
    generator's (and the judge's) own error from what retrieval costs. Not the retrieval `oracle`
    (each setup's chunks that cover the evidence): stage 6 already measures that ceiling for free.

The prompt (prompts/rag_answer.md) shows each passage's text with its document's title, but no
section path, so every chunker gets the same kind of input (chunk text only until 2026-10-21: the
generator then abstained on questions naming a setting the passage doesn't repeat, see the prompt's
notes). `Answer.contexts` holds the same title line (`context`), so the contexts Ragas judges
faithfulness against are exactly what the generator read. Its notes explain the other rules.

Abstention: only when no passage holds any of the information asked for, the prompt asks for exactly
ABSTENTION_PHRASE; when the passages hold part of it, for that part, with nothing on what is missing.
`is_abstention` marks every answer, because Ragas scores an abstention like a wrong answer
(relevancy and factual correctness 0) and its faithfulness at random; stage 8 handles them in code
(see ragas_eval.py).

Re-runs: the generator's answers are cached in answer_cache.jsonl, keyed by a hash of everything
that decides them (`answer_key`: the messages with their titles, the generator's model, settings and
Ollama digest, the seed, ANSWER_VERSION). A re-run asks only what is new: new questions, a question shown other
chunks, or everything after a prompt or generator change. The cache is shared by every run_id and
data dir (paths.llm_cache_dir): a new run asks only the requests no run has made. The answer files
are rewritten from the cache every time, so they always match the current questions and retrieval.

Reads:  data/qa/filtered.jsonl (+ raw.provenance.json), data/parsed/, data/chunks/,
        runs/<run_id>/retrieval/<primary embedder>/*.jsonl (up to date: `chunkeval retrieve`)
Writes: runs/<run_id>/answers/<corpus>_<setup>.jsonl   one `Answer` per question, in question order;
            <setup> is "<method>_<ovl>" or "gold_passage"; a .provenance.json next to each
        data/llm_cache/answer_cache.jsonl   the generator's answers (append-only cache, see above)
        runs/<run_id>/answer_summary.json
Prompt: prompts.rag_answer (prompts/rag_answer.md)
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Callable

from chunkeval.core.config import Secrets
from chunkeval.core.llm import ChatLLM, LLMOutputError, LLMResult, ollama_model_digest
from chunkeval.core.prompts import load_sections
from chunkeval.core.schema import REPO_ROOT, Config
from chunkeval.core.store import append_jsonl, read_jsonl, run_dir, write_json, write_jsonl
from chunkeval.evaluation.retrieval import (file_provenance, load_hits, load_questions, provenance_path,
                                            questions_sha256, retrieval_path, stored_provenance, variant_chunks,
                                            variants)
from chunkeval.evaluation.retrieval_metrics import question_metrics
from chunkeval.qa.filtering import filter_paths, load_docs
from chunkeval.qa.generation import QAPair, qa_paths

if TYPE_CHECKING:
    from chunkeval.corpus.parsing import CanonicalDoc

log = logging.getLogger(__name__)

# The setup name of the baseline answered from each question's own passage.
GOLD_PASSAGE = "gold_passage"

# What the prompt tells the model to reply when the context doesn't answer the question
# (rule 7 of prompts/rag_answer.md; `load_prompt` checks the two agree).
ABSTENTION_PHRASE = "The context does not contain the answer."

# The shipped prompt, the default of `load_prompt`; the stage reads cfg.prompts.rag_answer.
PROMPT_PATH = REPO_ROOT / "prompts" / "rag_answer.md"
# The sections of the prompt file and the placeholders each must use ("system" is sent as is).
PROMPT_SECTIONS: dict[str, set[str] | None] = {
    "system": None,
    "user": {"context", "question"},
    "passage": {"number", "title", "text"},
}

# Bump when a change in this module changes what the generator is asked or how its answer is read:
# every cached answer then misses its key and is asked again. 2: passages with their document's title.
ANSWER_VERSION = 2


@dataclass
class Shown:
    """What the generator is shown for one question in one setup."""
    chunk_ids: list[str]      # in rank order; empty for the gold passage
    texts: list[str]          # the passages, in that order
    titles: list[str]         # each passage's document title
    evidence: bool            # some passage overlaps the gold evidence (span_hit at top_k); gold: always

    @property
    def passages(self) -> list[tuple[str, str]]:
        """(title, text) of each passage, in rank order."""
        return list(zip(self.titles, self.texts))


@dataclass
class Answer:
    """One line of an answer file: what stage 8 hands to Ragas, and what the summary counts."""
    question_id: str
    corpus: str
    setup: str                # "<method>_<ovl>" or "gold_passage"
    response: str             # Ragas `response`; "" when `error`
    contexts: list[str]       # Ragas `retrieved_contexts`: exactly the passages the generator read (`context`)
    chunk_ids: list[str]      # their chunk ids, in rank order (empty for the gold passage)
    evidence_shown: bool      # some passage overlaps the gold evidence
    abstained: bool           # `is_abstention(response)`
    error: str | None         # why the answer is unusable (cut off at max_tokens, context full), else None
    model: str
    prompt_tokens: int
    completion_tokens: int


# --- the prompt -----------------------------------------------------------------------------------

@functools.cache
def load_prompt(path: Path = PROMPT_PATH) -> dict[str, str]:
    """The sections of the answer prompt file, by name (read once per process); exactly those of
    PROMPT_SECTIONS, each using exactly its placeholders, and the system section must contain
    ABSTENTION_PHRASE word for word, or `is_abstention` would look for a phrase never asked for."""
    sections = load_sections(path, PROMPT_SECTIONS)
    if ABSTENTION_PHRASE not in sections["system"]:
        raise ValueError(f"{path}: the system section must contain the abstention phrase {ABSTENTION_PHRASE!r}")
    return sections


def build_messages(question: str, passages: list[tuple[str, str]],
                   prompt_path: Path = PROMPT_PATH) -> list[dict[str, str]]:
    """System + user chat messages for one question, from the prompt file (cfg.prompts.rag_answer).

    The system message holds every rule, so it is the same for every call; the user message holds
    the passages, (title, text) pairs numbered in the order given (rank order), then the question.
    """
    prompt = load_prompt(prompt_path)
    context = "\n\n".join(Template(prompt["passage"]).substitute(number=i, title=title, text=text)
                          for i, (title, text) in enumerate(passages, start=1))
    user = Template(prompt["user"]).substitute(context=context, question=question)
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user.strip()}]


def context(title: str, text: str) -> str:
    """One passage as Ragas reads it (`Answer.contexts`): the title line of the prompt's `passage`
    section, then the text; without the number and quotes, which only separate passages there."""
    return f"Document: {title}\n{text}"


# --- abstentions ----------------------------------------------------------------------------------

# Typographic apostrophes and quotes -> ASCII, so "doesn’t" reads like "doesn't".
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_SOURCE = r"(?:the\s+)?(?:provided\s+|given\s+|retrieved\s+)?(?:context|passages?|texts?|documents?)"
_NOT = r"(?:does|do|did)\s*(?:not|n't)"
# Paraphrases of the phrase, at the start of the answer. "There is no information" needs "in the
# context": alone it can be a real answer ("farmers had no information on prices").
_PARAPHRASE = re.compile(
    rf"(?:{_SOURCE}\s+{_NOT}\s+(?:contain|provide|include|mention|state|specify|give|say|answer|have)\b"
    rf"|{_SOURCE}\s+(?:contains?|provides?|includes?|gives?|has)\s+no\b"
    rf"|(?:the\s+)?answer\s+(?:is\s+not|isn't|cannot\s+be)\s+(?:found\s+|given\s+)?in\s+{_SOURCE}"
    rf"|there\s+is\s+no\s+(?:information|mention|answer)\b[^.!?]*\bin\s+{_SOURCE}"
    rf"|i\s+(?:cannot|can't|am\s+unable\s+to|do\s+not|don't)\s+(?:answer|know|determine|find|tell)\b)")
_NEXT_SENTENCE = re.compile(r"[.!?]\s+\S")


def _normalize(text: str) -> str:
    """Lower case, ASCII quotes, single spaces, without leading markup or quotes."""
    return " ".join(text.translate(_QUOTES).split()).casefold().lstrip("*_\"' ")


def is_abstention(response: str) -> bool:
    """Whether the answer declines to answer, so the abstention rate can be counted exactly.

    Two rules, because a partial answer must not count (the prompt asks for its facts, without the phrase):
      - the answer starts with ABSTENTION_PHRASE (case, quotes and markup ignored): it is reserved
        for abstaining, so anything after it is an explanation;
      - or the WHOLE answer is one sentence that starts like a paraphrase of it ("The provided
        context does not mention ..."). A paraphrase followed by more text is more likely a partial
        answer written the other way round, so it counts as an answer.
    """
    text = _normalize(response)
    if text.startswith(_normalize(ABSTENTION_PHRASE).rstrip(".")):
        return True
    return bool(_PARAPHRASE.match(text)) and not _NEXT_SENTENCE.search(text)


# --- what each question is shown ------------------------------------------------------------------

def doc_titles(cfg: Config, corpus: str) -> dict[str, str]:
    """doc_id -> title of every parsed document of `corpus` (`parse` applied parsing.title_overrides):
    a retrieved chunk can come from any document, also one without questions."""
    titles = {}
    for path in sorted((cfg.paths.data_dir / "parsed" / corpus).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        titles[doc["doc_id"]] = doc["title"]
    return titles


def titles_sha256(titles: dict[str, str]) -> str:
    """sha256 of `doc_titles`, recorded with the answers: a retitled document changes what the
    generator is shown, and the chunk and retrieval hashes don't see it (chunks hold text only)."""
    return hashlib.sha256(json.dumps(titles, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def retrieved(cfg: Config, corpus: str, variant: str, pairs: list[QAPair],
              titles: dict[str, str]) -> tuple[dict[str, Shown], dict[str, Any]]:
    """What each question of `pairs` (the corpus's whole question set) is shown in one setup: the
    top_k chunks the primary embedder retrieved, in rank order, with their text from the chunk
    file and their document's title from `titles` (`doc_titles`). Also returns the retrieval
    file's provenance, recorded with the answers.

    Refuses a retrieval file that is missing or stale (retrieved before the questions, the chunks or
    the query settings changed): its chunk ids could then point to other text than it ranked."""
    embedder, k = cfg.embedding.primary, cfg.retrieval.top_k
    chunks, index = variant_chunks(cfg, corpus, variant)
    path = retrieval_path(cfg, embedder, corpus, variant)
    expected = file_provenance(cfg, embedder, corpus, variant, index["chunks_sha256"], pairs)
    if stored_provenance(path) != expected:
        raise RuntimeError(f"{path} is missing or older than the current questions, chunks or query settings: "
                           f"run `chunkeval retrieve` first")
    hits = load_hits(path)
    text = {c.chunk_id: c.text for c in chunks}
    shown = {}
    for qa in pairs:
        top = hits[qa.id][:k]
        shown[qa.id] = Shown([h.chunk_id for h in top], [text[h.chunk_id] for h in top],
                             [titles[h.doc_id] for h in top], bool(question_metrics(qa, top, k)["span_hit"]))
    return shown, expected


def gold_passage(qa: QAPair, doc: "CanonicalDoc") -> Shown:
    """The passage `qa` was generated from: its unit's spans as the generator saw them (one, or A and
    B of a two-span unit), each one passage with the document's title, without the section path the
    generator had."""
    spans = qa.meta["unit_spans"]
    return Shown([], [doc.text[start:end] for start, end in spans], [doc.title] * len(spans), True)


# --- the generator --------------------------------------------------------------------------------

def answer_key(messages: list[dict[str, str]], generator: dict[str, Any]) -> str:
    """sha256 of everything that decides an answer: the messages (prompt, passages, question) and
    `generator` (model, settings, Ollama digest, seed, ANSWER_VERSION)."""
    blob = json.dumps({"messages": messages, "generator": generator}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ask(todo: dict[str, tuple[str, str, list[dict[str, str]]]], llm: ChatLLM,
        on_record: Callable[[dict[str, Any]], None] | None = None) -> tuple[dict[str, dict[str, Any]], int]:
    """Ask the generator every request of `todo` (key -> (question_id, setup, messages)); return the
    new records by key and the number of API errors.

    A record: {"key", "question_id", "setup" (the first that asked it; identical messages are asked
    once), "model", "response", "error": None or why the answer is unusable (LLMOutputError: the same
    request at temperature 0 gives the same answer, so it is kept, not asked again),
    "prompt_tokens", "completion_tokens"}. Each goes to `on_record` as soon as it arrives, so a crash
    loses only the requests in flight. A request that failed (timeout, server down) gives no
    record: nothing to cache, the next run asks it again.
    """
    keys = list(todo)
    records: dict[str, dict[str, Any]] = {}
    api_errors = 0

    def save(i: int, result: LLMResult | Exception) -> None:
        nonlocal api_errors
        question_id, setup, _ = todo[keys[i]]
        record = {"key": keys[i], "question_id": question_id, "setup": setup, "model": llm.model, "response": "",
                  "error": None, "prompt_tokens": 0, "completion_tokens": 0}
        if isinstance(result, LLMResult):
            record.update(model=result.model, response=result.text.strip(), prompt_tokens=result.prompt_tokens,
                          completion_tokens=result.completion_tokens)
        elif isinstance(result, LLMOutputError):
            record["error"] = str(result)
        else:
            api_errors += 1
            return
        records[keys[i]] = record
        if on_record is not None:
            on_record(record)

    if keys:
        llm.chat_many([todo[key][2] for key in keys], on_result=save)
    return records, api_errors


# --- answer files ---------------------------------------------------------------------------------

def answer_path(cfg: Config, corpus: str, setup: str) -> Path:
    """runs/<run_id>/answers/<corpus>_<setup>.jsonl; <setup> is a variant name or GOLD_PASSAGE."""
    return cfg.paths.runs_dir / cfg.run_id / "answers" / f"{corpus}_{setup}.jsonl"


def answer_cache_path(cfg: Config) -> Path:
    """<paths.llm_cache_dir>/answer_cache.jsonl: one cache for every run_id and data dir."""
    return cfg.paths.llm_cache_dir / "answer_cache.jsonl"


def prompt_sha256(cfg: Config) -> str:
    """The answer prompt's hash, recorded with every answer file (and checked by `eval`)."""
    return hashlib.sha256(cfg.prompts.rag_answer.read_bytes()).hexdigest()


def write_answers(path: Path, answers: list[Answer], provenance: dict[str, Any]) -> None:
    """The provenance is removed first and written last, so it only ever sits next to the complete
    file it describes (as retrieval.write_hits does)."""
    meta = provenance_path(path)
    meta.unlink(missing_ok=True)
    write_jsonl(path, answers)
    write_json(meta, provenance)


def load_answers(path: Path) -> list[Answer]:
    """An answer file read back, in question order."""
    return [Answer(**record) for record in read_jsonl(path)]


def summarize(answers: list[Answer]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per corpus and setup: answered / abstained / errors, the abstention rate, and the two cases
    retrieval explains: abstained although the evidence was shown (an over-cautious generator, or
    evidence split so it looked incomplete), and answered without it (a hallucination risk, unless
    another document holds the same fact). Means of response words and token counts."""
    groups: dict[tuple[str, str], list[Answer]] = defaultdict(list)
    for a in answers:
        groups[(a.corpus, a.setup)].append(a)

    def mean(values: list[int], digits: int | None = None) -> float | None:
        return round(statistics.mean(values), digits) if values else None    # digits None: an int

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for (corpus, setup), group in groups.items():
        ok = [a for a in group if a.error is None]
        answered = [a for a in ok if not a.abstained]
        abstained = [a for a in ok if a.abstained]
        summary.setdefault(corpus, {})[setup] = {
            "questions": len(group), "answered": len(answered), "abstained": len(abstained),
            "errors": len(group) - len(ok), "abstention_rate": round(len(abstained) / len(group), 4),
            "evidence_shown": sum(a.evidence_shown for a in group),
            "abstained_with_evidence": sum(a.evidence_shown for a in abstained),
            "answered_without_evidence": sum(not a.evidence_shown for a in answered),
            "response_words_mean": mean([len(a.response.split()) for a in answered], 1),
            "prompt_tokens_mean": mean([a.prompt_tokens for a in ok]),
            "completion_tokens_mean": mean([a.completion_tokens for a in ok]),
        }
    return summary


# --- the stage ------------------------------------------------------------------------------------

def run(cfg: Config, secrets: Secrets, corpora: list[str]) -> None:
    """CLI entry (`chunkeval answer`): for each corpus of `corpora` with questions, answer every
    question in each (method, overlap) setup and from its gold passage; write one answer file per
    setup, and the summary (logged and written to runs/<run_id>/answer_summary.json).

    Everything the generator will be shown is assembled and checked first (documents, retrieval
    files), so a stale input stops the stage before any LLM call. Then only the answers the cache
    doesn't hold are asked. Answers without a result (API errors) stop the stage before any answer
    file is written: a re-run asks exactly those. The CLI already ran `check_ollama` before this
    and unloads the generator afterwards.
    """
    questions = load_questions(cfg)
    for corpus in corpora:
        if not questions.get(corpus):
            log.warning("%s: no questions in %s, nothing to answer", corpus, filter_paths(cfg)["filtered"])
    corpora = [c for c in corpora if questions.get(c)]
    docs = load_docs(cfg, [qa for c in corpora for qa in questions[c]], qa_paths(cfg)["provenance"])
    k = cfg.retrieval.top_k

    # 1. What each question is shown, per (corpus, setup), with the provenance of its answer file.
    setups: dict[tuple[str, str], tuple[dict[str, Shown], dict[str, Any]]] = {}
    for corpus in corpora:
        pairs = questions[corpus]
        titles = doc_titles(cfg, corpus)
        base = {"version": ANSWER_VERSION, "corpus": corpus, "questions_sha256": questions_sha256(pairs),
                "titles_sha256": titles_sha256(titles)}
        for _, _, variant in variants(cfg):
            shown, retrieval = retrieved(cfg, corpus, variant, pairs, titles)
            setups[(corpus, variant)] = shown, {**base, "setup": variant, "top_k": k, "retrieval": retrieval}
        shown = {qa.id: gold_passage(qa, docs[(corpus, qa.doc_id)]) for qa in pairs}
        setups[(corpus, GOLD_PASSAGE)] = shown, {**base, "setup": GOLD_PASSAGE, "top_k": None, "retrieval": None}

    # 2. The generator, asked only what the cache doesn't hold.
    llm = ChatLLM("rag_generator", cfg, secrets)
    generator = {**cfg.llm.rag_generator.model_dump(), "seed": cfg.seed, "version": ANSWER_VERSION,
                 "digest": ollama_model_digest(llm.base_url, llm.model) if llm.provider == "ollama" else None}
    cache_path = answer_cache_path(cfg)
    cache = {r["key"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    keys: dict[tuple[str, str, str], str] = {}          # (corpus, setup, question_id) -> answer key
    todo: dict[str, tuple[str, str, list[dict[str, str]]]] = {}
    for (corpus, setup), (shown, _) in setups.items():
        for qa in questions[corpus]:
            messages = build_messages(qa.question, shown[qa.id].passages, cfg.prompts.rag_answer)
            key = keys[(corpus, setup, qa.id)] = answer_key(messages, generator)
            if key not in cache and key not in todo:
                todo[key] = (qa.id, setup, messages)
    log.info("answers: %d of %d cached, %d to ask (%d setup(s) x questions, top_k %d)",
             sum(key in cache for key in keys.values()), len(keys), len(todo), len(setups), k)
    new, api_errors = ask(todo, llm, on_record=lambda record: append_jsonl(cache_path, record))
    cache.update(new)
    if api_errors:
        raise RuntimeError(f"{api_errors} question(s) got no answer (API errors, see the warnings above); no answer "
                           f"file written: re-run `chunkeval answer` to ask them again")

    # 3. The answer files, rewritten from the cache. Abstentions are marked here, not cached, so a
    # change to `is_abstention` applies without asking anything again.
    answers: list[Answer] = []
    for (corpus, setup), (shown, provenance) in setups.items():
        rows = []
        for qa in questions[corpus]:
            r, s = cache[keys[(corpus, setup, qa.id)]], shown[qa.id]
            rows.append(Answer(question_id=qa.id, corpus=corpus, setup=setup, response=r["response"],
                               contexts=[context(title, text) for title, text in s.passages],
                               chunk_ids=s.chunk_ids, evidence_shown=s.evidence,
                               abstained=r["error"] is None and is_abstention(r["response"]), error=r["error"],
                               model=r["model"], prompt_tokens=r["prompt_tokens"],
                               completion_tokens=r["completion_tokens"]))
        write_answers(answer_path(cfg, corpus, setup), rows, {**provenance, "generator": generator,
                      "prompt_sha256": prompt_sha256(cfg)})
        answers += rows

    summary = summarize(answers)
    for corpus, by_setup in summary.items():
        for setup, s in by_setup.items():
            log.info("%-7s %-21s %d questions: %d answered (%d without the evidence shown), %d abstained "
                     "(%d with it), %d errors | %s words, %s prompt tokens", corpus, setup, s["questions"],
                     s["answered"], s["answered_without_evidence"], s["abstained"], s["abstained_with_evidence"],
                     s["errors"], s["response_words_mean"], s["prompt_tokens_mean"])
    write_json(run_dir(cfg) / "answer_summary.json", summary)
