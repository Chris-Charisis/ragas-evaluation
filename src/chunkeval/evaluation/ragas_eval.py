"""Stage 8 — Ragas evaluation with an OpenAI judge, per question and setup, plus judge-free text metrics.

What is scored. For each corpus, its Ragas sample (`ragas_sample`: ragas.sample_per_corpus questions of
filtered.jsonl, or `--pilot N`) in every answer file of stage 7: the 6 (method, overlap) setups and the
gold passage, the same questions in each (paired data). The Ragas columns (the old script,
ragas_evaluation.py:470-486, used the gold answer as response and the source chunk as reference):
    user_input          <- the question
    retrieved_contexts  <- Answer.contexts: exactly the passages the generator read (document title line,
                           then the text), in rank order
    response            <- Answer.response, the RAG answer of THIS setup
    reference           <- the gold answer (QAPair.answer)

Metrics, with Ragas' own prompts, output parsers and scoring code (ragas 0.2.15):
  context_precision     LLMContextPrecisionWithReference: per retrieved chunk, "useful for reaching the
                        reference?"; average precision over the ranks
  context_recall        LLMContextRecall: the reference's sentences attributable to the contexts
  faithfulness          Faithfulness: the response's statements supported by the contexts
  answer_relevancy      ResponseRelevancy: a question generated back from the response, its cosine to
                        the question (embedding.primary); 0 when the judge calls the response noncommittal
  factual_correctness   FactualCorrectness (mode f1, atomicity and coverage "low": Ragas' defaults): the
                        response's claims checked against the reference, and the reference's against the response
Everything the judge returns on the way is kept, at no extra cost: the per-chunk verdicts, statement and
claim counts, the noncommittal flag (columns) and every reason, statement and claim (details.jsonl). For
factual correctness the counts give fc_precision (response claims the reference supports), fc_recall
(reference claims the response covers), fc_f1 of those two, and fc_recall_ragas, the recall inside Ragas'
f1: supported RESPONSE claims / (those + uncovered REFERENCE claims). It mixes the two claim sets, so it
is inflated when the response has more claims than the reference, as RAG answers do (35–43 words on the
dev docs, against 22 for the gold answers).
Judge-free, against the reference: rouge_l (F), token_f1 (SQuAD-style), answer_similarity (cosine,
embedding.primary), and response_words.

How Ragas is called: one Ragas prompt at a time (`Judge.ask`), not through `ragas.evaluate`, because
  - every judge call is cached in ragas_cache.jsonl as soon as it returns, keyed by its rendered prompt
    (instruction, examples, input) and the judge settings, so nothing is paid twice: a re-run asks only
    what is new, a crash loses only the calls in flight, and identical calls are asked once (on the dev
    docs: a chunk judged for the same question in several setups, 16% of the context-precision calls;
    the same answer in several setups, 27% of the answer-only calls; the gold answer's claims, once per
    question instead of once per setup);
  - an API error is told apart from an unusable output (`ragas.evaluate` makes both NaN): an API error
    caches nothing and stops the stage before anything is written, and the re-run asks only it; an
    output the parser can't read even after Ragas' fix-up requests is cached with its error (the same
    call at temperature 0 would fail again) and makes that metric NaN;
  - the judge is `JudgeLLM`, an `openai` client with the config's model, temperature and seed. Ragas'
    LangchainLLMWrapper sets temperature and n on a SHARED client before each call, so concurrent
    calls could go out with each other's settings.
Two deliberate departures from Ragas' defaults, as they would only cost money:
  - answer relevancy generates 1 question per response, not `strictness` 3: in 0.2.15 those are 3
    separate calls at temperature ~0, i.e. the same question three times;
  - a claim check with no claims to check isn't sent (Ragas sends the empty list).

Cost. Before any call, the stage counts the input tokens of every call it doesn't have cached (the
prompts as they will be sent, o200k tokenizer), estimates their output tokens (the cache's mean per
prompt once it has a few, else OUTPUT_GUESS) and refuses to start above ragas.max_cost_usd; `--dry-run`
stops after the estimate. While running, it stops asking once the new spend reaches the cap. Each cache
line holds the tokens and USD of its call; cost.json has the totals.

Abstentions (decided 2026-10-11; `Answer.abstained`, see answering.is_abstention). Ragas 0.2.15 scores an
abstention as relevancy 0 (noncommittal) and factual correctness 0, and its faithfulness at random (NaN,
0 or 1, depending on which statements the judge extracts). So:
  - abstentions aren't sent to the judge for the response metrics: answer relevancy and every factual-
    correctness score are 0 in code (what Ragas gives, without the noise), and so is the text metrics'
    similarity to the reference; context precision and recall don't read the response, so they are
    scored as usual;
  - faithfulness is NaN for abstentions: report it next to the abstention rate, plus a view with
    abstentions = 1 (an abstention makes no unsupported claim), in analysis.py;
  - abstentions are never dropped from every metric: a setup that abstains on its hard questions would
    look better on the rest, and the paired comparison needs the same questions in every setup.
Answers with `error` set (cut off, context full) have no response: their response metrics are NaN. The
summary counts every NaN per metric and setup.

Ragas 0.2.x sends usage events to its developers unless RAGAS_DO_NOT_TRACK is set: set below, before
ragas is imported.

Reads:  runs/<run_id>/answers/<corpus>_<setup>.jsonl (up to date: `chunkeval answer`), data/qa/filtered.jsonl
Writes: runs/<run_id>/ragas/per_question.parquet   one row per (question_id, corpus, setup)
        runs/<run_id>/ragas/details.jsonl          per row: the judge's verdicts, reasons, statements, claims
        runs/<run_id>/ragas/cost.json              calls, tokens and USD: this run, and all the table rests on
        runs/<run_id>/ragas_cache.jsonl            every judge call (append-only cache, see above)
        runs/<run_id>/ragas_summary.json           means and NaN counts per corpus and setup
"""
from __future__ import annotations

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"   # before ragas is imported, see the docstring

import asyncio
import contextvars
import functools
import hashlib
import json
import logging
import math
import re
import string
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import openai
import ragas
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel
from ragas.exceptions import LLMDidNotFinishException, RagasOutputParserException
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                           ResponseRelevancy)
from ragas.metrics._answer_relevance import ResponseRelevanceInput
from ragas.metrics._context_precision import QAC, Verification
from ragas.metrics._context_recall import QCA, ContextRecallClassification
from ragas.metrics._factual_correctness import ClaimDecompositionInput, fbeta_score
from ragas.metrics._faithfulness import NLIStatementInput, NLIStatementOutput, StatementGeneratorInput
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig

from chunkeval.core.config import Secrets
from chunkeval.core.llm import ollama_model_digest
from chunkeval.core.schema import Config
from chunkeval.core.store import append_jsonl, read_jsonl, run_dir, write_json, write_jsonl, write_table
from chunkeval.corpus.embedding import Embedder, check_gpu_free, load_specs
from chunkeval.evaluation import answering
from chunkeval.evaluation.answering import GOLD_PASSAGE, Answer, answer_path, load_answers
from chunkeval.evaluation.retrieval import (file_provenance, load_questions, questions_sha256, stored_provenance,
                                            variant_chunks, variants)
from chunkeval.evaluation.retrieval_metrics import question_attributes
from chunkeval.qa.filtering import filter_paths
from chunkeval.qa.generation import QAPair

log = logging.getLogger(__name__)

# Bump when a change in this module changes what the judge is asked or how its answer is read: every
# cached call then misses its key and is asked (and paid) again.
RAGAS_EVAL_VERSION = 1

# The Ragas metrics, with their default settings, used for their prompts (each with its instruction,
# few-shot examples and output parser) and their scoring code.
_CP, _CR, _FAITH = LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness()
_FC = FactualCorrectness()                    # mode f1, atomicity and coverage "low" (picks the claim examples)
PROMPTS: dict[str, PydanticPrompt] = {
    "context_precision": _CP.context_precision_prompt,     # (question, one chunk, reference) -> verdict
    "context_recall": _CR.context_recall_prompt,           # (question, contexts, reference) -> sentence attributions
    "statements": _FAITH.statement_generator_prompt,       # (question, response) -> statements
    "nli": _FAITH.nli_statements_prompt,                   # (premise, statements) -> verdicts; also FC's (same prompt)
    "question": ResponseRelevancy().question_generation,   # response -> a question + noncommittal
    "claims": _FC.claim_decomposition_prompt,              # a text -> claims
}

# Output tokens per call, for the estimate, while the cache holds fewer than MEASURED_MIN calls of
# that prompt: rough guesses (a verdict with its reason; a list about as long as the text it splits).
OUTPUT_GUESS = {"context_precision": 60, "context_recall": 110, "statements": 90, "nli": 170, "question": 45,
                "claims": 75}
MEASURED_MIN = 5

ROW_KEY = ["question_id", "corpus", "setup"]
# The table's score columns: the Ragas metrics, factual correctness' parts, the text metrics.
METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "factual_correctness",
           "fc_precision", "fc_recall", "fc_f1", "fc_recall_ragas", "rouge_l", "token_f1", "answer_similarity"]
# Scores of an abstention: what Ragas gives it, set in code instead of asking the judge.
ABSTENTION_SCORES = {"answer_relevancy": 0.0, "factual_correctness": 0.0, "fc_precision": 0.0, "fc_recall": 0.0,
                     "fc_f1": 0.0, "fc_recall_ragas": 0.0, "rouge_l": 0.0, "token_f1": 0.0, "answer_similarity": 0.0}

# An API error after the openai SDK's retries: nothing to cache, the next run asks again.
API_ERRORS = (openai.APIError, asyncio.TimeoutError)
# The judge answered, but its output can't be used even after Ragas' fix-up requests: cached as such.
OUTPUT_ERRORS = (RagasOutputParserException, OutputParserException, LLMDidNotFinishException)

Record = dict[str, Any]          # one line of ragas_cache.jsonl


# --- the judge ------------------------------------------------------------------------------------

@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


# The usage of the judge call running in the current asyncio task (`Judge._call` sets it, `JudgeLLM` adds
# to it), so each cached call carries its own tokens, Ragas' fix-up requests included.
_USAGE: contextvars.ContextVar[Usage] = contextvars.ContextVar("judge_usage")


@dataclass
class JudgeLLM(BaseRagasLLM):
    """The LLM behind Ragas' prompts: one chat request per generation, with the config's model,
    temperature, seed and max_tokens, sent as a single user message (what LangchainLLMWrapper sends).

    Not LangchainLLMWrapper: it sets `temperature` and `n` on a shared ChatOpenAI before each request,
    so concurrent requests could go out with each other's settings. Ragas asks for temperature ~0
    (1e-8) for every prompt used here; the config's temperature is sent instead. Ragas' own retries
    are off (run_config max_retries 1): the openai SDK retries, honouring Retry-After.
    """
    client: Any = None
    model: str = ""
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None      # sent only when set, as core.llm.ChatLLM does

    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        raise NotImplementedError("JudgeLLM is asynchronous only")

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
        params: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt.to_string()}],
                                  "temperature": self.temperature, "seed": self.seed}
        if n != 1:
            params["n"] = n
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if stop:
            params["stop"] = stop
        response = await self.client.chat.completions.create(**params)
        usage = _USAGE.get(None)
        if usage is not None and response.usage is not None:
            details = getattr(response.usage, "prompt_tokens_details", None)
            usage.requests += 1
            usage.input_tokens += response.usage.prompt_tokens
            usage.cached_tokens += getattr(details, "cached_tokens", 0) or 0
            usage.output_tokens += response.usage.completion_tokens
            usage.model = response.model
        return LLMResult(generations=[[Generation(text=c.message.content or "",
                                                  generation_info={"finish_reason": c.finish_reason})
                                       for c in response.choices]])

    def is_finished(self, response: LLMResult) -> bool:
        """Every generation ended by itself; one cut off at max_tokens makes Ragas raise
        LLMDidNotFinishException (an unusable output)."""
        return all((g.generation_info or {}).get("finish_reason") == "stop" for g in response.generations[0])


def judge_client(cfg: Config, secrets: Secrets) -> Any:
    """The async `openai` client of the judge's provider (as core.llm.ChatLLM picks it)."""
    judge = cfg.llm.judge
    if judge.provider == "ollama":
        base_url, api_key = secrets.ollama_base_url, "ollama"
    else:
        base_url, api_key = "https://api.openai.com/v1", secrets.openai_api_key.get_secret_value()
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=cfg.ragas.timeout,
                              max_retries=cfg.ragas.max_retries)


def judge_settings(cfg: Config, secrets: Secrets) -> dict[str, Any]:
    """Everything besides the prompt that decides a judge call: the judge's model and settings, the
    seed, the Ragas version (its prompts and parsers), RAGAS_EVAL_VERSION, and an Ollama judge's digest."""
    judge = cfg.llm.judge
    settings = {**judge.model_dump(), "seed": cfg.seed, "ragas": ragas.__version__, "version": RAGAS_EVAL_VERSION}
    if judge.provider == "ollama":
        settings["digest"] = ollama_model_digest(secrets.ollama_base_url, judge.model)
    return settings


def call_key(prompt: str, settings: dict[str, Any]) -> str:
    """sha256 of one judge call: the rendered prompt (instruction, examples, input) and the settings."""
    blob = json.dumps({"prompt": prompt, "judge": settings}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def usable(record: Record | None) -> bool:
    return record is not None and record["error"] is None


class CachedJudge:
    """Answers Ragas' prompts from the cache only. A call that isn't cached gives None and is noted in
    `missing` (key -> (prompt kind, rendered prompt)); subclasses decide what else happens to it."""

    def __init__(self, cache: dict[str, Record], settings: dict[str, Any]) -> None:
        self.cache, self.settings = cache, settings
        self.missing: dict[str, tuple[str, str]] = {}
        self.cached: set[str] = set()          # keys found in the cache

    async def ask(self, kind: str, data: BaseModel) -> Record | None:
        text = PROMPTS[kind].to_string(data)
        key = call_key(text, self.settings)
        if key in self.cache:
            self.cached.add(key)
            return self.cache[key]
        self.missing[key] = (kind, text)
        return await self.on_missing(kind, data, key)

    async def on_missing(self, kind: str, data: BaseModel, key: str) -> Record | None:
        return None

    async def aclose(self) -> None:
        pass


class EstimatingJudge(CachedJudge):
    """Asks nothing: notes every call a run would make. Where a later call depends on the output
    (the statements or claims to check), it stands in the text's sentences for them."""

    async def on_missing(self, kind: str, data: BaseModel, key: str) -> Record | None:
        if kind == "statements":
            return {"key": key, "output": {"statements": sentences(data.answer)}, "error": None}
        if kind == "claims":
            return {"key": key, "output": {"claims": sentences(data.response)}, "error": None}
        return None


class Judge(CachedJudge):
    """Asks the judge what the cache doesn't hold, each call once (concurrent askers of the same call
    share its request), at most ragas.max_workers at a time. Each result is cached and handed to
    `on_record` as soon as it arrives.

    A call without a result gives None: an API error (counted in `api_errors`) or a call not started
    because the new spend reached ragas.max_cost_usd (`over_budget`)."""

    def __init__(self, cfg: Config, secrets: Secrets, cache: dict[str, Record], settings: dict[str, Any],
                 on_record: Callable[[Record], None]) -> None:
        super().__init__(cache, settings)
        judge = cfg.llm.judge
        self.client = judge_client(cfg, secrets)
        self.llm = JudgeLLM(run_config=RunConfig(max_retries=1, timeout=cfg.ragas.timeout), client=self.client,
                            model=judge.model, temperature=judge.temperature, seed=cfg.seed,
                            max_tokens=judge.max_tokens, reasoning_effort=judge.reasoning_effort)
        self.price = cfg.ragas.prices.get(judge.model) if judge.provider == "openai" else None
        self.cap = cfg.ragas.max_cost_usd
        self.on_record = on_record
        self.semaphore = asyncio.Semaphore(cfg.ragas.max_workers)
        self.pending: dict[str, asyncio.Future] = {}
        self.asked = self.api_errors = self.over_budget = 0
        self.spent = 0.0

    def cost(self, usage: Usage) -> float:
        if self.price is None:
            return 0.0
        return (usage.input_tokens * self.price.input + usage.output_tokens * self.price.output) / 1e6

    async def on_missing(self, kind: str, data: BaseModel, key: str) -> Record | None:
        if key not in self.pending:
            self.pending[key] = asyncio.ensure_future(self._call(kind, data, key))
        return await self.pending[key]

    async def _call(self, kind: str, data: BaseModel, key: str) -> Record | None:
        async with self.semaphore:
            if self.spent >= self.cap:
                self.over_budget += 1
                return None
            usage = Usage()
            _USAGE.set(usage)           # this task's own context: concurrent calls don't share it
            record: Record = {"key": key, "prompt": kind, "output": None, "error": None}
            try:
                record["output"] = (await PROMPTS[kind].generate(llm=self.llm, data=data)).model_dump()
            except OUTPUT_ERRORS as e:
                record["error"] = f"{type(e).__name__}: {e}"[:500]
            except API_ERRORS as e:
                self.api_errors += 1
                self.spent += self.cost(usage)          # a fix-up request may have gone through
                log.warning("judge: %s call failed: %s", kind, e)
                return None
            record.update(model=usage.model, requests=usage.requests, input_tokens=usage.input_tokens,
                          cached_tokens=usage.cached_tokens, output_tokens=usage.output_tokens,
                          cost_usd=round(self.cost(usage), 8))
            self.spent += record["cost_usd"]
            self.cache[key] = record
            self.on_record(record)
            self.asked += 1
            if self.asked % 200 == 0:
                log.info("judge: %d calls, $%.2f so far", self.asked, self.spent)
            return record

    async def aclose(self) -> None:
        await self.client.close()


# --- what the judge is asked per row --------------------------------------------------------------

def scores_response(a: Answer) -> bool:
    """Whether the response metrics ask the judge: not for an abstention or an answer without response."""
    return a.error is None and not a.abstained


async def judge_row(judge: CachedJudge, qa: QAPair, a: Answer) -> dict[str, Any]:
    """The judge records one row rests on (None where a call has no result), by name:
      context_precision   one per retrieved chunk, in rank order
      context_recall
    and when the response is scored:
      faithfulness        (the response's statements, their check against the contexts)
      question            the question generated back from the response (relevancy)
      response_claims     (the response's claims, their check against the reference)
      reference_claims    (the reference's claims, their check against the response)
    """
    contexts = "\n".join(a.contexts)
    jobs: dict[str, Awaitable] = {
        "context_precision": asyncio.gather(*(judge.ask("context_precision", QAC(question=qa.question, context=c,
                                                                                 answer=qa.answer))
                                              for c in a.contexts)),
        "context_recall": judge.ask("context_recall", QCA(question=qa.question, context=contexts, answer=qa.answer)),
    }
    if scores_response(a):
        jobs["faithfulness"] = _checked(judge, judge.ask("statements", StatementGeneratorInput(
            question=qa.question, answer=a.response)), "statements", contexts)
        jobs["question"] = judge.ask("question", ResponseRelevanceInput(response=a.response))
        jobs["response_claims"] = _checked(judge, judge.ask("claims", ClaimDecompositionInput(response=a.response)),
                                           "claims", qa.answer)
        jobs["reference_claims"] = _checked(judge, judge.ask("claims", ClaimDecompositionInput(response=qa.answer)),
                                            "claims", a.response)
    return dict(zip(jobs, await asyncio.gather(*jobs.values())))


async def _checked(judge: CachedJudge, extract: Awaitable, field: str,
                   premise: str) -> tuple[Record | None, Record | None]:
    """(the record of a statement or claim list, the record of its check against `premise`). Nothing to
    check, nothing asked: no statements makes faithfulness NaN (as in Ragas), no claims count 0."""
    record = await extract
    items = record["output"][field] if usable(record) else []
    if not items:
        return record, None
    return record, await judge.ask("nli", NLIStatementInput(context=premise, statements=items))


def records_of(judged: dict[str, Any]) -> list[Record | None]:
    """Every record of one row (`judge_row`'s result), None for a call without a result. A check that
    wasn't needed (no statements or claims to check) isn't listed."""
    records = [*judged["context_precision"], judged["context_recall"]]
    if "question" in judged:
        records.append(judged["question"])
    for name in ("faithfulness", "response_claims", "reference_claims"):
        if name in judged:
            extract, check = judged[name]
            records.append(extract)
            if usable(extract) and any(extract["output"].values()):     # it had items: their check is needed
                records.append(check)
    return records


async def judge_rows(judge: CachedJudge, items: list[tuple[QAPair, Answer]]) -> list[dict[str, Any]]:
    try:
        return list(await asyncio.gather(*(judge_row(judge, qa, a) for qa, a in items)))
    finally:
        await judge.aclose()


# --- scores ---------------------------------------------------------------------------------------

def _output(record: Record | None) -> Any:
    return record["output"] if record is not None and record["error"] is None else {"error": (record or {}).get("error")}


def judged_scores(a: Answer, judged: dict[str, Any], cosine: Callable[[str, str], float],
                  question: str) -> dict[str, Any]:
    """The Ragas scores of one row from its judge records, with Ragas' own scoring code, plus what the
    judge returned on the way (counts, verdicts). NaN where a record holds an unusable output."""
    nan = math.nan
    s: dict[str, Any] = {}

    cp = judged["context_precision"]
    if all(usable(r) for r in cp):
        verdicts = [int(bool(r["output"]["verdict"])) for r in cp]
        s["context_precision"] = _CP._calculate_average_precision([Verification(**r["output"]) for r in cp])
        s["cp_verdicts"] = "".join(map(str, verdicts))           # by rank: "10010"
        s["cp_useful"] = sum(verdicts)
    else:
        s["context_precision"], s["cp_verdicts"], s["cp_useful"] = nan, None, None

    cr = judged["context_recall"]
    if usable(cr):
        classes = [ContextRecallClassification(**c) for c in cr["output"]["classifications"]]
        s["context_recall"] = _CR._compute_score(classes) if classes else nan
        s["cr_sentences"], s["cr_attributed"] = len(classes), sum(bool(c.attributed) for c in classes)
    else:
        s["context_recall"], s["cr_sentences"], s["cr_attributed"] = nan, None, None

    if not scores_response(a):
        none = {k: None for k in ("faith_statements", "faith_supported", "rr_cosine", "rr_noncommittal",
                                  "fc_response_claims", "fc_response_supported", "fc_reference_claims",
                                  "fc_reference_covered")}
        scores = {m: (ABSTENTION_SCORES[m] if a.abstained else nan) for m in ABSTENTION_SCORES}
        return {**s, "faithfulness": nan, **scores, **none}

    statements, check = judged["faithfulness"]
    s["faithfulness"], s["faith_statements"], s["faith_supported"] = nan, None, None
    if usable(statements) and (not statements["output"]["statements"] or usable(check)):
        if check is None:                                  # no statements: NaN, as in Ragas
            s["faith_statements"], s["faith_supported"] = 0, 0
        else:
            nli = NLIStatementOutput(**check["output"])
            s["faithfulness"] = _FAITH._compute_score(nli)
            s["faith_statements"] = len(nli.statements)
            s["faith_supported"] = sum(bool(v.verdict) for v in nli.statements)

    q = judged["question"]
    s["answer_relevancy"], s["rr_cosine"], s["rr_noncommittal"] = nan, None, None
    if usable(q) and q["output"]["question"]:
        s["rr_cosine"] = cosine(question, q["output"]["question"])
        s["rr_noncommittal"] = bool(q["output"]["noncommittal"])
        s["answer_relevancy"] = s["rr_cosine"] * int(not s["rr_noncommittal"])

    s.update(factual_correctness(judged["response_claims"], judged["reference_claims"]))
    return s


def _verdicts(extract: Record | None, check: Record | None) -> list[bool] | None:
    """The NLI verdicts of a claim list (empty without claims); None if a record is unusable."""
    if not usable(extract):
        return None
    if not extract["output"]["claims"]:
        return []
    return [bool(v["verdict"]) for v in check["output"]["statements"]] if usable(check) else None


def factual_correctness(response_claims: tuple, reference_claims: tuple) -> dict[str, Any]:
    """Ragas' f1 (FactualCorrectness, mode f1: fbeta_score, rounded to 2 decimals as Ragas does) and its
    parts from the same verdicts: precision = response claims the reference supports; recall =
    reference claims the response covers; fc_f1 of those two (unrounded); fc_recall_ragas = the recall
    inside Ragas' f1, tp / (tp + fn) with tp counted in response claims and fn in reference claims."""
    supported, covered = _verdicts(*response_claims), _verdicts(*reference_claims)
    if supported is None or covered is None:
        return {k: math.nan for k in ("factual_correctness", "fc_precision", "fc_recall", "fc_f1", "fc_recall_ragas")} | \
               {k: None for k in ("fc_response_claims", "fc_response_supported", "fc_reference_claims",
                                  "fc_reference_covered")}
    tp, fp, fn = sum(supported), len(supported) - sum(supported), len(covered) - sum(covered)
    precision = tp / len(supported) if supported else math.nan
    recall = sum(covered) / len(covered) if covered else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"factual_correctness": float(np.round(fbeta_score(tp, fp, fn, _FC.beta), 2)),
            "fc_precision": precision, "fc_recall": recall, "fc_f1": f1,
            "fc_recall_ragas": tp / (tp + fn) if tp + fn else 0.0,
            "fc_response_claims": len(supported), "fc_response_supported": tp,
            "fc_reference_claims": len(covered), "fc_reference_covered": sum(covered)}


# --- judge-free text metrics ----------------------------------------------------------------------

@functools.cache
def _rouge():
    from rouge_score import rouge_scorer        # imported here: it pulls in nltk
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(response: str, reference: str) -> float:
    """ROUGE-L F-measure (longest common subsequence of stemmed, lower-cased words)."""
    return _rouge().score(reference, response)["rougeL"].fmeasure


_PUNCTUATION = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _squad_words(text: str) -> list[str]:
    """SQuAD's answer normalisation: lower case, no punctuation, no articles, single spaces."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION)
    return _ARTICLES.sub(" ", text).split()


def token_f1(response: str, reference: str) -> float:
    """SQuAD token F1: the harmonic mean of the shared words' share of each text (words counted with
    multiplicity)."""
    pred, gold = _squad_words(response), _squad_words(reference)
    common = sum((Counter(pred) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def cosines(cfg: Config, secrets: Secrets, questions: list[tuple[str, str]],
            answers: list[tuple[str, str]]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Cosines of each text pair with embedding.primary (unit vectors: dot products). Question pairs (the
    question, the one generated back for relevancy) are encoded as queries, answer pairs (response,
    reference) as documents: each side the same way. The model is loaded only if there is a pair."""
    if not questions and not answers:
        return {}, {}
    check_gpu_free(secrets)
    with Embedder(load_specs(cfg)[cfg.embedding.primary]) as model:
        def pair_cosines(pairs, encode):
            texts = list(dict.fromkeys(t for pair in pairs for t in pair))
            vectors = dict(zip(texts, encode(texts, cfg.embedding.batch_size))) if texts else {}
            return {(x, y): float(np.dot(vectors[x], vectors[y])) for x, y in pairs}
        return pair_cosines(questions, model.encode_queries), pair_cosines(answers, model.encode_documents)


@functools.cache
def _segmenter():
    import pysbd
    return pysbd.Segmenter(language="en", clean=False)


def sentences(text: str) -> list[str]:
    """The sentences of `text` (pysbd), for the estimate's stand-in statements and claims."""
    return [s.strip() for s in _segmenter().segment(text) if s.strip()] or [text]


# --- inputs ---------------------------------------------------------------------------------------

def ragas_sample(pairs: list[QAPair], n: int, seed: int) -> list[QAPair]:
    """The n questions of a corpus Ragas scores, in file order: those with the lowest sha256(seed|id).

    A hash rank, not a shuffle: the sample of n is the start of the sample of n + 1, so the `--pilot`
    questions are part of the full sample (and paid for once), and adding questions to filtered.jsonl
    changes the sample only where a new one ranks in."""
    ranked = sorted(pairs, key=lambda qa: hashlib.sha256(f"{seed}|{qa.id}".encode("utf-8")).hexdigest())
    chosen = {qa.id for qa in ranked[:n]}
    return [qa for qa in pairs if qa.id in chosen]


def setups(cfg: Config) -> list[tuple[str | None, bool | None, str]]:
    """(method, overlap, setup) of every answer file of a corpus: the variants, then the gold passage."""
    return [*variants(cfg), (None, None, GOLD_PASSAGE)]


def load_corpus_answers(cfg: Config, corpus: str, pairs: list[QAPair]) -> dict[str, dict[str, Answer]]:
    """setup -> question_id -> Answer for a corpus's whole question set `pairs`. Refuses an answer file
    that is missing or was made for other questions, retrieval, top_k, document titles or answer prompt
    than the current ones (what `chunkeval answer` would redo), so no row scores an answer to something else."""
    prompt_sha = hashlib.sha256(answering.PROMPT_PATH.read_bytes()).hexdigest()
    titles_sha = answering.titles_sha256(answering.doc_titles(cfg, corpus))
    by_setup = {}
    for method, overlap, setup in setups(cfg):
        path = answer_path(cfg, corpus, setup)
        stored = stored_provenance(path)
        current = stored is not None and stored.get("version") == answering.ANSWER_VERSION and \
            stored.get("questions_sha256") == questions_sha256(pairs) and stored.get("prompt_sha256") == prompt_sha \
            and stored.get("titles_sha256") == titles_sha
        if current and setup != GOLD_PASSAGE:
            sha = variant_chunks(cfg, corpus, setup)[1]["chunks_sha256"]
            current = stored["top_k"] == cfg.retrieval.top_k and stored["retrieval"] == file_provenance(
                cfg, cfg.embedding.primary, corpus, setup, sha, pairs)
        if not current:
            raise RuntimeError(f"{path} is missing or older than the current questions, retrieval, document "
                               f"titles or answer prompt: run `chunkeval answer` first")
        by_setup[setup] = {a.question_id: a for a in load_answers(path)}
    return by_setup


# --- cost -----------------------------------------------------------------------------------------

@functools.cache
def _encoding():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")          # gpt-4o, gpt-4.1 and later


def count_tokens(text: str) -> int:
    """Input tokens of one prompt sent as a single user message (+7: the chat format's own tokens)."""
    return len(_encoding().encode(text, disallowed_special=())) + 7


def estimate(cfg: Config, missing: dict[str, tuple[str, str]], cache: dict[str, Record]) -> dict[str, Any]:
    """Calls, tokens and USD of the `missing` calls (key -> (prompt kind, rendered prompt)). Input tokens
    are counted; output tokens are the cache's mean for that prompt once it holds MEASURED_MIN calls of
    it, else OUTPUT_GUESS. Fix-up requests for unreadable outputs aren't foreseen."""
    measured: dict[str, list[int]] = defaultdict(list)
    for r in cache.values():
        measured[r["prompt"]].append(r["output_tokens"])
    out_per_call = {kind: statistics.mean(measured[kind]) if len(measured[kind]) >= MEASURED_MIN else guess
                    for kind, guess in OUTPUT_GUESS.items()}
    by_prompt: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    for kind, text in missing.values():
        b = by_prompt[kind]
        b["calls"] += 1
        b["input_tokens"] += count_tokens(text)
        b["output_tokens"] += out_per_call[kind]
    judge = cfg.llm.judge
    price = cfg.ragas.prices.get(judge.model) if judge.provider == "openai" else None
    for b in by_prompt.values():
        b["output_tokens"] = round(b["output_tokens"])
        b["cost_usd"] = (b["input_tokens"] * price.input + b["output_tokens"] * price.output) / 1e6 if price else 0.0
    total = {k: sum(b[k] for b in by_prompt.values()) for k in ("calls", "input_tokens", "output_tokens", "cost_usd")}
    return {"by_prompt": dict(by_prompt), **total}


def cost_totals(records: list[Record]) -> dict[str, Any]:
    """Calls, requests, tokens and USD of `records` (cache lines), in all and by prompt."""
    fields = ("requests", "input_tokens", "cached_tokens", "output_tokens", "cost_usd")
    by_prompt: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, **{f: 0 for f in fields}})
    for r in records:
        b = by_prompt[r["prompt"]]
        b["calls"] += 1
        for f in fields:
            b[f] += r[f]
    total = {k: sum(b[k] for b in by_prompt.values()) for k in ("calls", *fields)}
    return {**{k: round(v, 4) if k == "cost_usd" else v for k, v in total.items()},
            "by_prompt": {k: {**b, "cost_usd": round(b["cost_usd"], 4)} for k, b in by_prompt.items()}}


# --- outputs --------------------------------------------------------------------------------------

def ragas_cache_path(cfg: Config) -> Path:
    """runs/<run_id>/ragas_cache.jsonl, outside ragas/ so that folder holds the stage's results only."""
    return cfg.paths.runs_dir / cfg.run_id / "ragas_cache.jsonl"


def details(qa: QAPair, a: Answer, judged: dict[str, Any]) -> dict[str, Any]:
    """One line of details.jsonl: the texts and everything the judge returned for a row (an unusable
    output as {"error": ...}), to read why a score is what it is."""
    d: dict[str, Any] = {"question_id": qa.id, "corpus": qa.corpus, "setup": a.setup, "question": qa.question,
                         "reference": qa.answer, "response": a.response, "chunk_ids": a.chunk_ids,
                         "context_precision": [_output(r) for r in judged["context_precision"]],
                         "context_recall": _output(judged["context_recall"])}
    for name in ("faithfulness", "response_claims", "reference_claims"):
        if name in judged:
            extract, check = judged[name]
            d[name] = {"extracted": _output(extract), "checked": None if check is None else _output(check)}
    if "question" in judged:
        d["question_generated"] = _output(judged["question"])
    return d


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per corpus and setup: questions, abstentions, answers with an error, and each metric's mean over
    its non-NaN values with the NaN count; analysis.py adds the confidence intervals."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["corpus"], r["setup"])].append(r)
    summary: dict[str, Any] = {}
    for (corpus, setup), group in groups.items():
        entry: dict[str, Any] = {"questions": len(group), "abstained": sum(r["abstained"] for r in group),
                                 "answer_errors": sum(r["answer_error"] for r in group)}
        for m in METRICS:
            values = [r[m] for r in group if not math.isnan(r[m])]
            entry[m] = {"mean": round(statistics.mean(values), 4) if values else None,
                        "nan": len(group) - len(values)}
        summary.setdefault(corpus, {})[setup] = entry
    return summary


def score_rows(cfg: Config, secrets: Secrets, scored: list[tuple[QAPair, Answer, dict[str, Any]]]) -> list[dict[str, Any]]:
    """The table's rows from each (question, answer, judge records): the judged scores, the text
    metrics, and the columns analysis.py groups by. The cosines are computed in one embedder session."""
    question_pairs, answer_pairs = [], []
    for qa, a, j in scored:
        if scores_response(a):
            if usable(j["question"]) and j["question"]["output"]["question"]:
                question_pairs.append((qa.question, j["question"]["output"]["question"]))
            answer_pairs.append((a.response, qa.answer))
    question_cos, answer_cos = cosines(cfg, secrets, list(dict.fromkeys(question_pairs)),
                                       list(dict.fromkeys(answer_pairs)))
    variant_of = {setup: (method, overlap) for method, overlap, setup in setups(cfg)}
    rows = []
    for qa, a, j in scored:
        method, overlap = variant_of[a.setup]
        row = {"question_id": qa.id, "corpus": qa.corpus, "setup": a.setup, "method": method, "overlap": overlap,
               **judged_scores(a, j, lambda x, y: question_cos[(x, y)], qa.question)}
        if scores_response(a):
            row.update(rouge_l=rouge_l(a.response, qa.answer), token_f1=token_f1(a.response, qa.answer),
                       answer_similarity=answer_cos[(a.response, qa.answer)])
        row.update(response_words=len(a.response.split()), abstained=a.abstained, answer_error=a.error is not None,
                   evidence_shown=a.evidence_shown, n_contexts=len(a.contexts), **question_attributes(qa))
        rows.append(row)
    return rows


# --- the stage ------------------------------------------------------------------------------------

def run(cfg: Config, secrets: Secrets, corpora: list[str], pilot: int | None = None, dry_run: bool = False) -> None:
    """CLI entry (`chunkeval eval`): score the Ragas sample of each corpus of `corpora` (or its first
    `pilot` questions) in every setup and the gold passage, and write the table, details, cost and summary.

    1. Inputs: each corpus's answer files must be up to date (`load_corpus_answers`).
    2. Estimate: the calls the cache doesn't hold, their tokens and USD; refused above
       ragas.max_cost_usd; `dry_run` stops here, with nothing asked or written.
    3. The judge is asked what is missing. An API error, or the cap reached while running, stops the
       stage before anything is written; what was paid is cached, and the re-run asks only the rest.
    4. The table covers every corpus with questions, like the retrieval table: a corpus outside
       `corpora` comes from the cache only, and is left out (with a warning) if any of its calls is missing.
    """
    questions = load_questions(cfg)
    for corpus in corpora:
        if not questions.get(corpus):
            log.warning("%s: no questions in %s, nothing to evaluate", corpus, filter_paths(cfg)["filtered"])
    corpora = [c for c in corpora if questions.get(c)]
    if pilot:
        log.warning("pilot: only the first %d questions of each corpus's sample; the table covers just these", pilot)

    # 1. The rows: (question, answer) per setup, for each corpus's sample.
    items: dict[str, list[tuple[QAPair, Answer]]] = {}
    left_out: list[str] = []
    for corpus in cfg.paths.corpora:
        pairs = questions.get(corpus)
        if not pairs:
            continue
        try:
            answers = load_corpus_answers(cfg, corpus, pairs)
        except (RuntimeError, FileNotFoundError) as e:
            if corpus in corpora:
                raise
            left_out.append(f"{corpus} ({e})")
            continue
        n = pilot if pilot and corpus in corpora else cfg.ragas.sample_per_corpus
        sample = ragas_sample(pairs, n, cfg.seed)
        items[corpus] = [(qa, answers[setup][qa.id]) for _, _, setup in setups(cfg) for qa in sample]
        if corpus in corpora:
            log.info("%s: %d of %d questions x %d setups = %d rows", corpus, len(sample), len(pairs),
                     len(setups(cfg)), len(items[corpus]))
    in_scope = [item for c in corpora for item in items.get(c, [])]

    # 2. What isn't cached, and what it would cost.
    settings = judge_settings(cfg, secrets)
    cache_path = ragas_cache_path(cfg)
    cache = {r["key"]: r for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    estimator = EstimatingJudge(cache, settings)
    asyncio.run(judge_rows(estimator, in_scope))
    cost = estimate(cfg, estimator.missing, cache)
    cap = cfg.ragas.max_cost_usd
    log.info("judge %s: %d call(s) to ask, %d cached: ~%.2fM input + ~%.2fM output tokens, ~$%.2f (cap $%.2f)",
             cfg.llm.judge.model, cost["calls"], len(estimator.cached), cost["input_tokens"] / 1e6,
             cost["output_tokens"] / 1e6, cost["cost_usd"], cap)
    for kind, b in sorted(cost["by_prompt"].items()):
        log.info("  %-17s %6d calls  %9d input tokens  ~$%.3f", kind, b["calls"], b["input_tokens"], b["cost_usd"])
    if dry_run:
        log.info("dry run: nothing asked, nothing written")
        return
    if cost["cost_usd"] > cap:
        raise RuntimeError(f"the judge calls would cost ~${cost['cost_usd']:.2f}, above ragas.max_cost_usd "
                           f"(${cap:.2f}): use --pilot N or a smaller ragas.sample_per_corpus, or raise the cap")

    # 3. Ask what is missing; then every row's records, the in-scope ones complete.
    new: list[Record] = []
    if estimator.missing:
        def save(record: Record) -> None:
            append_jsonl(cache_path, record)
            new.append(record)
        judge = Judge(cfg, secrets, cache, settings, on_record=save)
        asyncio.run(judge_rows(judge, in_scope))
        spent = sum(r["cost_usd"] for r in new)
        log.info("judge: %d call(s) asked, $%.4f (estimate $%.4f)", len(new), spent, cost["cost_usd"])
        if judge.api_errors or judge.over_budget:
            why = (f"{judge.api_errors} call(s) failed (API errors, see the warnings above)" if judge.api_errors
                   else f"the cap ragas.max_cost_usd (${cap:.2f}) was reached with {judge.over_budget} call(s) left")
            raise RuntimeError(f"{why}; nothing written, the {len(new)} paid call(s) are cached: re-run "
                               f"`chunkeval eval` to ask only the rest")
    judged_rows: dict[str, list[dict[str, Any]]] = {}
    for corpus, rows in items.items():
        judged = asyncio.run(judge_rows(CachedJudge(cache, settings), rows))
        if any(r is None for j in judged for r in records_of(j)):
            if corpus in corpora:
                raise RuntimeError(f"{corpus}: judge calls missing right after asking them")
            left_out.append(f"{corpus} (not every judge call is cached: run `chunkeval eval --corpus {corpus}`)")
            continue
        judged_rows[corpus] = judged
    if left_out:
        log.warning("left out of the table: %s", "; ".join(left_out))
    if not judged_rows:
        log.warning("no corpus has rows to score: nothing written")
        return

    # 4. Scores, then the outputs.
    scored = [(qa, a, j) for corpus, judged in judged_rows.items() for (qa, a), j in zip(items[corpus], judged)]
    rows = score_rows(cfg, secrets, scored)
    used = {r["key"]: r for _, _, j in scored for r in records_of(j)}
    out = run_dir(cfg, "ragas")
    path = write_table(out / "per_question.parquet", rows, unique_by=ROW_KEY)
    write_jsonl(out / "details.jsonl", (details(qa, a, j) for qa, a, j in scored))
    n_questions = len({(r["corpus"], r["question_id"]) for r in rows})
    table = cost_totals(list(used.values()))
    price = cfg.ragas.prices.get(cfg.llm.judge.model)
    write_json(out / "cost.json", {
        "judge": cfg.llm.judge.model, "price_per_mtok": price.model_dump() if price else None,
        "this_run": {"estimate_usd": round(cost["cost_usd"], 4), **cost_totals(new)},
        "table": {"rows": len(rows), "questions": n_questions,
                  "cost_per_question_usd": round(table["cost_usd"] / n_questions, 5) if n_questions else None,
                  **table},
        "note": "cached_tokens are priced as input tokens: the bill is at most this"})
    summary = summarize(rows)
    write_json(run_dir(cfg) / "ragas_summary.json", {"pilot": pilot, "by_corpus": summary})
    log.info("%d rows written to %s; all the table's judge calls cost $%.4f ($%.5f per question, all setups)",
             len(rows), path, table["cost_usd"], table["cost_usd"] / n_questions if n_questions else 0)
    for corpus, by_setup in summary.items():
        for setup, s in by_setup.items():
            fmt = {m: "  nan" if s[m]["mean"] is None else f"{s[m]['mean']:.3f}" for m in METRICS}
            nans = sum(s[m]["nan"] for m in METRICS)
            log.info("%-7s %-21s %3d q, %2d abstained | precision %s recall %s faithfulness %s relevancy %s "
                     "factual %s (P %s R %s) | rougeL %s F1 %s sim %s | %d NaN", corpus, setup, s["questions"],
                     s["abstained"], fmt["context_precision"], fmt["context_recall"], fmt["faithfulness"],
                     fmt["answer_relevancy"], fmt["factual_correctness"], fmt["fc_precision"], fmt["fc_recall"],
                     fmt["rouge_l"], fmt["token_f1"], fmt["answer_similarity"], nans)
