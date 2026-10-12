"""The `eval` stage: Ragas' scores reproduced call by call, the judge cache, the cost guard, the table.

Offline throughout: the judge is a fake `openai` client (`FakeJudge`) that recognises each Ragas prompt by
its instruction, reads the input from the rendered prompt and answers in valid JSON, deterministically: a
statement is "supported" when all its words occur in the premise. The stage tests reuse the offline
`answer` setup of test_answering.py (tiny_doc, 3 questions, one variant + the gold passage), with gold
answers and a fake generator that answers from what it is shown; the embedder is FakeEmbedder (hash vectors).
"""
import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pandas as pd
import pytest
from ragas import SingleTurnSample
from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                           ResponseRelevancy)
from ragas.metrics._faithfulness import NLIStatementInput
from ragas.run_config import RunConfig

from chunkeval.core.config import Secrets
from chunkeval.core.store import read_jsonl, run_dir, write_jsonl
from chunkeval.evaluation import ragas_eval
from chunkeval.evaluation.answering import ABSTENTION_PHRASE, GOLD_PASSAGE, Answer
from chunkeval.evaluation.ragas_eval import (METRICS, PROMPTS, Judge, JudgeLLM, judge_row, judge_settings,
                                            judged_scores, ragas_cache_path, ragas_sample, rouge_l, sentences,
                                            token_f1)
from chunkeval.qa.filtering import filter_paths
from chunkeval.qa.generation import QAPair, load_pairs
from test_answering import REPLICATES, answer, answers  # noqa: F401  (fixtures used below)
from test_retrieval import VARIANT, FakeEmbedder, hash_vectors, offline, retrieve, stage, words  # noqa: F401

SECRETS = Secrets(openai_api_key="sk-test")
GOLD = ["Each site had four replicates of every treatment.",
        "Irrigation doubled the harvest in the Sahel.",
        "Weeds were the main constraint in rainfed plots. Seed supply remained a bottleneck in remote villages."]
TWO_CLAIMS = "Each site had four replicates of every treatment. The trials ran for two seasons."


# --- the fake judge -------------------------------------------------------------------------------

def supported(premise: str, statement: str) -> int:
    return int(words(statement) <= words(premise))


def fake_judge(prompt: str) -> str:
    """A valid answer to each Ragas prompt; Ragas' fix-up prompt (for an unreadable output) gets garbage."""
    kind = next((k for k, p in PROMPTS.items() if p.instruction in prompt), None)
    if kind is None:
        return "still not JSON"
    data = json.loads(prompt.rsplit("input: ", 1)[1].rsplit("\nOutput:", 1)[0])
    if kind == "context_precision":
        out = {"reason": "r", "verdict": supported(data["context"], data["answer"])}
    elif kind == "context_recall":
        out = {"classifications": [{"statement": s, "reason": "r", "attributed": supported(data["context"], s)}
                                   for s in sentences(data["answer"])]}
    elif kind == "statements":
        out = {"statements": sentences(data["answer"])}
    elif kind == "nli":
        out = {"statements": [{"statement": s, "reason": "r", "verdict": supported(data["context"], s)}
                              for s in data["statements"]]}
    elif kind == "question":
        out = {"question": f"What about {data['response'][:40]}?", "noncommittal": 0}
    else:   # claims
        out = {"claims": sentences(data["response"])}
    return json.dumps(out)


def kind_of(prompt: str) -> str | None:
    return next((k for k, p in PROMPTS.items() if p.instruction in prompt), None)


class FakeJudge:
    """Stands in for the judge's AsyncOpenAI client: answers each prompt with `respond(prompt)`, a string
    or an exception. `prompts` records every prompt sent, `params` every request's parameters."""

    def __init__(self, respond=fake_judge, prompt_tokens=None):
        self.respond, self.prompt_tokens = respond, prompt_tokens
        self.prompts: list[str] = []
        self.params: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **params):
        prompt = params["messages"][0]["content"]
        self.prompts.append(prompt)
        self.params.append(params)
        out = self.respond(prompt)
        if isinstance(out, Exception):
            raise out
        usage = SimpleNamespace(prompt_tokens=self.prompt_tokens or len(prompt) // 4, completion_tokens=len(out) // 4,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        choice = SimpleNamespace(message=SimpleNamespace(content=out), finish_reason="stop")
        return SimpleNamespace(model=params["model"], choices=[choice], usage=usage)

    async def close(self):
        pass

    def asked(self, kind: str) -> list[str]:
        return [p for p in self.prompts if kind_of(p) == kind]


def cosine(x: str, y: str) -> float:
    a, b = hash_vectors("bge_m3", [x, y], 1024)
    return float(np.dot(a, b))


class FakeRagasEmbeddings:
    """What ResponseRelevancy needs, with the vectors FakeEmbedder gives bge_m3 (no prompt)."""

    def embed_query(self, text):
        return hash_vectors("bge_m3", [text], 1024)[0].tolist()

    def embed_documents(self, texts):
        return hash_vectors("bge_m3", texts, 1024).tolist()


# --- Ragas' scores, call by call ------------------------------------------------------------------

def test_fc_checks_claims_with_the_faithfulness_prompt():
    """One `nli` prompt serves both metrics, so its calls can be shared: FactualCorrectness' NLI prompt
    renders exactly like Faithfulness'."""
    data = NLIStatementInput(context="Seeds were sown in May.", statements=["Seeds were sown.", "It rained."])
    assert FactualCorrectness().nli_prompt.to_string(data) == PROMPTS["nli"].to_string(data)


def test_scores_equal_ragas_own_metrics(cfg, monkeypatch):
    """Each of the 5 metrics, run by Ragas itself (`single_turn_ascore`) and by the stage's per-call
    pipeline on the same judge, sends the same prompts and gives the same score."""
    question = "How many replicates of every treatment did each Sahel trial site have?"
    contexts = ["Weeds were the main constraint in rainfed plots.", GOLD[0],
                "Seed supply remained a bottleneck.", f"{GOLD[0]} Soils were sampled."]    # verdicts 0 1 0 1
    sample = SingleTurnSample(user_input=question, retrieved_contexts=contexts, response=TWO_CLAIMS,
                              reference=GOLD[0])
    ragas_judge = FakeJudge()
    llm = JudgeLLM(run_config=RunConfig(max_retries=1), client=ragas_judge, model="judge", seed=1)
    metrics = {"context_precision": LLMContextPrecisionWithReference(), "context_recall": LLMContextRecall(),
               "faithfulness": Faithfulness(), "answer_relevancy": ResponseRelevancy(strictness=1),
               "factual_correctness": FactualCorrectness()}
    metrics["answer_relevancy"].embeddings = FakeRagasEmbeddings()
    expected = {}
    for name, metric in metrics.items():
        metric.llm = llm
        expected[name] = asyncio.run(metric.single_turn_ascore(sample))

    ours = FakeJudge()
    monkeypatch.setattr(ragas_eval, "judge_client", lambda cfg, secrets: ours)
    qa = QAPair(id="d:q0", doc_id="d", corpus="papers", unit_id="q0", question=question, answer=GOLD[0],
                question_type="factoid", evidence=[], meta={})
    a = Answer(question_id=qa.id, corpus="papers", setup=VARIANT, response=TWO_CLAIMS, contexts=contexts,
               chunk_ids=[], evidence_shown=True, abstained=False, error=None, model="g", prompt_tokens=1,
               completion_tokens=1)
    judge = Judge(cfg, SECRETS, {}, judge_settings(cfg, SECRETS), on_record=lambda r: None)
    judged = asyncio.run(judge_row(judge, qa, a))
    scores = judged_scores(a, judged, cosine, question)

    assert sorted(ours.prompts) == sorted(ragas_judge.prompts)               # 4 + 1 + 2 + 1 + 4 prompts
    for name, value in expected.items():
        assert scores[name] == pytest.approx(value, abs=1e-6), name
    assert expected["context_precision"] == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert expected["faithfulness"] == 0.5 and expected["factual_correctness"] == 0.67
    assert (scores["fc_precision"], scores["fc_recall"], scores["fc_recall_ragas"]) == (0.5, 1.0, 1.0)
    assert scores["fc_f1"] == pytest.approx(2 / 3) and scores["cp_verdicts"] == "0101"


def test_judge_sends_the_config_settings_and_counts_usage(cfg, monkeypatch):
    client = FakeJudge()
    monkeypatch.setattr(ragas_eval, "judge_client", lambda cfg, secrets: client)
    records = []
    judge = Judge(cfg, SECRETS, {}, judge_settings(cfg, SECRETS), on_record=records.append)
    data = NLIStatementInput(context="Seeds were sown in May.", statements=["Seeds were sown."])
    record = asyncio.run(judge.ask("nli", data))
    params = client.params[0]
    assert params["model"] == cfg.llm.judge.model and params["temperature"] == cfg.llm.judge.temperature
    assert params["seed"] == cfg.seed and "n" not in params and [m["role"] for m in params["messages"]] == ["user"]
    assert record["output"]["statements"][0]["verdict"] == 1 and records == [record]
    assert record["input_tokens"] == len(client.prompts[0]) // 4 and record["requests"] == 1
    price = cfg.ragas.prices[cfg.llm.judge.model]
    assert record["cost_usd"] == pytest.approx((record["input_tokens"] * price.input
                                                + record["output_tokens"] * price.output) / 1e6)


def test_ragas_sample_is_nested_and_in_file_order():
    pairs = [QAPair(id=f"d:q{i}", doc_id="d", corpus="papers", unit_id=f"q{i}", question="?", answer="-",
                    question_type="factoid", evidence=[], meta={}) for i in range(30)]
    small, large = ragas_sample(pairs, 10, seed=42), ragas_sample(pairs, 11, seed=42)
    assert len(small) == 10 and set(q.id for q in small) < set(q.id for q in large)
    assert small == sorted(small, key=pairs.index) and ragas_sample(pairs, 99, seed=42) == pairs
    assert ragas_sample(pairs, 10, seed=7) != small


def test_text_metrics():
    assert token_f1("The yield was 3.4 t/ha.", "yield was 3.4 t/ha") == 1.0     # articles, punctuation, case
    assert token_f1("four replicates", "two seasons") == 0.0
    assert token_f1("rice maize millet sorghum", "rice maize") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert token_f1("a rice", "the rice") == 1.0                                  # articles don't count
    assert rouge_l(GOLD[0], GOLD[0]) == 1.0 and rouge_l("Weeds.", GOLD[0]) == 0.0


def test_ragas_telemetry_is_off():
    from ragas._analytics import do_not_track
    assert do_not_track()


# --- the stage ------------------------------------------------------------------------------------

def generator_answer(user: str) -> str:
    """The fake generator: answers from the passages it is shown, or abstains."""
    if REPLICATES in user:
        return TWO_CLAIMS
    if "irrigation doubled the harvest" in user:
        return GOLD[1]
    return ABSTENTION_PHRASE


@pytest.fixture
def evaluated(answers, monkeypatch):
    """The offline `answer` setup with gold answers, answered; the fake judge, embedder and token count.
    Returns (cfg, judge client)."""
    cfg, generator, pairs = answers
    for qa, gold in zip(pairs, GOLD):
        qa.answer = gold
    write_jsonl(filter_paths(cfg)["filtered"], pairs)           # the answer isn't part of the questions' hash
    generator.respond = generator_answer
    answer(cfg)
    client = FakeJudge()
    monkeypatch.setattr(ragas_eval, "judge_client", lambda cfg, secrets: client)
    monkeypatch.setattr(ragas_eval, "Embedder", FakeEmbedder)
    monkeypatch.setattr(ragas_eval, "check_gpu_free", lambda secrets: None)
    monkeypatch.setattr(ragas_eval, "count_tokens", lambda text: len(text) // 4)
    return cfg, client


def evaluate(cfg, **options) -> pd.DataFrame | None:
    ragas_eval.run(cfg, SECRETS, corpora=["papers"], **options)
    path = run_dir(cfg, "ragas") / "per_question.parquet"
    return pd.read_parquet(path) if path.exists() else None


def test_run_scores_every_question_in_every_setup(evaluated):
    cfg, client = evaluated
    table = evaluate(cfg)
    assert len(table) == 3 * 2 and not table.duplicated(["question_id", "corpus", "setup"]).any()
    assert set(table["setup"]) == {VARIANT, GOLD_PASSAGE} and set(METRICS) <= set(table.columns)
    gold = table[table["setup"] == GOLD_PASSAGE].set_index("question_id")
    assert gold["method"].isna().all() and table.loc[table["setup"] == VARIANT, "overlap"].all()

    # the replicates question from its own passage: two claims, one in the gold answer
    row = gold.loc["tiny:q0"]
    assert row["context_precision"] == pytest.approx(1.0) and (row["cp_verdicts"], row["context_recall"]) == ("1", 1.0)
    assert (row["faithfulness"], row["faith_statements"], row["faith_supported"]) == (0.5, 2, 1)
    assert (row["fc_precision"], row["fc_recall"], row["fc_recall_ragas"], row["factual_correctness"]) == \
           (0.5, 1.0, 1.0, 0.67)
    assert row["answer_relevancy"] == row["rr_cosine"] and not row["rr_noncommittal"]
    assert 0 < row["token_f1"] < 1 and 0 < row["rouge_l"] < 1 and row["response_words"] == len(TWO_CLAIMS.split())

    # abstentions: no response metric asked, scored as Ragas would (faithfulness left out)
    abstained = table[table["abstained"]]
    assert len(abstained) > 0
    assert (abstained[["answer_relevancy", "factual_correctness", "fc_recall", "token_f1"]] == 0).all().all()
    assert abstained["faithfulness"].isna().all() and abstained["context_precision"].notna().all()
    assert not any(ABSTENTION_PHRASE in p for p in client.asked("question") + client.asked("statements"))

    details = list(read_jsonl(run_dir(cfg, "ragas") / "details.jsonl"))
    assert len(details) == 6 and details[0].keys() >= {"question", "reference", "response", "context_precision"}
    cache = list(read_jsonl(ragas_cache_path(cfg)))
    assert len(cache) == len(client.prompts)                        # one line per call, no fix-ups needed
    cost = json.loads((run_dir(cfg, "ragas") / "cost.json").read_text(encoding="utf-8"))
    assert cost["this_run"]["calls"] == cost["table"]["calls"] == len(cache)
    assert cost["this_run"]["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in cache), abs=1e-4)
    summary = json.loads((run_dir(cfg) / "ragas_summary.json").read_text(encoding="utf-8"))["by_corpus"]["papers"]
    assert summary[GOLD_PASSAGE]["faithfulness"]["nan"] == summary[GOLD_PASSAGE]["abstained"]


def test_identical_calls_are_asked_once(evaluated):
    """No prompt is sent twice: the same answer in both setups has its claims and its question asked
    once, and a gold answer's claims are asked once per question, not once per setup."""
    cfg, client = evaluated
    table = evaluate(cfg)
    assert len(client.prompts) == len(set(client.prompts))
    scored = table[~table["abstained"]]
    responses = {d["response"] for d in read_jsonl(run_dir(cfg, "ragas") / "details.jsonl") if "question_generated" in d}
    assert len(client.asked("question")) == len(responses) < len(scored)
    assert len(client.asked("claims")) < 2 * len(scored)               # per row: the response's and the reference's


def test_rerun_asks_only_what_changed(evaluated, answers):
    cfg, client = evaluated
    evaluate(cfg)
    first = len(client.prompts)
    evaluate(cfg)
    assert len(client.prompts) == first                              # everything cached

    _, generator, _ = answers
    generator.respond = lambda user: GOLD[0] if REPLICATES in user else generator_answer(user)
    cfg.llm.rag_generator.max_tokens = 256                           # another generator: every answer asked again,
    answer(cfg)                                                      # only the replicates one changes
    table = evaluate(cfg)
    new = client.prompts[first:]
    assert new and not any(kind_of(p) in ("context_precision", "context_recall") for p in new)
    q0 = table[(table["question_id"] == "tiny:q0") & ~table["abstained"]]
    assert len(q0) and (q0["fc_precision"] == 1.0).all()


def test_api_error_writes_nothing_and_the_rerun_asks_only_it(evaluated):
    cfg, client = evaluated
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1"))
    client.respond = lambda p: error if kind_of(p) == "context_recall" else fake_judge(p)
    with pytest.raises(RuntimeError, match=r"call\(s\) failed .* re-run"):
        evaluate(cfg)
    assert not (run_dir(cfg, "ragas") / "per_question.parquet").exists()
    paid = list(read_jsonl(ragas_cache_path(cfg)))
    assert paid and not any(r["prompt"] == "context_recall" for r in paid)

    client.respond = fake_judge
    before = len(client.prompts)
    table = evaluate(cfg)
    assert {kind_of(p) for p in client.prompts[before:]} == {"context_recall"}
    assert table["context_recall"].notna().all()


def test_unreadable_output_is_cached_and_scores_nan(evaluated):
    """Ragas tries to fix an unreadable output (3 fix-up requests), then gives up: the call is cached
    with its error, only its metric is NaN, and a re-run doesn't pay for it again."""
    cfg, client = evaluated
    client.respond = lambda p: "not JSON" if kind_of(p) == "question" else fake_judge(p)
    table = evaluate(cfg)
    scored = table[~table["abstained"]]
    assert scored["answer_relevancy"].isna().all() and scored["factual_correctness"].notna().all()
    failed = [r for r in read_jsonl(ragas_cache_path(cfg)) if r["prompt"] == "question"]
    assert failed and all(r["error"] and r["requests"] == 4 for r in failed)
    before = len(client.prompts)
    evaluate(cfg)
    assert len(client.prompts) == before
    summary = json.loads((run_dir(cfg) / "ragas_summary.json").read_text(encoding="utf-8"))["by_corpus"]["papers"]
    assert summary[GOLD_PASSAGE]["answer_relevancy"]["nan"] == (~table[table["setup"] == GOLD_PASSAGE]["abstained"]).sum()


def test_dry_run_and_the_cost_cap_ask_nothing(evaluated):
    cfg, client = evaluated
    assert evaluate(cfg, dry_run=True) is None and client.prompts == []
    assert not ragas_cache_path(cfg).exists()

    cfg.ragas.max_cost_usd = 1e-6
    with pytest.raises(RuntimeError, match="above ragas.max_cost_usd"):
        evaluate(cfg)
    assert client.prompts == [] and not (run_dir(cfg, "ragas") / "per_question.parquet").exists()


def test_the_cap_stops_a_run_whose_estimate_was_too_low(evaluated, monkeypatch):
    """The spend is checked before every call: once it reaches the cap, nothing more is asked; what was
    paid is cached and nothing is written."""
    cfg, client = evaluated
    monkeypatch.setattr(ragas_eval, "count_tokens", lambda text: 0)        # estimate: $0
    client.prompt_tokens = 10_000_000                                     # each call: $4 at $0.40 per 1M
    with pytest.raises(RuntimeError, match="the cap ragas.max_cost_usd .* was reached"):
        evaluate(cfg)
    assert 1 <= len(client.prompts) <= cfg.ragas.max_workers + 1
    assert len(list(read_jsonl(ragas_cache_path(cfg)))) == len(client.prompts)
    assert not (run_dir(cfg, "ragas") / "per_question.parquet").exists()


def test_pilot_scores_a_prefix_of_the_sample_and_the_full_run_reuses_it(evaluated):
    cfg, client = evaluated
    table = evaluate(cfg, pilot=1)
    assert table["question_id"].nunique() == 1
    pilot_prompts = len(client.prompts)
    table = evaluate(cfg)
    assert table["question_id"].nunique() == 3
    assert not set(client.prompts[:pilot_prompts]) & set(client.prompts[pilot_prompts:])


def test_a_corpus_outside_corpus_comes_from_the_cache(evaluated, caplog):
    """`eval --corpus reports` keeps the papers rows in the table, from the cache alone (nothing asked).
    With a papers call missing from the cache, papers is left out with a warning, still asking nothing."""
    cfg, client = evaluated
    full = evaluate(cfg)                                             # papers scored and cached
    asked = len(client.prompts)
    ragas_eval.run(cfg, SECRETS, corpora=["reports"])                # reports has no questions here
    assert len(client.prompts) == asked
    pd.testing.assert_frame_equal(pd.read_parquet(run_dir(cfg, "ragas") / "per_question.parquet"), full)

    write_jsonl(ragas_cache_path(cfg), list(read_jsonl(ragas_cache_path(cfg)))[1:])   # one papers call lost
    with caplog.at_level(logging.WARNING):
        ragas_eval.run(cfg, SECRETS, corpora=["reports"])
    assert len(client.prompts) == asked
    assert "left out of the table: papers (not every judge call is cached" in caplog.text
    assert "nothing written" in caplog.text


def test_stale_answers_are_refused(evaluated):
    cfg, client = evaluated
    pairs = load_pairs(filter_paths(cfg)["filtered"])
    pairs[0].question = "How many replicates did each Sahel site have?"
    write_jsonl(filter_paths(cfg)["filtered"], pairs)
    retrieve(cfg)                                                    # retrieved again, not answered again
    with pytest.raises(RuntimeError, match="run `chunkeval answer` first"):
        evaluate(cfg)
    assert client.prompts == []
