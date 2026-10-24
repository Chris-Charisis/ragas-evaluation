"""llm.py — offline logic tests (fake client) + live tests against Ollama / OpenAI.

Offline tests run everywhere in milliseconds. Live tests are marked and skip themselves when
the service is unavailable:
  - `ollama`: needs the local server with the configured models pulled; loads them onto the GPU
    (~1 min) and unloads them again at the end. They also pin down Ollama BEHAVIOUR this code
    relies on (thinking switch, silent truncation, ...) — rerun them after upgrading Ollama.
  - `openai`: only free metadata calls (key valid, judge snapshots not retired).
Fast run: pytest -m "not ollama and not openai"
"""
import threading
import time
from types import SimpleNamespace

import pytest
import requests

from chunkeval.core import store
from chunkeval.core.config import Secrets, load_secrets
from chunkeval.core.llm import (ChatLLM, LLMOutputError, LLMResult, _native_url, _ollama_tag,
                           _parse_json, check_ollama, unload_ollama_model)

SECRETS = Secrets(openai_api_key="sk-test-not-real", ollama_base_url="http://localhost:11434/v1")
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["reason", "score"],
    "properties": {"reason": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 5}},
}


# ======================================================================================
# Offline: helpers
# ======================================================================================

@pytest.mark.parametrize("url, native", [
    ("http://localhost:11434/v1", "http://localhost:11434"),
    ("http://localhost:11434/v1/", "http://localhost:11434"),
    ("http://localhost:11434", "http://localhost:11434"),
])
def test_native_url(url, native):
    assert _native_url(url) == native


def test_ollama_tag():
    assert _ollama_tag("llama3.2") == "llama3.2:latest"
    assert _ollama_tag("qwen3.5:9b-q8_0") == "qwen3.5:9b-q8_0"


@pytest.mark.parametrize("text", ['{"reason": "r", "score": 3}',
                                  '```json\n{"reason": "r", "score": 3}\n```',
                                  '```\n{"reason": "r", "score": 3}\n```'])
def test_parse_json_accepts_plain_and_fenced(text):
    assert _parse_json(text, SCHEMA, "role") == {"reason": "r", "score": 3}


@pytest.mark.parametrize("text, needle", [
    ("Hello!", "not valid JSON"),
    ('{"reason": "r", "score": "3"}', "doesn't match the schema"),    # string instead of integer
    ('{"reason": "r", "score": 9}', "doesn't match the schema"),      # out of range
    ('{"score": 3}', "doesn't match the schema"),                     # missing field
])
def test_parse_json_rejects(text, needle):
    with pytest.raises(LLMOutputError, match=needle):
        _parse_json(text, SCHEMA, "role")


# ======================================================================================
# Offline: ChatLLM construction
# ======================================================================================

def test_ollama_role(cfg):
    llm = ChatLLM("qa_generator", cfg, SECRETS)
    assert llm.base_url == SECRETS.ollama_base_url
    assert (llm.model, llm.seed, llm.concurrency) == (cfg.llm.qa_generator.model, cfg.seed, cfg.ollama.concurrency)
    assert llm.context_length == cfg.ollama.context_length
    assert llm.reasoning_effort == "none"


def test_openai_role(cfg):
    llm = ChatLLM("judge", cfg, SECRETS)
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.concurrency == 1                    # sequential by default: 429 risk
    assert llm.context_length is None              # no truncation check for OpenAI
    assert llm.reasoning_effort == cfg.llm.judge.reasoning_effort


def test_api_key_not_stored_on_object(cfg):
    llm = ChatLLM("judge", cfg, SECRETS)
    assert not any("sk-test-not-real" in repr(v) for k, v in vars(llm).items() if k != "client")
    assert llm.client.api_key == "sk-test-not-real"    # the real value reached the client


@pytest.mark.parametrize("role, secrets, needle", [
    ("qa_generator", Secrets(openai_api_key="k"), "OLLAMA_BASE_URL"),
    ("judge", Secrets(ollama_base_url="u"), "OPENAI_API_KEY"),
])
def test_missing_secret_raises(cfg, role, secrets, needle):
    with pytest.raises(ValueError, match=needle):
        ChatLLM(role, cfg, secrets)


def test_unknown_role_raises(cfg):
    with pytest.raises(AttributeError):
        ChatLLM("generator", cfg, SECRETS)


# ======================================================================================
# Offline: chat / chat_many with a fake OpenAI client
# ======================================================================================

def fake_response(content="ok", finish="stop", prompt_tokens=10, completion_tokens=5, reasoning=None):
    message = SimpleNamespace(content=content, model_extra={"reasoning": reasoning} if reasoning else {})
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=message)],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                           model="fake-model")


class FakeClient:
    """Stands in for `openai.OpenAI`: records every request, answers with `self.respond(params)`."""

    def __init__(self):
        self.calls: list[dict] = []
        self.respond = lambda params: fake_response()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        result = self.respond(params)
        if isinstance(result, Exception):
            raise result
        return result


def with_fake(llm: ChatLLM) -> FakeClient:
    llm.client = FakeClient()
    return llm.client


MSGS = [{"role": "user", "content": "hi"}]


def test_chat_sends_role_settings(cfg):
    llm = ChatLLM("qa_critic", cfg, SECRETS); fake = with_fake(llm)
    result = llm.chat(MSGS)
    sent = fake.calls[0]
    assert sent["model"] == cfg.llm.qa_critic.model and sent["messages"] == MSGS
    assert sent["temperature"] == cfg.llm.qa_critic.temperature and sent["seed"] == cfg.seed
    assert sent["max_tokens"] == cfg.llm.qa_critic.max_tokens
    assert sent["reasoning_effort"] == "low"
    assert "response_format" not in sent
    assert result == LLMResult(text="ok", parsed=None, prompt_tokens=10, completion_tokens=5, model="fake-model")


def test_chat_omits_unset_optional_params(cfg):
    """Non-reasoning OpenAI models reject `reasoning_effort`; max_tokens=None means no limit."""
    cfg.llm.judge.reasoning_effort = None          # as for a non-reasoning judge
    llm = ChatLLM("judge", cfg, SECRETS); fake = with_fake(llm)
    llm.chat(MSGS)
    assert "reasoning_effort" not in fake.calls[0] and "max_tokens" not in fake.calls[0]


def test_chat_overrides_win(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    llm.chat(MSGS, temperature=0.9, max_tokens=7)
    assert fake.calls[0]["temperature"] == 0.9 and fake.calls[0]["max_tokens"] == 7


def test_chat_json_schema(cfg):
    llm = ChatLLM("qa_critic", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response('{"reason": "clear", "score": 5}')
    result = llm.chat(MSGS, json_schema=SCHEMA)
    assert fake.calls[0]["response_format"]["json_schema"]["schema"] == SCHEMA
    assert result.parsed == {"reason": "clear", "score": 5}


def test_chat_json_invalid_raises(cfg):
    llm = ChatLLM("qa_critic", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("**Rating:** 5/5")      # what Qwen does with thinking off
    with pytest.raises(LLMOutputError, match="not valid JSON"):
        llm.chat(MSGS, json_schema=SCHEMA)


def test_chat_context_overflow_raises(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("", finish="length", prompt_tokens=8191, completion_tokens=1)
    with pytest.raises(LLMOutputError, match="context window full"):   # not the "cut off" message
        llm.chat(MSGS)


@pytest.mark.parametrize("prompt_tokens, completion_tokens", [
    (4098, 8),          # Ollama 0.40.6 cuts an over-long prompt to half the window, reply ends normally
    (4096, 0),          # exactly half: counts as truncated, since the two can't be told apart
    (3500, 4692),       # a prompt that fits, but the reply fills the window
])
def test_chat_truncated_prompt_raises(cfg, prompt_tokens, completion_tokens):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("ok", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    with pytest.raises(LLMOutputError, match="context window full"):
        llm.chat(MSGS)


def test_chat_prompt_under_half_the_window_passes(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("ok", prompt_tokens=cfg.ollama.context_length // 2 - 1, completion_tokens=500)
    assert llm.chat(MSGS).text == "ok"


def test_no_overflow_check_for_openai(cfg):
    llm = ChatLLM("judge", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("fine", prompt_tokens=50_000)
    assert llm.chat(MSGS).text == "fine"


def test_chat_cut_off_raises(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("A long answer that st", finish="length")
    with pytest.raises(LLMOutputError, match="cut off") as e:
        llm.chat(MSGS)
    assert "reasoning" not in str(e.value)


def test_chat_cut_off_by_reasoning_gives_hint(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("", finish="length", reasoning="Let me think... " * 50)
    with pytest.raises(LLMOutputError, match="reasoning_effort"):
        llm.chat(MSGS)


def test_chat_think_block_in_content_raises(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: fake_response("<think>hmm</think>Paris")
    with pytest.raises(LLMOutputError, match="think"):
        llm.chat(MSGS)


def test_chat_many_order_errors_and_callback(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)

    def respond(params):
        i = int(params["messages"][0]["content"])
        time.sleep(0.01 * (5 - i))                  # later items finish first
        if i == 2:
            return fake_response("cut", finish="length")
        if i == 4:
            return RuntimeError("API down")         # stands in for an SDK error after retries
        return fake_response(f"answer {i}")
    fake.respond = respond

    seen = []
    results = llm.chat_many([[{"role": "user", "content": str(i)}] for i in range(5)],
                            on_result=lambda i, r: seen.append((i, threading.current_thread())))
    assert [r.text for r in results if isinstance(r, LLMResult)] == ["answer 0", "answer 1", "answer 3"]
    assert isinstance(results[2], LLMOutputError) and isinstance(results[4], RuntimeError)
    assert sorted(i for i, _ in seen) == [0, 1, 2, 3, 4]
    assert all(t is threading.main_thread() for _, t in seen)     # safe to append to a file there


def test_chat_many_respects_concurrency(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    active, peak, lock = 0, 0, threading.Lock()

    def respond(params):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return fake_response()
    fake.respond = respond
    llm.chat_many([MSGS] * 10)
    assert peak == cfg.ollama.concurrency


def test_chat_many_interrupt_cancels_queued(cfg):
    llm = ChatLLM("rag_generator", cfg, SECRETS); fake = with_fake(llm)
    fake.respond = lambda p: (time.sleep(0.02), fake_response())[1]

    def stop(i, r):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        llm.chat_many([MSGS] * 50, on_result=stop)
    time.sleep(0.1)                                 # let already-running requests finish
    assert len(fake.calls) <= 2 * cfg.ollama.concurrency      # the queued ones never ran


# ======================================================================================
# Live: Ollama
# ======================================================================================

def _ollama_available(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        return requests.get(f"{_native_url(base_url)}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def _loaded(base_url: str) -> list[str]:
    return [m["name"] for m in requests.get(f"{_native_url(base_url)}/api/ps", timeout=5).json()["models"]]


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """Real config + secrets + one client per local role; unloads every model afterwards."""
    from chunkeval.core.config import load_config
    from conftest import DEFAULT_CONFIG

    secrets = load_secrets()
    if not _ollama_available(secrets.ollama_base_url):
        pytest.skip("Ollama server not reachable (OLLAMA_BASE_URL)")
    cfg = load_config(DEFAULT_CONFIG)
    cfg.paths.runs_dir = tmp_path_factory.mktemp("runs")
    clients = {role: ChatLLM(role, cfg, secrets)
               for role, llm_cfg in cfg.llm if llm_cfg.provider == "ollama"}
    yield SimpleNamespace(cfg=cfg, secrets=secrets, clients=clients)
    for llm in clients.values():                    # leave the GPU free, even after a failure
        unload_ollama_model(llm.base_url, llm.model)


@pytest.mark.ollama
def test_live_check_ollama(live):
    check_ollama(live.cfg, live.secrets.ollama_base_url)
    report = {r["model"]: r for r in store.read_jsonl(store.run_dir(live.cfg) / "ollama_models.jsonl")}
    for role, llm in live.clients.items():
        r = report[_ollama_tag(llm.model)]
        assert r["quantization_level"] and len(r["digest"]) == 64
        assert r["context_length"] >= live.cfg.ollama.context_length
        # A dense generator must never be split with the CPU. For Gemma that takes the derived tag of
        # ollama/gemma4-allgpu.Modelfile: the official one keeps a layer on the CPU (99.5%).
        if role != "qa_critic":
            assert r["gpu_share"] == 1.0, f"{llm.model} only {r['gpu_share']:.1%} on GPU"
    assert _loaded(live.secrets.ollama_base_url) == []          # each model unloaded after its check


@pytest.mark.ollama
@pytest.mark.parametrize("role", ["qa_generator", "rag_generator", "qa_critic"])
def test_live_json_answer(live, role):
    msgs = [{"role": "system", "content": "Rate the question's clarity. Reply with ONLY a JSON object, "
                                          'no markdown: {"reason": "<one sentence>", "score": <integer 1-5>}'},
            {"role": "user", "content": "What was the average maize yield in Kenya in 2019, in t/ha?"}]
    result = live.clients[role].chat(msgs, json_schema=SCHEMA)
    assert 1 <= result.parsed["score"] <= 5
    assert result.prompt_tokens > 0 and result.completion_tokens > 0


@pytest.mark.ollama
def test_live_thinking_is_off_for_qwen(live):
    """Pins pitfall 2: without reasoning_effort="none", Qwen3.5 spends max_tokens on thinking."""
    gen = live.clients["rag_generator"]
    result = gen.chat([{"role": "user", "content": "What is 17*23? Answer with just the number."}])
    assert result.text.strip() == "391" and result.completion_tokens < 20
    with pytest.raises(LLMOutputError, match="reasoning"):
        gen.chat([{"role": "user", "content": "What is 17*23?"}], reasoning_effort=None, max_tokens=100)


@pytest.mark.ollama
def test_live_deterministic(live):
    gen = live.clients["rag_generator"]
    msgs = [{"role": "user", "content": "Define chunk overlap in one sentence."}]
    assert gen.chat(msgs).text == gen.chat(msgs).text


@pytest.mark.ollama
def test_live_context_overflow_detected(live):
    """Pins pitfall 1: Ollama truncates an over-long prompt silently; `chat` must notice."""
    with pytest.raises(LLMOutputError, match="context window full"):
        live.clients["rag_generator"].chat([{"role": "user", "content": "word " * 12000}])


@pytest.mark.ollama
def test_live_cut_off_detected(live):
    with pytest.raises(LLMOutputError, match="cut off"):
        live.clients["rag_generator"].chat([{"role": "user", "content": "Write 400 words on RAG."}], max_tokens=16)


@pytest.mark.ollama
def test_live_resume_after_interrupt(live, tmp_path):
    """End to end, the way the LLM stages use it: chat_many + on_result + append_jsonl + done_ids."""
    gen = live.clients["rag_generator"]
    out = tmp_path / "answers.jsonl"
    items = [{"id": f"q{i}", "q": f"Name one crop grown in African country number {i}, one word."} for i in range(8)]
    sent: list[str] = []

    def run_stage(stop_after: int | None = None) -> None:
        done = store.done_ids(out)
        todo = [it for it in items if it["id"] not in done]
        sent.extend(it["id"] for it in todo)

        def save(i: int, result) -> None:
            if isinstance(result, LLMResult):
                store.append_jsonl(out, {"id": todo[i]["id"], "answer": result.text})
                if stop_after and len(store.done_ids(out)) >= stop_after:
                    raise KeyboardInterrupt         # simulated Ctrl+C
        gen.chat_many([[{"role": "user", "content": it["q"]}] for it in todo], on_result=save)

    with pytest.raises(KeyboardInterrupt):
        run_stage(stop_after=3)
    finished = store.done_ids(out)
    assert len(finished) >= 3

    sent.clear()
    run_stage()
    assert set(sent) == {it["id"] for it in items} - finished     # finished items not re-sent
    assert sorted(store.done_ids(out)) == sorted(it["id"] for it in items)
    assert len(list(store.read_jsonl(out))) == len(items)         # each stored exactly once


@pytest.mark.ollama
def test_live_unload(live):
    gen = live.clients["rag_generator"]
    gen.chat(MSGS)                                   # make sure it is loaded
    assert _ollama_tag(gen.model) in _loaded(gen.base_url)
    unload_ollama_model(gen.base_url, gen.model)
    assert _ollama_tag(gen.model) not in _loaded(gen.base_url)
    unload_ollama_model(gen.base_url, gen.model)     # not loaded: quick no-op


# ======================================================================================
# Live: OpenAI (free metadata endpoint only)
# ======================================================================================

@pytest.mark.openai
def test_live_openai_judge_snapshot_exists(cfg):
    """Key is valid and the judge model hasn't been retired — check before an eval run."""
    secrets = load_secrets()
    if secrets.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY not set")
    llm = ChatLLM("judge", cfg, secrets)
    assert llm.client.models.retrieve(llm.model).id == llm.model
