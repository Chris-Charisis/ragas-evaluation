"""One OpenAI-compatible chat client for every LLM role (local Ollama or OpenAI).

Why: Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1. Using the `openai` SDK
with a different `base_url` means generator, critic, RAG generator and judge share one code path;
switching a role between Ollama and the OpenAI API is a config change (cfg.llm.<role>.provider).

Typical use in a local stage:
    check_ollama(cfg, secrets.ollama_base_url)             # preflight: fail in seconds
    llm = ChatLLM("qa_generator", cfg, secrets)
    llm.chat_many(batch, json_schema=..., on_result=save)  # save = append_jsonl per item
    unload_ollama_model(llm.base_url, llm.model)           # free the GPU for the next stage
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable
import json
import logging
import time

import jsonschema
from openai import OpenAI
import requests

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config
from chunkeval.core.store import run_dir, write_jsonl

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """One chat answer plus its token usage (feeds the cost estimate and context-token metric)."""
    text: str
    parsed: dict[str, Any] | None      # JSON output when a schema was requested
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLMOutputError(RuntimeError):
    """The model answered, but the answer can't be used (cut off, context overflow, bad JSON).

    Separate from API errors so callers can count and log these per item instead of
    stopping the whole run.
    """


def _parse_json(text: str, schema: dict[str, Any], role: str) -> dict[str, Any]:
    """Parse a structured answer and validate it against the schema that was requested."""
    # When the schema isn't enforced (Qwen3.5 with thinking off) the JSON may come wrapped
    # in a markdown code fence.
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"[{role}] answer is not valid JSON ({e}): {text[:200]!r}") from e
    # Valid JSON can still miss a required field or have a wrong type, e.g. "score": "5".
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as e:
        raise LLMOutputError(f"[{role}] JSON doesn't match the schema: {e.message}") from e
    return parsed


def _native_url(base_url: str) -> str:
    """OpenAI-compatible URL -> native API URL: "http://localhost:11434/v1" -> "http://localhost:11434"."""
    return base_url.rstrip("/").removesuffix("/v1")


def _ollama_tag(model: str) -> str:
    """Ollama lists an untagged name with ":latest" ("llama3.2" -> "llama3.2:latest")."""
    return model if ":" in model else f"{model}:latest"


def _loaded_models(native: str) -> dict[str, dict[str, Any]]:
    """Models currently in memory (`GET /api/ps`), by name."""
    r = requests.get(f"{native}/api/ps", timeout=10)
    r.raise_for_status()
    return {m["name"]: m for m in r.json()["models"]}


def _unload_and_wait(native: str, model: str, timeout_s: float = 60) -> None:
    """Unload `model` and wait until it is really gone.

    The `keep_alive: 0` request returns immediately while the model is still listed in /api/ps
    (with `expires_at` = now), so measuring the next model right away could see both loaded.
    """
    requests.post(f"{native}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30).raise_for_status()
    deadline = time.monotonic() + timeout_s
    while model in _loaded_models(native):
        if time.monotonic() > deadline:
            raise RuntimeError(f"Ollama did not unload {model} within {timeout_s:.0f}s")
        time.sleep(0.5)


def check_ollama(cfg: Config, base_url: str) -> None:
    """Preflight check before a long local run — fail in seconds, not after 2 hours.

    For every model used by an "ollama" role:
      1. it is pulled (`GET /api/tags`);
      2. loaded with the server defaults, its context length is >= cfg.ollama.context_length
         (pitfall 1 in `chat`: a too-small window truncates prompts silently);
      3. its GPU share (size_vram / size) is reported. Below 100% is expected for gpt-oss:20b
         and logged as a warning, not an error: a MoE generator alternative also offloads by design.
    Then writes model, quantization, size and digest of each model to
    runs/<run_id>/ollama_models.jsonl — they are part of the experiment definition.

    Each model is loaded and unloaded in turn, so it is measured alone, the way a stage runs it.
    All problems are collected and raised together in one RuntimeError.
    """
    # `base_url` is the OpenAI-compatible endpoint (.../v1); the checks need the native API.
    native = _native_url(base_url)

    # 1. One request answers both "is the server up?" and "which models are pulled?".
    try:
        r = requests.get(f"{native}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama is not reachable at {native} (systemctl status ollama)") from e
    pulled = {m["name"]: m for m in r.json()["models"]}

    # Several roles can share one model (qa_generator and rag_generator): check each model once.
    roles_by_model: dict[str, list[str]] = {}
    for role, llm_cfg in cfg.llm:          # iterating a pydantic model yields (field name, value)
        if llm_cfg.provider == "ollama":
            roles_by_model.setdefault(_ollama_tag(llm_cfg.model), []).append(role)

    missing = [m for m in roles_by_model if m not in pulled]
    if missing:
        raise RuntimeError("Ollama models not pulled: " + ", ".join(missing)
                           + "\nRun: " + " && ".join(f"ollama pull {m}" for m in missing))

    problems: list[str] = []
    report: list[dict[str, Any]] = []
    for model, roles in roles_by_model.items():
        # 2. Load it with an empty request and NO options: passing num_ctx here would hide a
        # wrong server default, because /v1 requests always get the server default.
        # Generous read timeout: the first load reads the weights from disk.
        r = requests.post(f"{native}/api/generate", json={"model": model}, timeout=(5, 600))
        if not r.ok:
            problems.append(f"{model}: failed to load ({r.status_code}: {r.text.strip()})")
            continue
        loaded = _loaded_models(native).get(model)
        if loaded is None:
            problems.append(f"{model}: loaded, but missing from /api/ps")
            continue

        ctx = loaded.get("context_length")
        if ctx is None:
            log.warning("%s: /api/ps reports no context_length; can't verify the window", model)
        elif ctx < cfg.ollama.context_length:
            problems.append(f"{model}: context length {ctx} < {cfg.ollama.context_length} — set "
                            f"OLLAMA_CONTEXT_LENGTH={cfg.ollama.context_length} (sudo systemctl edit ollama)")

        # 3. A model that doesn't fit in VRAM is split with the CPU and runs several times slower.
        gpu_share = loaded["size_vram"] / loaded["size"]
        if gpu_share < 1:
            # One decimal: gemma4:12b-it-q8_0's one layer on the CPU is 99.5%, which "%.0f" shows as 100%.
            log.warning("%s (%s): only %.1f%% on GPU, the rest runs on the CPU",
                        model, ", ".join(roles), 100 * gpu_share)
        else:
            log.info("%s (%s): 100%% on GPU, context %s", model, ", ".join(roles), ctx)

        details = pulled[model]["details"]
        report.append({
            "model": model,
            "roles": roles,
            "quantization_level": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "family": details.get("family"),
            # A tag can be re-pushed with new weights; the digest pins the exact file used.
            "digest": pulled[model]["digest"],
            "size_bytes": loaded["size"],
            "gpu_share": round(gpu_share, 3),
            "context_length": ctx,
        })
        _unload_and_wait(native, model)

    if problems:
        raise RuntimeError("Ollama preflight failed:\n  - " + "\n  - ".join(problems))
    write_jsonl(run_dir(cfg) / "ollama_models.jsonl", report)


class ChatLLM:
    """Thin wrapper around `openai.OpenAI(base_url=..., api_key=...)` for one role."""

    def __init__(self, role: str, cfg: Config, secrets: Secrets) -> None:
        """Build the client for one role of the `llm:` config section (e.g. "qa_generator").

        The role's `provider` picks the endpoint:
          - "ollama" -> secrets.ollama_base_url, with a dummy api_key (Ollama ignores it, but the
            SDK refuses to start without one);
          - "openai" -> api.openai.com with secrets.openai_api_key.
        Model, temperature, max_tokens, seed and reasoning effort are stored for `chat`, the
        default concurrency for `chat_many`.
        """
        # getattr(cfg.llm, "judge") == cfg.llm.judge; `cfg.llm` is a pydantic model, not a dict.
        # AttributeError here means the role name is misspelled.
        llm_cfg = getattr(cfg.llm, role)
        self.role = role
        self.provider = llm_cfg.provider

        # The checks below repeat `require_secrets` on purpose: without them a missing
        # OLLAMA_BASE_URL makes the SDK fall back to api.openai.com, and a local model name
        # fails there with a confusing 401/404 instead of "OLLAMA_BASE_URL is not set".
        if self.provider == "ollama":
            if not secrets.ollama_base_url:
                raise ValueError(f"role '{role}' uses Ollama, but OLLAMA_BASE_URL is not set in .env")
            self.base_url = secrets.ollama_base_url
            api_key = "ollama"
            # The seed of the config: makes local runs repeatable at temperature 0.
            self.seed = cfg.seed
        else:  # "openai" — the only other value pydantic's Literal allows
            if secrets.openai_api_key is None:
                raise ValueError(f"role '{role}' uses OpenAI, but OPENAI_API_KEY is not set in .env")
            # Explicit, so an OPENAI_BASE_URL in the environment can't redirect the judge.
            self.base_url = "https://api.openai.com/v1"
            # SecretStr: the real value is read only here, where the client is built.
            # The key is not kept on `self`, so printing the object never shows it.
            api_key = secrets.openai_api_key.get_secret_value()
            # OpenAI treats the seed as best effort (same seed -> mostly the same output).
            self.seed = cfg.seed

        self.model = llm_cfg.model
        self.temperature = llm_cfg.temperature
        self.max_tokens = llm_cfg.max_tokens       # None = no explicit limit
        self.reasoning_effort = llm_cfg.reasoning_effort   # None = don't send the parameter
        # Only Ollama has a context window we configure; `chat` uses it to detect truncation.
        self.context_length = cfg.ollama.context_length if self.provider == "ollama" else None
        # Default for `chat_many`. Ollama: must match OLLAMA_NUM_PARALLEL. OpenAI: sequential,
        # because every extra parallel request raises the chance of a 429 and the safe level
        # depends on the account's rate-limit tier — pass `concurrency=` explicitly to go faster.
        # (The judge doesn't come through here: Ragas calls it, with `ragas.max_workers`.)
        self.concurrency = cfg.ollama.concurrency if self.provider == "ollama" else 1

        # timeout: generous, because the first request to Ollama also loads the model into VRAM
        # and queued requests wait for a free slot. max_retries: the SDK retries connection
        # errors, timeouts, 429 and 5xx with exponential backoff — `chat_many` relies on that.
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=300, max_retries=3)

    def chat(self, messages: list[dict[str, str]], json_schema: dict | None = None,
             **overrides: Any) -> LLMResult:
        """Send one chat completion.

        `json_schema`: request structured output and return it parsed and validated in
        `LLMResult.parsed`. `overrides`: any `chat.completions.create` parameter for this call
        only (e.g. `temperature=0.3`); they win over the role's config.

        Raises `LLMOutputError` when the model answered but the answer can't be used (cut off,
        context overflow, invalid JSON). API errors (timeouts, 5xx) come from the `openai` SDK
        after its own retries.

        Behaviour measured on Ollama 0.30.4 (qwen3.5:9b-q8_0, gpt-oss:20b), re-checked on 0.40.6
        (2026-10-19, also gemma4:12b-it-q8_0) — re-check after upgrades:
          1. CONTEXT TRUNCATION IS SILENT. 0.30.4: a 12k-token prompt came back with prompt_tokens =
             8191 (the window) and no error; the start of the prompt (system message) was cut.
             0.40.6: any prompt over the window is cut to HALF of it (prompt_tokens 4098 for 8.5k or
             12k tokens; the log says "truncating input prompt", the reply nothing), keeping only
             its first 5 tokens, so the system message is lost again. Native /api/chat refuses
             instead with `truncate: false`, but /v1 ignores that option. Detected below from
             `usage`: a prompt of half the window or more counts as truncated.
          2. THINKING: by default Qwen3.5 thinks first and spent all 400 max_tokens on it — empty
             content, finish_reason "length". On /v1 only `reasoning_effort="none"` turns it off
             (`think: false` in extra_body and Qwen's `/no_think` are ignored). Thinking goes to a
             separate `reasoning` field, never into `content`, and counts in completion_tokens.
          3. STRUCTURED OUTPUT: with thinking OFF, Qwen3.5 IGNORES the schema (on /v1 and on native
             /api/chat) and may answer in markdown. So the prompt itself must spell out the JSON
             format; the schema is still sent (enforced for OpenAI and for thinking models) and
             the answer is always validated against it here.
          4. DETERMINISM: temperature 0 + fixed seed gave identical outputs across repeated calls.
        """
        params: dict[str, Any] = {
            "model": self.model,              # Ollama tag ("qwen3.5:9b-q8_0"), not a Hugging Face id
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        # Optional parameters are sent only when set: None would be rejected or mean something else.
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None:
            # "name" is required by the API; it only labels the schema.
            params["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "schema": json_schema}}
        params.update(overrides)

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        text = choice.message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # Pitfall 1. Checked before "length" below, because an overflow also ends with "length"
        # and this is the more useful message. A truncated prompt reports half the window (0.40.6)
        # or all of it (0.30.4), so half the window is the usable prompt budget. Largest prompts on
        # the dev docs: 2,166 tokens (answer), 1,410 (critic), ~1,500 (gen-qa); at most 5 chunks of
        # 512 tokens + instructions, ~3,100.
        if self.context_length and (prompt_tokens >= self.context_length // 2
                                    or prompt_tokens + completion_tokens >= self.context_length):
            raise LLMOutputError(
                f"[{self.role}] context window full ({prompt_tokens} prompt + {completion_tokens} "
                f"output tokens; window {self.context_length}, prompts must stay under half of it): "
                f"the prompt was probably truncated and its start (the system message) lost. "
                f"Shorten it or raise OLLAMA_CONTEXT_LENGTH and ollama.context_length.")

        # Pitfall 2: the answer was cut at max_tokens. Unusable for JSON, and a cut-off RAG answer
        # would be scored as if it were complete. Same request at temperature 0 = same result,
        # so this is raised rather than retried.
        if choice.finish_reason == "length":
            reasoning = (choice.message.model_extra or {}).get("reasoning")
            hint = " — most of it was reasoning; check reasoning_effort" if reasoning and len(text) < 50 else ""
            raise LLMOutputError(f"[{self.role}] answer cut off at max_tokens="
                                 f"{params.get('max_tokens')} ({completion_tokens} tokens){hint}")

        # Ollama puts thinking in a separate field; a <think> block in the content would mean
        # a template or version change that breaks that, so stop instead of storing it.
        if "<think>" in text:
            raise LLMOutputError(f"[{self.role}] reasoning leaked into the content (<think> block)")

        parsed = _parse_json(text, json_schema, self.role) if json_schema is not None else None
        return LLMResult(text=text, parsed=parsed, prompt_tokens=prompt_tokens,
                         completion_tokens=completion_tokens, model=response.model)

    def chat_many(self, batch: list[list[dict[str, str]]], json_schema: dict | None = None,
                  concurrency: int | None = None,
                  on_result: Callable[[int, LLMResult | Exception], None] | None = None,
                  ) -> list[LLMResult | Exception]:
        """Run many `chat` requests concurrently; return one result per request, in input order.

        A failed request returns its exception in its slot instead of losing the batch (the old
        code did `except: continue` and silently dropped 64 items at a time). Check with
        `isinstance(r, Exception)`.

        `concurrency`: defaults to `self.concurrency` — for Ollama `ollama.concurrency`, which
        must equal OLLAMA_NUM_PARALLEL: Ollama runs that many requests at once per model and
        queues the rest, so more client threads add no throughput.

        `on_result(index, result)`: called as soon as each request finishes, in completion order,
        always from the calling thread (so it's safe to append to a file there). Use it with
        `store.append_jsonl` so a crash keeps every finished item, not just finished batches.
        """
        concurrency = concurrency or self.concurrency
        results: list[LLMResult | Exception | None] = [None] * len(batch)
        start = time.monotonic()

        pool = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = {pool.submit(self.chat, messages, json_schema): i for i, messages in enumerate(batch)}
            # as_completed yields each future when it finishes, so on_result runs right away
            # instead of waiting for the slowest item of the batch.
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result: LLMResult | Exception = future.result()
                except Exception as e:     # LLMOutputError, or an API error after the SDK's retries
                    log.warning("[%s] request %d failed: %s", self.role, i, e)
                    result = e
                results[i] = result
                if on_result is not None:
                    on_result(i, result)
        except BaseException:
            # Ctrl+C or an error in on_result: drop the queued requests. A plain `with` block
            # would wait for ALL of them to run first, which can mean hours on a large batch.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()

        # Throughput is this project's bottleneck with Ollama: log it so the pilot can
        # estimate the full run (tokens/s × total expected tokens). Only successful requests
        # count — a failed one carries no usage — so many failures understate the real rate.
        elapsed = time.monotonic() - start
        ok = [r for r in results if isinstance(r, LLMResult)]
        out_tokens = sum(r.completion_tokens for r in ok)
        log.info("[%s] %d/%d ok in %.0fs — %d output tokens, %.1f tokens/s, concurrency %d",
                 self.role, len(ok), len(batch), elapsed, out_tokens,
                 out_tokens / elapsed if elapsed else 0.0, concurrency)
        return results  # type: ignore[return-value]   # every slot is filled by now


def ollama_model_digest(base_url: str, model: str) -> str:
    """Digest of a pulled model (`GET /api/tags`), for the provenance of what a model made: a tag
    can be re-pushed with new weights, the digest pins the exact file.

    `base_url` is the OpenAI-compatible URL. Raises `requests.RequestException` when the server is
    unreachable and RuntimeError when the model isn't pulled.
    """
    r = requests.get(f"{_native_url(base_url)}/api/tags", timeout=10)
    r.raise_for_status()
    pulled = {m["name"]: m["digest"] for m in r.json()["models"]}
    tag = _ollama_tag(model)
    if tag not in pulled:
        raise RuntimeError(f"Ollama model {tag} is not pulled (ollama pull {tag})")
    return pulled[tag]


def loaded_ollama_models(base_url: str) -> list[str]:
    """Models Ollama holds in memory right now — what `ollama ps` lists.

    `base_url` is the OpenAI-compatible URL. Raises `requests.RequestException` when the server
    is unreachable, so callers decide whether that matters (`embedding.check_gpu_free` doesn't mind).
    """
    return list(_loaded_models(_native_url(base_url)))


def unload_ollama_model(base_url: str, model: str) -> None:
    """Free the GPU after a local LLM stage, before embedding or switching models.

    Why: Ollama keeps a model in VRAM for OLLAMA_KEEP_ALIVE (5 min by default). Embedding
    (bge-m3 + Qwen3-Embedding) and the next LLM need that memory now, not in 5 minutes — and if two
    LLMs end up loaded together, Ollama pushes layers to the CPU instead of failing.

    `base_url` is the OpenAI-compatible URL (`secrets.ollama_base_url` or `ChatLLM.base_url`).
    Waits until the model is really gone; a model that isn't loaded is a quick no-op.
    """
    _unload_and_wait(_native_url(base_url), _ollama_tag(model))
    log.info("unloaded %s", model)
