# chunkeval — chunking evaluation v2

This package evaluates chunking strategies for the RAG system on two corpora:
- **papers**: structured scientific papers
- **reports**: free-format documents

The package is a **scaffold**. Each module has docstrings, signatures and `TODO(you)` hints, and every body is left for you to implement. The previous pipeline, [ragas_evaluation.py](ragas_evaluation.py), stays as a reference. The previous results are in `ragas_results_analysis_new.docx`.

> **Security:** `ragas_evaluation.py` contains a hard-coded HF token and OpenAI key (ggshield flagged both). Rotate them, put the new ones in `.env` (copy `.env.example`), and never commit `.env`.

## Stages

| # | CLI command | Module | Input → output |
|---|---|---|---|
| 1 | `parse` | `corpus/parsing.py` | PDFs → `data/parsed/<corpus>/<doc_id>.json` (CanonicalDoc) |
| 2 | `chunk` | `corpus/chunking/*` | CanonicalDoc → `data/chunks/<corpus>/<method>_<ovl>.jsonl` + size stats |
| 2b | `embed` | `corpus/embedding.py` | chunks → cached vectors `data/embeddings/<embedder>/…` (bge-m3 + Qwen3, GPU) |
| 3 | `index` | `corpus/indexing.py` | chunks + both vectors → Weaviate `{corpus}_{method}_{ovl}_v2` (named vectors) |
| 4 | `gen-qa` | `qa/generation.py` | CanonicalDoc → `data/qa/raw.jsonl` (with evidence char spans) |
| 5 | `filter-qa` | `qa/filtering.py` | raw → `data/qa/filtered.jsonl` (+ `rejected.jsonl` with reasons) |
| 6 | `retrieve` | `evaluation/retrieval.py` + `evaluation/retrieval_metrics.py` | questions × 12 collections × 2 embedders → `runs/<id>/retrieval/<embedder>/*.jsonl` + span metrics |
| 7 | `answer` | `evaluation/answering.py` | retrieved contexts → RAG responses |
| 8 | `eval` | `evaluation/ragas_eval.py` | samples → per-question Ragas scores |
| 9 | `report` | `analysis.py` | all per-question scores → tables with bootstrap CIs and paired diffs |

Every stage reads `configs/default.yaml` and writes its own artefacts, so you can re-run any stage without repeating the earlier ones.

### Running a stage

```bash
uv run chunkeval --help                                # list the stages
uv run chunkeval parse --help                          # the options of one stage
uv run chunkeval parse --corpus papers --limit 2       # smoke test: 2 PDFs of one corpus
uv run chunkeval parse                                 # every corpus in the config
uv run chunkeval gen-qa --config configs/other.yaml -v # another config, debug logging
```

`uv run python -m chunkeval.cli …` does the same without the console script. Both work from any directory, because `uv sync` installs the package (see [Package layout](#package-layout)).

**Options:**
- **On every stage:** `--config` (default `configs/default.yaml`, relative to the repo root) and `-v` for debug logging.
- **`--corpus NAME`** (repeatable; default: every corpus) **and `--limit N`** (the first N documents per corpus, sorted by name, so a smoke test always picks the same ones). For now these exist on `parse`, `chunk`, `index` and `gen-qa`.

**What the CLI does around every stage,** so stage code doesn't have to:
1. It validates the config and checks this stage's secrets, so a problem stops the run in the first second.
2. It saves the resolved config to `runs/<run_id>/config.yaml`, the record of what the run used.
3. It logs to the console and to `runs/<run_id>/logs/<stage>_<timestamp>.log`.
4. **For stages with local LLM roles** (`gen-qa`, `filter-qa`, `answer`): it runs `check_ollama` first, then unloads the models afterwards, even if the stage fails.
5. It calls the stage's `run(cfg, secrets, **options)` and logs how long it took. A failure is logged with its traceback and exits with code 1; Ctrl+C exits with 130.

**Adding a stage option:**
1. Register it in `cli.OPTIONS`.
2. List it in the stage's entry in `cli.STAGES`.
3. Add a keyword argument with the same name to that stage's `run`.

`tests/test_cli.py` fails if these three steps don't match.

## Package layout

Module paths in this README are relative to `src/chunkeval/`. The folders follow the pipeline order:

```
src/chunkeval/
├── cli.py              entry point, one subcommand per stage (`uv run chunkeval …`)
├── core/               shared by every stage, no pipeline logic
│   ├── schema.py       pydantic models of configs/*.yaml (`Config`) and REPO_ROOT
│   ├── config.py       load_config / save_config, secrets from .env, collection_name
│   ├── store.py        JSONL / Parquet persistence, resumable appends, run folders
│   └── llm.py          one chat client for every LLM role (Ollama or OpenAI)
├── corpus/             stages 1–3: PDFs → indexed chunks
│   ├── parsing.py
│   ├── chunking/       base.py, character.py, sentence.py, recursive.py
│   ├── embedding.py
│   └── indexing.py
├── qa/                 stages 4–5: the ground-truth questions
│   ├── generation.py
│   └── filtering.py
├── evaluation/         stages 6–8
│   ├── retrieval.py
│   ├── retrieval_metrics.py
│   ├── answering.py
│   └── ragas_eval.py
└── analysis.py         stage 9: report tables
```

**Conventions:**
- **Always import with the full path,** for example `from chunkeval.core.store import write_jsonl` or `from chunkeval.qa.generation import QAPair`. Never use a bare `import config`, which only works by accident from inside the folder.
- **The package is installed in editable mode** by `uv sync`, through `[build-system]` in `pyproject.toml`. That's why no `PYTHONPATH=src` is needed anywhere, and why code changes take effect without reinstalling. After a fresh clone, or if `import chunkeval` fails, run `uv sync` once.
- **Relative paths** (configs, `data/`, `runs/`, `.env`) resolve against `REPO_ROOT`, the folder that holds `pyproject.toml`. It is searched upward from `core/schema.py`, so moving modules between folders can't break it.
- **`core/` is imported by every stage.** After changing anything there, run the whole test suite.
- **A new module** goes into the folder of its pipeline phase. A new subfolder needs an `__init__.py`.

## Key decisions (summary of the plan)

1. **Ground truth that doesn't depend on any chunker.** Q-A pairs are generated from paragraph groups of the Docling structure, not from one of the chunkers under test. Each answer must quote its evidence word for word. The quote is located in the canonical text and becomes a **character span**. This allows deterministic retrieval metrics, and any pair whose quote can't be found is dropped, which filters out hallucinated pairs by construction.
2. **One canonical text per document.** Every chunker and the Q-A generator work on the same string, so all character offsets are comparable.
3. **Chunk sizes are measured in bge-m3 tokens**, targeting 256 with a cap of 512. The cap is an experimental choice, not a model limit, since bge-m3 reads up to 8192 tokens. The grid is 3 methods (character / sentence / recursive-chapter) × overlap on/off × 2 corpora, which gives 12 collections.
4. **Two embedders, stored as named vectors in the same 12 collections:**
   - **Primary: `BAAI/bge-m3`.** Used for everything: span metrics, RAG answers, Ragas. It needs no query instruction.
   - **Secondary: `Qwen/Qwen3-Embedding-0.6B`.** Only the free span metrics are computed with it. Its job is to check whether the chunker ranking holds under a different embedder, and whether the embedder or the chunker is the bigger lever. Its queries need an `Instruct: …\nQuery:` prompt (set in the config).
   - Both models produce 1024-dim vectors, so a mix-up never raises an error. Always pass vectors together with their embedder name (see `embedding.py`).
5. **Correct Ragas inputs:**
   - `user_input` = question
   - `retrieved_contexts` = top-k chunks
   - `response` = RAG answer
   - `reference` = gold answer

   The old script used the gold answer as `response` and the source chunk as `reference`.
6. **Hybrid models:**
   - a local Qwen model on Ollama (default `qwen3.5:9b-q8_0`) generates the Q-A pairs and the RAG answers
   - local `gpt-oss:20b` on Ollama is the critic
   - a pinned OpenAI model is the judge, plus one bridge run with gpt-4o for comparison with the previous round
7. **Metrics:**
   - span-based retrieval metrics: hit@k, span recall@k, MRR, IoU and precision, context tokens
   - the 4 Ragas metrics plus factual correctness
   - cheap generation metrics
   - per-question storage, bootstrap CIs, paired comparisons, and a budget-matched view

## Suggested implementation order
1. `core/`: `schema.py`, `config.py`, `store.py`, `llm.py`, and the `cli.py` skeleton (done). Add each stage's own CLI options when you write that stage. Configure Ollama (see the section below), pull the models, and send one test request.
2. `corpus/parsing.py`, then the chunkers in `corpus/chunking/`, then `tests/test_chunking.py`.
3. `corpus/embedding.py` and `tests/test_embedding.py`. Embed with both models while no Ollama model is loaded (`ollama ps` should be empty), because they share the 16 GB GPU.
4. Start Weaviate with `docker compose up -d` (see the section below), then write `corpus/indexing.py`. Check that object counts match chunk counts and that both named vectors are present.
5. `qa/generation.py`, then `qa/filtering.py`. Hand-label 50 pairs.
6. `evaluation/retrieval.py` and `evaluation/retrieval_metrics.py` on all 12 configs × 2 embedders. This step has **no API cost** and is the first decision point.
7. `evaluation/answering.py` (primary embedder only). Run a Ragas pilot on 20 questions to check cost, then the full run, then the gpt-4o bridge run.
8. `analysis.py`, including the embedder robustness table.

## Testing

The tests import `chunkeval` as the installed package, so run `uv sync` once (see [Package layout](#package-layout)). pytest is in the dev dependency group, which `uv sync` installs too.

```bash
uv run pytest                                   # everything (~25 s with Ollama running)
uv run pytest -m "not ollama and not openai"    # offline tests only (~1 s): run after every change
uv run pytest -m ollama                         # live Ollama tests only: run after upgrading Ollama or changing models
uv run pytest -m openai                         # OpenAI key + judge snapshots: run before an eval run
uv run pytest tests/test_store.py -v            # one file, one line per test
uv run pytest -k crash                          # tests whose name contains "crash"
```

`.venv/bin/python -m pytest …` works the same way, without uv.

**What is covered:**

| File | Tests | Needs |
|---|---|---|
| [tests/test_config.py](tests/test_config.py) | Config loading (from any working directory) and every validation rule: unknown keys, types, ranges, `TODO` placeholders, cross-section references. Also the `save_config` round trip, secrets (masking, per-stage requirements), and collection names. | nothing |
| [tests/test_store.py](tests/test_store.py) | Atomic `write_jsonl`, crash-tolerant `append_jsonl` / `done_ids` (the resume mechanism), corruption detection, `write_table` (parquet, duplicate keys, CSV fallback) and `run_dir`. | nothing |
| [tests/test_llm.py](tests/test_llm.py), offline | `ChatLLM` setup per provider. `chat` and `chat_many` logic against a fake client: the parameters sent, every `LLMOutputError` case, result order, errors returned in place, concurrency limit, and Ctrl+C cancellation. | nothing |
| [tests/test_llm.py](tests/test_llm.py), marker `ollama` | Preflight (`check_ollama`), JSON answers from every local role, determinism, resume after an interrupted run, and unloading. Also pins the Ollama behaviour the code relies on: `reasoning_effort="none"` turns Qwen's thinking off, and too-long prompts are truncated silently. | Ollama running, models pulled, `OLLAMA_BASE_URL` |
| [tests/test_cli.py](tests/test_cli.py) | Every stage in `cli.STAGES` points to a `run` that accepts the options the CLI passes. Argument parsing, and what `main` does around a stage: config saved, log file written, exit codes 1 and 130, missing secrets stop the run before any work, and the Ollama check before and unload after an LLM stage, even when it fails. | nothing |
| [tests/test_llm.py](tests/test_llm.py), marker `openai` | The API key is valid and the pinned judge snapshots still exist. It only calls a free metadata endpoint, so there's **no cost**. | `OPENAI_API_KEY` |
| `test_chunking.py`, `test_embedding.py`, … | Placeholders (`pytest.skip("TODO(you)")`) for the modules that aren't written yet. They show as `s`. | — |

**Good to know:**
- **Live tests skip themselves** when their service isn't available. If Ollama isn't running, `uv run pytest` still passes, and the live tests show as skipped instead of failing.
- **Tests never touch real data.** The `cfg` fixture points `runs_dir` to a temporary folder, so nothing is written to `runs/` or `data/`.
- **The `ollama` tests load the models onto the GPU** (about 20 s) and unload them at the end, even when a test fails. Don't run them while embedding is using the GPU.
- **A generator must be 100% on the GPU.** `test_live_check_ollama` fails if `qa_generator` or `rag_generator` is partly offloaded to the CPU. If you switch to one of the MoE generators, which offload by design, relax that assertion.
- **Writing new tests:**
  - Use the fixtures from `conftest.py`: `cfg` (a validated config) and `raw_config` (the YAML as a dict, for building invalid configs).
  - Mark tests that need a service: `@pytest.mark.ollama`, `@pytest.mark.openai`, or `@pytest.mark.gpu` for embedding models.
  - The markers are registered in `pyproject.toml`.

## Local LLMs (Ollama)

Ollama runs as a system service here (version 0.30.4). The code talks to its OpenAI-compatible endpoint, `http://localhost:11434/v1`, through the same `openai` client it uses for the judge (see `llm.py`). A containerised alternative is in `docker-compose.yml`, off by default; see [Moving Ollama to Docker](#moving-ollama-to-docker-optional) below.

**One-time setup (system service):**
```bash
sudo systemctl edit ollama          # add the [Service] lines below, then:
sudo systemctl restart ollama
ollama pull qwen3.5:9b-q8_0         # Q-A generator + RAG generator (default, see configs/default.yaml)
ollama pull gpt-oss:20b             # critic
```
```ini
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=8192"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

**Why these settings:**
- **Context length.** Ollama's default context window is small. When a prompt is too long, it is cut from the start *without an error*, so the system prompt quietly disappears. 8192 tokens leaves room for every prompt in this project.
- **Parallel slots.** Each slot reserves its own context memory, so two slots is the safe number for `gpt-oss:20b` on this GPU. Set `ollama.concurrency` in the config to the same value; extra client threads only queue.
- **Flash attention + q8_0 KV cache.** Together they roughly halve the context memory at negligible quality cost. That memory is what lets the models fit.

**Quantization.** Every Ollama tag is a quantized GGUF.
- **Budget:** about 14.4 GB of GPU memory is free; the display uses about 1.6 GB.
- **What fits, and why the defaults were chosen:**
  - **Generator default: `qwen3.5:9b-q8_0`** (10 GB). Q8 is effectively lossless, and the model fits fully on the GPU with room for context.
  - **Alternatives for the generator** (all listed with sizes in `configs/default.yaml`):
    - `qwen3:14b`: a larger model, but older and at 4-bit
    - the mixture-of-experts models `qwen3:30b-a3b-instruct-2507-q4_K_M` and `qwen3.5:35b-a3b-int4`: they don't fit, but they run acceptably with part of the model on the CPU, because only about 3B parameters are active per token
  - **Critic: `gpt-oss:20b`** (14 GB). It exists only in its native MXFP4 quantization, so a few layers run on the CPU. Being mixture-of-experts, it tolerates this.
  - **Not viable:** `qwen3:14b-q8_0` (16 GB) and `qwen3.5:27b` (17 GB, dense). A dense model that doesn't fit is very slow.
- **Choose the generator with a pilot on about 50 generation units.** Compare three things:
  - the share of evidence quotes that can be located
  - the share of pairs the critic accepts
  - tokens per second

**Running notes:**
- **One model on the GPU at a time.** Before each `embed` run, and between generator and critic stages, unload the loaded model with `ollama stop <model>`. Otherwise Ollama keeps it for 5 minutes, and the next model or the embedders get pushed onto the CPU.
- **Check placement with `ollama ps`.** It should show `100% GPU` for the generator.
- **Turn thinking off for generation.** Qwen and gpt-oss are reasoning models, so set this in `llm.py`. Alternatively use an `-instruct-2507` tag, which never thinks.
- **Embeddings stay in Python** (sentence-transformers, fp16), not in Ollama. That gives full control over query prompts and pooling, and keeps the vectors free of extra quantization.

### Moving Ollama to Docker (optional)

The system service is the default. Its advantages: it's already installed, it uses the GPU without an extra layer, and with a single server two Ollamas can't compete for GPU memory.

Switch to the container if any of these apply:
- you want the whole stack portable to another machine with `docker compose`
- another project needs different Ollama settings at the same time
- you want to pin this project's Ollama version on its own

The `ollama` service in `docker-compose.yml` uses Compose profile `ollama` and:
- **Image:** pinned to `ollama/ollama:0.30.4`, the same version as the system install
- **GPU:** reserved through the NVIDIA runtime, which Docker here already has
- **Settings:** the same server settings as the systemd override, plus `OLLAMA_MAX_LOADED_MODELS=1`
- **Models:** stored in the named volume `chunkeval_ollama_models`

> **Rule:** never run both servers with models loaded. On one 16 GB GPU they push each other's layers to the CPU without any error.

**0. Check GPU access from containers** (one time):
```bash
docker run --rm --gpus all ubuntu nvidia-smi         # should list the RTX 5070 Ti
```

**1. Record the current state**, so you can check the move afterwards:
```bash
ollama --version
ollama list                                           # note the ID column: model digests
```

**2. Get the models into the container volume.** Pick one:
- **Re-pull only this project's models.** This is cleaner, because the volume holds only what chunkeval uses:
  ```bash
  ollama ps                                           # nothing should be loaded on the system server;
                                                      # if something is: ollama stop <model>
  echo "OLLAMA_HOST_PORT=11435" >> .env               # temporary port, so it can run next to the system service
  docker compose --profile ollama up -d ollama
  docker compose exec ollama ollama pull qwen3.5:9b-q8_0
  docker compose exec ollama ollama pull gpt-oss:20b
  ```
- **Or copy the models you already have** (no download; this copies *all* system models, including other projects'):
  ```bash
  docker volume create chunkeval_ollama_models
  docker run --rm -v /usr/share/ollama/.ollama/models:/src:ro -v chunkeval_ollama_models:/dst \
      alpine sh -c "mkdir -p /dst/models && cp -a /src/. /dst/models/"
  ```

**3. Optional side-by-side test** on port 11435, while the system service is still installed but has nothing loaded:
```bash
curl -s localhost:11435/api/version
docker compose exec ollama ollama list                # same IDs as step 1 → identical models
OLLAMA_BASE_URL=http://localhost:11435/v1 uv run chunkeval gen-qa --limit 2   # smoke test (once --limit exists in cli.py)
docker compose exec ollama ollama ps                  # should show 100% GPU for the generator
```

**4. Switch over:**
```bash
docker compose --profile ollama down                  # stop the test container
sed -i '/^OLLAMA_HOST_PORT=/d' .env                   # back to the default port 11434
sudo systemctl disable --now ollama                   # stop the system service and keep it from starting at boot
docker compose --profile ollama up -d                 # Weaviate + Ollama on 11434
curl -s localhost:11434/api/version                   # answered by the container now
```
Nothing changes in the code or in `.env`, because `OLLAMA_BASE_URL` still points to `localhost:11434`. The `ollama` command-line tool on the host keeps working as a client: `ollama ps`, `ollama stop` and `ollama list` now talk to the container. Don't run `ollama serve` on the host, though; that would start a second server.

**5. Where the settings live now.** Server settings are in the `environment:` block of `docker-compose.yml`; the systemd override no longer has any effect. Keep `OLLAMA_CONTEXT_LENGTH` and `OLLAMA_NUM_PARALLEL` equal to `ollama.context_length` and `ollama.concurrency` in `configs/default.yaml`, because `llm.check_ollama` checks them.

**6. Free disk space** (optional, and only once the container has worked for a while): the system models stay in `/usr/share/ollama/.ollama/models`. Other projects may still use them, so delete them deliberately (`ollama rm <model>` while the system service runs), not with `rm -rf`.

**Rollback:**
```bash
docker compose --profile ollama down                  # the model volume is kept
sudo systemctl enable --now ollama                    # system service back on port 11434
```

## Weaviate (Docker)

[docker-compose.yml](docker-compose.yml) runs one local Weaviate node for the 12 v2 collections.

```bash
# 1. once: put a NEW random key in .env (e.g. `openssl rand -hex 32`) as WEAVIATE_API_KEY=...
docker compose up -d                                   # start
curl -s http://localhost:8080/v1/.well-known/ready -o /dev/null -w "%{http_code}\n"   # 200 = ready
docker compose logs -f weaviate                        # start-up / errors
docker compose down                                    # stop, keep data
docker compose down -v                                 # stop and DELETE all collections
# careful: `docker compose --profile ollama down -v` would ALSO delete the downloaded Ollama models
```

**Why it's set up this way:**
- **The version is pinned** to `1.39.8`, so a server update can't change retrieval behaviour in the middle of an experiment. weaviate-client 4.22 requires server 1.27 or newer. If you upgrade, do it on purpose and re-run the smoke test.
- **Both ports are exposed:** 8080 for REST and 50051 for gRPC. The v4 Python client uses gRPC for queries and batch imports, so a missing 50051 mapping shows up as timeouts. Both ports are bound to `127.0.0.1` only.
- **No vectorizer modules** (`DEFAULT_VECTORIZER_MODULE=none`). The vectors are computed in Python (`embedding.py`) and sent with each object. The container needs no GPU and downloads no models, so the GPU stays free for Ollama and embedding.
- **Named vectors.** Each object carries two vectors, `bge_m3` and `qwen3_06b`, in the same collection. Every query must say which one to search with `target_vector=...` (see `retrieval.py`). Both vectors have to be present when the collection is created, so run `embed` with both models before `index`.
- **API-key authentication with RBAC.** The key comes from `WEAVIATE_API_KEY` in `.env` and maps to the root user `chunkeval`. `docker compose` refuses to start if the key is missing. In Python, connect with `weaviate.connect_to_local(..., auth_credentials=Auth.api_key(key))`.
- **A dedicated named volume**, `chunkeval_weaviate_data`. Don't reuse Weaviate volumes from other projects.
- **The database is disposable.** Chunks and cached embeddings live in `data/`, so `chunkeval index` rebuilds every collection. You don't need Weaviate backups. Use `down -v` to start clean.
- **Memory:** `GOMEMLIMIT=6GiB` is a soft cap. 12 collections with two 1024-dim vectors per chunk, from a few hundred documents, need a few GB at most. Raise it if you run the chunk-size sweep.

**Collection names:** `{corpus}_{method}_{overlap}_v2` (see `configs/default.yaml`). Weaviate capitalises the first letter, so `papers_sentence_overlap_v2` is stored as `Papers_sentence_overlap_v2`. `config.collection_name` should normalise names so lookups match.

**Problems:**

| Symptom | Fix |
|---|---|
| Connection timeouts or gRPC errors | Check that port 50051 is mapped |
| Query error about a missing or unknown vector | `target_vector` must be `bge_m3` or `qwen3_06b`, matching the names in the config |
| 401 errors | The key in `.env` doesn't match the one the container started with. Restart the container after changing `.env`. |
| `port is already allocated` | Another Weaviate is running. Check with `docker ps`. |

## Dependencies
- **Installed with uv:** all the pipeline libraries, plus pytest in the dev group, plus `chunkeval` itself in editable mode.
  - **On a fresh clone,** run `uv sync`.
  - **On the existing `.venv`,** use `uv sync --inexact`. It also holds packages that aren't in the lockfile (CUDA 12 libraries left over from the torch upgrade), and a plain `uv sync` would uninstall them.
- **ragas** is still 0.2.15. Upgrading changes its API, so read the migration notes first.
- **Ollama** serves the local LLMs. It's a system service, not a Python package; the code reaches it with the `openai` client.

## Future work

**Models partly on the CPU.** At the moment, `check_ollama` only logs a warning when a model isn't 100% on the GPU. `test_live_check_ollama` hard-codes that both generators must be 100%. There are two situations to handle:
- **Deliberate:** a MoE generator (only about 3B parameters active per token) is partly on the CPU by design and stays usable.
- **Accidental:** a dense model is pushed partly onto the CPU and gets several times slower. Typical causes: another model still loaded, embedding models running on the GPU, or a higher `OLLAMA_NUM_PARALLEL` or context length. Each parallel slot reserves its own KV cache.

Planned handling:
1. Add a per-role `min_gpu_share` to `LLMModel`, defaulting to `1.0`. Lower it explicitly in the YAML when a MoE model is chosen, so the decision is recorded in the config.
2. Make `check_ollama` raise an error below that threshold, with a diagnosis: other loaded models, and the KV-cache size from context length × parallel slots.
3. Make `check_ollama` refuse to measure while unrelated models are loaded. It should name them and suggest `ollama stop <model>`, not unload them silently.
4. Make the live test read the threshold from the config instead of hard-coding the generator roles.

If a dense model is offloaded by accident, fix it in this order:
1. Free the GPU (check with `ollama ps` and `nvidia-smi`).
2. Lower `OLLAMA_NUM_PARALLEL`.
3. Use a smaller quantization.
4. Lower the context length, but only if the longest prompts still fit. A window that's too small truncates prompts silently.

Offload that starts in the middle of a run shows up as a drop in the tokens/s that `chat_many` logs for each batch.
