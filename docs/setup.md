# Setup: Ollama and Weaviate

The two local services the pipeline needs: Ollama for the three local LLMs, Weaviate for the 12
collections. To set up a whole new machine step by step, follow [rerun.md](rerun.md); this file
explains the settings.

## Local LLMs (Ollama)

Ollama runs as a system service here (version 0.40.6, upgraded from 0.30.4 on 2026-10-19 because `gemma4` needs 0.30.5 or newer; the dev run used 0.30.4). The code talks to its OpenAI-compatible endpoint, `http://localhost:11434/v1`, through the same `openai` client it uses for the judge (see `llm.py`). A containerised alternative is in `docker-compose.yml`, off by default; see [Moving Ollama to Docker](#moving-ollama-to-docker-optional) below.

**One-time setup (system service):**
```bash
sudo systemctl edit ollama          # add the [Service] lines below, then:
sudo systemctl restart ollama
ollama pull gemma4:12b-it-q8_0      # Q-A generator, then the all-GPU tag the config uses:
ollama create gemma4-allgpu:12b-it-q8_0 -f ollama/gemma4-allgpu.Modelfile
ollama pull qwen3.5:9b-q8_0         # RAG generator
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
- **Context length.** Ollama's default context window is small. When a prompt is too long, it is cut *without an error*, so the system prompt quietly disappears. 0.30.4 cut it from the start down to the window; 0.40.6 cuts it to half the window, keeping only its first 5 tokens and the end. `chat` therefore treats any prompt of half the window or more as truncated and raises (`LLMOutputError`). 8192 tokens gives a 4,096-token prompt budget, almost twice the largest prompt on the dev docs (2,166 tokens, an answer from 5 chunks).
- **Parallel slots.** Each slot reserves its own context memory, so two slots is the safe number for `gpt-oss:20b` on this GPU. Set `ollama.concurrency` in the config to the same value; extra client threads only queue.
- **Flash attention + q8_0 KV cache.** Together they roughly halve the context memory at negligible quality cost. That memory is what lets the models fit.

**Quantization.** Every Ollama tag is a quantized GGUF.
- **Budget:** about 14.4 GB of GPU memory is free; the display uses about 1.6 GB.
- **What fits, and why the defaults were chosen:**
  - **Q-A generator: `gemma4:12b-it-q8_0`** (13 GB, dense; chosen 2026-10-19), run as the derived tag `gemma4-allgpu:12b-it-q8_0`. Q8 is effectively lossless.
    - **Why the derived tag:** with the official tag, Ollama's memory fit puts 1 of the 48 layers on the CPU (99.5% on GPU; `offloaded 48/49 layers` in `journalctl -u ollama`). The model fits: fully loaded it would leave 1,104 MiB of VRAM free, but the fit wants a 1,206 MiB margin, so it moves one 219 MiB layer.
    - **The fix:** [ollama/gemma4-allgpu.Modelfile](../ollama/gemma4-allgpu.Modelfile) adds `num_gpu 49` to the official tag. Same weight files, template, renderer and sampling defaults; only Gemma is affected, unlike lowering the margin server-wide (`LLAMA_ARG_FIT_TARGET`). All on the GPU it leaves ~1.2 GB free (15.1 of 16.3 GB used) and decodes 16% faster: 55 vs 47 tokens/s for one request.
    - **Build it once,** after pulling the official tag: `ollama create gemma4-allgpu:12b-it-q8_0 -f ollama/gemma4-allgpu.Modelfile`. If something else holds more GPU memory when it loads, the load fails in `check_ollama`'s preflight instead of spilling onto the CPU.
  - **RAG generator: `qwen3.5:9b-q8_0`** (10 GB), also the Q-A generator of the dev run. Q8 is effectively lossless, and the model fits fully on the GPU with room for context.
  - **Alternatives for the generators** (all listed with sizes in `configs/default.yaml`):
    - `qwen3:14b`: a larger model, but older and at 4-bit
    - the mixture-of-experts models `qwen3:30b-a3b-instruct-2507-q4_K_M` and `qwen3.5:35b-a3b-int4`: they don't fit, but they run acceptably with part of the model on the CPU, because only about 3B parameters are active per token
  - **Critic: `gpt-oss:20b`** (14 GB). It exists only in its native MXFP4 quantization. On 0.30.4 a few layers ran on the CPU, which a mixture-of-experts model tolerates; on 0.40.6 it fits entirely on the GPU.
  - **Not viable:** `qwen3:14b-q8_0` (16 GB) and `qwen3.5:27b` (17 GB, dense). A dense model that doesn't fit is very slow.
- **How the Q-A generator was chosen** (Gemma 4 over Qwen, 2026-10-19): a pilot on the same ~100 dev units, comparing the share of evidence quotes that can be located, the share of pairs the critic accepts, and tokens per second ([qa.md](qa.md#q-a-generation-gen-qa)). Repeat it this way before switching generators.

**Running notes:**
- **One model on the GPU at a time.** Ollama keeps a model loaded for 5 minutes after its last request, and the next model or the embedders would then be pushed onto the CPU. So the CLI unloads a stage's models when the stage ends, even when it fails, and `embed` and `retrieve` refuse to start while Ollama holds a model. After a test stopped with Ctrl+C or a manual `ollama run`, unload it yourself with `ollama stop <model>`. `embed` checks this itself: before loading each embedder it asks Ollama what is loaded and stops with the `ollama stop` command to run. It never unloads a model itself, because the model may belong to another project.
- **Check placement with `ollama ps`.** It should show `100% GPU` for the generator. It rounds: the official Gemma tag's one layer on the CPU shows as `0%/100% CPU/GPU`, so read `check_ollama`'s log line (one decimal) or `ollama_models.jsonl` (`gpu_share`).
- **Thinking is off for generation.** Qwen, Gemma 4 and gpt-oss are reasoning models. Each role's `reasoning_effort` in the config is sent with every request (`llm.py`): `"none"` for both generators, `"low"` for the critic, whose `max_tokens` leaves room for it. The judge, `gpt-6-luna`, is a reasoning model too: with `"none"` it accepts temperature 0 and the seed, and thinks not at all. An `-instruct-2507` tag, which never thinks, would be the alternative.
- **Embeddings stay in Python** (sentence-transformers, fp16), not in Ollama. That gives full control over query prompts and pooling, and keeps the vectors free of extra quantization.

### Moving Ollama to Docker (optional)

The system service is the default. Its advantages: it's already installed, it uses the GPU without an extra layer, and with a single server two Ollamas can't compete for GPU memory.

Switch to the container if any of these apply:
- you want the whole stack portable to another machine with `docker compose`
- another project needs different Ollama settings at the same time
- you want to pin this project's Ollama version on its own

The `ollama` service in `docker-compose.yml` uses Compose profile `ollama` and:
- **Image:** pinned to `ollama/ollama:0.40.6`, the same version as the system install
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
  docker compose exec ollama ollama pull gemma4:12b-it-q8_0
  docker compose cp ollama/gemma4-allgpu.Modelfile ollama:/tmp/gemma4-allgpu.Modelfile
  docker compose exec ollama ollama create gemma4-allgpu:12b-it-q8_0 -f /tmp/gemma4-allgpu.Modelfile
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
OLLAMA_BASE_URL=http://localhost:11435/v1 uv run pytest -m ollama   # every role answers, generators 100% on the GPU;
                                                      # writes nothing to data/ or runs/
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

[docker-compose.yml](../docker-compose.yml) runs one local Weaviate node for the 12 v2 collections.

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
- **Memory:** `GOMEMLIMIT=6GiB` is a soft cap. The full corpus's 12 collections hold 84,447 objects with two 1024-dim vectors each, about 0.7 GB of vectors. Raise it if you run the chunk-size sweep.

**Collection names:** `{namespace}{corpus}_{method}_{overlap}_v2` (see `configs/default.yaml`). The namespace is empty for `data/`, and for another data dir its name (`data/size512` gives `size512_papers_sentence_overlap_v2`), so two data dirs never share a collection. Weaviate capitalises the first letter, so `papers_sentence_overlap_v2` is stored as `Papers_sentence_overlap_v2`. `config.collection_name` returns names in that form, so lookups match.

**Problems:**

| Symptom | Fix |
|---|---|
| Connection timeouts or gRPC errors | Check that port 50051 is mapped |
| Query error about a missing or unknown vector | `target_vector` must be `bge_m3` or `qwen3_06b`, matching the names in the config |
| 401 errors | The key in `.env` doesn't match the one the container started with. Restart the container after changing `.env`. |
| `port is already allocated` | Another Weaviate is running. Check with `docker ps`. |
