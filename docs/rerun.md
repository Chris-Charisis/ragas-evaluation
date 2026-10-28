# Clean rerun on another machine

How to set up chunkeval v2 from scratch on a new Linux machine and run the whole pipeline, either on the
same corpus (150 papers + 150 reports) or on a new one. Every command is meant to be run as written, from
the repo root unless the block says otherwise. The "why" behind each setting is in the other docs (start from
[README.md](../README.md#documentation)) and in the comments of `configs/default.yaml`; this file is
only the procedure.

Blocks marked **[source]** run on the machine that has the current corpus; everything else runs on the
**new machine**. The paths assume the repo is cloned to `~/ragas-evaluation`.

**Time and cost.** About 7 hours of unattended local compute up to the paid stage (most of it Q-A
generation, critic and RAG answers), then the Ragas judge: roughly 10 hours and ~$34 at the current sample
size (the dry run's estimate on the full corpus), with a hard cap in the config. Only `eval` costs money; nothing before it calls a paid API.

## Contents

1. [Requirements](#1-requirements)
2. [Source machine: publish the code](#2-source-machine-publish-the-code)
3. [New machine: system setup](#3-new-machine-system-setup)
4. [New machine: Ollama and the models](#4-new-machine-ollama-and-the-models)
5. [New machine: the project](#5-new-machine-the-project)
6. [The corpus](#6-the-corpus)
7. [Settings for the run](#7-settings-for-the-run)
8. [Run the pipeline](#8-run-the-pipeline)
9. [Where the results are](#9-where-the-results-are)
10. [Troubleshooting](#10-troubleshooting)
11. [Reference: the source machine](#11-reference-the-source-machine)

## 1. Requirements

| What | Needed | Why |
|---|---|---|
| GPU | NVIDIA with **16 GB** VRAM, nothing else using more than ~1.5 GB | The Q-A generator (Gemma 4 12B, Q8) and the critic (gpt-oss 20B) must sit 100% on the GPU. A smaller GPU needs other models: see [Troubleshooting](#10-troubleshooting). |
| NVIDIA driver | **580 or newer** (CUDA 13.0) | torch 2.14.1 is the cu130 build |
| OS | Ubuntu 24.04 (any systemd Linux works; the commands use apt) | |
| RAM | 32 GB recommended (the source machine has 60 GB) | Docling while parsing; Weaviate is capped at 6 GiB |
| Disk | ~60 GB free | Ollama models 35 GB, embedding and Docling models ~6 GB, `.venv` ~7 GB, corpus and artefacts ~2 GB |
| Internet | yes, on the first run | models are downloaded on first use |
| Accounts | GitHub access to the repo; an **OpenAI API key** (only for `eval` and `pytest -m openai`); optionally a Hugging Face token | the models are public; a token only raises download rate limits |

## 2. Source machine: publish the code

`data/` and `runs/` are git-ignored, so the new machine gets the code from git and the corpus by copy
(step 6). First make sure everything the pipeline needs is pushed. **[source]**

```bash
cd ~/ragas-evaluation          # wherever the repo lives on the source machine
git status --short
```

Nothing may be listed under `src/`, `tests/`, `prompts/`, `configs/`, `ollama/`, `docs/`, `README.md`,
`docker-compose.yml`, `pyproject.toml` or `uv.lock`. If something is, commit and push it:

```bash
git add src tests prompts configs ollama docs README.md docker-compose.yml pyproject.toml uv.lock
git status --short             # check: no .env, no ragas_evaluation.py (it holds hard-coded keys)
git commit -m "Pipeline state for the rerun"
git push origin main
```

## 3. New machine: system setup

**3.1 Packages** (git, the word lists the `audit` stage needs, rsync for the corpus copy):

```bash
sudo apt update
sudo apt install -y git curl openssl rsync wamerican wbritish
ls /usr/share/dict/american-english /usr/share/dict/british-english
```

**3.2 NVIDIA driver.** Check first:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

The driver must be 580 or newer. If `nvidia-smi` is missing or older:

```bash
sudo ubuntu-drivers list          # shows the available versions
sudo apt install -y nvidia-driver-580
sudo reboot
```

**3.3 Docker** (for Weaviate):

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# log out and back in (or reboot) so the group applies, then:
docker run --rm hello-world
docker compose version
```

**3.4 uv** (installs Python 3.12 itself when the project needs it; the version used is pinned here):

```bash
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
source "$HOME/.local/bin/env"
uv --version                      # uv 0.7.19
```

## 4. New machine: Ollama and the models

**4.1 Install Ollama 0.40.6** (the version the code was tested with; Gemma 4 needs 0.30.5 or newer):

```bash
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.6 sh
ollama --version                  # ollama version is 0.40.6
```

**4.2 Server settings.** `check_ollama` refuses to run an LLM stage when these don't match
`configs/default.yaml` (`ollama.context_length`, `ollama.concurrency`):

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=8192"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
systemctl show ollama --property=Environment     # the four settings must be listed
```

**4.3 Pull the models** (~35 GB) and build the all-GPU Gemma tag. The Modelfile is in the repo, so clone
first (step 5.1) or run the `ollama create` line after it:

```bash
ollama pull gemma4:12b-it-q8_0      # Q-A generator (base of the derived tag)
ollama pull qwen3.5:9b-q8_0         # RAG generator
ollama pull gpt-oss:20b             # critic
ollama create gemma4-allgpu:12b-it-q8_0 -f ~/ragas-evaluation/ollama/gemma4-allgpu.Modelfile
ollama list
```

Compare the IDs with the source machine's. A different ID means the registry tag was updated since
2026-10-20, i.e. a different model: the run still works and records the new digest in its provenance,
but its numbers aren't strictly comparable with the source run's.

| Tag | ID on the source machine |
|---|---|
| `gemma4:12b-it-q8_0` | `41c402fdddc2` |
| `gemma4-allgpu:12b-it-q8_0` | `b3d5f4123b2d` (built locally from the line above) |
| `qwen3.5:9b-q8_0` | `a8ee95bc3592` |
| `gpt-oss:20b` | `17052f91a42e` |

`num_gpu 49` in the Modelfile is the layer count of Gemma 4 12B plus its output layer, not a property of
the source GPU: on any GPU that fits the model it puts every layer there.

## 5. New machine: the project

**5.1 Clone and install** (~7 GB of packages, including torch and its CUDA libraries):

```bash
git clone git@github.com:Chris-Charisis/ragas-evaluation.git ~/ragas-evaluation
cd ~/ragas-evaluation
uv sync
uv run chunkeval --help             # lists 11 stages, parse … report
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.14.1+cu130 True <your GPU>
```

**5.2 Secrets.** `.env` is git-ignored and never leaves the machine. Create it with a new Weaviate key
and type the OpenAI key without it landing in the shell history:

```bash
cp .env.example .env && chmod 600 .env
sed -i "s/^WEAVIATE_API_KEY=.*/WEAVIATE_API_KEY=$(openssl rand -hex 32)/" .env
read -rsp "OpenAI API key: " KEY && sed -i "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=$KEY|" .env && unset KEY; echo
grep -c '^OPENAI_API_KEY=sk-' .env    # 1
```

`OLLAMA_BASE_URL=http://localhost:11434/v1` is already set by the template. `HF_TOKEN` may stay empty.

**5.3 Weaviate** (version pinned to 1.39.8 in `docker-compose.yml`):

```bash
docker compose up -d
curl -s http://localhost:8080/v1/.well-known/ready -o /dev/null -w "%{http_code}\n"    # 200
```

**5.4 Tests**, in this order (the GPU tests must run while Ollama has no model loaded):

```bash
uv run pytest -m "not ollama and not openai and not gpu and not weaviate"   # ~45 s, offline
uv run pytest -m gpu         # first run downloads bge-m3 and Qwen3-Embedding (~5.5 GB)
uv run pytest -m weaviate    # touches only *_pytest collections
uv run pytest -m ollama      # loads each model once; checks 100% GPU for the generators
uv run pytest -m openai      # key and judge model; free (metadata endpoint only)
```

Everything must pass. A test that shows as skipped means its service is missing (Ollama not running, no
GPU, Weaviate down, no key): fix that before going on.

## 6. The corpus

Pick one case.

### 6A. The same corpus as the source machine

Copy the PDFs and `rename_map.csv` (666 MB). **[source]**, with your user and host:

```bash
rsync -a --mkpath --info=progress2 ~/ragas-evaluation/data/corpora/ USER@NEW_HOST:~/ragas-evaluation/data/corpora/
```

Then on the new machine, check every file against the checksums in the map (its lines end in `\r\n`,
hence the `tr`):

```bash
tail -n +2 data/corpora/rename_map.csv | tr -d '\r' \
  | awk -F, '{print $4"  data/corpora/"$1"/"$3}' | md5sum -c --quiet && echo "all PDFs match the map"
ls data/corpora/papers | wc -l; ls data/corpora/reports | wc -l      # 150 and 150
```

`configs/titles.csv` comes with the repo and fits this corpus: leave it as it is.

### 6B. A new corpus

The document id is the file name without `.pdf`, and it must be unique across the corpora (`parse` refuses a clash). Copy the
new PDFs in as `paper_<i>.pdf` / `report_<i>.pdf`, numbered in natural sort order, and record the
original names and checksums. Set `src` to the folder with the new PDFs and run once per corpus
(`corpus=papers prefix=paper`, then `corpus=reports prefix=report`):

```bash
mkdir -p data/corpora
[ -f data/corpora/rename_map.csv ] || echo "corpus,old_name,new_name,md5" > data/corpora/rename_map.csv

corpus=papers; prefix=paper; src=/path/to/new/papers
mkdir -p "data/corpora/$corpus"
i=0; find "$src" -maxdepth 1 -type f -iname '*.pdf' -print0 | sort -zV | while IFS= read -r -d '' f; do
  i=$((i + 1)); new="${prefix}_$i.pdf"
  cp --update=none "$f" "data/corpora/$corpus/$new"
  echo "$corpus,$(basename "$f"),$new,$(md5sum < "$f" | cut -d' ' -f1)"
done >> data/corpora/rename_map.csv
```

Original names containing a comma would break the CSV: rename those first. Verify with the checksum
command of 6A. Then empty the title overrides, which are keyed by document id and would retitle the new
files:

```bash
printf 'corpus,doc_id,title\n' > configs/titles.csv
```

The corpus names stay `papers` and `reports`. Other names would need the keys of `paths.corpora` and
`parsing.body` in `configs/default.yaml` renamed, and the first column of `configs/titles.csv`.

To add a corpus next to `papers` and `reports` instead of replacing them, on its own or in one run with
them, see [configs for other experiments](development.md#configs-for-other-experiments).

## 7. Settings for the run

Everything a stage writes outside `data/` goes to `runs/<run_id>/`, including the labelling sample. The
answer and judge caches are in `data/llm_cache/`, shared by every run: their keys hold everything that
decides a call, so a hit is always the same call. Give this run its own id, so its results are never
mixed with another machine's or another corpus's:

```bash
sed -i -E 's/^run_id: "[^"]*"/run_id: "full_data-v2.0"/' configs/default.yaml
grep -n '^run_id' configs/default.yaml
```

`qa.docs_per_corpus` must be at least the number of PDFs per corpus, or `gen-qa` uses a seeded sample of
the documents. It is 150, which fits the current corpus; for a new corpus:

```bash
N=$(ls data/corpora/reports/*.pdf | wc -l)   # use the larger of the two corpora
sed -i -E "s/^(  docs_per_corpus: )[0-9]+/\1$N/" configs/default.yaml
grep -n 'docs_per_corpus' configs/default.yaml
```

Leave everything else as it is: it was decided on purpose (see the comments next to each setting in the config).

Settle them before the first stage: from then on, a stage under this `run_id` refuses when a setting
differs from the config the run recorded, until you bump `run_id` or pass `--accept-config`
([the guard](development.md#the-guard-on-a-runs-config)).

## 8. Run the pipeline

Run the stages in this order. Each one logs to the console and to `runs/<run_id>/logs/`, and can be
re-run: it redoes only what is missing or out of date. Long stages are best run inside `tmux` or `screen`
so a closed terminal doesn't stop them.

The times are from the source machine (RTX 5070 Ti, 300 documents, 2026-10-20); "est." marks the one
not measured on the full corpus yet.

| Step | Stage | Time |
|---|---|---|
| 8.1 | `parse` | 32 min |
| 8.2 | `audit` | 7 s |
| 8.3 | `chunk` | 1 min |
| 8.4 | `embed` | 15 min |
| 8.5 | `index` | 40 s |
| 8.6 | `gen-qa` | 2 h |
| 8.7 | `filter-qa` | 1 h 25 min |
| 8.8 | `retrieve` | 1 min |
| 8.9 | `answer` | 2 h 25 min |
| 8.10 | `eval` (paid) | ~10 h (est.) |
| 8.11 | `report` | 15 s |

### 8.1 Parse

```bash
uv run chunkeval parse
```

The first run downloads Docling's layout and OCR models (~0.5 GB). Check the last lines of the log:
`papers: N parsed, …, 0 failed` and the same for reports, and that `data/parsed/*/_failures.jsonl` is
empty or absent. Warnings about OCR ("converting again with full-page OCR", "OCR doesn't fix it") are
normal: the parser decides per document and records it in `meta`.

### 8.2 Audit, until the corpus is final

```bash
uv run chunkeval audit
less runs/full_data-v2.0/audit.txt
```

The stage changes nothing; its warnings are what needs a decision:

| Warning | What to do |
|---|---|
| `not fully converted` | Open the PDF. Usually acceptable (on the source corpus: report_102, 24 pages kept). |
| `broken text layer?` (garbled text in most of a document) | Replace or remove the PDF. |
| `not in English` | Replace or remove the PDF: its questions would mix languages with the English prompts. |
| `byte-identical PDFs`, `near-duplicate pairs` | Remove one copy. A pair sharing only front matter can stay. |
| `suspicious titles without an override` | Add `corpus,doc_id,title` rows to `configs/titles.csv`, with the title on the PDF's first page. |

Also read the BODY SELECTION part of `audit.txt`: a document that lost most of its text to
`parsing.body` points to a missing start heading or a drop rule that went too far ([the full-corpus
audit](corpus.md#audit-audit)).

After any change, run both again. Only changed or retitled PDFs are re-parsed, and the JSON of a removed
PDF is deleted:

```bash
uv run chunkeval parse && uv run chunkeval audit
```

Repeat until only accepted warnings are left. **The corpus must be final before the next step:** adding
or removing a document later changes the top-k of every question, and everything from `chunk` on would
have to be redone.

### 8.3 Chunk

```bash
uv run chunkeval chunk
```

Check: the log shows 6 variants per corpus, `words` about 150–170 per chunk (~145 for the section chunker, `recursive` in the log, on reports),
and only a handful of `window … > max_tokens` warnings (9 on the source corpus). The numbers are in
`data/chunks/<corpus>/_size_stats.json`.

### 8.4 Embed

The embedders and an Ollama model don't fit on the GPU together, so nothing may be loaded:

```bash
ollama ps                          # must list no model; else: ollama stop <name>
uv run chunkeval embed
```

### 8.5 Index

```bash
uv run chunkeval index
```

Check: 12 lines `… objects with vectors ['bge_m3', 'qwen3_06b'] indexed and verified`. If the Weaviate
volume already holds collections from another corpus, the stage refuses ("exists, built from other
inputs"); then rebuild them all:

```bash
uv run chunkeval index --recreate
```

### 8.6 Q-A generation

A smoke test first (~1 min including the Ollama preflight); the full run resumes it:

```bash
uv run chunkeval gen-qa --corpus papers --limit 1
uv run chunkeval gen-qa
```

Check: the preflight line `gemma4-allgpu:12b-it-q8_0 (qa_generator): 100% on GPU, context 8192`, then
about 65–70 tokens/s per batch. The summary is in `runs/<run_id>/gen_qa_summary.json` (pair rate per
corpus, drops by reason). If the stage stops (Ctrl+C, crash, API error), run the same command again: it
asks only the units that have no record yet.

### 8.7 Q-A filtering and hand labelling

```bash
uv run chunkeval filter-qa
```

This writes the final question set (`data/qa/filtered.jsonl`), the rejections with their reasons, and a
blind sample of 50 pairs to label: `runs/<run_id>/filter_qa_labelling.csv`.

1. Open the CSV (LibreOffice, or any editor) and fill in `label`: 1 = good pair, 0 = bad. The
   [labelling guide](../Labelling%20guide%20Q-A%20pairs%20for%20the%20agricultural%20retrieval%20evaluation.md),
   if it was copied over, explains what counts as good.
2. Compare the labels with the filter's decisions ([hand labelling](qa.md#q-a-filtering-filter-qa):
   precision from the kept pairs, a rough recall from the rejected ones).
3. If the critic rejects too many good pairs, lower the threshold to the agreed fallback, then re-run.
   This asks the critic nothing again:

```bash
sed -i -E 's/^(  critic_min_score: )4/\13/' configs/default.yaml     # only if step 3 says so
uv run chunkeval filter-qa
```

Check the yield in `runs/<run_id>/filter_qa_summary.json`: kept questions per corpus against
`ragas.sample_per_corpus` (500).

### 8.8 Retrieval and span metrics (free)

```bash
ollama ps                          # must be empty (filter-qa unloads the critic itself)
uv run chunkeval retrieve
```

Summary in `runs/<run_id>/retrieval_summary.json`. If `retrieve` warns that a question fits all `k_max`
hits into the token budget, raise `retrieval.k_max` and re-run.

### 8.9 RAG answers (free)

```bash
uv run chunkeval answer
uv run chunkeval report            # a free, retrieval-only report before paying for the judge
```

Summary in `runs/<run_id>/answer_summary.json`. The abstention rate on `gold_passage` should be near 0;
`runs/<run_id>/reports/latest/report.md` lists the questions to read by hand.

### 8.10 Ragas evaluation (the only paid stage)

```bash
uv run pytest -m openai                      # key and judge model still valid (free)
uv run chunkeval eval --dry-run              # cost of every judge call not cached yet: no API call
```

Before going on: check that OpenAI hasn't announced a shutdown of the judge, `gpt-6-luna`
(<https://developers.openai.com/api/docs/deprecations>), and that the dry run's estimate is below
`ragas.max_cost_usd` (100). If it isn't, lower `ragas.sample_per_corpus` (the sample is nested, so a
smaller one is the start of a larger one) or raise the cap on purpose. Then:

```bash
uv run chunkeval eval --pilot 20             # 20 questions per corpus: the real cost per question
cat runs/full_data-v2.0/ragas/cost.json      # compare with the estimate
uv run chunkeval eval                        # the whole sample; reuses what the pilot paid for
```

The stage never pays for a call twice: every judge call is cached in `data/llm_cache/ragas_cache.jsonl`
as it returns, and an interrupted run continues where it stopped. It stops by itself when the spend
reaches the cap.

### 8.11 Report

```bash
uv run chunkeval report
```

## 9. Where the results are

| What | Where |
|---|---|
| The report | `runs/<run_id>/reports/latest/report.md`, with `cases.md`, `tables/*.csv`, `figures/*.png` |
| What the run used | `runs/<run_id>/config.yaml`, `reports/latest/run_record.json` |
| Per-question tables | `runs/<run_id>/metrics/retrieval_per_question.parquet`, `runs/<run_id>/ragas/per_question.parquet` |
| The question set | `data/qa/filtered.jsonl` (rejected: `data/qa/rejected.jsonl`) |
| The corpus audit | `runs/<run_id>/audit.txt`, `audit.json` |
| Logs | `runs/<run_id>/logs/<stage>_<timestamp>.log` |

To bring a run back to the source machine, copy `runs/<run_id>/` and `data/qa/` (both git-ignored):

```bash
rsync -a ~/ragas-evaluation/runs/full_data-v2.0 USER@SOURCE_HOST:~/ragas-evaluation/runs/
rsync -a ~/ragas-evaluation/data/qa/ USER@SOURCE_HOST:~/ragas-evaluation/data/qa_full_data-v2.0/
```

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Driver older than 580, or no reboot after installing it (3.2). |
| `docker: permission denied` | Not in the `docker` group yet: log out and back in (3.3). |
| `Ollama is not reachable` | `sudo systemctl status ollama`; `sudo systemctl restart ollama`. |
| `Ollama models not pulled: …` | Pull the tag, or build `gemma4-allgpu` (4.3). |
| Context length or parallel slots don't match | The override isn't active: re-run the 4.2 block, then `systemctl show ollama --property=Environment`. |
| A generator below 100% on GPU (`check_ollama`, `pytest -m ollama`) | Something else holds GPU memory: `ollama ps`, `nvidia-smi`, stop it. On a GPU under 16 GB the models don't fit: a smaller quantization or the MoE alternatives in `configs/default.yaml` change the experiment (new `run_id`; [models partly on the CPU](future-work.md#models-partly-on-the-cpu)). |
| `embed` or `retrieve` stops because Ollama holds a model | `ollama stop <name>`, then re-run. |
| `audit`: English word lists missing | `sudo apt install wamerican wbritish` (3.1). |
| Weaviate 401 | `.env`'s key changed after the container started: `docker compose up -d --force-recreate`. |
| Weaviate timeouts or gRPC errors | Port 50051 not mapped or the container down: `docker compose ps`. |
| `index`: `exists, built from other inputs` | Collections from another corpus or `--limit` run: `uv run chunkeval index --recreate`. |
| `gen-qa` refused, naming settings that differ | A setting, the prompt or a document changed since the corpus's records were written. Undo the change, or start that corpus over with `uv run chunkeval gen-qa --corpus <corpus> --recreate` (deletes its Q-A records only). |
| `run chunkeval retrieve first` (from `answer` or `report`) | The questions or chunks changed after `retrieve`: run the stage it names, then continue. |
| `eval` refuses to start | The estimate is above `ragas.max_cost_usd`: lower `ragas.sample_per_corpus` or raise the cap deliberately (8.10). |
| Throughput drops in the middle of an LLM stage | A model was pushed partly onto the CPU (another GPU process started): free the GPU; the stage resumes on re-run. |

## 11. Reference: the source machine

Versions the 2026-10-20 run used, for comparison:

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.5 LTS |
| GPU, driver | RTX 5070 Ti 16 GB, driver 580.178.04 (CUDA 13.0) |
| RAM, CPU | 60 GB, 32 threads |
| Docker, Compose, NVIDIA Container Toolkit | 29.9.1, v5.6.0, 1.20.1 (the toolkit only for the optional Ollama container) |
| uv, Python | 0.7.19, 3.12.3 |
| Ollama | 0.40.6 (system service) |
| Weaviate | 1.39.8 (pinned in `docker-compose.yml`) |
| torch, docling, ragas | 2.14.1, 2.133.0, 0.2.15 (all Python versions are pinned in `uv.lock`) |
| Embedders | `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B`, each pinned to a Hub commit in `configs/default.yaml` |
| Judge | `gpt-6-luna` (OpenAI, `reasoning_effort: "none"`; no dated snapshot); until 2026-10-21 `gpt-4.1-mini-2025-04-14` |
