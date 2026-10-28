# Ragas evaluation pipeline

Which way of chunking documents gives the best retrieval and the best answers in our agricultural RAG
system? Chunkeval module compares 3 chunking methods (fixed-size character windows, whole sentences, and whole sentences
within one section), each with and without overlap, on two corpora:
- **papers**: 150 structured scientific papers
- **reports**: 150 free-format reports

It generates a question set from the documents themselves, with the evidence of each answer located in
the text. Then every question is retrieved against every chunker with two embedders and answered by a
local LLM. Retrieval is scored with exact span metrics and the answers with Ragas, and every comparison
is paired by question, with bootstrap intervals.

It replaces the previous round's pipeline, `ragas_evaluation.py`, whose results are in
`ragas_results_analysis_new.docx`. Both are kept locally as a reference and are not in the repository.

> **Security:** `ragas_evaluation.py` contains a hard-coded HF token and OpenAI key (ggshield flagged both). Rotate them, put the new ones in `.env` (copy `.env.example`), and never commit `.env` or that script.

## Quick start

Needs Linux, an NVIDIA GPU with 16 GB, Docker, [uv](https://docs.astral.sh/uv/) and Ollama with the
project's models. Setting up a machine from scratch, step by step: [docs/rerun.md](docs/rerun.md).

```bash
uv sync                            # installs the package and its dependencies (uv.lock)
cp .env.example .env               # then fill in OPENAI_API_KEY and a new WEAVIATE_API_KEY
docker compose up -d               # Weaviate
uv run pytest -m "not ollama and not openai and not gpu and not weaviate"   # offline tests, ~45 s
uv run chunkeval --help            # the stages
```

## Stages

Run them in this order. Each reads `configs/default.yaml` and writes its own artefacts, so any stage can
be re-run without repeating the earlier ones; it redoes only what is missing or out of date.

| # | Command | What it does | Details |
|---|---|---|---|
| 1 | `parse` | PDFs → `data/parsed/<corpus>/<doc_id>.json`: one canonical text per document, with its blocks and sections (Docling) | [corpus](docs/corpus.md#parse-parse) |
| 1b | `audit` | flags documents to fix or replace before `chunk` (conversion, OCR, titles, garbled text, other languages, duplicates) → `runs/<id>/audit.txt` | [corpus](docs/corpus.md#audit-audit) |
| 2 | `chunk` | the 6 chunkers (fixed-size, sentence, section × overlap) → `data/chunks/<corpus>/<method>_<ovl>.jsonl` + size stats | [corpus](docs/corpus.md#chunk-chunk) |
| 2b | `embed` | chunk vectors with bge-m3 and Qwen3-Embedding (GPU), cached in `data/embeddings/` | [corpus](docs/corpus.md#embed-embed) |
| 3 | `index` | chunks + both vectors → 12 Weaviate collections `{corpus}_{method}_{ovl}_v2` | [corpus](docs/corpus.md#index-index) |
| 4 | `gen-qa` | question-answer pairs with located evidence spans (local LLM) → `data/qa/raw.jsonl` | [questions](docs/qa.md#q-a-generation-gen-qa) |
| 5 | `filter-qa` | checks, a local critic and de-duplication → the question set `data/qa/filtered.jsonl` | [questions](docs/qa.md#q-a-filtering-filter-qa) |
| 6 | `retrieve` | every question × 12 collections × (2 embedders + oracle + random) → span metrics, `runs/<id>/metrics/` | [evaluation](docs/evaluation.md#retrieval-and-span-metrics-retrieve) |
| 7 | `answer` | RAG answers from each setup's top chunks and from the gold passage (local LLM) → `runs/<id>/answers/` | [evaluation](docs/evaluation.md#rag-answers-answer) |
| 8 | `eval` | Ragas metrics with an OpenAI judge, cost-capped (**the only paid stage**) → `runs/<id>/ragas/` | [evaluation](docs/evaluation.md#ragas-evaluation-eval) |
| 9 | `report` | the report: tables, figures and paired comparisons with bootstrap CIs → `runs/<id>/reports/latest/` | [report](docs/report.md) |

`<id>` is `run_id` in the config: bump it for every new experiment.

## Running a stage

```bash
uv run chunkeval --help                                # list the stages
uv run chunkeval parse --help                          # the options of one stage
uv run chunkeval parse --corpus papers --limit 2       # smoke test: 2 PDFs of one corpus
uv run chunkeval parse                                 # every corpus in the config
uv run chunkeval gen-qa --config configs/other.yaml -v # another config, debug logging
```

`uv run python -m chunkeval.cli …` does the same without the console script. Both work from any directory, because `uv sync` installs the package (see [Development](docs/development.md#package-layout)).

**Options:**
- **On every stage:** `--config` (default `configs/default.yaml`, relative to the repo root) and `-v` for debug logging. A config for another experiment can start with `extends: default.yaml` and hold only what differs ([configs for other experiments](docs/development.md#configs-for-other-experiments)).
- **`--corpus NAME`** (repeatable; default: every corpus) **and `--limit N`** (the first N documents per corpus, sorted by name, so a smoke test always picks the same ones). For now these exist on `parse`, `chunk`, `index` and `gen-qa`; `audit`, `embed`, `filter-qa`, `retrieve`, `answer` and `eval` have `--corpus` only.
- **`--pilot N`** and **`--dry-run`** on `eval`: only the first N questions of each corpus's Ragas sample; or only the cost estimate, with no API call (see [Ragas evaluation](docs/evaluation.md#ragas-evaluation-eval)).
- **`--embedder NAME`** on `embed` (repeatable; default: `embedding.primary`, then `embedding.secondary`) and on `retrieve` (default: `retrieval.embedders`).
- **`--recreate`** on `index`: delete and rebuild the selected collections even when they exist (see [Index](docs/corpus.md#index-index)). On `gen-qa`: delete the Q-A records of the selected corpora, and generate them again; the other corpora's records stay (see [Q-A generation](docs/qa.md#q-a-generation-gen-qa)).

Every stage logs to the console and to `runs/<id>/logs/`, saves the config it used to `runs/<id>/config.yaml`, and stops in the first second on a config error or a missing secret ([what the CLI does around a stage](docs/development.md#the-cli)).

## Key decisions

1. **Ground truth that doesn't depend on any chunker.** Q-A pairs are generated from paragraph groups of the Docling structure, not from one of the chunkers under test. Each answer must quote its evidence word for word. The quote is located in the canonical text and becomes a **character span**. This allows deterministic retrieval metrics, and any pair whose quote can't be found is dropped, which filters out hallucinated pairs by construction.
2. **One canonical text per document.** Every chunker and the Q-A generator work on the same string, so all character offsets are comparable.
3. **Chunk sizes are measured in bge-m3 tokens**, with the tokenizer pinned to the same Hub commit as the bge-m3 embedder. Chunks target 256 tokens with a cap of 512, an experimental choice rather than a model limit (bge-m3 reads up to 8192). 256 tokens is about 160 words on the full corpus, not the previous round's 200; the reason for keeping it is in `configs/default.yaml`. The grid is 3 methods × overlap on/off × 2 corpora: 12 collections.
4. **Two embedders, stored as named vectors in the same 12 collections.** `BAAI/bge-m3` is the primary and is used for everything. `Qwen/Qwen3-Embedding-0.6B` gets the free span metrics only, to check whether the chunker ranking holds under another embedder, and whether the embedder or the chunker is the bigger lever. Both are pinned to a Hub commit ([corpus](docs/corpus.md#embed-embed)).
5. **Correct Ragas inputs:** `user_input` = question, `retrieved_contexts` = the top-k chunks, each with its document's title as the generator read it, `response` = the RAG answer, `reference` = the gold answer. The old script used the gold answer as `response` and the source chunk as `reference`.
6. **Local models where possible, one paid judge.** On Ollama: `gemma4:12b-it-q8_0` writes the Q-A pairs (as the derived all-GPU tag `gemma4-allgpu`), `gpt-oss:20b` is the critic, `qwen3.5:9b-q8_0` writes the RAG answers. Three different families, so no model judges its own text. The judge is OpenAI's `gpt-6-luna` with reasoning off, chosen over `gpt-4.1-mini-2025-04-14` after both judged the same pilot ([evaluation](docs/evaluation.md#the-judge-gpt-6-luna-instead-of-gpt-41-mini-2026-10-21)).
7. **Metrics:** span-based retrieval metrics (hit@k, span recall@k, MRR, IoU and precision, context tokens), the 4 Ragas metrics plus factual correctness, judge-free text metrics, all stored per question, with bootstrap CIs, paired comparisons and a budget-matched view.
8. **Never pay twice.** Every LLM stage keeps what it was told: `gen-qa` saves each unit's outcome as it arrives, and the critic, the RAG generator and the judge cache every call under a hash of what decides it. The answer and judge caches are shared by every run and data dir (`data/llm_cache/`). A re-run, a crash or a new `run_id` never asks again what was already answered, and the judge refuses to start above a cost cap (`ragas.max_cost_usd`).

## Documentation

| File | Contents |
|---|---|
| [docs/rerun.md](docs/rerun.md) | Setting up a new machine and running the whole pipeline from scratch, step by step |
| [docs/setup.md](docs/setup.md) | Ollama (server settings, models, quantization, moving it to Docker) and Weaviate |
| [docs/corpus.md](docs/corpus.md) | Parse, audit, chunk, embed, index; the full-corpus audit and what was decided |
| [docs/qa.md](docs/qa.md) | Q-A generation and filtering, hand labelling |
| [docs/evaluation.md](docs/evaluation.md) | Retrieval and span metrics, RAG answers, Ragas evaluation and its cost guard |
| [docs/report.md](docs/report.md) | The report: statistics, the ranking rule, what it shows |
| [docs/development.md](docs/development.md) | Package layout, conventions, the CLI, configs for other experiments, testing, dependencies |
| [docs/future-work.md](docs/future-work.md) | Extensions considered and left out for now |

Each stage's doc says what it does, what it writes, the decisions behind its settings, and what was found on the dev docs and on the full corpus.
