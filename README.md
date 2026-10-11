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
| 4 | `gen-qa` | `qa/generation.py` | CanonicalDoc → `data/qa/raw.jsonl` (with evidence char spans) + `gen_dropped.jsonl` (with reasons) |
| 5 | `filter-qa` | `qa/filtering.py` | raw → `data/qa/filtered.jsonl` (+ `rejected.jsonl` with reasons, `critic_scores.jsonl` cache) |
| 6 | `retrieve` | `evaluation/retrieval.py` + `evaluation/retrieval_metrics.py` | questions × 12 collections × (2 embedders + oracle + random) → `runs/<id>/retrieval/<retriever>/*.jsonl` + `runs/<id>/metrics/retrieval_per_question.parquet` |
| 7 | `answer` | `evaluation/answering.py` + `prompts/rag_answer.md` | questions × 6 setups per corpus (bge-m3's top `top_k` chunks) + the gold passage → `runs/<id>/answers/<corpus>_<setup>.jsonl` (+ `answer_cache.jsonl`) |
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
- **`--corpus NAME`** (repeatable; default: every corpus) **and `--limit N`** (the first N documents per corpus, sorted by name, so a smoke test always picks the same ones). For now these exist on `parse`, `chunk`, `index` and `gen-qa`; `embed`, `retrieve` and `answer` have `--corpus` only.
- **`--embedder NAME`** on `embed` (repeatable; default: `embedding.primary`, then `embedding.secondary`) and on `retrieve` (default: `retrieval.embedders`).
- **`--recreate`** on `index`: delete and rebuild the selected collections even when they exist (see [What `index` builds](#what-index-builds)). On `gen-qa`: delete every Q-A record, of all corpora, and generate again (see [Q-A generation](#q-a-generation-gen-qa)).

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
│   ├── llm.py          one chat client for every LLM role (Ollama or OpenAI)
│   └── prompts.py      prompts/*.md split into checked `## <name>` sections
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
3. **Chunk sizes are measured in bge-m3 tokens**, with the tokenizer pinned to the same Hub commit as the bge-m3 embedder (`chunking.size_tokenizer_revision`; the config refuses two different pins). Chunk sizes target 256 tokens with a cap of 512. The cap is an experimental choice, not a model limit, since bge-m3 reads up to 8192 tokens. 256 tokens is about 160 words on the dev docs, not the previous round's 200; the reason for keeping it is in `configs/default.yaml`. The grid is 3 methods (character / sentence / recursive-chapter) × overlap on/off × 2 corpora, which gives 12 collections.
4. **Two embedders, stored as named vectors in the same 12 collections:**
   - **Primary: `BAAI/bge-m3`.** Used for everything: span metrics, RAG answers, Ragas. It needs no query instruction.
   - **Secondary: `Qwen/Qwen3-Embedding-0.6B`.** Only the free span metrics are computed with it. Its job is to check whether the chunker ranking holds under a different embedder, and whether the embedder or the chunker is the bigger lever. Its queries need an `Instruct: …\nQuery:` prompt (set in the config).
   - Both models produce 1024-dim vectors, so a mix-up never raises an error. Always pass vectors together with their embedder name (see `embedding.py`).
   - **Both are pinned to a Hub commit** (`revision` in `configs/default.yaml`, a full 40-character hash; a branch like `main` is rejected). Otherwise an upstream push could change the vectors mid-experiment, and queries would be embedded by a different model than the documents. Changing a revision makes `chunkeval embed` re-embed that model's chunk files. The config comment shows how to look up the latest commit.
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

## Before the QA stage: audit the final corpus

Both corpora may still be replaced by a different set of documents, so every setting so far was checked on dev PDFs only: first paper_4, paper_8, report_15 and report_16, then (2026-10-11, to develop the Q-A stages) paper_42, paper_74, paper_93, report_119, report_123 and report_140. The chunking numbers below are from the first 4. Once the final corpus is fixed, run `chunkeval parse` and `chunkeval chunk` on all of it and go through this list before `gen-qa`. Don't tune any of these on the dev docs.

**Parsing**
- **What each document lost:** check `meta.body` in every parsed JSON (`start`, `end`, `dropped_sections`, `chars_dropped`) and the warnings in the parse log.
- **OCR thresholds:** recalibrate `parsing.ocr_max_*_ratio` from the `meta.text_layer.auto` ratios every parsed JSON keeps. Changing them re-parses only the PDFs whose OCR decision flips.
- **Reports' drop list:** `parsing.body.reports.drop_titles` was built from two AfricaRice annual reports. Extend it from the full corpus (candidates are noted in the config).
- **OCR retried without need (report_140).** Its first conversion had 12.3% repeated paragraphs, so it was converted again with full-page OCR; afterwards the parser still reported 9.2%, yet only one block (a table footnote) actually repeats. Check what `repeated_paragraph_ratio` counts there before recalibrating its threshold: if repeated short items trigger it, it sends PDFs with a good text layer through OCR, which is slower and can garble text (see paper_4 below).
- **Document titles.** report_119 (a book chapter) got the publisher's banner as its title: "We are IntechOpen, the world's leading publisher of Open Access books…". The title is shown to the Q-A generator and to the critic, so check the titles of the full corpus (`parsing.doc_title`).
- **Garbled OCR text (seen in paper_4).** paper_4 is converted with full-page OCR because its text layer is duplicated (`repeated_paragraph_ratio` 0.12). In 7 paragraphs of sections 2, 3.1, 3.2 and 4.1, the OCR output is nonsense, e.g. "The GanbanlPlatondastenegtls initiadedbovemgonicatienebaitks…". That is about 2,100 of its 16,700 tokens (13%). report_16, also OCR'd (for unreadable fonts), came out clean.
  - **Why it matters:** garbled chunks get indexed and compete in retrieval as noise. A Q-A unit drawn from them gives nonsense questions, and since the quote locator matches verbatim text, it can still "locate" a garbled quote. It also distorts the size stats (one garbled "sentence" alone is 529 tokens).
  - **A second kind, inside clean paragraphs:** paper_4 also has fragments of single letters mixed into otherwise normal paragraphs, e.g. "…potentially unique ar s s o e e o es ,sers s u ton Cryopreserved back-up collections…" or a paragraph opening "a re rs h n nd h- nae rnd s s u uun…". Their tokens per word is normal (1.36), and at block level neither tokens per word nor mean word length separates them from clean text (4.64 letters per word vs. 3.52 for a clean formula paragraph in paper_8). `sample_units` picked one such paragraph on the dev docs; the generation prompt tells the model to skip garbled text, and the critic sees the passage.
  - **Detecting it:** tokens per word is a good first signal: 3.4–17 in the garbled paragraphs vs. about 1.6 in normal text. But email contact lines in the reports also reach 3–4.5, so a second signal is needed, e.g. the share of words found in an English word list. That share would also catch the letter fragments, if measured per sentence rather than per block. Calibrate it on the full corpus. Every generation unit stores its `tokens_per_word` in `meta`. `sample_units` already drops a single "sentence" longer than `unit_max_tokens`: on the dev docs only two, both garbled in paper_4 (529 and 401 tokens). The rest of the 669-token garbled paragraph splits into normal-sized sentences, which can still end up in a unit.
  - **Options:** (a) measure it as a third `meta.text_layer` ratio and flag or drop garbled blocks; (b) retry OCR with another engine or a higher image resolution; (c) OCR only the pages with the duplicated layer and keep the text layer elsewhere; (d) exclude the document. Choose once you know how many documents of the final corpus are affected.

**Chunking** (the numbers are in `data/chunks/<corpus>/_size_stats.json`)
- **Words per chunk (`words_mean`):** about 160 words at 256 tokens on the dev docs. If it moves noticeably, update the note on `target_tokens` in `configs/default.yaml`.
- **Small recursive chunks on reports:** on the dev docs, recursive chunks average 192 tokens (vs. 256 for the other methods), and 5% are 40 tokens or less. Reports have many tiny sections (office addresses, "ISBN:", "Writing:"). Check how common small chunks are on the full corpus. The setting is `chunking.min_section_tokens`, which merges small sibling sections under the same parent only. Raising it trades section purity for size.
- **Character window:** it is calibrated per corpus on every run (1,073 chars for papers and 1,101 for reports on the dev docs). Check that the two values stay close, and that the chunk log has no "window … > max_tokens" warnings.
- **Sentence overlap:** it averages 42–47 tokens vs. 31 for the character windows, because it repeats at least one whole sentence. That rule is deliberate (the docx suggests a 1-sentence overlap). Report the measured `overlap_tokens_mean` next to the results.

**Q-A generation and filtering** (check after `gen-qa` and `filter-qa` have run on the full corpus)
- **Two-span units are off** (`qa.two_span_ratio: 0`, decided 2026-10-11). On the 10 dev docs, at a ratio of 0.2, 20 two-span units gave 3 pairs (same section 3 of 14, across sections 0 of 6; single units: 66 of 76), and `filter-qa` rejected all 3: one names its source ("described in the abstract"), one doesn't need passage B (`needs_both` 1), and one infers more than the passages say. They took a fifth of the generator calls for no question. The code stays; a ratio above 0 brings them back. Chunk boundaries are still tested by single units, whose evidence can span several paragraphs.
- **Critic threshold and yield.** With single units only, the 10 dev docs gave 84 pairs from 97 units, and `qa.critic_min_score: 4` keeps 35 of them (42%; at 3: 53). Calibrate the threshold with the 50 hand labels (see [Q-A filtering](#q-a-filtering-filter-qa)), then check the yield. At 4 it is 32% of units for papers and 40% for reports, so 100 documents × 10 units give ~320 and ~400 questions: papers only just above `ragas.sample_per_corpus` (300). Raise `qa.max_questions_per_doc` if the full corpus comes out lower.

**Retrieval** (check after `retrieve` has run on the full corpus)
- **Very long documents.** On the dev docs, report_123 (326k characters) holds 279 of the 420 reports chunks (66%), so most distractors in the reports collections come from one document. Check the chunks per document on the final corpus. If a few documents dominate, decide whether to cap them or to report `doc_hit` next to `span_hit`.
- **The budget view.** `retrieve` warns when a question fits all `k_max` hits into `budget_tokens`, because its budget k is then capped. Raise `retrieval.k_max` if it does.

## Testing

The tests import `chunkeval` as the installed package, so run `uv sync` once (see [Package layout](#package-layout)). pytest is in the dev dependency group, which `uv sync` installs too.

```bash
uv run pytest                                   # everything (~35 s with Ollama running and a free GPU)
uv run pytest -m "not ollama and not openai and not gpu and not weaviate"   # offline tests only (~5 s): run after every change
uv run pytest -m gpu                            # embedding models on the GPU (~10 s): run after changing embedding.py or the models
uv run pytest -m weaviate                       # the local Weaviate (~15 s): run after changing indexing.py or the Weaviate version
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
| [tests/test_chunking.py](tests/test_chunking.py) | The chunk invariants on all 6 configurations: `text == doc.text[start:end]`, full coverage, `max_tokens`, disjoint spans without overlap, the overlap budget. Also sentence boundaries around tricky abbreviations, the recursive section groups, the shared size rules (`pack`, `overlap_start`), each way `validate_chunks` fails, and the `chunk` stage's output files. When `data/parsed/` exists, the 6 configurations also run on the parsed dev docs with the config sizes. | the bge-m3 tokenizer (~20 MB, downloaded on first use) |
| [tests/test_embedding.py](tests/test_embedding.py), offline | The vector cache and the `embed` stage with a fake embedder. Each model is loaded once and only when a file needs it. A re-run embeds only changed chunk files: the cache compares content hashes, not mtimes, and a changed query prompt doesn't count. A new model revision re-embeds that model's files. A crashed run resumes. The loader refuses vectors whose chunk_ids differ (order included) or whose settings are stale. The Ollama guard names the loaded models and passes when Ollama is unreachable. | nothing |
| [tests/test_embedding.py](tests/test_embedding.py), marker `gpu` | Both real models: loaded at the pinned revision; float32 unit-norm output; Qwen3's instruction applied to queries only; two different vector spaces; a 3-passage retrieval sanity check; the same vector with and without padding (pooling + padding side); too-long input raises instead of being truncated; every chunk in `data/chunks/` fits each model's `max_seq_length`. | a CUDA GPU with no Ollama model loaded; the models (~4.5 GB bge-m3, ~1.2 GB Qwen3, downloaded on first use) |
| [tests/test_indexing.py](tests/test_indexing.py), offline | Every chunk field but `meta` is a property, and only `text` is searchable. `--limit` takes the first documents by doc_id, with their own vectors. Missing chunks or vectors name the stage to run. Provenance changes with the chunks, the embedder settings and `--limit`. | nothing |
| [tests/test_indexing.py](tests/test_indexing.py), marker `weaviate` | On 6 fake chunks with hash vectors: the collection gets both named vectors (flat, cosine) and its provenance, and every object matches its chunk and cached vectors. A chunk's vector finds that chunk first in its own space. doc_id filters match whole ids, and BM25 ignores section titles. A re-run skips an up-to-date collection and refuses a stale, incomplete or hand-made one until `--recreate`. Verification catches swapped names and shifted rows, and failed inserts raise. | Weaviate running (`docker compose up -d`), `WEAVIATE_API_KEY`; touches only `*_pytest` collections |
| [tests/test_generation.py](tests/test_generation.py) | Unit sampling on `tiny_doc` with small sizes: single units stay in one section and within the size bounds; sections that are too small get no unit; a paragraph over the max is cut into sentence groups; an over-long sentence is in no unit; two-span units are adjacent whole paragraphs (also across a section boundary); skippable sections are never used; the per-doc cap and the two-span ratio hold, and no two units overlap. The prompt file has exactly its sections and placeholders, checked on load: a missing section, a misspelt placeholder or a lone `$` fails. The chat messages: one system message shared by every call, then the title, section path, passage and question type in the user message, with passages A and B for a two-span unit. Which outputs `QA_JSON_SCHEMA` accepts (pairs, skips) and rejects, and that its question types are the config's. The generation loop against a fake LLM: pairs with located evidence, every drop reason, a two-span answer quoting one passage only, the factoid retry (single units only), the type rotation. The `gen-qa` stage: resume after a crash asks only the missing unit; drops are saved but API errors aren't (and the stage fails); other settings, another prompt, a re-parsed document or records without provenance are refused until `--recreate`. `load_pairs` reads back exactly what the stage wrote. When `data/parsed/` exists, the same unit invariants on the dev docs with the config sizes, reproducible by seed; a document with enough running text (quota × `unit_max_tokens`) fills the quota, spread over the document. | the bge-m3 tokenizer |
| [tests/test_filtering.py](tests/test_filtering.py) | Each deterministic check on its pattern, case-insensitively, and on what it must let through ("in the context of <topic>", "farmers had no information on prices", a number spelt out in the evidence). The critic's two calls: the passage call shows the evidence as located in the text, the question call only the question, and the system messages are shared. Which outputs each call's schema accepts (note first, reason before score) and rejects. `decide` ignores the notes, and the threshold applies per criterion. De-duplication: greedy, earlier wins, within a corpus only. The `filter-qa` stage against a fake critic and embedder: each pair ends at the right stage with its reasons; a re-run or a new threshold asks nothing; a changed answer asks only its passage call; another critic setting asks everything; an API error writes nothing, and the re-run asks only what is missing; an unusable answer rejects its pair and is cached; a re-parsed document is refused; the labelling sample is blind and never overwritten. | nothing |
| [tests/test_retrieval_metrics.py](tests/test_retrieval_metrics.py) | Every span metric on hand-computed toy cases: interval merging and overlap; a hit equal to the evidence scores 1 everywhere; the same offsets in another document score 0; overlapping hits and other documents' text count once in precision and IoU; `all_evidence` needs every quote touched (touching quotes are one); MRR and hit only see the top k; `budget_k` never returns 0. | nothing |
| [tests/test_retrieval.py](tests/test_retrieval.py), offline | The oracle's greedy cover (hand-computed) and the random baseline's seeding. Query vectors: encoded with the query prompt, cached, encoded again after a new question or prompt. The `retrieve` stage against a fake embedder and retriever: a file per retriever with ranked hits and no text, then the metrics table (one row per question, retriever and k, unique). A re-run asks nothing; a new query prompt redoes only that embedder; a new question set redoes everything; a mode switch is refused; a question's document without chunks, or chunks cut from another parse, stop the stage. The metrics leave out stale files outside the run's scope, with a warning, and fail on stale files inside it. | nothing |
| [tests/test_retrieval.py](tests/test_retrieval.py), marker `weaviate` | `WeaviateRetriever` on a `*_pytest` collection: a chunk's own vector finds it first with cosine 1, in the named vector asked; an unknown vector name fails; hybrid with alpha 0 is BM25; a collection built from other chunks, or a missing one, is refused. | Weaviate running, `WEAVIATE_API_KEY` |
| [tests/test_answering.py](tests/test_answering.py) | The prompt file must contain the abstention phrase the code looks for. The messages: one shared system message, then the passages numbered in rank order with their text only, then the question. `is_abstention` on the phrase (case, markup, an explanation after it), paraphrases alone, and what must stay an answer (a partial answer, a paraphrase followed by facts, "the soil does not contain…"). The `answer` stage on the offline `retrieve` setup against a fake generator: one file per setup and one for the gold passage, each answer holding exactly the top-k chunks it was shown (or its own paragraph) and the evidence flag; the summary counts. A re-run asks nothing, a dropped and restored question asks nothing, another generator setting asks everything; a stale retrieval file stops the stage before any call; an API error writes no answer file and the re-run asks only it; an unusable answer is kept with its error and cached. | nothing |
| [tests/test_quote_locator.py](tests/test_quote_locator.py) | `locate_quote`, the source of every evidence span. Exact matches, and normalized ones whose offsets still point into the original text: whitespace and paragraph breaks, typographic dashes and quotes, soft hyphens, case, an added full stop, line-break hyphenation. Fuzzy matches only above the threshold, widened to whole words, never against a span shorter than the quote. Only the unit's spans are searched, and too-short quotes are rejected. When `data/parsed/` exists, a random quote from every block of the dev docs, verbatim and with that drift (a quote occurring twice in its block is found at its first occurrence; an invisible character at its edge is left out of the span). | nothing |

**Good to know:**
- **Live tests skip themselves** when their service isn't available. If Ollama isn't running, `uv run pytest` still passes, and the live tests show as skipped instead of failing.
- **Tests never touch real data.** The `cfg` fixture points `runs_dir` to a temporary folder, so nothing is written to `runs/` or `data/`.
- **The `ollama` tests load the models onto the GPU** (about 20 s) and unload them at the end, even when a test fails. Don't run them while embedding is using the GPU.
- **The `gpu` tests load both embedders once per test module** and unload them at its end, so they're gone before the `ollama` tests run. They skip when there's no CUDA GPU or when Ollama holds a model.
- **A generator must be 100% on the GPU.** `test_live_check_ollama` fails if `qa_generator` or `rag_generator` is partly offloaded to the CPU. If you switch to one of the MoE generators, which offload by design, relax that assertion.
- **Writing new tests:**
  - Use the fixtures from `conftest.py`: `cfg` (a validated config) and `raw_config` (the YAML as a dict, for building invalid configs).
  - Mark tests that need a service: `@pytest.mark.ollama`, `@pytest.mark.openai`, `@pytest.mark.weaviate`, or `@pytest.mark.gpu` for embedding models.
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
- **One model on the GPU at a time.** Before each `embed` run, and between generator and critic stages, unload the loaded model with `ollama stop <model>`. Otherwise Ollama keeps it for 5 minutes, and the next model or the embedders get pushed onto the CPU. `embed` checks this itself: before loading each embedder it asks Ollama what is loaded and stops with the `ollama stop` command to run. It never unloads a model itself, because the model may belong to another project.
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

### What `index` builds

```bash
uv run chunkeval index --corpus papers --limit 1    # smoke test: the chunks of one paper
uv run chunkeval index --recreate                   # all 12 collections, replacing the smoke test
```

- **One object per chunk.** Its properties are the chunk's fields (all but `meta`), and its uuid is derived from `chunk_id`. It carries both named vectors, read from the `embed` cache; `index` never loads a model, so it can run while Ollama holds the GPU.
- **Exact search.** Both named vectors use a flat (brute-force) index with cosine distance, not HNSW. An approximate index can miss a true neighbour, and which one depends on insertion order, so a rebuild could move the retrieval metrics. At ~55 chunks per document, a collection is a few thousand objects, where brute force takes milliseconds per query.
- **BM25 sees only `text`.** The ids and `section_path` are filterable (matched whole, so `paper_4` doesn't match `paper_4_suppl`) but not searchable, so the hybrid follow-up scores only the text the generator reads.
- **Provenance and re-runs.** Each collection's description holds a JSON record of what it was built from: a hash of the chunks, `--limit`, and the settings of both embedders (model id, revision, …). On a re-run:
  - a collection with the same provenance that passes verification is **skipped**;
  - any other existing collection is **refused**, with the inputs that differ named (e.g. after `chunk` or `embed` changed something, or after a `--limit` smoke test). `--recreate` rebuilds it.
- **Verification after every build.** The object count must equal the chunk count. For 8 objects spread over the collection, the properties and both stored vectors must equal the chunk file and the cache. Both models give 1024-dim unit vectors, so this is what catches swapped names or rows shifted by one. Any failure stops the stage.

**Problems:**

| Symptom | Fix |
|---|---|
| Connection timeouts or gRPC errors | Check that port 50051 is mapped |
| Query error about a missing or unknown vector | `target_vector` must be `bge_m3` or `qwen3_06b`, matching the names in the config |
| 401 errors | The key in `.env` doesn't match the one the container started with. Restart the container after changing `.env`. |
| `port is already allocated` | Another Weaviate is running. Check with `docker ps`. |

## Q-A generation (`gen-qa`)

```bash
uv run chunkeval gen-qa --corpus papers --limit 1   # smoke test: one paper, ~20 s after the ~25 s Ollama preflight
uv run chunkeval gen-qa                             # every corpus; resumes where an earlier run stopped
uv run chunkeval gen-qa --recreate                  # delete ALL Q-A records (every corpus) and start over
```

**What it does,** per document (`qa/generation.py`):
- **Units** (`sample_units`): up to `qa.max_questions_per_doc`, spread over the whole document, of which `two_span_ratio` are two-span. No two units overlap.
  - A **single** unit is consecutive paragraphs of one section, packed towards the middle of `unit_min_tokens..unit_max_tokens`.
  - A **two-span** unit is two adjacent whole paragraphs, also across a section boundary; the question must need both. **Off by default** (`two_span_ratio: 0`): they rarely gave a question (see [Before the QA stage](#before-the-qa-stage-audit-the-final-corpus)).
  - A paragraph over the max is cut into sentence groups. A single "sentence" over the max is never used (garbled OCR in practice).
  - Sections that `skip_section_titles` names, or with less text than the minimum, give no unit.
- **One question per call,** with the prompt in [prompts/qa_generation.md](prompts/qa_generation.md); its notes explain the file format.
  - Question types rotate per document. `numeric` passes to the next type for a unit without a digit.
  - A single unit's skip of another type is asked once more as `factoid`; a two-span skip is final.
- **Evidence** (`locate_quote`): each quote is searched only inside the unit, first exactly, then normalized, then fuzzy at `qa.fuzzy_match_threshold`. A pair needs every quote located, and in a two-span unit at least one quote in each passage.

**What it writes.** One line per unit, appended as soon as the unit's outcome is final, so a crash loses only the units in flight:
- **`data/qa/raw.jsonl`:** one `QAPair` per line. `question_type` is the type the model labelled; `meta` holds the types asked, the unit's spans, section path, size and `tokens_per_word`, and the token usage.
- **`data/qa/gen_dropped.jsonl`:** one `DroppedUnit` per line, with a `reason` (`skip`, `bad_output`, `quote_not_found`, `two_span_one_side`) and the model's output, so the drop rate per corpus and reason can be reported.
- **API errors are not saved** (timeouts, server down). The stage fails at the end, and a re-run asks exactly those units again.
- **Summary:** logged and written to `runs/<run_id>/gen_qa_summary.json`. Per corpus: units, pairs, pair rate, drops by reason, how the evidence was located, and the question types.

**Resume and provenance.** `data/qa/raw.provenance.json` records what decides a unit's outcome:
- the generator (model, settings, Ollama digest), the prompt file's hash, the seed, the size tokenizer, `skip_section_titles`, the `qa` settings that shape units, and `GENERATION_VERSION` (bumped when the code changes its output);
- a text hash of every document that has records.

On a re-run:
- with the same values, only units that are in neither file are asked;
- with any difference, the run is refused and the differing keys are named. `--recreate` deletes the records of every corpus, because the settings are shared.

`qa.docs_per_corpus` and `--limit` only choose which documents are used, so the full run resumes a `--limit` smoke test.

**On the dev docs** (`qwen3.5:9b-q8_0`; 36 hand-made units, plus the paper_4 smoke run):
- **Output format:** valid JSON every time, never in code fences. Every quote was located, almost all exactly.
- **Speed:** 56–64 tokens/s at concurrency 2, so about 20 s per document of 10 units, or about 1 h for 200 documents.
- **Skips:**
  - `numeric` asked of passages whose only digits are years or names like "F2";
  - two-span pairs across a section boundary (0 of 4 gave a pair: the model calls them unrelated; `meta.crosses_section` lets the pilot measure this);
  - garbled OCR text, correctly.
- **Left to `filter-qa`:** some questions still refer to the source ("according to the passage", "the researchers") or are vague.
- **All 10 dev docs** (2026-10-11): with two-span units at 0.2, 96 units gave 69 pairs (72%), of which two-span only 3 from 20 (see [Before the QA stage](#before-the-qa-stage-audit-the-final-corpus)). With single units only, 97 units gave 84 pairs (87%) in 3 min 11 s.

**For the generator pilot** (step 5 of the implementation order): run each candidate with `--recreate` on the same documents, then compare `pair_rate`, the drop reasons and tokens/s from the summary, plus the critic's acceptance (`keep_rate` in the `filter-qa` summary). The 10 dev docs give 96 units with the default config.

## Q-A filtering (`filter-qa`)

```bash
uv run chunkeval filter-qa      # every pair of raw.jsonl; asks the critic only what isn't cached
```

**What it does** (`qa/filtering.py`), in three layers, each only on what the one before kept:
1. **Deterministic checks** (`deterministic_checks`), free, before any LLM call:
   - **On the question:** a reference to its source ("this study", "according to the passage", "described in the abstract", "for the study", "the authors"), a pronoun opening, a figure, table or section number, no final "?", two questions in one, fewer than 5 or more than 40 words. These are case-insensitive regexes; the old `phrase_blacklist` was case-sensitive. "in the context of <topic>" is allowed, because good questions use it to name their topic.
   - **On the answer:** an abstention ("the passage does not mention…", "N/A"). Also, every number in the answer must appear in the located evidence or in the question, in digits or in words there. A number found only elsewhere in the passage means the gold evidence misses part of the answer; on the dev docs this caught "less than 10 kg", which was in the passage but not in the quote.
   - An answer that repeats a quote word for word is fine, as long as it answers the question (decided 2026-10-11).
2. **The critic** (`llm.qa_critic`, gpt-oss:20b, a different family from the Qwen generator), with the prompt in [prompts/qa_critic.md](prompts/qa_critic.md). Its notes explain each choice. It is called **twice per pair**:
   - **The passage call** sees the passage, question, answer and evidence, and judges the answer: `groundedness`, `answerability`, plus `needs_both` for a two-span pair, when they are on (one passage alone mustn't answer it).
   - **The question call** sees the question alone, as a user of the retrieval system does, and judges the question: `standalone`, `specificity` (one clear answer among hundreds of documents), `usefulness`.
   - Each call first writes a note the scores rest on: `question_asks_for` and `unresolved_references`. Then it gives a reason and a score from 1 to 5 per criterion.
   - **The decision is made in code** (`decide`): every criterion must reach `qa.critic_min_score`. Changing the threshold asks nothing again.
3. **De-duplication** (`deduplicate`): questions are embedded with bge-m3 (`embedding.primary`) once the critic is unloaded. A question is removed when its cosine to an already kept question of the same corpus is above `qa.dedup_cosine_threshold`. The pair kept is a two-span one (the scarce kind), else the one with the higher total critic score.

Every pair that passes is kept, with no down-sampling (decided 2026-10-11). `filtered.jsonl` is the final question set: every later stage evaluates exactly its question ids, and Ragas draws its own subset of it.

**What it writes:**
- **`data/qa/filtered.jsonl`:** the kept `QAPair`s in document order, with the critic's output in `meta.filter.critic`.
- **`data/qa/rejected.jsonl`:** every other pair, with `meta.filter`: `stage` (`checks`, `critic`, `dedup`), `reasons` (e.g. `source_reference: 'this study'`, `critic_specificity: 3`, `duplicate_of: <id> (0.95)`) and the critic's output, so the filter can be audited.
- **`data/qa/critic_scores.jsonl`:** the critic's answers, one line per pair and call, appended as they arrive. This is a cache: each line is keyed by a hash of everything that decides it (the messages, the schema, the critic's model, settings and Ollama digest, `CRITIC_VERSION`). A re-run asks only what is new: new pairs, a changed pair's call, or every call after a prompt or critic change. An answer the critic couldn't produce (cut off, invalid JSON) is cached too and rejects its pair (`critic_bad_output`). API errors stop the stage before anything is written; a re-run asks exactly those.
- **Summary:** logged and written to `runs/<run_id>/filter_qa_summary.json`. Per corpus: pairs, kept, `keep_rate`, rejections by stage and by reason, each criterion's mean and count below the minimum, and the kept pairs by question type and kind.
- **`runs/<run_id>/filter_qa_labelling.csv`:** 50 pairs for hand labelling (see below). It is written once and never overwritten; delete it to draw a new sample.

**Hand labelling, to calibrate the threshold.** Do it on the final corpus, with a new `run_id`.
- **The sample:** half of it is kept pairs, half pairs rejected by the checks or the critic (duplicates are good pairs, not a verdict). The rows are shuffled and carry the passage and evidence, but not the filter's decision, so the labels can't be biased by it.
- **Labelling:** fill in `label` (1 = good pair, 0 = bad). Then join by `id` with `filtered.jsonl` and `rejected.jsonl`. The kept pairs labelled good give the filter's precision; the rejected pairs labelled bad give a rough recall.
- **The threshold:** if the critic rejects too many good pairs, try `critic_min_score: 3` (agreed fallback). Note that 3 also lets through `groundedness` 3, an answer with one inferred detail.

**On the dev docs**, single units only (84 pairs from 10 documents, 2026-10-11; 2 min 33 s): 65 pass the checks, 35 the critic, none removed as duplicates. That is 15 of 40 pairs for papers and 20 of 44 for reports; at `critic_min_score: 3` it would be 23 and 30.

The detailed findings below come from the first run (69 pairs, two-span units still on; about 1.1 s per critic call at concurrency 2):
- **Checks:** 13 rejected, 11 of them for naming their source. Every flagged reference was a real one.
- **Critic:** 27 of the 56 remaining pass. Most rejections are `specificity` (24 pairs below 4, e.g. "What factor was identified as a determinant of the quantity of rice supplied through collective marketing?") and `standalone` (13: "the two control villages", "these enterprises", "their full model"). It also caught answers that reverse the passage (gender and improved varieties in Nasarawa) or answer something else ("Why…" answered with a restatement).
- **Compared with a first reading of 33 pairs (7 bad, 8 borderline, 18 good):** 6 of 7 bad rejected, 4 of 18 good rejected. Two of those four are specificity errors (a question naming CADFund scored 3); the other two are defensible. The bad pair it kept answers "Which organisation has FAO worked under?" with "FAO"; the same critic rejected it under an earlier prompt, so its answers on borderline cases are not stable across prompt versions.
- **De-duplication:** nothing removed (no two questions above 0.92).
- **How the prompt got there:** a first version with one-line criteria passed 50 of 56 pairs with 5 on every criterion. Raising the reasoning effort to medium or high didn't help. Asking for a note before the scores fixed the answer criteria. Judging the question without the passage fixed `standalone` and `specificity`: with the passage in view, the critic resolved "the full model" from it and never listed an unresolved reference.

## Retrieval and span metrics (`retrieve`)

```bash
uv run chunkeval retrieve                          # every corpus, retrieval.embedders, both baselines; then the metrics
uv run chunkeval retrieve --embedder qwen3_06b     # only Qwen's files (e.g. after changing its query prompt)
```

No LLM and no API cost. The GPU is used only to encode questions that aren't cached, and the stage stops (`check_gpu_free`) while Ollama holds a model. On the 10 dev docs the whole stage takes 8 s, and a re-run with nothing changed takes 0.3 s.

**What it does** (`evaluation/retrieval.py`). Every question of `filtered.jsonl` gets its `retrieval.k_max` (20) best chunks from each of the 6 collections of its own corpus, for each retriever:
- **Each embedder** searches its own named vector (`target_vector`) with the question encoded by `Embedder.encode_queries`: bge-m3 without an instruction, Qwen3 with its `query_prompt`. The query vectors are cached in `data/embeddings/queries/<embedder>/<corpus>.npy`, keyed by the questions and the query settings (model, revision, prompt).
  - **`retrieval.mode: dense`** scores by cosine similarity.
  - **`hybrid`** fuses it with BM25 on `text`, weighted by `hybrid_alpha`: 1 is pure vector, 0 pure BM25.
- **Oracle**, per (corpus, method, overlap): the chunks of the question's own document that cover its evidence, chosen greedily by how many evidence characters each adds that aren't covered yet. It stops once all of the evidence is covered, so it usually has 1–2 hits. This is the ceiling a chunker allows, which separates "the chunker can't" from "the retriever didn't". Its `context_tokens` is how much text that chunker needs to hand over the whole evidence.
- **Random**: `k_max` chunks drawn from the variant's chunks of the whole corpus, seeded per question and variant. This is the chance level.

**Checks before anything is retrieved,** so no metric compares offsets into two different texts:
- The questions' documents still have the text `gen-qa` saw (the hashes in `raw.provenance.json`).
- Every one of those documents has chunks, and each of its chunks is `doc.text[start:end]`. This catches documents that were parsed after `chunk` ran, which happened on the dev docs on 2026-10-11.
- Each collection was built from exactly the current chunk file: its `index` provenance and object count are checked.

**What it writes:**
- **`runs/<run_id>/retrieval/<retriever>/<corpus>_<method>_<ovl>.jsonl`:** one line per question, `{"question_id", "hits": [...]}`, with each hit's rank, chunk_id, doc_id, offsets, n_tokens and score. The chunk text isn't stored: at 20 hits it would repeat each corpus many times over, and the chunk file has it by chunk_id. `<retriever>` is an embedder name, `oracle` or `random`.
- **A `.provenance.json` next to each file:** the chunk file's hash, a hash of the questions (ids, text, evidence spans), `k_max`, plus the query settings and mode for an embedder, or the seed for random.
  - **On a re-run,** an up-to-date file is skipped and a stale one is retrieved again.
  - **A different `mode` or `hybrid_alpha` is refused** instead: that is another experiment, so bump `run_id`.
- **`runs/<run_id>/metrics/retrieval_per_question.parquet`** (`evaluation/retrieval_metrics.py`): the long table, one row per (question_id, embedder, corpus, method, overlap, k_label).
  - `k_label` is each of `retrieval.k_values` (1, 3, 5, 10), plus `budget`: the largest k whose chunks fit in `retrieval.budget_tokens`.
  - The baselines are rows with `embedder` `oracle` or `random`.
  - Each row also carries the question's doc_id, type, kind, section, number of evidence intervals and evidence length, for the breakdowns.
  - It covers the whole configured grid. A file outside this run's `--corpus` / `--embedder` that is missing or stale is left out, with a warning naming it.
- **`runs/<run_id>/retrieval_summary.json`:** the means per corpus, retriever, variant and k. They are also logged at `top_k` and at the budget.

**The metrics, in characters** (decided 2026-10-11). The spans are character offsets into the same text, so the overlaps are exact and need no tokenizer; only `context_tokens` counts bge-m3 tokens. E is the evidence quotes merged into intervals, and R is the merged union of the top-k chunks: overlapping chunks count once, and other documents' text counts in |R|.

| Metric | Definition |
|---|---|
| `doc_hit` | some top-k chunk is from the question's document |
| `span_hit` | some top-k chunk of that document overlaps E |
| `all_evidence` | every interval of E is touched. Defined for every pair, not only two-span ones: 22 of the 35 dev pairs have 2–3 quotes. With one interval it equals `span_hit`. |
| `span_recall` | \|E ∩ R\| / \|E\| |
| `precision` | \|E ∩ R\| / \|R\| |
| `iou` | \|E ∩ R\| / \|E ∪ R\| |
| `mrr` | 1 / rank of the first chunk overlapping E, 0 if none in the top k (the mean is MRR@k) |
| `context_tokens` | sum of n_tokens of the top k; overlapping chunks count fully, since the generator reads both |

**On the dev docs** (35 questions, 2026-10-11):
- **The embedders:** at k=5, `span_hit` is 0.89 (papers) and 0.88 (reports) for bge-m3, and 0.90 and 0.91 for Qwen3. `span_recall` is 0.85–0.89.
- **The baselines:** random is about 0, and the oracle needs 1.1–1.2 chunks on average (at most 2).
- **Overlap in the oracle:** it shows the expected effect. The share of questions whose evidence needs two chunks drops with overlap: sentence, papers from 27% to 0%; character, reports from 25% to 10%.
- **Too few questions to rank chunkers:** only 4 of the 35 questions change their `span_hit@5` across the 6 configurations, so the dev set can't separate the methods. That needs the full question set and the paired CIs of `report`.

## RAG answers (`answer`)

```bash
uv run chunkeval answer                       # every corpus; asks the generator only what isn't cached
uv run chunkeval answer --corpus papers
```

**What it does** (`evaluation/answering.py`). Every question of `filtered.jsonl` is answered by `llm.rag_generator` (qwen3.5:9b-q8_0, temperature 0, thinking off), with the prompt in [prompts/rag_answer.md](prompts/rag_answer.md):
- **In each of the 6 setups of its corpus,** from the top `retrieval.top_k` (5) chunks that bge-m3 (`embedding.primary`) retrieved, in rank order. Qwen3-Embedding stays a retrieval-only check. There is no budget-matched variant: the 1,300-token budget fits 4.2–5.0 chunks on average, so it would nearly repeat the k=5 run.
- **From its gold passage** (setup `gold_passage`): the passage the question was generated from, which holds all its evidence. This is perfect retrieval, one answer per question for all setups. Stage 8 scores it like a setup, as the **ceiling** of the generation metrics:
  - setup − gold passage is what retrieval with that chunker costs, and 1 − gold passage is the generator's and the judge's own error;
  - a metric on which the gold passage doesn't beat the setups doesn't respond to retrieval (the previous round's relevancy was ~0.54 everywhere);
  - its context recall should be ~1 by construction, so the shortfall measures the judge's error rate;
  - a question answered wrongly even from its own passage points to a bad gold answer: list those for the audit.

  It is generous: one clean passage without distractors, which the same Qwen model wrote the question from. Not to be confused with the retrieval `oracle` (each setup's chunks that cover the evidence), whose ceiling stage 6 already measures for free.

**The prompt** (decided 2026-10-11; its notes explain each rule):
- **Chunk text only.** No document title or section path: section paths would favour the recursive chunker, and Ragas must judge faithfulness against exactly what the generator read.
- **Complete first, then concise:** every detail asked, with numbers and units as given, and nothing else. No reference to "the context" or passage numbers in the answer.
- **No citations** (see [Future work](#future-work)).

**Abstentions.** Only when no passage holds any of the information asked for, the generator must reply exactly "The context does not contain the answer." When the passages hold part of it, it gives that part and says nothing about what is missing (a first version that named the missing part backfired on the dev docs; see the prompt's notes). `is_abstention` marks an answer that starts with the phrase, or that is one sentence paraphrasing it ("The provided context does not mention…"); a paraphrase followed by facts counts as an answer. Why this matters:
- **Ragas 0.2.15 scores an abstention like a wrong answer:** answer relevancy 0 (noncommittal) and factual correctness 0. Its faithfulness comes out at random (NaN, 0 or 1, depending on what the judge extracts). Context precision and recall don't read the answer.
- **So stage 8 handles them in code** (planned in `ragas_eval.py`): relevancy and factual correctness 0 without asking the judge; faithfulness left out for abstentions and reported next to the abstention rate, plus a view with abstentions = 1. Abstentions are never dropped from every metric, because the paired comparison needs the same questions in every setup.
- **The summary separates the two kinds of miss** with `evidence_shown` (some passage overlaps the gold evidence, i.e. span_hit at top_k): `abstained_with_evidence` (an over-cautious generator, or evidence split so it looked incomplete) and `answered_without_evidence` (a hallucination risk, unless another document holds the same fact). On the gold passage the abstention rate should be about 0.

**What it writes:**
- **`runs/<run_id>/answers/<corpus>_<setup>.jsonl`:** one `Answer` per question, in question order. `<setup>` is a variant name (`character_no_overlap`, …) or `gold_passage`. Each answer holds the `response` and the `contexts` it was shown, in rank order, which are Ragas' `response` and `retrieved_contexts`, plus the chunk ids, `evidence_shown`, `abstained`, `error`, the model and the token counts. A `.provenance.json` next to each file records the questions hash, the retrieval file's provenance, `top_k`, the generator (with its Ollama digest) and the prompt file's hash.
- **`runs/<run_id>/answer_cache.jsonl`:** the generator's answers, one line per request, appended as they arrive. It is keyed by a hash of the messages and the generator (model, settings, digest, seed, `ANSWER_VERSION`), so a re-run asks only what is new: a new question, a question shown other chunks, or everything after a prompt or generator change. Identical requests (two setups showing the same chunks) are asked once. An answer the generator couldn't finish (cut off at `max_tokens`, context full) is cached with its `error` and has an empty response. API errors stop the stage before any answer file is written; a re-run asks exactly those.
- **`runs/<run_id>/answer_summary.json`:** per corpus and setup: answered, abstained, errors, the abstention rate, the two miss counts above, and mean answer words and token counts.

**Checks before any call:** the questions' documents still have the text `gen-qa` saw, and each bge-m3 retrieval file is up to date with the current questions, chunk file and query settings. A stale one stops the stage with "run `chunkeval retrieve` first", because its chunk ids could point to other text than it ranked.

**On the dev docs** (35 questions, 2026-10-11; 244 calls, as two setups showed the same chunks once; 3.5 min at 48 tokens/s, concurrency 2):
- **Size:** about 1,850 prompt tokens per answer from 5 chunks and 830 from the gold passage. Nothing was cut off at `max_tokens` 512. Answers from the chunks average 35–43 words per setup, from the gold passage 27–33; the gold answers average 22.
- **Abstentions:** 2–5 per setup, 55 of 245 in all, and all but one are the bare phrase. 3 questions abstain everywhere, gold passage included, so their abstentions say nothing about the chunkers. `answered_without_evidence` is 0–1 per setup.
- **The gold passage abstains on 7 of 35,** where about 0 was expected. Reading the passages:
  - **5 are weak pairs:** the question asks for something the passage doesn't state in that form: a "definition" whose answer is a percentage, a circular "why", a "how" the passage doesn't explain, a premise about the SDGs the passage never mentions, and a garbled question (paper_93:s19729, report_140:s27355, report_15:s17844, report_123:s73707, paper_4:s48236). The critic passed all of them.
  - **2 are an over-cautious generator:** paper_4:s24654 (the passage gives the reasons, hedged as "may be attributable to") and paper_74:s36976 ("threshold value" in the question, "more than 100 kg" in the passage).

  Three of the seven are in the labelling sample (paper_74, paper_93, report_123); check the rest when labelling. On the full corpus, list every question the gold passage abstains on for the audit.
- **The first version of the prompt** asked partial answers to name what was missing. It abstained less (42 in all, 5 on the gold passage), but 20 answers carried "the context does not say…" sentences, and several abstentions opened with the phrase, then gave the facts anyway. The prompt's notes have the details.

## Dependencies
- **Installed with uv:** all the pipeline libraries, plus pytest in the dev group, plus `chunkeval` itself in editable mode.
  - **On a fresh clone,** run `uv sync`.
  - **On the existing `.venv`,** use `uv sync --inexact`. It also holds packages that aren't in the lockfile (CUDA 12 libraries left over from the torch upgrade), and a plain `uv sync` would uninstall them.
- **ragas** is still 0.2.15. Upgrading changes its API, so read the migration notes first.
- **Ollama** serves the local LLMs. It's a system service, not a Python package; the code reaches it with the `openai` client.

## Future work

**Chunk-size sweep (`chunking.size_sweep`).** An optional extension for after the whole pipeline is complete, built only if we decide we want it. The config key exists and is validated, but no stage reads it yet.
- **What:** rechunk with all 3 methods × overlap on/off at other target sizes (e.g. `[128, 256, 512, 1024]`), index them, retrieve with the same questions, and score with the span metrics only: hit@k, span recall@k, MRR, IoU/precision, context tokens. No RAG answers and no Ragas, so no API cost.
- **Why:**
  - It checks whether the method ranking found at 256 holds at other sizes.
  - It tests whether 256 is a good size: 256 is inherited from the previous round, not tuned (see the comment in `configs/default.yaml`).
  - It measures the precision/recall trade-off against chunk size on a fixed corpus and a fixed metric. That is the controlled ablation the docx says its own size comparison lacked, since chunk size, judge and corpus all changed at once.
- **Open design points:**
  - **Compare at equal context.** At a fixed top-k, bigger chunks win recall almost automatically. Use the budget-matched view (`retrieval.budget_tokens`, scaled per size) as the main comparison.
  - **A cap per size.** 1024 is above `max_tokens: 512`, and the schema rejects a target above the cap. Use something like 2× the size per sweep size.
  - **Naming.** The extra chunk files and Weaviate collections need the size in their names, which `weaviate.collection_template` can't express yet.
  - **Weaviate memory.** Raise `GOMEMLIMIT` (see the Weaviate section).

**Citations in the RAG answers.** Considered for `answer` and left out for now (decided 2026-10-11).
- **What:** label the passages `[C1]`…`[C5]` and ask the generator to end each sentence with the passages it used. Keep the raw answer, give Ragas a copy with the markers stripped (they would be noise in faithfulness' statements and relevancy's generated questions), and store the cited chunk ids.
- **Why `C`:** 25 of 191 papers chunks and 17 of 420 reports chunks on the dev docs already contain reference markers like `[12]`, `[5–11]` or `[1, 2]`, so a plain `[3]` copied from a chunk would be read as a citation.
- **What it would give, without the judge:** whether the answer cites a chunk that overlaps the evidence; combined with `evidence_shown`, which questions had the evidence but were answered from other chunks (distraction), and which were answered from a wrong chunk (e.g. report_16 for a report_15 question, the near-duplicate case of the pilot); the rank of the chunks actually used.
- **Limits:** a 9B model's citations are approximate (wrong block, missing ones), so they are a diagnostic, not a headline metric; check ~20 by hand first. A prompt change re-asks every answer.

**Random-context check of the abstention rule.** Not implemented (decided 2026-10-11). Answer each question once more from `top_k` random chunks (the `random` retrieval files of one setup, ~35 calls on the dev docs) and count abstentions: nearly all should abstain. If many don't, the generator answers from irrelevant text and rule 7 of `prompts/rag_answer.md` isn't working. A cheap check of the prompt, best run on the dev docs after any change to it.

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
