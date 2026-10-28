# Development

## Package layout

Module paths in these docs are relative to `src/chunkeval/`. The folders follow the pipeline order:

```
src/chunkeval/
├── cli.py              entry point, one subcommand per stage (`uv run chunkeval …`)
├── core/               shared by every stage, no pipeline logic
│   ├── schema.py       pydantic models of configs/*.yaml (`Config`) and REPO_ROOT
│   ├── config.py       load_config (with `extends`) / save_config (guarded), secrets from .env, collection_name
│   ├── store.py        JSONL / Parquet persistence, resumable appends, run folders
│   ├── llm.py          one chat client for every LLM role (Ollama or OpenAI)
│   └── prompts.py      prompts/*.md split into checked `## <name>` sections
├── corpus/             stages 1–3: PDFs → indexed chunks
│   ├── parsing.py
│   ├── audit.py        the corpus checks between parse and chunk
│   ├── chunking/       base.py, character.py, sentence.py, recursive.py
│   ├── embedding.py
│   ├── indexing.py
│   └── removal.py      `remove-corpus`: everything of one corpus out of data/ and Weaviate
├── qa/                 stages 4–5: the ground-truth questions
│   ├── generation.py
│   └── filtering.py
├── evaluation/         stages 6–8
│   ├── retrieval.py
│   ├── retrieval_metrics.py
│   ├── answering.py
│   └── ragas_eval.py
└── analysis/           stage 9: the report
    ├── inputs.py       loads both tables and checks they belong together
    ├── stats.py        bootstrap, paired comparisons, the ranking rule
    ├── views.py        every table
    ├── plots.py, cases.py, report.py   figures, the case file, report.md
```

**Conventions:**
- **Always import with the full path,** for example `from chunkeval.core.store import write_jsonl` or `from chunkeval.qa.generation import QAPair`. Never use a bare `import config`, which only works by accident from inside the folder.
- **The package is installed in editable mode** by `uv sync`, through `[build-system]` in `pyproject.toml`. That's why no `PYTHONPATH=src` is needed anywhere, and why code changes take effect without reinstalling. After a fresh clone, or if `import chunkeval` fails, run `uv sync` once.
- **Relative paths** (configs, `data/`, `runs/`, `.env`) resolve against `REPO_ROOT`, the folder that holds `pyproject.toml`. It is searched upward from `core/schema.py`, so moving modules between folders can't break it.
- **`core/` is imported by every stage.** After changing anything there, run the whole test suite.
- **A new module** goes into the folder of its pipeline phase. A new subfolder needs an `__init__.py`.

## The CLI

**What the CLI does around every stage,** so stage code doesn't have to:
1. It validates the config and checks this stage's secrets, so a problem stops the run in the first second.
2. It saves the resolved config to `runs/<run_id>/config.yaml`, the record of what the run used. Paths under the repo are written relative to it, so the record still holds after the repo moves. When the run recorded one before, [the guard](#the-guard-on-a-runs-config) compares the two first.
3. It logs to the console and to `runs/<run_id>/logs/<stage>_<timestamp>.log`.
4. **For stages with local LLM roles** (`gen-qa`, `filter-qa`, `answer`): it runs `check_ollama` first, then unloads the models afterwards, even if the stage fails.
5. It calls the stage's `run(cfg, secrets, **options)` and logs how long it took. A failure is logged with its traceback and exits with code 1; Ctrl+C exits with 130.

**Adding a stage option:**
1. Register it in `cli.OPTIONS`.
2. List it in the stage's entry in `cli.STAGES`.
3. Add a keyword argument with the same name to that stage's `run`.

`tests/test_cli.py` fails if these three steps don't match.

### The guard on a run's config

A stage under a `run_id` that already recorded a config compares the two before any work (`config.save_config`). When a setting that can change a result differs, the stage **refuses**: it exits with code 1, names each changed key (`qa.critic_min_score: 4 -> 3`), and writes the refusal to the log. Otherwise the stages would replace the run's results with ones made under the new settings, without asking: `filter-qa` rewrites the question set, `retrieve` redoes stale files, `answer` and `eval` rewrite their tables. Two ways on:
- **A new experiment:** bump `run_id`. The old run stays as it was.
- **A change to this run:** re-run with `--accept-config`. The old record moves to `runs/<run_id>/config_history/config_<when it was replaced>.yaml`, so the run folder keeps every config its results were made under.

What doesn't count as a change, and is only logged:
- **Settings that can't change a result** (`config.UNGUARDED`): `paths.runs_dir`, `ollama.concurrency`, `ragas.max_workers`, `timeout`, `max_retries`, `max_cost_usd` and `prices`. A setting not on this list is guarded, so a new one is safe by default.
- **A setting the code added or removed** since the record was written: a key one side doesn't have, in a section. A new setting keeps the old behaviour. The exception is a setting added with a default and set to something else (`title_overrides`), which is a change.

A key added to a map the config fills in is a change like any other: a corpus in `paths.corpora`, a model in `embedding.models`. `--corpus` and `--limit` aren't part of the config, so a smoke test never trips the guard. Neither does `remove-corpus`: it isn't part of a run, so it neither saves nor checks a record.

`full_data-v1.0` was recorded before `extends` and the data-dir namespaces. Its first stage under this code refuses on two keys whose form changed but whose effect didn't: `embedding.cache_dir` (now derived) and `weaviate.collection_template` (now `{namespace}…`, empty for `data/`). One `--accept-config` clears both.

## Configs for other experiments

A config can start with `extends: <file>` (relative to its own folder) and hold only what differs. `load_config` reads the base first and puts the file on top:
- mappings merge key by key, at any depth;
- a list or a value replaces the base's;
- `null` removes the key, so the field's default applies (`title_overrides: null`), or leaves an entry out (`reports: null` under `paths.corpora`).

A base can extend another one. `runs/<run_id>/config.yaml` holds the merged result.

**What two configs share:**

| What | Where | Shared by |
|---|---|---|
| Parsed docs, chunks, the question set (`data/qa/`) | `paths.data_dir` | every config with that data dir |
| Chunk, query and de-duplication vectors | `embedding.cache_dir`; when unset, `<data_dir>/embeddings` | the same |
| Weaviate collections | `weaviate.collection_template`: `{namespace}` is empty for `data/`, and the data dir's name elsewhere (`data/size512` gives `Size512_papers_sentence_overlap_v2`), or `weaviate.namespace` | the same |
| Answer and judge caches | `paths.llm_cache_dir` (`data/llm_cache/`) | every config: a call is keyed by everything that decides it, so a hit is always the same call |
| Retrieval files, answers, Ragas table, reports | `runs/<run_id>/` | one run |
| Prompts | `prompts.*` | every config pointing at the file; the records hold its text's hash, not its path |

A data dir other than `data/` without `{namespace}` in the template is refused when the config loads: its `index --recreate` would replace the collections of `data/`. `weaviate.namespace: ""` shares them on purpose.

**1. Same documents, another setting.**
- **A setting used after the question set** (retrieval mode, k, the RAG generator or its prompt, the judge, the Ragas sample, the analysis): a new `run_id` is enough.
  ```yaml
  extends: default.yaml
  run_id: "full_data-hybrid-v1.0"
  retrieval: {mode: "hybrid"}
  ```
  Then `retrieve`, `answer`, `eval --dry-run`, `eval`, `report`, each with `--config`. The shared caches serve every call both runs make, for example the gold-passage answers and their judge calls.
- **A setting that builds `data/`** (parsing, chunk sizes, embedders, `gen-qa` or `filter-qa` settings): also give it its own data dir, so `data/` stays as the finished run used it.
  ```yaml
  extends: default.yaml
  run_id: "size512-v1.0"
  paths: {data_dir: "data/size512"}
  chunking: {target_tokens: 512, max_tokens: 1024}
  ```
  Copy in what doesn't change: `cp -a data/parsed data/size512/` skips Docling, and `cp -a data/qa data/size512/` keeps the same questions. The vectors and collections are its own.
- **A new chunking method** needs code: a `chunking.base.Chunker` subclass, its key in `schema.Chunking.methods` and in `chunking.CHUNKERS`, its arguments in `build_chunker`, and its report name in `analysis/inputs.py`. Then add it to `chunking.methods`. The existing variants come out unchanged, so `embed` and `index` skip them and `eval` pays only for the new setup's rows.

**2. A new corpus on its own.** Give it its own data dir, so nothing of `papers` and `reports` is read or written:
```yaml
extends: default.yaml
run_id: "theses-v1.0"
paths:
  data_dir: "data/theses"
  corpora: {papers: null, reports: null, theses: "data/corpora/theses"}
parsing: {body: {papers: null, reports: null}, title_overrides: null}
```
Run every stage with `--config` on it. It gets its own question set (its own generator, critic and thresholds if wanted), its own labelling sample and its own report. It shares the Weaviate server (under its own collection names), Ollama, the GPU and the LLM caches.

**3. A new corpus with some or all of the existing ones.** Keep `data/`, so the corpora share one question set, made the same way:
1. Add the corpus to `paths.corpora` (a `parsing.body` rule is optional). Give its files their own prefix: question ids are `<doc_id>:<unit_id>`, so `parse` refuses a document id another corpus already has.
2. Run `parse` through `gen-qa` with `--corpus <new>`, then `filter-qa --corpus <new>`. The other corpora's lines in `filtered.jsonl` and `rejected.jsonl` are carried over unchanged.
3. Use a new `run_id`, then run `retrieve`, `answer`, `eval` and `report`. The shared caches serve papers and reports, so the judge pays only for the new corpus (check with `eval --dry-run`).

For only some of the existing corpora, an overlay leaves the others out (`paths.corpora: {reports: null}` and `parsing.body: {reports: null}`), with its own `run_id`: under the old one, `retrieve` and `eval` would rewrite that run's tables without them. `gen-qa` and `filter-qa` record their settings per corpus, and the report warns when the corpora of a run were made with different ones.

A corpus left out of a config is still in `data/`: `filter-qa` carries its lines over, its collections stay, and `parse` keeps its doc ids reserved. Adding it back costs nothing.

### Removing a corpus for good

```bash
uv run chunkeval remove-corpus reports         # lists what it would remove; removes nothing
uv run chunkeval remove-corpus reports --yes   # removes it
```

**What goes** (`corpus/removal.py`), from the config's data dir and Weaviate:
- `data/parsed/<corpus>/` and `data/chunks/<corpus>/`;
- its vectors: `data/embeddings/<embedder>/<corpus>/`, `queries/<embedder>/<corpus>.*` and `dedup/<embedder>/<corpus>.*`;
- its lines in `raw.jsonl`, `gen_dropped.jsonl`, `filtered.jsonl` and `rejected.jsonl`, and its entries in both provenance files of `data/qa/`. The other corpora's lines stay as they are;
- its Weaviate collections, for every chunking method and overlap under the data dir's namespace, but only those whose provenance says `index` built them from its chunks.

Its doc ids are then free again.

**What stays, on purpose:**
- **The inputs:** its PDFs, the configs and `configs/titles.csv`. The command edits no input, so everything it removes can be rebuilt from them. Delete `data/corpora/<corpus>/` and its rows in `rename_map.csv` and `titles.csv` yourself if you want them gone too.
- **`runs/`:** the results of finished runs, the record of what was measured.
- **The caches keyed by content** (`data/llm_cache/`, `data/qa/critic_scores.jsonl`): an entry only ever serves the identical call, so a leftover can't affect another corpus. If the corpus comes back, nothing is asked or paid for again.

**Refused** while any config with the same data dir still lists the corpus (the one given, or any of `configs/*.yaml`): the next `parse` would bring it back, or another experiment still needs it. Take it out of `paths.corpora` and `parsing.body` of each first. A name with nothing to remove is refused too (a typo). Weaviate must be running, so the list is complete before anything is removed. The command doesn't overwrite the run's `config.yaml`, and it can be run again after a crash.

## Testing

The tests import `chunkeval` as the installed package, so run `uv sync` once (see [Package layout](#package-layout)). pytest is in the dev dependency group, which `uv sync` installs too.

```bash
uv run pytest                                   # everything (needs Ollama, a free GPU, Weaviate and the OpenAI key)
uv run pytest -m "not ollama and not openai and not gpu and not weaviate"   # offline tests only (~45 s): run after every change
uv run pytest -m gpu                            # embedding models on the GPU (~10 s): run after changing embedding.py or the models
uv run pytest -m weaviate                       # the local Weaviate (~15 s): run after changing indexing.py or the Weaviate version
uv run pytest -m ollama                         # live Ollama tests only: run after upgrading Ollama or changing models
uv run pytest -m openai                         # OpenAI key + judge model (free): run before an eval run
uv run pytest tests/test_store.py -v            # one file, one line per test
uv run pytest -k crash                          # tests whose name contains "crash"
```

`.venv/bin/python -m pytest …` works the same way, without uv.

**What is covered:**

| File | Tests | Needs |
|---|---|---|
| [tests/test_config.py](../tests/test_config.py) | Config loading (from any working directory) and every validation rule: unknown keys, types, ranges, `TODO` placeholders, cross-section references, missing prompt files, a namespace that isn't a name, another data dir without `{namespace}`. `extends`: mappings merge, lists and values replace, `null` removes a key, the base is found next to the file, chains work, a cycle is refused. Another data dir gets its own vector cache and collection names but shares the LLM caches. Also the `save_config` round trip (paths under the repo saved relative), its guard (a changed setting refused until accepted, the old record kept; the unguarded settings; a setting the code added or removed only logged, unless added at another value than its default; a corpus added refused; an older absolute record the same config), secrets (masking, per-stage requirements), and collection names. | nothing |
| [tests/test_store.py](../tests/test_store.py) | Atomic `write_jsonl`, crash-tolerant `append_jsonl` / `done_ids` (the resume mechanism), corruption detection, `filter_jsonl` (one corpus out of a shared file), `write_table` (parquet, duplicate keys, CSV fallback) and `run_dir`. | nothing |
| [tests/test_llm.py](../tests/test_llm.py), offline | `ChatLLM` setup per provider. `chat` and `chat_many` logic against a fake client: the parameters sent, every `LLMOutputError` case, result order, errors returned in place, concurrency limit, and Ctrl+C cancellation. | nothing |
| [tests/test_llm.py](../tests/test_llm.py), marker `ollama` | Preflight (`check_ollama`), JSON answers from every local role, determinism, resume after an interrupted run, and unloading. Also pins the Ollama behaviour the code relies on: `reasoning_effort="none"` turns Qwen's thinking off, and too-long prompts are truncated silently (and caught by `chat`). Generators must be 100% on the GPU (for Gemma, the derived `gemma4-allgpu` tag). | Ollama running, models pulled, `OLLAMA_BASE_URL` |
| [tests/test_cli.py](../tests/test_cli.py) | Every stage in `cli.STAGES` points to a `run` that accepts the options the CLI passes. Argument parsing, and what `main` does around a stage: config saved, log file written, exit codes 1 and 130, missing secrets stop the run before any work, a changed config stops it too (logged) until `--accept-config`, and the Ollama check before and unload after an LLM stage, even when it fails. `remove-corpus` needs the corpus named (no default) and doesn't overwrite the run's `config.yaml`. | nothing |
| [tests/test_removal.py](../tests/test_removal.py) | `remove-corpus` on two corpora's artefacts with a fake Weaviate: without `--yes` nothing changes; with it, the corpus's folders, vectors, lines, provenance entries and collections go, while the other corpus's files stay byte for byte, a collection with its name but built from other chunks stays, and its doc ids are free again. Refused while the config given, or a `configs/*.yaml` with the same data dir, lists it (not one with another data dir); a misspelt name finds nothing. | nothing |
| [tests/test_llm.py](../tests/test_llm.py), marker `openai` | The API key is valid and the judge model still exists. It only calls a free metadata endpoint, so there's **no cost**. | `OPENAI_API_KEY` |
| [tests/test_parsing.py](../tests/test_parsing.py) | Body selection on hand-built documents: papers from the Introduction (or Background, or Results first) to the end markers, a paper without anchors kept whole, an end marker before the start ignored; reports' dropped sections with their subsections, never more than `DROP_MAX_SHARE` of the document, title prefixes ("Annex 2: …"); lead labels (a parenthesis isn't one, the first quarter is front matter). `title_matches`, `ocr_helps`, and the CanonicalDoc invariants: offsets within the text, blocks sorted and disjoint, sections containing their children, page furniture dropped. The doc-id check: distinct names pass, and a PDF named like another corpus's PDF, parsed document (also of a corpus left out of the config) or document with Q-A records is refused before anything is parsed. | nothing |
| [tests/test_audit.py](../tests/test_audit.py) | Each audit check on hand-made text: the language of a block (French, Portuguese, English, an e-mail domain), the title flags (author list, line numbers, banner, short, long, shared, the doc id, and what must pass: "Deliverable 4.4.1.1", "TOLERANCE1", years), garbled and other-language shares, near-duplicates. The `audit` stage on a small parsed corpus: both files written and `data/` untouched; byte-identical PDFs, French and garbled documents warned; an overridden title not listed; missing parsed docs or word lists stop it with the fix. | nothing (a fake token counter and word list) |
| [tests/test_chunking.py](../tests/test_chunking.py) | The chunk invariants on all 6 configurations: `text == doc.text[start:end]`, full coverage, `max_tokens`, disjoint spans without overlap, the overlap budget. Also sentence boundaries around tricky abbreviations, the section chunker's groups, the shared size rules (`pack`, `overlap_start`), each way `validate_chunks` fails, and the `chunk` stage's output files. When `data/parsed/` exists, the 6 configurations also run on the first 2 parsed documents of each corpus with the config sizes. | the bge-m3 tokenizer (~20 MB, downloaded on first use) |
| [tests/test_embedding.py](../tests/test_embedding.py), offline | The vector cache and the `embed` stage with a fake embedder. Each model is loaded once and only when a file needs it. A re-run embeds only changed chunk files: the cache compares content hashes, not mtimes, and a changed query prompt doesn't count. A new model revision re-embeds that model's files. A crashed run resumes. The loader refuses vectors whose chunk_ids differ (order included) or whose settings are stale. The Ollama guard names the loaded models and passes when Ollama is unreachable. | nothing |
| [tests/test_embedding.py](../tests/test_embedding.py), marker `gpu` | Both real models: loaded at the pinned revision; float32 unit-norm output; Qwen3's instruction applied to queries only; two different vector spaces; a 3-passage retrieval sanity check; the same vector with and without padding (pooling + padding side); too-long input raises instead of being truncated; every chunk in `data/chunks/` fits each model's `max_seq_length`. | a CUDA GPU with no Ollama model loaded; the models (~4.5 GB bge-m3, ~1.2 GB Qwen3, downloaded on first use) |
| [tests/test_indexing.py](../tests/test_indexing.py), offline | Every chunk field but `meta` is a property, and only `text` is searchable. `--limit` takes the first documents by doc_id, with their own vectors. Missing chunks or vectors name the stage to run. Provenance changes with the chunks, the embedder settings and `--limit`. | nothing |
| [tests/test_indexing.py](../tests/test_indexing.py), marker `weaviate` | On 6 fake chunks with hash vectors: the collection gets both named vectors (flat, cosine) and its provenance, and every object matches its chunk and cached vectors. A chunk's vector finds that chunk first in its own space. doc_id filters match whole ids, and BM25 ignores section titles. A re-run skips an up-to-date collection and refuses a stale, incomplete or hand-made one until `--recreate`. Verification catches swapped names and shifted rows, and failed inserts raise. | Weaviate running (`docker compose up -d`), `WEAVIATE_API_KEY`; touches only `*_pytest` collections |
| [tests/test_generation.py](../tests/test_generation.py) | Unit sampling on `tiny_doc` with small sizes: single units stay in one section and within the size bounds; sections that are too small get no unit; a paragraph over the max is cut into sentence groups; an over-long sentence is in no unit; two-span units are adjacent whole paragraphs (also across a section boundary); skippable sections are never used; the per-doc cap and the two-span ratio hold, and no two units overlap. The prompt file has exactly its sections and placeholders, checked on load: a missing section, a misspelt placeholder or a lone `$` fails. The chat messages: one system message shared by every call, then the title, section path, passage and question type in the user message, with passages A and B for a two-span unit. Which outputs `QA_JSON_SCHEMA` accepts (pairs, skips) and rejects, and that its question types are the config's. The generation loop against a fake LLM: pairs with located evidence, every drop reason, a two-span answer quoting one passage only, the factoid retry (single units only), the type rotation. The `gen-qa` stage: resume after a crash asks only the missing unit; drops are saved but API errors aren't (and the stage fails); other settings, another prompt text (a copy elsewhere is the same prompt), a re-parsed document or records without provenance are refused until `--recreate`. Settings are per corpus: a second corpus can use others, each corpus is checked against its own, and `--recreate` deletes only the selected corpora's records. A provenance file in the old format (one `settings` for every corpus) resumes. `load_pairs` reads back exactly what the stage wrote. When `data/parsed/` exists, the same unit invariants on every parsed document with the config sizes (~25 s on the full corpus), reproducible by seed; a document with enough running text (quota × `unit_max_tokens`) fills the quota, spread over the document. | the bge-m3 tokenizer |
| [tests/test_filtering.py](../tests/test_filtering.py) | Each deterministic check on its pattern, case-insensitively, and on what it must let through ("in the context of <topic>", "farmers had no information on prices", a number spelt out in the evidence). The critic's two calls: the passage call shows the evidence as located in the text, the question call only the question, and the system messages are shared. Which outputs each call's schema accepts (note first, reason before score) and rejects. `decide` ignores the notes, and the threshold applies per criterion. De-duplication: greedy, earlier wins, within a corpus only. The `filter-qa` stage against a fake critic and embedder: each pair ends at the right stage with its reasons; a re-run or a new threshold asks nothing; a changed answer asks only its passage call; another critic setting asks everything; an API error writes nothing, and the re-run asks only what is missing; an unusable answer rejects its pair and is cached; a re-parsed document is refused; the labelling sample is blind and never overwritten. `--corpus` leaves the other corpora's lines byte for byte as they were, records each corpus's filter settings, and flags a carried corpus whose pairs in `raw.jsonl` changed. Each question is encoded once for the de-duplication: a re-run encodes nothing, a changed question only itself. | nothing |
| [tests/test_retrieval_metrics.py](../tests/test_retrieval_metrics.py) | Every span metric on hand-computed toy cases: interval merging and overlap; a hit equal to the evidence scores 1 everywhere; the same offsets in another document score 0; overlapping hits and other documents' text count once in precision and IoU; `all_evidence` needs every quote touched (touching quotes are one); MRR and hit only see the top k; `budget_k` never returns 0. | nothing |
| [tests/test_retrieval.py](../tests/test_retrieval.py), offline | The oracle's greedy cover (hand-computed) and the random baseline's seeding. Query vectors: encoded with the query prompt, cached, encoded again after a new question or prompt. The `retrieve` stage against a fake embedder and retriever: a file per retriever with ranked hits and no text, then the metrics table (one row per question, retriever and k, unique). A re-run asks nothing; a new query prompt redoes only that embedder; a new question set redoes everything; a mode switch is refused; a question's document without chunks, or chunks cut from another parse, stop the stage. The metrics leave out stale files outside the run's scope, with a warning, and fail on stale files inside it. | nothing |
| [tests/test_retrieval.py](../tests/test_retrieval.py), marker `weaviate` | `WeaviateRetriever` on a `*_pytest` collection: a chunk's own vector finds it first with cosine 1, in the named vector asked; an unknown vector name fails; hybrid with alpha 0 is BM25; a collection built from other chunks, or a missing one, is refused. | Weaviate running, `WEAVIATE_API_KEY` |
| [tests/test_answering.py](../tests/test_answering.py) | The prompt file must contain the abstention phrase the code looks for. The messages: one shared system message, then the passages numbered in rank order, each with its document's title and text, then the question; the contexts Ragas reads carry the same title line. `is_abstention` on the phrase (case, markup, an explanation after it), paraphrases alone, and what must stay an answer (a partial answer, a paraphrase followed by facts, "the soil does not contain…"). The `answer` stage on the offline `retrieve` setup against a fake generator: one file per setup and one for the gold passage, each answer holding exactly the top-k chunks it was shown (or its own paragraph) and the evidence flag; the summary counts. A re-run asks nothing, a dropped and restored question asks nothing, another generator setting asks everything, a retitled document asks its passages again and changes the recorded titles hash; the titles cover documents without questions; a stale retrieval file stops the stage before any call; an API error writes no answer file and the re-run asks only it; an unusable answer is kept with its error and cached. | nothing |
| [tests/test_ragas_eval.py](../tests/test_ragas_eval.py) | Each of the 5 Ragas metrics run by Ragas itself (`single_turn_ascore`) and by the stage's per-call pipeline on the same fake judge: the same prompts and the same scores. Factual correctness checks claims with faithfulness' prompt, so the two share calls. The judge request (the config's model, temperature and seed) and each call's tokens and cost. The sample is nested and in file order; the text metrics; telemetry is off. The `eval` stage against a fake judge and embedder: a row per question and setup, one of them hand-computed; abstentions scored in code and never sent; no prompt sent twice; a re-run asks nothing, a new answer asks only its response calls; an API error writes nothing and the re-run asks only it; an unreadable output is cached after Ragas' 3 fix-up requests and scores NaN; a dry run and an estimate above the cap ask nothing; the cap stops a run whose estimate was too low; the full run reuses the pilot's calls; a corpus outside `--corpus` comes from the cache alone, or is left out with a warning when a call is missing; stale answers are refused, and so are answers made with other document titles. | nothing |
| [tests/test_analysis.py](../tests/test_analysis.py) | The statistics on hand-computed cases:<br>• the bootstrap drops and counts NaN and is seeded per cell;<br>• the document-level interval is wider when one document carries the effect;<br>• paired differences align by question and drop NaN pairs;<br>• Holm by hand;<br>• the ranking rule needs both the interval and Holm (the dev example passes neither);<br>• "few documents" is flagged when the effect sits in one of 29 documents;<br>• the smallest detectable difference.<br><br>Also each answer's outcome and the tercile labels.<br><br>The `report` stage on the offline pipeline (one variant, 3 questions):<br>• it writes every file;<br>• every row lands in exactly one outcome;<br>• each run writes a new folder (`-2` within the same second), keeps the earlier one and moves `latest`;<br>• without a Ragas table, the report covers retrieval only;<br>• a stale evidence flag, `details.jsonl` or retrieval table is refused with the stage to re-run;<br>• the run record names each corpus's `gen-qa` and `filter-qa` settings, and the report warns when they differ between corpora.<br><br>When the run has a Ragas table, the stage also runs on a copy of it, checked on invariants only. | nothing |
| [tests/test_quote_locator.py](../tests/test_quote_locator.py) | `locate_quote`, the source of every evidence span. Exact matches, and normalized ones whose offsets still point into the original text: whitespace and paragraph breaks, typographic dashes and quotes, soft hyphens, case, an added full stop, line-break hyphenation. Fuzzy matches only above the threshold, widened to whole words, never against a span shorter than the quote. Only the unit's spans are searched, and too-short quotes are rejected. When `data/parsed/` exists, a random quote from every block of every parsed document, verbatim and with that drift (a quote occurring twice in its block is found at its first occurrence; an invisible character at its edge is left out of the span). | nothing |

**Good to know:**
- **Live tests skip themselves** when their service isn't available. If Ollama isn't running, `uv run pytest` still passes, and the live tests show as skipped instead of failing.
- **Tests never touch real data.** The `cfg` fixture points `runs_dir` and `llm_cache_dir` to a temporary folder, and the stage tests point `data_dir` there too (the vector caches follow it), so nothing is written to `runs/` or `data/`.
- **The `ollama` tests load the models onto the GPU** (about 20 s) and unload them at the end, even when a test fails. Don't run them while embedding is using the GPU.
- **The `gpu` tests load both embedders once per test module** and unload them at its end, so they're gone before the `ollama` tests run. They skip when there's no CUDA GPU or when Ollama holds a model.
- **A generator must be 100% on the GPU.** `test_live_check_ollama` fails if `qa_generator` or `rag_generator` is partly offloaded to the CPU. If you switch to one of the MoE generators, which offload by design, relax that assertion.
- **Writing new tests:**
  - Use the fixtures from `conftest.py`: `cfg` (a validated config) and `raw_config` (the YAML as a dict, for building invalid configs).
  - Mark tests that need a service: `@pytest.mark.ollama`, `@pytest.mark.openai`, `@pytest.mark.weaviate`, or `@pytest.mark.gpu` for embedding models.
  - The markers are registered in `pyproject.toml`.

## Dependencies

- **Declared directly** (2026-10-12): `numpy`, `pandas`, `pyarrow`, `scipy` and `tabulate`, which `analysis/` imports and which came only through other packages before. They were added with `uv add --no-sync` and installed with `uv sync --inexact`, which kept the extra CUDA libraries.
- **Installed with uv:** all the pipeline libraries, plus pytest in the dev group, plus `chunkeval` itself in editable mode.
  - **On a fresh clone,** run `uv sync`.
  - **On the existing `.venv`,** use `uv sync --inexact`. It also holds packages that aren't in the lockfile (CUDA 12 libraries left over from the torch upgrade), and a plain `uv sync` would uninstall them.
- **ragas** is still 0.2.15. Upgrading changes its API, so read the migration notes first. `eval` uses its prompts, parsers and scoring code directly, and caches every judge call under the Ragas version, so an upgrade re-asks (and re-pays) every call; `tests/test_ragas_eval.py` checks the stage still matches Ragas' own metrics. Ragas 0.2.x sends usage data to its developers unless `RAGAS_DO_NOT_TRACK` is set: `ragas_eval.py` sets it before importing ragas.
- **Ollama** serves the local LLMs. It's a system service, not a Python package; the code reaches it with the `openai` client.
