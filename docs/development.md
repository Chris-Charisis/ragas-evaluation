# Development

## Package layout

Module paths in these docs are relative to `src/chunkeval/`. The folders follow the pipeline order:

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
│   ├── audit.py        the corpus checks between parse and chunk
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
2. It saves the resolved config to `runs/<run_id>/config.yaml`, the record of what the run used.
3. It logs to the console and to `runs/<run_id>/logs/<stage>_<timestamp>.log`.
4. **For stages with local LLM roles** (`gen-qa`, `filter-qa`, `answer`): it runs `check_ollama` first, then unloads the models afterwards, even if the stage fails.
5. It calls the stage's `run(cfg, secrets, **options)` and logs how long it took. A failure is logged with its traceback and exits with code 1; Ctrl+C exits with 130.

**Adding a stage option:**
1. Register it in `cli.OPTIONS`.
2. List it in the stage's entry in `cli.STAGES`.
3. Add a keyword argument with the same name to that stage's `run`.

`tests/test_cli.py` fails if these three steps don't match.

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
| [tests/test_config.py](../tests/test_config.py) | Config loading (from any working directory) and every validation rule: unknown keys, types, ranges, `TODO` placeholders, cross-section references. Also the `save_config` round trip, secrets (masking, per-stage requirements), and collection names. | nothing |
| [tests/test_store.py](../tests/test_store.py) | Atomic `write_jsonl`, crash-tolerant `append_jsonl` / `done_ids` (the resume mechanism), corruption detection, `write_table` (parquet, duplicate keys, CSV fallback) and `run_dir`. | nothing |
| [tests/test_llm.py](../tests/test_llm.py), offline | `ChatLLM` setup per provider. `chat` and `chat_many` logic against a fake client: the parameters sent, every `LLMOutputError` case, result order, errors returned in place, concurrency limit, and Ctrl+C cancellation. | nothing |
| [tests/test_llm.py](../tests/test_llm.py), marker `ollama` | Preflight (`check_ollama`), JSON answers from every local role, determinism, resume after an interrupted run, and unloading. Also pins the Ollama behaviour the code relies on: `reasoning_effort="none"` turns Qwen's thinking off, and too-long prompts are truncated silently (and caught by `chat`). Generators must be 100% on the GPU (for Gemma, the derived `gemma4-allgpu` tag). | Ollama running, models pulled, `OLLAMA_BASE_URL` |
| [tests/test_cli.py](../tests/test_cli.py) | Every stage in `cli.STAGES` points to a `run` that accepts the options the CLI passes. Argument parsing, and what `main` does around a stage: config saved, log file written, exit codes 1 and 130, missing secrets stop the run before any work, and the Ollama check before and unload after an LLM stage, even when it fails. | nothing |
| [tests/test_llm.py](../tests/test_llm.py), marker `openai` | The API key is valid and the judge model still exists. It only calls a free metadata endpoint, so there's **no cost**. | `OPENAI_API_KEY` |
| [tests/test_parsing.py](../tests/test_parsing.py) | Body selection on hand-built documents: papers from the Introduction (or Background, or Results first) to the end markers, a paper without anchors kept whole, an end marker before the start ignored; reports' dropped sections with their subsections, never more than `DROP_MAX_SHARE` of the document, title prefixes ("Annex 2: …"); lead labels (a parenthesis isn't one, the first quarter is front matter). `title_matches`, `ocr_helps`, and the CanonicalDoc invariants: offsets within the text, blocks sorted and disjoint, sections containing their children, page furniture dropped. | nothing |
| [tests/test_audit.py](../tests/test_audit.py) | Each audit check on hand-made text: the language of a block (French, Portuguese, English, an e-mail domain), the title flags (author list, line numbers, banner, short, long, shared, the doc id, and what must pass: "Deliverable 4.4.1.1", "TOLERANCE1", years), garbled and other-language shares, near-duplicates. The `audit` stage on a small parsed corpus: both files written and `data/` untouched; byte-identical PDFs, French and garbled documents warned; an overridden title not listed; missing parsed docs or word lists stop it with the fix. | nothing (a fake token counter and word list) |
| [tests/test_chunking.py](../tests/test_chunking.py) | The chunk invariants on all 6 configurations: `text == doc.text[start:end]`, full coverage, `max_tokens`, disjoint spans without overlap, the overlap budget. Also sentence boundaries around tricky abbreviations, the section chunker's groups, the shared size rules (`pack`, `overlap_start`), each way `validate_chunks` fails, and the `chunk` stage's output files. When `data/parsed/` exists, the 6 configurations also run on the first 2 parsed documents of each corpus with the config sizes. | the bge-m3 tokenizer (~20 MB, downloaded on first use) |
| [tests/test_embedding.py](../tests/test_embedding.py), offline | The vector cache and the `embed` stage with a fake embedder. Each model is loaded once and only when a file needs it. A re-run embeds only changed chunk files: the cache compares content hashes, not mtimes, and a changed query prompt doesn't count. A new model revision re-embeds that model's files. A crashed run resumes. The loader refuses vectors whose chunk_ids differ (order included) or whose settings are stale. The Ollama guard names the loaded models and passes when Ollama is unreachable. | nothing |
| [tests/test_embedding.py](../tests/test_embedding.py), marker `gpu` | Both real models: loaded at the pinned revision; float32 unit-norm output; Qwen3's instruction applied to queries only; two different vector spaces; a 3-passage retrieval sanity check; the same vector with and without padding (pooling + padding side); too-long input raises instead of being truncated; every chunk in `data/chunks/` fits each model's `max_seq_length`. | a CUDA GPU with no Ollama model loaded; the models (~4.5 GB bge-m3, ~1.2 GB Qwen3, downloaded on first use) |
| [tests/test_indexing.py](../tests/test_indexing.py), offline | Every chunk field but `meta` is a property, and only `text` is searchable. `--limit` takes the first documents by doc_id, with their own vectors. Missing chunks or vectors name the stage to run. Provenance changes with the chunks, the embedder settings and `--limit`. | nothing |
| [tests/test_indexing.py](../tests/test_indexing.py), marker `weaviate` | On 6 fake chunks with hash vectors: the collection gets both named vectors (flat, cosine) and its provenance, and every object matches its chunk and cached vectors. A chunk's vector finds that chunk first in its own space. doc_id filters match whole ids, and BM25 ignores section titles. A re-run skips an up-to-date collection and refuses a stale, incomplete or hand-made one until `--recreate`. Verification catches swapped names and shifted rows, and failed inserts raise. | Weaviate running (`docker compose up -d`), `WEAVIATE_API_KEY`; touches only `*_pytest` collections |
| [tests/test_generation.py](../tests/test_generation.py) | Unit sampling on `tiny_doc` with small sizes: single units stay in one section and within the size bounds; sections that are too small get no unit; a paragraph over the max is cut into sentence groups; an over-long sentence is in no unit; two-span units are adjacent whole paragraphs (also across a section boundary); skippable sections are never used; the per-doc cap and the two-span ratio hold, and no two units overlap. The prompt file has exactly its sections and placeholders, checked on load: a missing section, a misspelt placeholder or a lone `$` fails. The chat messages: one system message shared by every call, then the title, section path, passage and question type in the user message, with passages A and B for a two-span unit. Which outputs `QA_JSON_SCHEMA` accepts (pairs, skips) and rejects, and that its question types are the config's. The generation loop against a fake LLM: pairs with located evidence, every drop reason, a two-span answer quoting one passage only, the factoid retry (single units only), the type rotation. The `gen-qa` stage: resume after a crash asks only the missing unit; drops are saved but API errors aren't (and the stage fails); other settings, another prompt, a re-parsed document or records without provenance are refused until `--recreate`. `load_pairs` reads back exactly what the stage wrote. When `data/parsed/` exists, the same unit invariants on every parsed document with the config sizes (~25 s on the full corpus), reproducible by seed; a document with enough running text (quota × `unit_max_tokens`) fills the quota, spread over the document. | the bge-m3 tokenizer |
| [tests/test_filtering.py](../tests/test_filtering.py) | Each deterministic check on its pattern, case-insensitively, and on what it must let through ("in the context of <topic>", "farmers had no information on prices", a number spelt out in the evidence). The critic's two calls: the passage call shows the evidence as located in the text, the question call only the question, and the system messages are shared. Which outputs each call's schema accepts (note first, reason before score) and rejects. `decide` ignores the notes, and the threshold applies per criterion. De-duplication: greedy, earlier wins, within a corpus only. The `filter-qa` stage against a fake critic and embedder: each pair ends at the right stage with its reasons; a re-run or a new threshold asks nothing; a changed answer asks only its passage call; another critic setting asks everything; an API error writes nothing, and the re-run asks only what is missing; an unusable answer rejects its pair and is cached; a re-parsed document is refused; the labelling sample is blind and never overwritten. | nothing |
| [tests/test_retrieval_metrics.py](../tests/test_retrieval_metrics.py) | Every span metric on hand-computed toy cases: interval merging and overlap; a hit equal to the evidence scores 1 everywhere; the same offsets in another document score 0; overlapping hits and other documents' text count once in precision and IoU; `all_evidence` needs every quote touched (touching quotes are one); MRR and hit only see the top k; `budget_k` never returns 0. | nothing |
| [tests/test_retrieval.py](../tests/test_retrieval.py), offline | The oracle's greedy cover (hand-computed) and the random baseline's seeding. Query vectors: encoded with the query prompt, cached, encoded again after a new question or prompt. The `retrieve` stage against a fake embedder and retriever: a file per retriever with ranked hits and no text, then the metrics table (one row per question, retriever and k, unique). A re-run asks nothing; a new query prompt redoes only that embedder; a new question set redoes everything; a mode switch is refused; a question's document without chunks, or chunks cut from another parse, stop the stage. The metrics leave out stale files outside the run's scope, with a warning, and fail on stale files inside it. | nothing |
| [tests/test_retrieval.py](../tests/test_retrieval.py), marker `weaviate` | `WeaviateRetriever` on a `*_pytest` collection: a chunk's own vector finds it first with cosine 1, in the named vector asked; an unknown vector name fails; hybrid with alpha 0 is BM25; a collection built from other chunks, or a missing one, is refused. | Weaviate running, `WEAVIATE_API_KEY` |
| [tests/test_answering.py](../tests/test_answering.py) | The prompt file must contain the abstention phrase the code looks for. The messages: one shared system message, then the passages numbered in rank order, each with its document's title and text, then the question; the contexts Ragas reads carry the same title line. `is_abstention` on the phrase (case, markup, an explanation after it), paraphrases alone, and what must stay an answer (a partial answer, a paraphrase followed by facts, "the soil does not contain…"). The `answer` stage on the offline `retrieve` setup against a fake generator: one file per setup and one for the gold passage, each answer holding exactly the top-k chunks it was shown (or its own paragraph) and the evidence flag; the summary counts. A re-run asks nothing, a dropped and restored question asks nothing, another generator setting asks everything, a retitled document asks its passages again and changes the recorded titles hash; the titles cover documents without questions; a stale retrieval file stops the stage before any call; an API error writes no answer file and the re-run asks only it; an unusable answer is kept with its error and cached. | nothing |
| [tests/test_ragas_eval.py](../tests/test_ragas_eval.py) | Each of the 5 Ragas metrics run by Ragas itself (`single_turn_ascore`) and by the stage's per-call pipeline on the same fake judge: the same prompts and the same scores. Factual correctness checks claims with faithfulness' prompt, so the two share calls. The judge request (the config's model, temperature and seed) and each call's tokens and cost. The sample is nested and in file order; the text metrics; telemetry is off. The `eval` stage against a fake judge and embedder: a row per question and setup, one of them hand-computed; abstentions scored in code and never sent; no prompt sent twice; a re-run asks nothing, a new answer asks only its response calls; an API error writes nothing and the re-run asks only it; an unreadable output is cached after Ragas' 3 fix-up requests and scores NaN; a dry run and an estimate above the cap ask nothing; the cap stops a run whose estimate was too low; the full run reuses the pilot's calls; a corpus outside `--corpus` comes from the cache alone, or is left out with a warning when a call is missing; stale answers are refused, and so are answers made with other document titles. | nothing |
| [tests/test_analysis.py](../tests/test_analysis.py) | The statistics on hand-computed cases:<br>• the bootstrap drops and counts NaN and is seeded per cell;<br>• the document-level interval is wider when one document carries the effect;<br>• paired differences align by question and drop NaN pairs;<br>• Holm by hand;<br>• the ranking rule needs both the interval and Holm (the dev example passes neither);<br>• "few documents" is flagged when the effect sits in one of 29 documents;<br>• the smallest detectable difference.<br><br>Also each answer's outcome and the tercile labels.<br><br>The `report` stage on the offline pipeline (one variant, 3 questions):<br>• it writes every file;<br>• every row lands in exactly one outcome;<br>• each run writes a new folder (`-2` within the same second), keeps the earlier one and moves `latest`;<br>• without a Ragas table, the report covers retrieval only;<br>• a stale evidence flag, `details.jsonl` or retrieval table is refused with the stage to re-run.<br><br>When the run has a Ragas table, the stage also runs on a copy of it, checked on invariants only. | nothing |
| [tests/test_quote_locator.py](../tests/test_quote_locator.py) | `locate_quote`, the source of every evidence span. Exact matches, and normalized ones whose offsets still point into the original text: whitespace and paragraph breaks, typographic dashes and quotes, soft hyphens, case, an added full stop, line-break hyphenation. Fuzzy matches only above the threshold, widened to whole words, never against a span shorter than the quote. Only the unit's spans are searched, and too-short quotes are rejected. When `data/parsed/` exists, a random quote from every block of every parsed document, verbatim and with that drift (a quote occurring twice in its block is found at its first occurrence; an invisible character at its edge is left out of the span). | nothing |

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

## Dependencies

- **Declared directly** (2026-10-12): `numpy`, `pandas`, `pyarrow`, `scipy` and `tabulate`, which `analysis/` imports and which came only through other packages before. They were added with `uv add --no-sync` and installed with `uv sync --inexact`, which kept the extra CUDA libraries.
- **Installed with uv:** all the pipeline libraries, plus pytest in the dev group, plus `chunkeval` itself in editable mode.
  - **On a fresh clone,** run `uv sync`.
  - **On the existing `.venv`,** use `uv sync --inexact`. It also holds packages that aren't in the lockfile (CUDA 12 libraries left over from the torch upgrade), and a plain `uv sync` would uninstall them.
- **ragas** is still 0.2.15. Upgrading changes its API, so read the migration notes first. `eval` uses its prompts, parsers and scoring code directly, and caches every judge call under the Ragas version, so an upgrade re-asks (and re-pays) every call; `tests/test_ragas_eval.py` checks the stage still matches Ragas' own metrics. Ragas 0.2.x sends usage data to its developers unless `RAGAS_DO_NOT_TRACK` is set: `ragas_eval.py` sets it before importing ragas.
- **Ollama** serves the local LLMs. It's a system service, not a Python package; the code reaches it with the `openai` client.
