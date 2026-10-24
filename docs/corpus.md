# Corpus: parse, audit, chunk, embed, index

Stages 1–3 (`src/chunkeval/corpus/`): PDFs → one canonical text per document → chunks → vectors →
12 Weaviate collections. Weaviate itself is set up as described in [setup.md](setup.md#weaviate-docker).

## The dev docs and the full corpus

- **The final corpus** (fixed 2026-10-20): 150 papers (`paper_1` … `paper_150`) and 150 reports
  (`report_1` … `report_150`) in `data/corpora/`. The files were renamed on 2026-10-19;
  `data/corpora/rename_map.csv` maps the original names to the new ones, with each PDF's MD5.
- **The dev docs.** Before the final corpus existed, every setting was developed on 10 dev PDFs:
  first paper_4, paper_8, report_15 and report_16, then (2026-10-11, to develop the Q-A stages)
  paper_42, paper_74, paper_93, report_119, report_123 and report_140. The chunking numbers marked
  "dev docs" are from the first 4.
- **None of the dev docs is in the final corpus,** and their ids are the dev set's own: dev report_15
  ("Toward rice-based food systems transformation in Africa") is not today's report_15. Every
  finding marked "dev docs" in these docs refers to those 10 documents. Their outputs are archived
  in `data/_dev_archive/` and `runs/v2-pilot/`.

## Parse (`parse`)

```bash
uv run chunkeval parse --corpus papers --limit 2    # smoke test
uv run chunkeval parse                              # every corpus; only new or changed PDFs are parsed
```

**What it does** (`corpus/parsing.py`), per PDF, with Docling:
1. Converts the PDF and keeps the body items in reading order (`parsing.include_labels`); page
   headers, footers and page numbers never reach the text.
2. If the text layer is broken (unreadable characters, or a duplicated hidden layer:
   `parsing.ocr_max_*_ratio`), converts it again with full-page OCR, and keeps the OCR output only
   if it helps (`ocr_helps`, see [the audit](#audit-audit)).
3. Infers heading levels (`parsing.heading_levels`: font size, numbering, or flat).
4. Keeps the body only (`parsing.body[corpus]`): papers from the Introduction to the references,
   reports without contents, front and back matter. What went is in `meta.body`.
5. Joins the pieces into `CanonicalDoc.text`, with the offsets of every block and section. Every
   chunker and the Q-A generator work on this one string, so all character offsets are comparable.

The title comes from Docling, or from [configs/titles.csv](../configs/titles.csv)
(`parsing.title_overrides`) where Docling's is wrong.

**What it writes:** `data/parsed/<corpus>/<doc_id>.json` (doc_id = the file name without `.pdf`) and
`data/parsed/<corpus>/_failures.jsonl` (the PDFs that failed in the last run, with the error; one
failure doesn't stop the corpus).

**Cache.** A JSON is reused while the PDF's MD5, `PARSER_VERSION`, the Docling version, the parsing
settings and its title override are unchanged; otherwise the PDF is parsed again. So changing an OCR
threshold re-parses only the PDFs whose OCR decision flips, and a new title override only that
document. A JSON whose PDF is gone from the corpus folder is deleted. The full corpus takes 32 min.

### What the dev docs raised, and where it stands

- **What each document lost:** check `meta.body` in every parsed JSON (`start`, `end`, `dropped_sections`, `chars_dropped`) and the warnings in the parse log.
  - **Now:** the `audit` stage reports it (BODY SELECTION); the parser fixes it led to are listed under [the audit](#audit-audit).
- **OCR thresholds:** recalibrate `parsing.ocr_max_*_ratio` from the `meta.text_layer.auto` ratios every parsed JSON keeps. Changing them re-parses only the PDFs whose OCR decision flips.
  - **Now:** kept at 0.01 (replacement characters) and 0.05 (repeated paragraphs). Instead, OCR output is kept only when it helps (`ocr_helps`, 2026-10-20): 13 documents keep full-page OCR, 24 more were converted again and kept their text layer.
- **Reports' drop list:** `parsing.body.reports.drop_titles` was built from two AfricaRice annual reports. Extend it from the full corpus (candidates are noted in the config).
  - **Now:** extended from the whole corpus (see [the audit](#audit-audit)).
- **OCR retried without need (dev report_140).** Its first conversion had 12.3% repeated paragraphs, so it was converted again with full-page OCR; afterwards the parser still reported 9.2%, yet only one block (a table footnote) actually repeats. Check what `repeated_paragraph_ratio` counts there before recalibrating its threshold: if repeated short items trigger it, it sends PDFs with a good text layer through OCR, which is slower and can garble text (see paper_4 below).
  - **Now:** a document whose ratio OCR doesn't fix keeps its text layer (`ocr_helps`).
- **Document titles.** Dev report_119 (a book chapter) got the publisher's banner as its title: "We are IntechOpen, the world's leading publisher of Open Access books…". The title is shown to the Q-A generator and to the critic, so check the titles of the full corpus (`parsing.doc_title`).
  - **Now:** 51 titles are replaced by `configs/titles.csv`, and the `audit` stage flags suspicious titles that have no override.
- **Garbled OCR text (dev paper_4).** Dev paper_4 is converted with full-page OCR because its text layer is duplicated (`repeated_paragraph_ratio` 0.12). In 7 paragraphs of sections 2, 3.1, 3.2 and 4.1, the OCR output is nonsense, e.g. "The GanbanlPlatondastenegtls initiadedbovemgonicatienebaitks…". That is about 2,100 of its 16,700 tokens (13%). Dev report_16, also OCR'd (for unreadable fonts), came out clean.
  - **Why it matters:** garbled chunks get indexed and compete in retrieval as noise. A Q-A unit drawn from them gives nonsense questions, and since the quote locator matches verbatim text, it can still "locate" a garbled quote. It also distorts the size stats (one garbled "sentence" alone is 529 tokens).
  - **A second kind, inside clean paragraphs:** dev paper_4 also has fragments of single letters mixed into otherwise normal paragraphs, e.g. "…potentially unique ar s s o e e o es ,sers s u ton Cryopreserved back-up collections…" or a paragraph opening "a re rs h n nd h- nae rnd s s u uun…". Their tokens per word is normal (1.36), and at block level neither tokens per word nor mean word length separates them from clean text (4.64 letters per word vs. 3.52 for a clean formula paragraph in dev paper_8). `sample_units` picked one such paragraph on the dev docs; the generation prompt tells the model to skip garbled text, and the critic sees the passage.
  - **Detecting it:** tokens per word is a good first signal: 3.4–17 in the garbled paragraphs vs. about 1.6 in normal text. But email contact lines in the reports also reach 3–4.5, so a second signal is needed, e.g. the share of words found in an English word list. Every generation unit stores its `tokens_per_word` in `meta`. `sample_units` already drops a single "sentence" longer than `unit_max_tokens`: on the dev docs only two, both garbled in dev paper_4 (529 and 401 tokens). The rest of the 669-token garbled paragraph splits into normal-sized sentences, which can still end up in a unit.
  - **Options considered:** (a) measure it as a third `meta.text_layer` ratio and flag or drop garbled blocks; (b) retry OCR with another engine or a higher image resolution; (c) OCR only the pages with the duplicated layer and keep the text layer elsewhere; (d) exclude the document.
  - **Now:** the `audit` stage measures both signals per block (tokens per word above 3, or under 60% English words). On the full corpus the only real OCR garbling is paper_8's (6.4% of its text; it keeps OCR because its duplicated layer drops from 12% to 1% repeated paragraphs); the other flagged blocks are mostly author lists and affiliations. Not changed (2026-10-20): none of the options was needed.

## Audit (`audit`)

```bash
uv run chunkeval audit           # after parse, before chunk (~7 s)
```

**What it does** (`corpus/audit.py`). It checks the parsed corpus and flags; it never fixes or drops
anything. The checks: conversion failures, OCR, what each document lost to `parsing.body`, drop-list
candidates, suspicious titles (except those already in `configs/titles.csv`), garbled text and other
languages per text block, byte-identical PDFs and near-duplicates (8-word shingles), and size.

**What it writes:** `runs/<run_id>/audit.txt` (the summary, one part per check) and `audit.json` (the
numbers per document). What needs a decision is also logged as a warning: not fully converted, a
broken text layer (garbled text in most of a document), not in English, duplicates, titles without an
override. For each, replace or remove the PDF, or add a title override, then run `parse` and `audit`
again until only accepted warnings are left. The corpus must be final before `chunk`: adding a
document later changes the top-k of every question.

The garbled-text check needs the English word lists in `/usr/share/dict` (`sudo apt install wamerican wbritish`).

**The full-corpus audit (2026-10-20).** 150 papers and 150 reports were parsed (32 min, no failure; report_102 only partly converted, 24 pages kept). What was found and decided:
- **Parser fixes** (`PARSER_VERSION` 4; each with a test in `tests/test_parsing.py`):
  - Papers in Results-first order (no Introduction heading, Methods last) started at "Methods" and lost their Results and Discussion: 68–83% of paper_99, 111, 117 and 141. `start_titles` now include "results" and "results and discussion".
  - A paragraph opening with "(Additional file 1: Table S1)" counted as back matter and cut paper_64 and paper_71 in their Results. A lead label must now start with a letter.
  - PLOS ONE's first-page "Competing Interests:" footnote, placed after the Introduction heading, cut paper_75 to 921 characters. Lead labels in the first quarter of the body are ignored (`FRONT_LABEL_SHARE`).
  - A "Contents" or "Acknowledgments" heading in a larger font than the chapters after it dropped them all as its subsections: 50–99% of report_112, 127, 5, 103 and 114. A dropped section now takes its subsections only while they stay under 20% of the document (`DROP_MAX_SHARE`), else only its own text (`meta.body.dropped_own_text_only`).
  - Full-page OCR is kept only if it brings each triggering ratio under its threshold or at least halves it (`ocr_helps`). 24 of the 37 documents sent through OCR keep their text layer: their repeats are on the page (a form template printed 5 times, herbarium entries in an annex), not a duplicated layer, and the parser's de-duplication removes them anyway. They keep their exact text, without OCR noise.
  - The PDF's MD5 is part of the parse cache key, so a new file under an old name is parsed again, and a JSON whose PDF is gone from the corpus folder is deleted.
- **Reports' drop list,** extended from the section titles of the whole corpus: contents and list variants ("table of content", "content", lists of figures and tables), front and back matter (disclaimer, citation, about the authors, contact, copyright, credits, keywords, glossary, index, board of trustees, singular acknowledgment), abstracts in other languages (résumé, resumen, mots clés), and annexes and appendices by their first word (`drop_title_prefixes`: "Annex 2: Workshop agenda"), mostly agendas, participant lists and questionnaires.
- **Titles:** 51 documents had no title (8 fell back to the file name), a banner ("RTB Working Paper", "SPOTLIGHT", "1"), an author list, manuscript line numbers ("CAN 1 CARBOHYDRATE…"), a footnote marker ("1[C][W][OPEN]"), letter-spaced text or a French title on an English paper. [configs/titles.csv](../configs/titles.csv) replaces them with the title on the first page (`parsing.title_overrides`).
- **Not changed:** no near-duplicates (no pair reaches a containment of 0.2; report_103 and report_114, two RTB working papers, shared 49% only while the parser had cut both down to their front matter), no byte-identical PDFs, and no document dominates its corpus (largest 2.1% of the text, unlike dev report_123 on the dev docs). The garbled-text check flags mostly author lists and affiliations, which stay. paper_77 has words split by spaces ("fami ly Bartonel laceae", ~20% of its blocks) and keeps them.

## Chunk (`chunk`)

```bash
uv run chunkeval chunk           # every corpus, all 6 variants (~1 min)
```

**What it does** (`corpus/chunking/`). Every document is cut by 3 methods, each with and without
overlap (`chunking.methods`, `chunking.overlap`), to `chunking.target_tokens` (256) with a cap of
`max_tokens` (512):
- **fixed-size** (`character.py`): the baseline. Windows of a fixed number of characters, calibrated per
  corpus so the mean chunk is the target, each cut moved to the nearest whitespace.
- **sentence** (`sentence.py`): whole sentences (pysbd) packed towards the target.
- **section** (`recursive.py`): the same packing, but never across a section boundary; tiny sibling
  sections under one parent are merged (`chunking.min_section_tokens`).

**Names** (decided 2026-10-22): the report and these docs say fixed-size and section. The config, the
file names, chunk ids, Weaviate collections and stored tables keep the keys `character` and `recursive`
(`chunking.methods`, `analysis.baseline`), so nothing had to be recomputed. Why: "character" named the
unit, not the rule, and with "recursive" it read as LangChain's `CharacterTextSplitter` and
`RecursiveCharacterTextSplitter`, which cut differently.

Overlap is about `overlap_ratio` (12%) of the target: characters for the fixed-size windows, at least
one whole sentence for the other two. Sizes are measured in bge-m3 tokens, with the tokenizer pinned
to the same Hub commit as the bge-m3 embedder (`chunking.size_tokenizer_revision`; the config refuses
two different pins). Every chunk satisfies `text == doc.text[start:end]`.

**What it writes:** `data/chunks/<corpus>/<method>_<overlap|no_overlap>.jsonl` and
`data/chunks/<corpus>/_size_stats.json` (size distribution per variant, for the report).

### The dev docs' checks on the full corpus (2026-10-20)

- **Words per chunk (`words_mean`):** about 160 words at 256 tokens on the dev docs. On the full corpus: 158–159 for papers and 172 for reports (fixed-size and sentence), so the note on `target_tokens` in `configs/default.yaml` still holds.
- **Small section chunks on reports:** on the dev docs, section chunks average 192 tokens (vs. 256 for the other methods), and 5% are 40 tokens or less. Reports have many tiny sections (office addresses, "ISBN:", "Writing:"). On the full corpus: 213 tokens on average and 2% at 40 tokens or less (papers: 244 and 0.2%). The setting is `chunking.min_section_tokens` (40), which merges small sibling sections under the same parent only; raising it trades section purity for size.
- **Fixed-size window:** it is calibrated per corpus on every run: 1,027 chars for papers and 1,133 for reports on the full corpus (dev docs: 1,073 and 1,101). The chunk log has 9 "window … > max_tokens" warnings (dense tables in paper_21, paper_48, paper_89, report_26 and report_143); those windows are split into token windows.
- **Sentence overlap:** it averages 44 tokens on papers and 40 on reports vs. 31 for the fixed-size windows (dev docs: 42–47 vs. 31), because it repeats at least one whole sentence. That rule is deliberate (the docx suggests a 1-sentence overlap). Report the measured `overlap_tokens_mean` next to the results.

## Embed (`embed`)

```bash
ollama ps                        # must be empty: the embedders need the GPU
uv run chunkeval embed           # both models, every chunk file (~15 min on the full corpus)
```

**What it does** (`corpus/embedding.py`). It embeds every chunk file with both models, in fp16 on the
GPU (identical top-5 neighbours to fp32 on the dev chunks, at half the memory):
- **Primary: `BAAI/bge-m3`.** Used for everything: span metrics, RAG answers, Ragas. It needs no query instruction.
- **Secondary: `Qwen/Qwen3-Embedding-0.6B`.** Only the free span metrics are computed with it. Its job is to check whether the chunker ranking holds under a different embedder, and whether the embedder or the chunker is the bigger lever. Its queries need an `Instruct: …\nQuery:` prompt (set in the config).
- Both models produce 1024-dim vectors, so a mix-up never raises an error. Always pass vectors together with their embedder name.
- **Both are pinned to a Hub commit** (`revision` in `configs/default.yaml`, a full 40-character hash; a branch like `main` is rejected). Otherwise an upstream push could change the vectors mid-experiment, and queries would be embedded by a different model than the documents. The config comment shows how to look up the latest commit.

**What it writes:** per model, corpus and variant, `data/embeddings/<embedder>/<corpus>/<variant>.npy`
(the vectors, in chunk-file order) and a `.json` with the chunk ids and the settings that made them.

**Cache.** A re-run embeds only the chunk files whose content changed (content hashes, not file
times), or all of a model's files after its `revision` changes. Before loading each model the stage
asks Ollama what is loaded, and stops with the `ollama stop` command to run if a model holds the GPU;
it never unloads one itself.

## Index (`index`)

```bash
uv run chunkeval index --corpus papers --limit 1    # smoke test: the chunks of one paper
uv run chunkeval index --recreate                   # all 12 collections, replacing the smoke test
```

- **One object per chunk.** Its properties are the chunk's fields (all but `meta`), and its uuid is derived from `chunk_id`. It carries both named vectors, read from the `embed` cache; `index` never loads a model, so it can run while Ollama holds the GPU.
- **Exact search.** Both named vectors use a flat (brute-force) index with cosine distance, not HNSW. An approximate index can miss a true neighbour, and which one depends on insertion order, so a rebuild could move the retrieval metrics. On the full corpus a collection holds 5,170–9,097 objects (35–61 chunks per document), where brute force still takes milliseconds per query.
- **BM25 sees only `text`.** The ids and `section_path` are filterable (matched whole, so `paper_4` doesn't match `paper_4_suppl`) but not searchable, so the hybrid follow-up scores only the text the generator reads.
- **Provenance and re-runs.** Each collection's description holds a JSON record of what it was built from: a hash of the chunks, `--limit`, and the settings of both embedders (model id, revision, …). On a re-run:
  - a collection with the same provenance that passes verification is **skipped**;
  - any other existing collection is **refused**, with the inputs that differ named (e.g. after `chunk` or `embed` changed something, after a `--limit` smoke test, or with collections from another corpus). `--recreate` rebuilds it.
- **Verification after every build.** The object count must equal the chunk count. For 8 objects spread over the collection, the properties and both stored vectors must equal the chunk file and the cache. Both models give 1024-dim unit vectors, so this is what catches swapped names or rows shifted by one. Any failure stops the stage.
- **On the full corpus** (2026-10-20): 84,447 objects in the 12 collections, indexed and verified in 38 s.
