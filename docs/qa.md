# Questions: generation and filtering

Stages 4–5 (`src/chunkeval/qa/`): question-answer pairs generated from the parsed documents, each answer
with its evidence located in the canonical text, then filtered down to the question set every later
stage evaluates (`data/qa/filtered.jsonl`).

## Q-A generation (`gen-qa`)

```bash
uv run chunkeval gen-qa --corpus papers --limit 1   # smoke test: one paper, ~20 s after the ~25 s Ollama preflight
uv run chunkeval gen-qa                             # every corpus; resumes where an earlier run stopped
uv run chunkeval gen-qa --corpus papers --recreate  # delete the papers records only and start papers over
```

**What it does,** per document (`qa/generation.py`), with `llm.qa_generator` (`gemma4-allgpu:12b-it-q8_0` since 2026-10-19):
- **Units** (`sample_units`): up to `qa.max_questions_per_doc`, spread over the whole document, of which `two_span_ratio` are two-span. No two units overlap.
  - A **single** unit is consecutive paragraphs of one section, packed towards the middle of `unit_min_tokens..unit_max_tokens`.
  - A **two-span** unit is two adjacent whole paragraphs, also across a section boundary; the question must need both. **On, at 0.2** since 2026-10-19: off with Qwen, which rarely gave a question from them; back on with Gemma, whose single-unit evidence rarely crosses a chunk boundary (see [Checks on the full corpus](#checks-on-the-full-corpus)).
  - A paragraph over the max is cut into sentence groups. A single "sentence" over the max is never used (garbled OCR in practice).
  - Sections that `skip_section_titles` names, or with less text than the minimum, give no unit.
- **One question per call,** with the prompt `prompts.qa_generation` ([prompts/qa_generation.md](../prompts/qa_generation.md)); its notes explain the file format.
  - Question types rotate per document. `numeric` passes to the next type for a unit without a digit.
  - A single unit's skip of another type is asked once more as `factoid`; a two-span skip is final.
- **Evidence** (`locate_quote`): each quote is searched only inside the unit, first exactly, then normalized, then fuzzy at `qa.fuzzy_match_threshold`. A pair needs every quote located, and in a two-span unit at least one quote in each passage.

**What it writes.** One line per unit, appended as soon as the unit's outcome is final, so a crash loses only the units in flight:
- **`data/qa/raw.jsonl`:** one `QAPair` per line. `question_type` is the type the model labelled; `meta` holds the types asked, the unit's spans, section path, size and `tokens_per_word`, and the token usage.
- **`data/qa/gen_dropped.jsonl`:** one `DroppedUnit` per line, with a `reason` (`skip`, `bad_output`, `quote_not_found`, `two_span_one_side`) and the model's output, so the drop rate per corpus and reason can be reported.
- **API errors are not saved** (timeouts, server down). The stage fails at the end, and a re-run asks exactly those units again.
- **Summary:** logged and written to `runs/<run_id>/gen_qa_summary.json`. Per corpus: units, pairs, pair rate, drops by reason, how the evidence was located, and the question types.

**Resume and provenance.** `data/qa/raw.provenance.json` records what decides a unit's outcome:
- per corpus, the generator (model, settings, Ollama digest), the prompt text's hash, the seed, the size tokenizer, `skip_section_titles`, the `qa` settings that shape units, and `GENERATION_VERSION` (bumped when the code changes its output);
- a text hash of every document that has records.

On a re-run, each corpus asked for (`--corpus`, default every configured corpus) is checked against its own settings:
- with the same values, only units that are in neither file are asked;
- with any difference, the run is refused and the corpus and the differing keys are named. `--recreate` deletes the records of the corpora asked for only; the other corpora's records stay as they are.

So a new corpus can be generated with other settings than the existing ones; the report then warns that the corpora's questions were made differently. A provenance file from before 2026-10-28 holds one set of settings for every corpus; it is read as that set for each, and the next run writes it per corpus.

`qa.docs_per_corpus` and `--limit` only choose which documents are used, so the full run resumes a `--limit` smoke test.

**On the full corpus** (2026-10-20, `gemma4-allgpu`, Ollama 0.40.6; 2 h 00 min at 62–68 tokens/s, concurrency 2):
- **Yield:** papers 2,173 units → 1,914 pairs (88%), reports 2,137 → 1,796 (84%); 3,710 pairs in all.
- **Drops:** mostly skips (229 papers, 309 reports). Unlocated quotes are rare (15 and 19), and so are bad outputs (2 and 3, all answers cut off at `max_tokens` 512).
- **Evidence:** 97–98% of quotes located exactly; the rest normalized (12 papers, 40 reports) or fuzzy (43 and 42).
- **Two-span units** give fewer pairs than single ones, and far fewer across a section boundary: same section 164 of 308 (papers) and 180 of 327 (reports), across sections 41 of 132 and 40 of 116.
- **Question types** (as labelled by the model): factoid 898 / 809, reasoning 467 / 488, definition 270 / 329, numeric 279 / 170 (papers / reports); about 420 units per corpus were retried as `factoid` after a skip.

**On the dev docs** (`qwen3.5:9b-q8_0`, the generator until 2026-10-19; 36 hand-made units, plus the dev paper_4 smoke run):
- **Output format:** valid JSON every time, never in code fences. Every quote was located, almost all exactly.
- **Speed:** 56–64 tokens/s at concurrency 2, so about 20 s per document of 10 units, or about 1 h for 200 documents.
- **Skips:**
  - `numeric` asked of passages whose only digits are years or names like "F2";
  - two-span pairs across a section boundary (0 of 4 gave a pair: the model calls them unrelated; `meta.crosses_section` lets the pilot measure this);
  - garbled OCR text, correctly.
- **Left to `filter-qa`:** some questions still refer to the source ("according to the passage", "the researchers") or are vague.
- **All 10 dev docs** (2026-10-11): with two-span units at 0.2, 96 units gave 69 pairs (72%), of which two-span only 3 from 20 (see [Checks on the full corpus](#checks-on-the-full-corpus)). With single units only, 97 units gave 84 pairs (87%) in 3 min 11 s.
- **Gemma 4 on the same 97 units** (`gemma4:12b-it-q8_0`, Ollama 0.40.6, 2026-10-19; a scratch run, the same seed gives the same units): 93 pairs (96%) in 3 min 15 s after the preflight, about Qwen's time, at 65–74 tokens/s (concurrency 2); the full corpus's up to 4,500 units would take ~2.5 h. Its only drops are 4 skips: no `bad_output` (it follows the JSON schema) and no `quote_not_found` (Qwen: 6).
  - **It quotes one sentence where Qwen quoted two.** 84 of its 93 pairs have a single evidence span (Qwen: 38 of 84), and the evidence is shorter (mean 218 vs 272 characters).
  - **So its evidence rarely crosses a chunk boundary,** except for the fixed-size chunker, which cuts inside sentences. Among the kept pairs, the evidence needs two or more chunks (the oracle) for 14% of Gemma's questions with `fixed-size` (Qwen: 17–26%), but for 0–2% with `sentence` and `section` (Qwen: 6–23%). Since chunker differences come from boundary cases, Gemma's question set separates the sentence and section variants, and overlap within them, much less.
  - **Reasoning questions made to combine two sentences** (prompt change, same 97 units, `gemma4-allgpu`, 2026-10-19): reasoning pairs with 2+ evidence spans rose from 6 of 25 to 13 of 22, with the same yield (93 pairs, 44 kept) and good questions. But the two sentences are neighbours (1–3 characters apart), so the evidence still needs 2+ chunks for only 0–2% of the kept questions with `sentence` and `section`; `fixed-size` 9–20%. No gain where it was needed. Generation took 2 min 45 s, 16% faster than the official tag.
  - **Two-span units back on, at 0.2** (with the prompt change; 96 units, of which 20 two-span): 12 two-span pairs from 20 units (8 skips: unrelated paragraphs, contact details, garbled text) and 5 kept; single units 70 pairs, 31 kept. In all 82 pairs, 36 kept (44%). The kept set's evidence needs 2+ chunks for 19% (sentence), 14% (section) and 36% (fixed-size) of the questions without overlap, and 6%, 8% and 19% with it, so the boundary cases are back and overlap shows its effect. Adopted: `qa.two_span_ratio: 0.2`.

**How the generator was chosen** (Gemma 4 over Qwen, 2026-10-19, above): run each candidate with `--recreate` on the same documents, then compare `pair_rate`, the drop reasons and tokens/s from the summary, plus the critic's acceptance (`keep_rate` in the `filter-qa` summary). Repeat it this way if the generator is ever changed. The 10 dev docs give 96 units with the default config.

## Q-A filtering (`filter-qa`)

```bash
uv run chunkeval filter-qa                    # every corpus in the config; asks the critic only what isn't cached
uv run chunkeval filter-qa --corpus reports   # reports only: the other corpora's lines stay as they are
```

**What it does** (`qa/filtering.py`), in three layers, each only on what the one before kept:
1. **Deterministic checks** (`deterministic_checks`), free, before any LLM call:
   - **On the question:** a reference to its source ("this study", "according to the passage", "described in the abstract", "for the study", "the authors"), a pronoun opening, a figure, table or section number, no final "?", two questions in one, fewer than 5 or more than 40 words. These are case-insensitive regexes; the old `phrase_blacklist` was case-sensitive. "in the context of <topic>" is allowed, because good questions use it to name their topic.
   - **On the answer:** an abstention ("the passage does not mention…", "N/A"). Also, every number in the answer must appear in the located evidence or in the question, in digits or in words there. A number found only elsewhere in the passage means the gold evidence misses part of the answer; on the dev docs this caught "less than 10 kg", which was in the passage but not in the quote.
   - An answer that repeats a quote word for word is fine, as long as it answers the question (decided 2026-10-11).
2. **The critic** (`llm.qa_critic`, gpt-oss:20b, a different family from the Gemma generator), with the prompt `prompts.qa_critic` ([prompts/qa_critic.md](../prompts/qa_critic.md)). Its notes explain each choice. It is called **twice per pair**:
   - **The passage call** sees the passage, question, answer and evidence, and judges the answer: `groundedness`, `answerability`, plus `needs_both` for a two-span pair, when they are on (one passage alone mustn't answer it).
   - **The question call** sees the question alone, as a user of the retrieval system does, and judges the question: `standalone`, `specificity` (one clear answer among hundreds of documents), `usefulness`.
   - Each call first writes a note the scores rest on: `question_asks_for` and `unresolved_references`. Then it gives a reason and a score from 1 to 5 per criterion.
   - **The decision is made in code** (`decide`): every criterion must reach `qa.critic_min_score`. Changing the threshold asks nothing again.
3. **De-duplication** (`deduplicate`): questions are embedded with bge-m3 (`embedding.primary`) once the critic is unloaded. A question is removed when its cosine to an already kept question of the same corpus is above `qa.dedup_cosine_threshold`. The pair kept is a two-span one (the scarce kind), else the one with the higher total critic score. The vectors are cached by question text (`dedup_vectors`), so only new questions are encoded: fp16 batches pad differently when other questions are in them, which would move a cosine in its last digits on every run. The first run after 2026-10-28 encodes every question once more, since the 2026-10-20 vectors weren't kept.

Every pair that passes is kept, with no down-sampling (decided 2026-10-11). `filtered.jsonl` is the final question set: every later stage evaluates exactly its question ids, and Ragas draws its own subset of it.

**Per corpus.** Every layer works within one corpus, so `--corpus` filters only the corpora asked for, and carries the other corpora's lines of `filtered.jsonl` and `rejected.jsonl` over exactly as the last run wrote them: what a full run would give them. A warning names a carried corpus whose pairs in `raw.jsonl` have changed since (`gen-qa` ran for it), with the command to filter it again. A corpus that is in `raw.jsonl` but not in the config is carried over too.

**What it writes:**
- **`data/qa/filtered.jsonl`:** the kept `QAPair`s in document order, with the critic's output in `meta.filter.critic`.
- **`data/qa/rejected.jsonl`:** every other pair, with `meta.filter`: `stage` (`checks`, `critic`, `dedup`), `reasons` (e.g. `source_reference: 'this study'`, `critic_specificity: 3`, `duplicate_of: <id> (0.95)`) and the critic's output, so the filter can be audited.
- **`data/qa/filtered.provenance.json`:** per corpus, what decided its lines: the critic (model, settings, Ollama digest, seed, `CRITIC_VERSION`), the critic prompt's hash, `critic_min_score`, the de-duplication's embedder and `dedup_cosine_threshold`. The report shows it per corpus and warns when the corpora of a run differ. A corpus last filtered before 2026-10-28 has no entry until it is filtered again.
- **`data/embeddings/dedup/<embedder>/<corpus>.npy`** (+ `.json`): the de-duplication's question vectors, keyed by question text.
- **`data/qa/critic_scores.jsonl`:** the critic's answers, one line per pair and call, appended as they arrive. This is a cache: each line is keyed by a hash of everything that decides it (the messages, the schema, the critic's model, settings and Ollama digest, `CRITIC_VERSION`). A re-run asks only what is new: new pairs, a changed pair's call, or every call after a prompt or critic change. An answer the critic couldn't produce (cut off, invalid JSON) is cached too and rejects its pair (`critic_bad_output`). API errors and prompts over the context window stop the stage before anything is written; a re-run asks exactly those.
- **Summary:** logged and written to `runs/<run_id>/filter_qa_summary.json`. Per corpus: pairs, kept, `keep_rate`, rejections by stage and by reason, each criterion's mean and count below the minimum, and the kept pairs by question type and kind.
- **`runs/<run_id>/filter_qa_labelling.csv`:** 50 pairs for hand labelling (see below), drawn from the corpora this run filtered. It is written once and never overwritten; delete it to draw a new sample.

**Hand labelling, to calibrate the threshold.** Do it on the final corpus, with a new `run_id`.
- **The sample:** half of it is kept pairs, half pairs rejected by the checks or the critic (duplicates are good pairs, not a verdict). The rows are shuffled and carry the passage and evidence, but not the filter's decision, so the labels can't be biased by it.
- **Labelling:** fill in `label` (1 = good pair, 0 = bad). Then join by `id` with `filtered.jsonl` and `rejected.jsonl`. The kept pairs labelled good give the filter's precision; the rejected pairs labelled bad give a rough recall.
- **The threshold:** if the critic rejects too many good pairs, try `critic_min_score: 3` (agreed fallback). Note that 3 also lets through `groundedness` 3, an answer with one inferred detail.

**On the full corpus** (2026-10-20; 1 h 25 min, 2 critic calls per pair that passed the checks):
- **Kept:** papers 852 of 1,914 pairs (44.5%), reports 904 of 1,796 (50.3%): 1,756 questions, above `ragas.sample_per_corpus` (500) in both corpora.
- **Checks** rejected 345 papers pairs and 173 reports pairs, mostly for naming their source (259 and 99: Gemma's "in the study", "of the survey") and two questions in one (67 each).
- **Critic** rejected 717 and 714, mostly on `specificity` (602 and 575 below 4) and `standalone` (442 and 431). `groundedness` and `answerability` average 4.9–5.0: the answers are almost always supported.
- **De-duplication** removed 5 reports questions, none in papers.
- **Kept by kind:** 37 two-span pairs in papers and 77 in reports (4% and 9% of the kept questions); `needs_both` averages 3.4 for papers and 4.3 for reports.
- **The labelling sample** is in `runs/full_data-v1.0/filter_qa_labelling.csv` (50 pairs), not labelled yet.

**On the dev docs**, single units only (84 pairs from 10 documents, 2026-10-11; 2 min 33 s): 65 pass the checks, 35 the critic, none removed as duplicates. That is 15 of 40 pairs for papers and 20 of 44 for reports; at `critic_min_score: 3` it would be 23 and 30.

**Gemma 4's pairs on the dev docs** (93 pairs, 2026-10-19; 3 min): 82 pass the checks, 43 the critic (46%; Qwen 42%), none removed as duplicates: 20 of 45 for papers and 23 of 48 for reports. The rejections follow Qwen's pattern (specificity 35, standalone 27 below 4), but Gemma names its source more often in papers (8 of 45, e.g. "in the study", "of the survey"). The critic prompt was tuned on Qwen's pairs; check Gemma's when hand labelling.

The detailed findings below come from the first run (69 pairs, two-span units still on; about 1.1 s per critic call at concurrency 2):
- **Checks:** 13 rejected, 11 of them for naming their source. Every flagged reference was a real one.
- **Critic:** 27 of the 56 remaining pass. Most rejections are `specificity` (24 pairs below 4, e.g. "What factor was identified as a determinant of the quantity of rice supplied through collective marketing?") and `standalone` (13: "the two control villages", "these enterprises", "their full model"). It also caught answers that reverse the passage (gender and improved varieties in Nasarawa) or answer something else ("Why…" answered with a restatement).
- **Compared with a first reading of 33 pairs (7 bad, 8 borderline, 18 good):** 6 of 7 bad rejected, 4 of 18 good rejected. Two of those four are specificity errors (a question naming CADFund scored 3); the other two are defensible. The bad pair it kept answers "Which organisation has FAO worked under?" with "FAO"; the same critic rejected it under an earlier prompt, so its answers on borderline cases are not stable across prompt versions.
- **De-duplication:** nothing removed (no two questions above 0.92).
- **How the prompt got there:** a first version with one-line criteria passed 50 of 56 pairs with 5 on every criterion. Raising the reasoning effort to medium or high didn't help. Asking for a note before the scores fixed the answer criteria. Judging the question without the passage fixed `standalone` and `specificity`: with the passage in view, the critic resolved "the full model" from it and never listed an unresolved reference.

## Checks on the full corpus

Raised on the dev docs, to check once `gen-qa`, `filter-qa` and `retrieve` have run on the full corpus.

- **Two-span units are on again** (`qa.two_span_ratio: 0.2`, decided 2026-10-19, with Gemma 4 as the generator).
  - **Why they were off** (2026-10-11, Qwen): on the 10 dev docs, at a ratio of 0.2, 20 two-span units gave 3 pairs (same section 3 of 14, across sections 0 of 6; single units: 66 of 76), and `filter-qa` rejected all 3: one names its source ("described in the abstract"), one doesn't need passage B (`needs_both` 1), and one infers more than the passages say. They took a fifth of the generator calls for no question.
  - **Why they are back:** Gemma's single units are answered from one sentence, which the sentence and section chunkers never cut, so their evidence crossed a chunk boundary for 0–2% of the questions (see [Q-A generation](#q-a-generation-gen-qa)). With Gemma, the same 20 two-span units give 12 pairs and 5 kept, and the evidence of all 5 needs two chunks with the sentence and section chunkers without overlap (2–3 of 5 with overlap).
  - **Check on the full corpus:** the two-span keep rate per corpus (dev: papers 4 of 10 units, reports 1 of 10), and the hand labels of the two-span pairs in the labelling sample. On the dev docs, 3 of their 7 rejections were two questions joined with "and what" (`compound_question`).
    - **Full corpus (2026-10-20):** 37 kept of 440 two-span units in papers (8%) and 77 of 443 in reports (17%); 51 papers pairs were rejected because one passage alone answers them (`needs_both`). Hand labels still to do.
- **Critic threshold and yield.** With single units only, the 10 dev docs gave 84 pairs from 97 units, and `qa.critic_min_score: 4` keeps 35 of them (42%; at 3: 53). Calibrate the threshold with the 50 hand labels (see [Q-A filtering](#q-a-filtering-filter-qa)), then check the yield. At 4 it is 32% of units for papers and 40% for reports, so 100 documents × 10 units give ~320 and ~400 questions: papers only just above `ragas.sample_per_corpus` (300 at the time). Raise `qa.max_questions_per_doc` if the full corpus comes out lower.
  - **With Gemma 4** (the Q-A generator since 2026-10-19) and two-span units at 0.2, 96 dev units give 36 kept questions: 30% of units for papers and 44% for reports. At 150 documents × 15 units (3 of them two-span) that is up to ~680 and ~990 questions, above `ragas.sample_per_corpus` (500) in both. Every question still gets the free retrieval metrics and an answer from `answer`.
    - **Full corpus (2026-10-20):** 852 and 904 kept, 39% and 42% of units, at `critic_min_score: 4`. The threshold waits for the hand labels.
  - **Check how often the evidence crosses a chunk boundary** (the oracle's share of questions needing 2+ chunks, per variant, after `retrieve`). On the dev docs, Gemma's kept questions give 19% (sentence), 14% (section) and 36% (fixed-size) without overlap, and 6–19% with it (Qwen, single units only: 23%, 9%, 26%). Without two-span units it was 0–2% for sentence and section.
    - **Full corpus (2026-10-20):** without overlap, 5.6% (sentence), 4.8% (section) and 23.6% (fixed-size) for papers, 9.7%, 3.8% and 24.6% for reports; with overlap 2.3–5.1% for sentence and section, 15.6–16.0% for fixed-size. Lower than on the dev docs, because two-span pairs are only 4–9% of the kept questions (dev: 14%). So the sentence and section variants differ on few boundary questions; the paired comparisons of `report` show whether that is enough.
