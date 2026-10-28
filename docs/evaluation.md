# Evaluation: retrieval, answers, Ragas

Stages 6–8 (`src/chunkeval/evaluation/`): every question of `data/qa/filtered.jsonl` is retrieved
against every collection (free), answered from each setup's top chunks by a local LLM (free), and a
seeded sample of the questions is scored by the Ragas judge (paid).

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

**On the full corpus** (852 papers and 904 reports questions, 2026-10-20; 1 min 2 s). Means at k = 5, bge-m3 (`span_hit` / `span_recall`):

| Variant | papers | reports |
|---|---|---|
| `fixed-size_no_overlap` | 0.86 / 0.83 | 0.83 / 0.78 |
| `fixed-size_overlap` | 0.89 / 0.86 | 0.83 / 0.80 |
| `sentence_no_overlap` | 0.86 / 0.85 | 0.81 / 0.79 |
| `sentence_overlap` | 0.88 / 0.87 | 0.83 / 0.81 |
| `section_no_overlap` | 0.90 / 0.89 | 0.85 / 0.84 |
| `section_overlap` | 0.89 / 0.89 | 0.85 / 0.84 |

- **Qwen3-Embedding** is 1–6 points lower everywhere (`span_hit@5` 0.84–0.87 papers, 0.78–0.80 reports). On papers it also puts section first; on reports its six variants are within 0.78–0.80.
- **The baselines:** random is 0; the oracle needs 1.0–1.2 chunks on average.
- **The budget view:** no question fits all `k_max` hits into `budget_tokens`; the budget k is 4.5–5.7.
- These are means: which differences hold is for the paired comparisons of `report` (`runs/full_data-v1.0/reports/latest/report.md`, retrieval only until `eval` has run).

**On the dev docs** (35 questions, 2026-10-11):
- **The embedders:** at k=5, `span_hit` is 0.89 (papers) and 0.88 (reports) for bge-m3, and 0.90 and 0.91 for Qwen3. `span_recall` is 0.85–0.89.
- **The baselines:** random is about 0, and the oracle needs 1.1–1.2 chunks on average (at most 2).
- **Overlap in the oracle:** it shows the expected effect. The share of questions whose evidence needs two chunks drops with overlap: sentence, papers from 27% to 0%; fixed-size, reports from 25% to 10%.
- **Too few questions to rank chunkers:** only 4 of the 35 questions change their `span_hit@5` across the 6 configurations, so the dev set can't separate the methods. That needs the full question set and the paired CIs of `report`.

### Checks on the full corpus

- **Very long documents.** On the dev docs, dev report_123 (326k characters) holds 279 of the 420 reports chunks (66%), so most distractors in the reports collections come from one document. Check the chunks per document on the final corpus. If a few documents dominate, decide whether to cap them or to report `doc_hit` next to `span_hit`.
  - **Full corpus (2026-10-20):** no document dominates. The largest holds 1.4–1.5% of a papers collection's chunks (paper_18, paper_20) and 2.2–2.4% of a reports collection's (report_55); the five largest together 6.5% and 10%.
- **The budget view.** `retrieve` warns when a question fits all `k_max` hits into `budget_tokens`, because its budget k is then capped. Raise `retrieval.k_max` if it does.

## RAG answers (`answer`)

```bash
uv run chunkeval answer                       # every corpus; asks the generator only what isn't cached
uv run chunkeval answer --corpus papers
```

**What it does** (`evaluation/answering.py`). Every question of `filtered.jsonl` is answered by `llm.rag_generator` (qwen3.5:9b-q8_0, temperature 0, thinking off), with the prompt `prompts.rag_answer` ([prompts/rag_answer.md](../prompts/rag_answer.md)):
- **In each of the 6 setups of its corpus,** from the top `retrieval.top_k` (5) chunks that bge-m3 (`embedding.primary`) retrieved, in rank order. Qwen3-Embedding stays a retrieval-only check. There is no budget-matched variant: the 1,300-token budget fits 4.2–5.0 chunks on average, so it would nearly repeat the k=5 run.
- **From its gold passage** (setup `gold_passage`): the passage the question was generated from, which holds all its evidence. This is perfect retrieval, one answer per question for all setups. Stage 8 scores it like a setup, as the **ceiling** of the generation metrics:
  - setup − gold passage is what retrieval with that chunker costs, and 1 − gold passage is the generator's and the judge's own error;
  - a metric on which the gold passage doesn't beat the setups doesn't respond to retrieval (the previous round's relevancy was ~0.54 everywhere);
  - its context recall should be ~1 by construction, so the shortfall measures the judge's error rate;
  - a question answered wrongly even from its own passage points to a bad gold answer: list those for the audit.

  It is generous: one clean passage without distractors, the one the question was written from. On the dev docs the same Qwen model also wrote the question; on the full corpus Gemma 4 does, so the gold passage's misses are less likely to be one model agreeing with itself. Not to be confused with the retrieval `oracle` (each setup's chunks that cover the evidence), whose ceiling stage 6 already measures for free.

**The prompt** (decided 2026-10-11, the title and the setting rule 2026-10-21; its notes explain each rule):
- **Chunk text with its document's title, no section path.** Section paths would favour the section chunker; a title belongs to the document, so every chunker gets the same help. Ragas reads the same title line, so it judges faithfulness against exactly what the generator read. Until 2026-10-21 the passages were chunk text only. The generator then couldn't confirm the setting a question names, and abstained even on its own gold passage (see [the trial](#the-title-and-the-setting-rule-2026-10-21)).
- **The setting rule.** A question may name the study's setting (place, crop or organism, organisation or project, period) that the passages don't repeat, and that isn't missing information. A passage whose text or title names a different setting is another study and is not used.
- **Complete first, then concise:** every detail asked, with numbers and units as given, and nothing else. No reference to "the context" or passage numbers in the answer.
- **No citations** (see [Future work](future-work.md#citations-in-the-rag-answers)).

**Abstentions.** Only when no passage holds any of the information asked for, the generator must reply exactly "The context does not contain the answer." When the passages hold part of it, it gives that part and says nothing about what is missing (a first version that named the missing part backfired on the dev docs; see the prompt's notes). `is_abstention` marks an answer that starts with the phrase, or that is one sentence paraphrasing it ("The provided context does not mention…"); a paraphrase followed by facts counts as an answer. Why this matters:
- **Ragas 0.2.15 scores an abstention like a wrong answer:** answer relevancy 0 (noncommittal) and factual correctness 0. Its faithfulness comes out at random (NaN, 0 or 1, depending on what the judge extracts). Context precision and recall don't read the answer.
- **So stage 8 handles them in code** (see [Ragas evaluation](#ragas-evaluation-eval)): relevancy and factual correctness 0 without asking the judge; faithfulness left out for abstentions and reported next to the abstention rate, plus a view with abstentions = 1. Abstentions are never dropped from every metric, because the paired comparison needs the same questions in every setup.
- **The summary separates the two kinds of miss** with `evidence_shown` (some passage overlaps the gold evidence, i.e. span_hit at top_k): `abstained_with_evidence` (an over-cautious generator, or evidence split so it looked incomplete) and `answered_without_evidence` (a hallucination risk, unless another document holds the same fact). On the gold passage the abstention rate should be about 0.

**What it writes:**
- **`runs/<run_id>/answers/<corpus>_<setup>.jsonl`:** one `Answer` per question, in question order. `<setup>` is a variant's key (`character_no_overlap` for fixed-size without overlap, …) or `gold_passage`. Each answer holds the `response` and the `contexts` it was shown, in rank order, which are Ragas' `response` and `retrieved_contexts`, plus the chunk ids, `evidence_shown`, `abstained`, `error`, the model and the token counts. Each context is the passage as the generator read it: `Document: <title>`, a line break, then the text. A `.provenance.json` next to each file records the questions hash, a hash of every document title of the corpus, the retrieval file's provenance, `top_k`, the generator (with its Ollama digest) and the prompt file's hash. The titles hash is there because a retitled document (`parsing.title_overrides`) keeps its chunks and retrieval, yet changes what the generator is shown; `eval` refuses answers made with other titles.
- **`data/llm_cache/answer_cache.jsonl`** (`paths.llm_cache_dir`): the generator's answers, one line per request, appended as they arrive. It is keyed by a hash of the messages and the generator (model, settings, digest, seed, `ANSWER_VERSION`), so a re-run asks only what is new: a new question, a question shown other chunks or another title, or everything after a prompt or generator change. Identical requests (two setups showing the same chunks) are asked once. The cache is shared by every `run_id` and data dir, so a new run asks only the requests no run has made. An answer the generator couldn't finish (cut off at `max_tokens`, context full) is cached with its `error` and has an empty response. API errors stop the stage before any answer file is written; a re-run asks exactly those.
- **`runs/<run_id>/answer_summary.json`:** per corpus and setup: answered, abstained, errors, the abstention rate, the two miss counts above, and mean answer words and token counts.

**Checks before any call:** the questions' documents still have the text `gen-qa` saw, and each bge-m3 retrieval file is up to date with the current questions, chunk file and query settings. A stale one stops the stage with "run `chunkeval retrieve` first", because its chunk ids could point to other text than it ranked.

**On the full corpus, first run** (chunk text only, before the title and the setting rule; 1,756 questions × 7 answer files, 2026-10-20; 12,059 calls in 2 h 25 min at 54 tokens/s, concurrency 2; no errors, nothing cut off):
- **Abstentions per setup:** papers 10.9–17.6%, reports 15.0–21.9%, highest for `fixed-size_no_overlap` and lowest for section in both corpora. `answered_without_evidence` is 22–45 per setup.
- **The gold passage abstains on 78 of 852 papers questions (9.1%) and 123 of 904 reports questions (13.6%),** where about 0 was expected. 16 and 34 of them abstain in every setup.
  - **Most of them are good pairs** (a reading of 10, 2026-10-21): in 9, the passage states the answer almost word for word ("Only 23% of Cx. tritaeniorhynchus had taken a blood meal…"), yet the generator replies with the abstention phrase.
  - **The likely cause:** the question names the document's context, as the critic's `standalone` criterion requires ("in the Senegal River Delta during 2008-2009", "in Northern Borneo", "in Cameroon"), but the passage doesn't repeat it: in the 5 papers questions checked, the place or organism named in the question is absent from the passage while the answer is there. The generator sees chunk text only (no title), so it can't confirm the question's premise and treats it as missing information.
  - **It affects every setup:** 26–81 answers per setup abstained although the retrieved chunks hold the evidence (`abstained_with_evidence`).
  - **Changed before `eval` (2026-10-21):** the prompt now shows each passage's document title and has the setting rule, chosen in [a trial](#the-title-and-the-setting-rule-2026-10-21); every answer was asked again.

**On the dev docs** (35 questions, 2026-10-11; 244 calls, as two setups showed the same chunks once; 3.5 min at 48 tokens/s, concurrency 2):
- **Size:** about 1,850 prompt tokens per answer from 5 chunks and 830 from the gold passage. Nothing was cut off at `max_tokens` 512. Answers from the chunks average 35–43 words per setup, from the gold passage 27–33; the gold answers average 22.
- **Abstentions:** 2–5 per setup, 55 of 245 in all, and all but one are the bare phrase. 3 questions abstain everywhere, gold passage included, so their abstentions say nothing about the chunkers. `answered_without_evidence` is 0–1 per setup.
- **The gold passage abstains on 7 of 35,** where about 0 was expected. Reading the passages:
  - **5 are weak pairs:** the question asks for something the passage doesn't state in that form: a "definition" whose answer is a percentage, a circular "why", a "how" the passage doesn't explain, a premise about the SDGs the passage never mentions, and a garbled question (paper_93:s19729, report_140:s27355, report_15:s17844, report_123:s73707, paper_4:s48236). The critic passed all of them.
  - **2 are an over-cautious generator:** paper_4:s24654 (the passage gives the reasons, hedged as "may be attributable to") and paper_74:s36976 ("threshold value" in the question, "more than 100 kg" in the passage).

  Three of the seven are in the labelling sample (paper_74, paper_93, report_123); check the rest when labelling. On the full corpus, list every question the gold passage abstains on for the audit.
- **The first version of the prompt** asked partial answers to name what was missing. It abstained less (42 in all, 5 on the gold passage), but 20 answers carried "the context does not say…" sentences, and several abstentions opened with the phrase, then gave the facts anyway. The prompt's notes have the details.

### The title and the setting rule (2026-10-21)

**The cause, measured on all 201 gold-passage abstentions of the first run.** In 158 of them (79%), the question has a capitalised name or a year that the passage lacks. That holds for only 426 of the 1,555 gold-passage questions it answered (27%). Many of the other 43 lack a lowercase crop or organism ("chickpea", "oil palm", "mango"). The document title holds every missing name in 134 of the 158, some of them in 10, none in 14. The Q-A generator sees the title and is asked for questions that stand alone, so the questions name the study's setting; the RAG generator saw chunk text only.

**The trial** (local, free: 1,899 calls in 15 min, plus the first-run prompt's 201 gold-passage answers from the answer cache). Three prompts on three sets:
- **Prompts:**
  - the first-run prompt;
  - the setting rule (rules 2, 6 and 7) on chunk text only;
  - the setting rule with each passage's document title: the prompt now in use.
- **Sets:**
  - **gold:** the 201 gold-passage abstentions, from the same passages. They should be answered.
  - **random:** 250 questions (125 per corpus), the top 5 of the `random` retrieval file from other documents than the question's. They should abstain.
  - **other documents:** 250 questions, the top 5 bge-m3 chunks from other documents only. These are on-topic chunks of other studies, the case the setting rule could get wrong. They should abstain, unless another document holds the same fact. Only questions whose top 20 hold at least 5 chunks of other documents were drawn: for 113 of the 363 drawn, bge-m3 ranks their own document so high that no other study competes for the top 5.
  - In random and other documents, the setup rotates over the 6 chunkers.

| | first-run prompt | setting rule | rule + title |
|---|---|---|---|
| Gold: abstained, of 201 | 201 | 118 | **29** |
| Random: answered, of 250 | 0 | 0 | 0 |
| Other documents: real answers, of 250 | 14 | 17 | 17 |
| … of which new, with another study's facts | — | 3 | 2 |
| Other documents: long refusals counted as answers | 2 | 6 | 3 |
| Prompt tokens, 5 chunks / gold passage | 1,863 / 802 | 1,961 / 900 | 2,144 / 992 |

**Reading the answers:**
- **Gold, rule + title:**
  - **The new answers are right.** The answers furthest from the gold answer by word overlap are correct, only shorter ("300 ml", "three to seven", "trypanosomosis"). The furthest, paper_109:t6504 (quinoa), gives a different reason that the passage also holds.
  - **Over-caution remains.** Of the 29 still refused, about 6 are weak pairs (paper_45:s14, paper_74:s22062, report_52:s1473, report_143:s3627, report_86:s8196, report_25:t19692). The rest hold the answer.
- **Other documents:**
  - **About 15 answers are the same under all three prompts and mostly right,** because another document holds the fact: near-duplicate reports (report_84/86, report_89/90, report_111/112), sister papers (paper_87/97), common facts (Vietnam's renovation began in 1986, El Niño in Zambia in 2023/24).
  - **New with the setting rule:**
    - paper_102:s25224 (rule only): contig counts of another transcriptome study;
    - paper_86:s10457 (both): another document's FAO growth rate of camel milk, 5.2% against the gold 6.72%;
    - report_57:s9305 (both): the Africa RISING programme's general objective, given as the Tanzania study's.
- **Long refusals** ("The provided context does not contain… It mentions…") count as answers under `is_abstention`, because more text follows the paraphrase. They are rare: 8 of the first run's 10,437 answers.
- **Titles barely leak into the answers:** once ("the period described in the document").

**Decision:** rule + title. Prompts grow by about 15%: about 100 tokens come from the longer rules, 180 from the titles. Ragas reads the titles too, so the judge's input grows as well. The trial ran from a scratch script, not kept in the repo. It built the messages with `answering.build_messages` and the stage's generator settings, so the first-run prompt's answers to the 201 came straight from the answer cache.

**The re-run with this prompt** (2026-10-21; 12,060 calls in 2 h 36 min at 51 tokens/s, concurrency 2; no errors, nothing cut off). First-run figures in brackets:
- **The gold passage abstains on 5 of 852 papers questions (0.6%) and 28 of 904 reports questions (3.1%)** [78 and 123].
  - 29 of the 33 are the ones the trial left.
  - 3 that the first run answered abstain now (report_78:s29362, report_8:s71569, report_8:s7702); the last two passages hold the answer. 171 went the other way.
- **Abstentions per setup:** papers 8.3–12.1% [10.9–17.6%], reports 12.5–16.9% [15.0–21.9%]. Still highest for `fixed-size_no_overlap` and lowest for section in both corpora.
- **Abstained with the evidence shown:** papers 8–26 per setup [26–62], reports 14–48 [28–81].
- **Answered without the evidence shown:** papers 29–48 per setup [22–37], reports 37–56 [27–45]. Almost all of them were shown chunks of the question's own document that miss the evidence span; only 0–3 per setup were answered from other documents alone.
- **Long refusals counted as answers:** 14 of 10,995 answers.
- **Titles in the answers:** 39 answers to 13 questions say "titled …" or "the title". Nearly all name a project, work package or workshop the question asks about; one hedge says "The document identifies…".
- **Size:** 2,018–2,148 prompt tokens per answer from 5 chunks, 996–1,031 from the gold passage. Answers average 28–35 words from the chunks, 25–30 from the gold passage.

## Ragas evaluation (`eval`)

```bash
uv run chunkeval eval --dry-run          # the cost of every judge call not cached yet: no API call, nothing written
uv run chunkeval eval --pilot 20         # the first 20 questions of each corpus's sample: the real cost per question
uv run chunkeval eval                    # the whole sample; asks the judge only what isn't cached
```

The judge is OpenAI's `gpt-6-luna` (`llm.judge`, with `reasoning_effort: "none"`), so this is the only stage that costs money. It replaced `gpt-4.1-mini-2025-04-14` on 2026-10-21, after both judged the same pilot ([the comparison](#the-judge-gpt-6-luna-instead-of-gpt-41-mini-2026-10-21)); the dev-docs findings below were made with `gpt-4.1-mini`. There is no second run with gpt-4o to compare with the previous round (decided 2026-10-11: the evaluation focuses on v2 only). The previous round's Ragas run reportedly cost almost $900, so the stage is built to never pay for a call twice and never to spend more than `ragas.max_cost_usd`.

**What it scores** (`evaluation/ragas_eval.py`): for each corpus, `ragas.sample_per_corpus` questions of `filtered.jsonl` (all of them when there are fewer), in each of its 7 answer files: the 6 setups and the gold passage. Every setup is scored on the same questions, so the comparisons are paired.
- **The sample** is a seeded hash rank, not a shuffle: a smaller sample is the start of a larger one. The pilot's questions are part of the full run, and raising `sample_per_corpus` only adds questions.
- **`sample_per_corpus` is a cost guardrail** (decided 2026-10-12, at 300; 500 since, confirmed by the 2026-10-21 pilot), at ~$0.009 per question across its 7 setups with `gpt-6-luna` (~$0.037 with `gpt-4.1-mini`). The dev docs (15 and 20 questions) are scored in full. The full corpus has more questions than that in both corpora, so `eval` samples them, without stratification (see [Future work](future-work.md#a-stratified-ragas-sample)).
- **The Ragas columns:** `user_input` = the question, `retrieved_contexts` = the passages the generator read, `response` = the RAG answer, `reference` = the gold answer. The old script used the gold answer as `response` and the source chunk as `reference`.

**The metrics.** Ragas returns more than the score on the way, and everything is kept, since it costs nothing extra (decided 2026-10-11):

| Column | What it is |
|---|---|
| `context_precision` | Per retrieved chunk, the judge says whether it helps reach the gold answer; average precision over the ranks. `cp_verdicts` keeps the verdicts by rank (e.g. `10010`), `cp_useful` their sum. |
| `context_recall` | The gold answer's sentences the judge attributes to the contexts (`cr_sentences`, `cr_attributed`). |
| `faithfulness` | The response's statements the contexts support (`faith_statements`, `faith_supported`). |
| `answer_relevancy` | Cosine (bge-m3) between the question and a question the judge generates back from the response; 0 when the judge calls the response noncommittal (`rr_cosine`, `rr_noncommittal`). |
| `factual_correctness` | Ragas' f1, rounded to 2 decimals as Ragas does: the response's claims checked against the gold answer, and the gold answer's against the response. Ragas' defaults: mode f1, atomicity and coverage "low". |
| `fc_precision`, `fc_recall`, `fc_f1` | From the same verdicts: the response's claims the gold answer supports; the gold answer's claims the response covers; their F1. With the claim counts (`fc_response_claims`, …). |
| `fc_recall_ragas` | The recall inside Ragas' f1. It divides supported *response* claims by those plus uncovered *gold* claims, so it is inflated when the response has more claims than the gold answer, as RAG answers do (35–43 words against 22 on the dev docs). Compare it with `fc_recall`. |
| `rouge_l`, `token_f1`, `answer_similarity` | Judge-free, response against the gold answer: ROUGE-L F, SQuAD token F1, bge-m3 cosine. A sanity check on factual correctness. |

Each row also carries `abstained`, `answer_error`, `evidence_shown`, `response_words`, `n_contexts` and the question's attributes (`doc_id`, `question_type`, `kind`, `section`, evidence size) for the breakdowns.

**Factual correctness is reported both ways** (decided 2026-10-12), each with what it measures:
- **`factual_correctness`** is Ragas' standard score, comparable with other Ragas reports. It counts as wrong any true claim the gold answer doesn't state, such as the question's own wording, the source's name or an extra correct detail.
- **`fc_recall`** asks "is the right answer in it?": the share of the gold answer's claims the answer contains. `fc_precision` asks "does it say only that?", and `fc_f1` combines the two.

**For stage 9:** what the report does with this table is in [the report doc](report.md) and in [analysis/\_\_init\_\_.py](../src/chunkeval/analysis/__init__.py): the abstention views, the gold passage as the ceiling and as a check on the judge, the join with the retrieval table, and the questions to read by hand.

**Abstentions** (decided 2026-10-11, see [RAG answers](#rag-answers-answer)): the response metrics don't ask the judge. Relevancy, every factual-correctness score and the three text metrics are 0, which is what Ragas would give; faithfulness is NaN, to be reported next to the abstention rate, with a view where abstentions count as 1. Context precision and recall don't read the response, so they are scored as usual. An answer with an `error` has its response metrics NaN.

**How Ragas is called.** With Ragas' own prompts, output parsers and scoring code, but one prompt at a time instead of through `ragas.evaluate`:
- **Every judge call is cached** in `data/llm_cache/ragas_cache.jsonl` as soon as it returns, keyed by its rendered prompt and the judge settings. A re-run asks only what is new, a crash loses only the calls in flight, and identical calls are asked once: the same chunk judged for the same question in several setups, the same answer in several setups, the gold answer's claims (once per question, not once per setup). The cache is shared by every `run_id` and data dir, so another run pays only for the calls no run has made, e.g. a hybrid-retrieval run pays nothing for the gold-passage setup.
- **Until 2026-10-28 each run kept its own caches** (`runs/<run_id>/answer_cache.jsonl` and `ragas_cache.jsonl`). Those of `v2-pilot` and `full_data-v1.0` were merged into `data/llm_cache/` (24,607 answers, 61,225 judge calls, no key in two runs); the old files are left in place, unused. A dry run of `eval` on `full_data-v1.0` then found every call cached ($0.00).
- **API errors are not unusable outputs.** `ragas.evaluate` turns both into NaN. Here an API error caches nothing and stops the stage before anything is written; the re-run asks only it. An output Ragas can't parse, even after its 3 fix-up requests, is cached with its error (at temperature 0 it would fail again) and makes that metric NaN.
- **The judge client** sends the config's model, temperature and seed. It replaces Ragas' LangchainLLMWrapper, which sets the temperature and `n` on a shared client before each call, so concurrent calls could go out with each other's settings.
- **Checked against Ragas:** a test runs each of the 5 metrics through Ragas itself and through the stage on the same fake judge, and gets the same prompts and the same scores.
- **Two departures from Ragas' defaults,** both because they would only cost money: relevancy generates 1 question, not 3 (in 0.2.15 those are 3 separate calls at temperature ~0, i.e. the same question three times), and a claim check with no claims isn't sent.

**The cost guard:**
- **Before any call,** the stage counts the input tokens of every call it doesn't have cached (the prompts exactly as they will be sent) and estimates the output tokens: from the cache's mean per prompt once it has 5 calls of a kind, else a fixed guess. It logs the estimate per prompt and **refuses to start above `ragas.max_cost_usd`** ($100). `--dry-run` stops after the estimate.
- **While running,** it stops asking once the new spend reaches the cap. What was paid is cached, nothing is written, and the re-run continues.
- **Prices** are in `ragas.prices`, per judge model; a config whose OpenAI judge has no price fails at load.
- **The bill is at most what is reported:** prompt-cache hits are priced as full input. In practice there are none: every Ragas prompt's fixed part is under OpenAI's 1,024-token caching minimum.

**What it writes:**
- **`runs/<run_id>/ragas/per_question.parquet`:** one row per (question_id, corpus, setup). `method` and `overlap` are empty for the gold passage.
- **`runs/<run_id>/ragas/details.jsonl`:** per row, the texts and everything the judge returned: each chunk's verdict and reason, the attributed sentences, the statements and claims with their verdicts, the generated question. This is where to read why a score is what it is.
- **`runs/<run_id>/ragas/cost.json`:** calls, tokens and USD of this run (with its estimate) and of all the calls the table rests on, by prompt, plus the cost per question across all setups, to size `sample_per_corpus`.
- **`data/llm_cache/ragas_cache.jsonl`:** every judge call of every run, with its output or error, model, tokens and cost.
- **`runs/<run_id>/ragas_summary.json`:** per corpus and setup, each metric's mean and NaN count; the summary lines are also logged.
- **A pilot** writes the same files, covering only its questions; the full run rewrites them. The table covers every corpus with questions: a corpus outside `--corpus` comes from the cache only, and is left out with a warning if a call is missing.

**On the full corpus, the dry run with `gpt-4.1-mini`** (2026-10-21, with the answers of [the title and the setting rule](#the-title-and-the-setting-rule-2026-10-21); 500 of 852 papers and 500 of 904 reports questions × 7 setups = 7,000 rows): 57,002 judge calls, none cached, ~66.4M input and ~5.3M output tokens, **~$35.03** (cap $100). By prompt: context precision $14.65 (25,777 calls, one per chunk), NLI checks $10.65, context recall $6.72, claims $1.13, statements $1.09, generated questions $0.80. At ~1.6 calls/s, about 10 hours.
- **Against the first answers** (2026-10-20: 55,732 calls, ~$33.61): +$1.42. The contexts now carry the document titles, and about 180 more rows are answered rather than abstained, so they get the response metrics.
- **With `gpt-6-luna`** (2026-10-21, after its pilot): 54,744 calls to ask and 2,256 cached from the pilot, ~63.7M input and ~5.6M output tokens, **~$9.14**. At the pilot's ~1.05 calls/s (`max_workers: 2`), about 14.5 hours.

**On the full corpus, the full run** (2026-10-21/22, `gpt-6-luna`, `max_workers: 2`): 54,652 calls in 14 h 06 min (~1.08 calls/s) for **$8.77**, against the $9.14 estimate. Identical calls were asked once, so 92 fewer than estimated.
- **The table** rests on 56,908 calls, the pilot's included: **$9.13**, $0.0091 per question across its 7 setups. Context precision is 41% of it ($3.72), the NLI checks 32% ($2.94).
- **Every output parsed,** with no fix-up requests and no API errors. The only NaNs are the abstentions' faithfulness.
- **The results are in [the report doc](report.md)** ("On the full corpus").

**On the dev docs** (35 questions, 2026-10-11):
- **Calls and cost:** 245 rows need 2,064 judge calls, against 2,660 if identical calls were asked again (22% fewer). Context precision is about 40% of the cost (909 calls, one per chunk).
- **The pilot** (`--pilot 2`: 2 questions per corpus, 28 rows): 252 calls in 2 min 55 s for **$0.165**, 10% above its $0.149 estimate. Every output parsed, with no fix-up requests and no prompt-cache hits.
  - **Output tokens per call,** now used by the estimate: context precision 66, context recall 108, statements 118, claims 100, NLI checks 312, question 30.
- **The full dev run** (2026-10-12): the rest was 1,809 calls in 20 min for $1.124, against a $1.177 estimate. The table's 2,061 calls cost **$1.29** in all: $0.037 per question across its 7 setups. Every output parsed again.
  - **The full corpus,** at 300 questions per corpus, would be ~$22 and ~35,000 calls, about 6 h at `max_workers: 2` (~1.6 calls/s); at the current 500, ~$37 and ~10 h. `eval --dry-run` gives the exact estimate.
- **Ragas' factual correctness scores correct answers 0** when they restate the question or add a true detail, because the judge checks the answer's claims against the gold answer alone. On 2 of the pilot's reports questions, every retrieved setup scored 0:
  - an answer listing exactly the gold answer's six partners had every claim phrased "…provided climate change resources to CABEI's member countries", wording from the question that the gold answer lacks;
  - an answer correctly attributed a figure to António Guterres, whom the gold answer doesn't name.

  `fc_recall` was 1.0 for both, since every gold claim was in the answer. Ragas' f1 is 0 because its recall counts answer claims, of which none were "supported".
- **On the whole dev set,** 143 of the 190 answered rows contain every gold claim (`fc_recall` 1), yet their mean Ragas f1 is only 0.60, and 18 of them (6 questions) score 0. The answers hold 4.2 claims on average, the gold answers 2.7. Over the answered rows: Ragas f1 0.54, `fc_precision` 0.47, `fc_recall` 0.83. Hence both versions are reported (see above).
- **The judge is consistent with the spans:**
  - The gold passage scores exactly 1.000 on context precision and recall in both corpora: no attribution error.
  - On the retrieved setups, "some chunk useful" agrees with stage 6's `span_hit@5` in 91% of the rows.
  - Most disagreements are retrievals that do hold the answer: in 15 rows (4 questions) the judge found a useful chunk although none overlaps the gold evidence (context recall 0.54 there). The fact also appears elsewhere: 20 such chunks from the same document, 8 from report_16, the near-duplicate of report_15. The span metrics count those as misses.
- **Which metrics respond to retrieval** (answered rows, gold passage vs the setups):
  - **Factual correctness does:** `fc_recall` 0.90 vs 0.82, and 0.86 with the evidence shown vs 0.06 without.
  - **Faithfulness and relevancy don't:** faithfulness 0.96 vs 0.94, near its ceiling; relevancy 0.79 vs 0.81, as flat as in the previous round.
- **No chunker ranking:** the setups differ by a few questions, as in retrieval (the paired CIs are stage 9's).

### The judge: gpt-6-luna instead of gpt-4.1-mini (2026-10-21)

**Why it came up.** On 2026-10-14 OpenAI announced that `gpt-4.1-mini-2025-04-14` shuts down on 2027-01-14, and named `gpt-6-luna` as its replacement. `gpt-6-luna` costs a quarter as much: $0.10 / $0.50 per 1M input / output tokens, against $0.40 / $1.60. But it is a reasoning model without a dated snapshot, and the judge had been a dated, non-reasoning snapshot for exactly those two reasons.

**Its parameters,** from one test call:
- it accepts temperature 0 and the seed only with `reasoning_effort: "none"`, and then uses no reasoning tokens; without it, the API answers "'temperature' does not support 0.0 with this model";
- its responses report the model as plain `gpt-6-luna`, with no system fingerprint, so an update behind the name can't be seen.

**The comparison.** Both judges scored the 20-question pilot: 280 rows and 2,256 calls each, cached under each judge's own keys. Each also answered 60 context-precision calls a second time.

| | `gpt-4.1-mini` | `gpt-6-luna` |
|---|---|---|
| Pilot cost (estimate) | $1.48 ($1.40) | $0.36 ($0.40) |
| Pilot time at `max_workers: 2` | 27 min | 36 min |
| Unusable outputs, fix-up requests | 0, 0 | 0, 0 |
| Same verdict when asked again (of 60) | 58 | 59 |
| Gold passage: context precision, context recall | 0.975, 1.000 | 0.987, 1.000 |
| Gold-passage answers marked partly unfaithful (of 40) | 4 | 5 |
| Chunks judged useful: overlapping the evidence (225) / same document, no evidence (720) / other documents (255) | 96.0% / 33.5% / 5.5% | 93.8% / 11.2% / 0.4% |
| Means: context precision, context recall, faithfulness, answer relevancy, factual correctness | 0.785, 0.915, 0.979, 0.838, 0.698 | 0.736, 0.886, 0.950, 0.836, 0.650 |

The two judges agree on 83.5% of the 1,200 chunk verdicts (Cohen's kappa 0.63). Row by row, their scores correlate at 0.81 for context precision, 0.86 for context recall, 0.88 for answer relevancy, 0.79 for factual correctness, and only 0.24 for faithfulness, which sits near 1 for both.

**Reading the disagreements:**
- **Context precision: `gpt-4.1-mini` rewards chunks for being on-topic.**
  - The judges disagree on 167 same-document chunks without evidence, which `gpt-4.1-mini` judges useful and `gpt-6-luna` doesn't. In a random sample of 12, `gpt-6-luna` was right every time: the chunk is about the topic but doesn't hold the answer.
  - `gpt-4.1-mini` accepts a title page with the author list (report_91:s37475), and gives reasons like "although the exact sequence Xmv42 is not explicitly mentioned…" (paper_64:s26876).
  - Every chunker retrieves on-topic chunks, so this lifts every setup and narrows the differences between them.
- **The 8 evidence chunks `gpt-6-luna` rejected** each hold only part of the evidence (2–73%), and its reasons are accurate: the chunk ends before the formula, or lacks one of two figures. That is defensible, but a chunker that splits the evidence across two chunks loses a little context precision under it.
- **Faithfulness: neither is clearly better.**
  - On the gold passage, they mark mostly the same answers. Each judge also makes errors of its own:
    - `gpt-6-luna` on paper_96:s30009: its statement adds "across all affected areas";
    - `gpt-4.1-mini` on report_91:s37475: it reads a stated goal as a claim that the goal was reached.
  - Over all rows, `gpt-6-luna` extracts fewer statements and rejects more of them: 33 of 674, against 22 of 857. Only 1 of those 33 rejects a name the statement took from the question.
- **Factual correctness: `gpt-6-luna` is harsher.** 27 rows differ by 0.5 or more, and `gpt-4.1-mini` scores higher in 18 of them.
  - When an answer adds a true detail (a project's name, "in fish eggs"), `gpt-6-luna` keeps the claim whole and rejects all of it; `gpt-4.1-mini` splits off the extra part and rejects only that.
  - This deepens the factual-correctness weakness found on the dev docs (above). Read it together with `fc_precision`, `fc_recall` and the judge-free metrics.
- **Setup rankings:** the rank correlation between the judges' setup means ranges from 0.00 to 0.96 across corpora and metrics. Twenty questions per corpus are too few to separate the setups.

**Decision: `gpt-6-luna`, with `max_workers` kept at 2.**
- **For:** its context-precision verdicts follow the evidence; it costs a quarter as much (~$9 instead of ~$37 for the full sample); it has no announced shutdown.
- **Against:**
  - it isn't pinned, though every call is cached, so a finished run stays as it was;
  - it is slower: ~1.05 against ~1.38 calls/s at 2 workers;
  - its factual correctness is harsher.
- **What remains of `gpt-4.1-mini`:** its pilot calls stay in the cache and its price stays in `ragas.prices`.
- **Cost of the comparison:** about $1.90.
