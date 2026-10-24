# Report

Stage 9 (`src/chunkeval/analysis/`): the two per-question tables (retrieval, and Ragas when it has
run) → the report, with paired comparisons, bootstrap intervals and the questions to read by hand.

```bash
uv run chunkeval report          # no LLM and no API call: free, and safe to re-run (~10 s on the dev docs)
```

**What it writes** (`analysis/`): a new folder on every run, `runs/<run_id>/reports/<YYYYMMDD-HHMMSS>/`. The name uses the logs' timestamp format, with `-2` added when two runs fall in the same second. Earlier reports are kept: delete old folders by hand when they're no longer needed (~2.6 MB each on the dev docs). `reports/latest` is a symlink to the newest report, so one path always opens it. A folder appears only once it is complete: it is written as `<name>.tmp` and renamed at the end.
- **`report.md`:** the report, in reading order. It opens with how to read it and the run record, then covers the question set, retrieval, generation, breakdowns, chunk sizes, cost, planning and the questions to read by hand, and ends with the secondary views.
- **`cases.md`:** a card per question worth reading by hand (see below).
- **`run_record.json`:** what was measured: questions and documents per corpus, the Ragas sample, chunking settings, model and embedder pins, the answer prompt's hash, the analysis settings and the judge's cost.
- **`tables/*.csv`:** every table, with every metric. The Markdown shows only the headline ones.
- **`figures/*.png`:** the figures. Each one also has a table with the same numbers.
- **`secondary/`:** the secondary views' tables and figures.

**Checks before anything is computed** (`analysis/inputs.py`). The two long tables carry no provenance of their own, so the stage checks that they belong together and names the stage to re-run:
- the retrieval table holds exactly the questions of `filtered.jsonl` (else `retrieve`);
- the bge-m3 retrieval files are up to date (else `retrieve`);
- the Ragas table has every setup on the same questions and on the Ragas sample, and every retrieved row's `evidence_shown` equals the retrieval table's `span_hit` at `top_k` (else `answer`, then `eval`);
- `details.jsonl` has exactly the table's rows (else `eval`).

Without a Ragas table, the report covers retrieval only.

**Method names** (decided 2026-10-22; `analysis/inputs.py`). The report calls the chunkers fixed-size, sentence and section, so its setups are `fixed-size_no_overlap`, …, `section_overlap`. Everything is loaded and checked under the config's keys (`character`, `recursive`), then renamed: the stored tables, file names and `run_record.json` keep the keys, and the run record's table in `report.md` maps the two. Why the new names: [corpus](corpus.md#chunk-chunk).

**The ranking rule** (decided 2026-10-12; `analysis/stats.py`). Every comparison is paired: the difference is taken per question, then averaged. A setup is called higher or lower than the baseline only when both of these hold:
1. its paired bootstrap CI excludes 0;
2. its Wilcoxon signed-rank p-value, Holm-adjusted within its family, is below `1 − analysis.ci`. A family is one corpus, one metric and one k: the 5 setups compared with the baseline.

The reasoning:
- **Many comparisons.** The report makes well over 100 comparisons (6 setups, ~10 metrics, 2 corpora, two k views). A 95% CI excludes 0 by chance about 1 time in 20, so even if every chunker were identical, ~5 "winners" would appear.
- **The two tests answer different questions.** The CI says how big the difference is. The Wilcoxon test says whether the per-question differences lean one way more than chance would give; it uses only the questions that changed.
- **Holm** keeps the chance of even one false winner per family below alpha. It sorts the family's p-values, multiplies the smallest by 5, the next by 4, and so on, keeping them non-decreasing. It is less strict than multiplying them all by 5 (Bonferroni).
- **Dev example:** on reports, `fc_recall` of `sentence_overlap` − baseline is +0.16 with CI [0.03, 0.32], so the CI alone would rank it. But only 4 of the 20 questions change, and its p of 0.066 becomes 0.33 after Holm: no difference shown.

Each comparison also lists how many questions are better, equal or worse (`+ / = / −`), with the number of questions behind it and how many pairs were dropped for NaN.

**Intervals by document** (decided 2026-10-12). The bootstrap redraws questions, as if each were independent. But questions of one document fail or succeed together: a garbled OCR layer, or a near-duplicate document, affects them all.
- **What is added:** every paired comparison also gets a CI that redraws whole documents with all their questions.
- **The flag:** "few documents" marks a difference whose question-level CI excludes 0 but whose document-level CI doesn't. Such a difference rests on a few documents. Dev example: 4 of the 7 papers questions behind `section_no_overlap`'s +0.17 `fc_recall` are from paper_4.
- **Its limit:** the document-level CI needs many documents. Below 20 (`stats.MIN_CLUSTER_DOCS`) it is computed but marked unreliable, as on the dev docs (5 per corpus).

**What the report shows** (decided 2026-10-12):
- **Retrieval:**
  - the span metrics at `top_k` and at the token budget, with the oracle (chunks and tokens needed) and random as references;
  - the paired comparisons with the baseline;
  - the span metrics as k grows;
  - the questions whose result changes between setups: the chunker differences come only from these;
  - per document: hit rates and the share of the corpus's chunks;
  - distractors: which documents' chunks take the top-k slots of other documents' questions;
  - the embedder check: both rankings side by side, Kendall's tau, the chunker's gain against the embedder's, and the spread between setups under each embedder.
- **Generation:**
  - **where the answers end up:** each row by whether retrieval showed the evidence, then by what the answer did with it (all gold claims, some, none, abstained). Without the evidence, an answer with gold claims was found elsewhere (or guessed), and one without is a hallucination risk;
  - the means with the gold passage as the ceiling, and the paired comparisons with the baseline;
  - which metrics respond to retrieval: per question, gold passage − mean of the setups;
  - the judge's check on the gold passage;
  - the abstention views;
  - the judge against the spans: agreement, and where the judge's useful chunks come from;
  - the text metrics against `fc_recall`, the scores split by `evidence_shown`, and score distributions.
- **Breakdowns:** by question type, by number of evidence quotes (1 vs 2+: where chunk boundaries should matter most), by evidence length (terciles per corpus) and by kind (only when two-span pairs exist: single vs two-span, on again since 2026-10-19). There is no breakdown by section: on the dev docs, 35 questions have 33 sections.
- **The rest:**
  - chunk sizes and cost (the judge's and the generator's tokens);
  - **the smallest detectable difference** per metric, now and at `ragas.sample_per_corpus` questions, to size the final run before paying for the judge. It uses the spread of the per-question differences, at the level Holm requires of a family's smallest p-value, with 80% power;
  - the run record.
- **The case file** (`cases.md`). A question gets a card when:
  - the gold passage abstains or misses gold claims;
  - the judge scores the gold passage's context below 1;
  - Ragas' factual correctness scores a correct answer 0;
  - the setups differ on `span_hit` or `fc_recall`;
  - no setup retrieves the evidence;
  - or the judge and the spans disagree.

  Each card has the question, gold answer and evidence quotes. Per setup, it shows the top-k chunks (✓ overlaps the evidence, · same document, else the other document's id), the judge's verdicts by rank, the scores and the outcome. Then come the answers, with identical ones grouped.
- **Secondary views** (diagnostics, not results; in `secondary/` and at the end of `report.md`):
  - context tokens handed over against span recall;
  - answer quality by the rank of the first chunk holding the evidence;
  - metric correlations;
  - scores by answer length.
- **A small set is flagged, never skipped.** Every output is produced on the dev set too. Cells under 10 questions are marked †, and `report.md` opens with a warning below 100 questions per corpus.

**Figures:** colours checked with a colour-blind-safety validator.
- A chunking method keeps its hue in every figure (blue, orange, aqua), and overlap is the marker fill: filled with overlap, hollow without.
- The gold passage and the oracle are ink diamonds.
- The outcome bars use a blue ramp for "evidence shown" and an orange one for "not shown".

**On the dev docs** (35 questions, 2026-10-12; 8 s; 31 tables, 13 figures, 24 cards):
- **Generation comparisons:** none of 130 passes the ranking rule.
- **Retrieval comparisons:** 6 of 160 pass, all on `precision` and `iou`, for setups with overlap. That is mechanical: precision and IoU count text that overlapping chunks repeat once, while `context_tokens` (what the generator reads) counts it every time. The report says so next to the table.
- **The judge checked on the gold passage:** context precision and recall are 1 on all 35 gold rows.
  - Where the judge and the spans disagree, its useful chunks come from the same document outside the evidence (20) or from report_16 (8).
  - Ragas' factual correctness follows claim precision almost exactly (Spearman 0.99). Both fall with answer length (−0.3 to −0.46), while `fc_recall` hardly does (−0.13 to −0.23).
- **Which metrics respond to retrieval** (gold passage − setups):
  - shown for context precision in both corpora, and for factual correctness, `fc_precision`, ROUGE-L and token F1 on reports;
  - `fc_recall`'s gap (+0.07 papers, +0.10 reports) isn't shown with 15–20 questions.
- **Smallest detectable difference at 300 questions per corpus** (the sample size then), from the dev spread (recompute on the final corpus): `span_recall` 0.04–0.05, `factual_correctness` 0.05–0.07, `fc_recall` 0.07–0.10.

**On the full corpus** (2026-10-22, `reports/20261022-200259`; 20 s; 31 tables, 13 figures, 964 cards). Retrieval covers every question (852 papers, 904 reports); generation covers the Ragas sample (500 per corpus × 7 setups), judged by `gpt-6-luna`. The baseline is `fixed-size_no_overlap`.

- **The result:**
  - **Section chunking is the best choice in both corpora, with or without overlap.** It has the highest span recall at k = 5 (+0.060 papers, +0.056 reports over the baseline) and the highest or near-highest `fc_recall` (+0.061 to +0.065 papers, +0.054 to +0.062 reports). All four pass the ranking rule.
  - **Overlap helps fixed-size and sentence a little and section not at all.** `section_overlap` and `section_no_overlap` are within 0.01 of each other on every headline metric.
  - **`fixed-size_no_overlap` is the weakest setup.** `sentence_no_overlap` is no better on reports (`fc_recall` −0.001).
  - **The differences are small: 3–7 points at best.** Among the better setups (`sentence_overlap`, `fixed-size_overlap` and both section ones), they are below what 500 questions can detect.
- **Retrieval** (68 comparisons pass: 32 of them are the mechanical precision/IoU gain of overlap, and 7 are on context tokens):
  - **span_hit@5:** papers 0.86–0.90, reports 0.81–0.85. On papers, section passes on span_hit (+0.033 / +0.028) and MRR (+0.043 / +0.045), and every setup passes on span recall. On reports only span recall and `all_evidence` separate the setups: span recall `sentence_overlap` +0.028, section +0.056 / +0.054.
  - **At equal context** (1,300 tokens), the same ranking holds. On reports, section even hands over fewer tokens at k = 5: 1,199–1,236 against ~1,287, because its chunks are smaller (mean 213–219 tokens, p5 53–54).
  - **Fewer chunks hold the evidence:** section covers all of it with 1.02–1.05 chunks (248–265 tokens), fixed-size with 1.24–1.25 (318).
  - **Only a minority of questions separate the setups:** span_hit differs between setups on 181 of 852 papers questions (21%) and 246 of 904 reports questions (27%). No setup retrieves the evidence for 33 papers and 53 reports questions.
  - **Distractors are mostly near-duplicate documents:** report_84/86, report_89/90, report_111/112, paper_79/109, paper_18/20.
  - **The embedder is as big a lever as the chunker.** Qwen3-Embedding-0.6B is below bge-m3 in every setup (span recall −0.016 to −0.054), about as much as the best chunker gains over the baseline. Section stays in the top two under Qwen3 too (span recall Kendall tau 0.60 papers, 0.87 reports).
- **Generation** (35 comparisons pass):
  - **The gold passage is the ceiling:** `fc_recall` 0.84 papers, 0.79 reports; the setups reach 0.63–0.74.
  - **Every metric responds to retrieval except faithfulness.** The gold passage beats the setups by 0.12–0.13 on `fc_recall`, 0.24–0.28 on context precision, 0.13–0.19 on context recall and 0.08–0.09 on answer relevancy. That last one was flat (~0.54) in the previous round. Faithfulness stays at 0.92–0.95 everywhere, gold passage included.
  - **Papers:** every setup is above the baseline on `fc_recall` (+0.051 to +0.068). `sentence_overlap` and section also pass on context precision (+0.044) and context recall. Only `sentence_overlap` passes on Ragas' factual correctness (+0.040) and answer relevancy (+0.035).
  - **Reports:** `sentence_overlap` and section pass on `fc_recall` (+0.048 to +0.062). `fixed-size_overlap` and `section_overlap` pass on factual correctness (+0.048 / +0.050), section on context recall. Nothing passes on context precision.
  - **No setup passes on faithfulness or abstention.**
  - **Whether retrieval shows the evidence decides the answer.** With the evidence shown, `fc_recall` is 0.75–0.84; without it, 0.06–0.13, and 52–74% of those rows abstain. Evidence at ranks 2–5 is still used: `fc_recall` 0.74–0.82, against 0.80–0.83 at rank 1.
  - **Abstentions:** 9.6–13% per setup on papers, 14–19% on reports. The gold passage abstains on 3 and 18 of 500.
  - **Hallucination risk:** 12–24 rows per setup in 500 answer without the evidence shown and without a gold claim.
- **The judge:**
  - **It passes the check on the gold passage.** Context recall is 1 on 100% and 99% of the gold rows, context precision on 99% and 97%. Its chunk verdicts agree with span_hit in 92–96% of the rows (kappa 0.71–0.85).
  - **Where it finds a useful chunk the spans don't,** the chunk comes from the same document (209 papers, 230 reports) or from a near-duplicate (report_84, report_82, report_118).
  - **The ceiling understates the answers.** In a sample of the 267 gold-passage answers that miss a gold claim:
    - about half are real omissions by the generator ("subspecies", "instead of direct land acquisition");
    - about a third are the judge being strict ("only at 80%" read into "at 80%");
    - one is its own claim-splitting error (paper_95:s5612).

    The comparisons are paired, so this strictness affects every setup alike.
- **Read `fc_recall`, not Ragas' factual correctness.** 927 of the 4,300 answered rows that contain every gold claim score 0 on factual correctness.
  - In the sampled rows, the answer names a setting the gold answer leaves out ("in Europe", "of Kanjanaburi province", "in tsetse flies"). The judge then counts the whole claim as unsupported by the gold answer.
  - The answer prompt asks for the setting since 2026-10-21, and `gpt-6-luna` rejects whole claims (see [the judge comparison](evaluation.md#the-judge-gpt-6-luna-instead-of-gpt-41-mini-2026-10-21)).
  - Longer answers score lower on factual correctness (Spearman −0.24 / −0.31) and on faithfulness (−0.33 / −0.31), and less so on `fc_recall` (−0.15 / −0.26).
- **Breakdowns:**
  - **Two or more evidence quotes:** `fc_recall` is lower everywhere, gold passage included (0.74 against 0.86 on papers). On reports, section and `sentence_overlap` lead there (0.61–0.62 against 0.54).
  - **Long evidence** (upper tercile) lowers `fc_recall` by 0.06–0.17 against the short tercile.
  - **Definitions on reports** are the hardest type (0.50–0.61).
  - **Two-span pairs** (22 and 40 in the sample) are too few to read.
- **Smallest detectable difference at 500 questions per corpus:** `fc_recall` 0.062, factual correctness 0.053–0.054, context precision 0.048–0.050, answer relevancy 0.045–0.048, faithfulness 0.032–0.034. Detecting 0.05 on `fc_recall` would take ~770 questions per corpus, about all of them: papers has 852 and reports 904.

