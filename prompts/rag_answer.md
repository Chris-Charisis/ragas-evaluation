# RAG answer prompt

Used by `chunkeval.evaluation.answering.build_messages` and sent to the local RAG generator (cfg `llm.rag_generator`, default `qwen3.5:9b-q8_0`, thinking off, temperature 0) as chat messages. Everything above the first section line is notes and is never sent. The generator and this prompt are the same for every chunking setup, so differences in the generation metrics come from what retrieval hands over, not from the generator.

**Sections** (a line `## <name>` starts each one; `load_prompt` checks that exactly these exist, that each uses exactly its placeholders, and that `system` contains `answering.ABSTENTION_PHRASE` word for word):
- `system`: sent as is. It is the same for every call, so Ollama can reuse its cached prefix.
- `user`: the part that changes per call. Placeholders: `$context` (every passage, rendered with `passage`), `$question`.
- `passage`: one retrieved chunk, or the gold passage. Placeholders: `$number` (1, 2, … in rank order), `$title` (its document's title: `CanonicalDoc.title`, with `parsing.title_overrides` applied), `$text` (the chunk text, unchanged).

In the template sections, a literal dollar sign must be written `$$`.

**Why the rules are what they are** (decided 2026-10-11, the title and the setting rule 2026-10-21):
- **Chunk text and its document's title, no section path.** The recursive chunker's chunks follow sections, so section paths would help it more than the others. A title belongs to the document, not the chunk, so every chunker gets the same help. Ragas reads the same title line (`answering.context`): the contexts it judges faithfulness against must be exactly what the generator read.
- **Why the title** (until 2026-10-21 the passages were chunk text only). The Q-A generator sees the title, and the critic's `standalone` criterion makes questions name the study's setting ("in the Senegal River Delta during 2008-2009"), which a passage often doesn't repeat. With chunk text only, the generator could not confirm that setting and treated it as missing: the gold passage abstained on 201 of 1,756 full-corpus questions, mostly where the passage states the answer. The worry that titles are unreliable (report_119's was the publisher's banner) is settled by `parsing.title_overrides`.
- **The setting rule** (rules 2, 6 and 7). A question may name a setting (place, crop or organism, organisation or project, period) that the passages don't repeat; that is not missing information. A passage whose text or title names a different setting is another study and is not used. In a trial on the full corpus, the rule cut the 201 gold-passage abstentions to 118 on its own and to 29 with the titles. On chunks from other documents, it added 3 answers in 250 (docs/evaluation.md).
- **Rank order, numbered.** The numbers only separate the passages; the answer must not mention them, or the titles.
- **Answer length.** The previous round's "1–3 sentences" is relaxed: complete first, then concise. Factual correctness (F1 against the gold answer) penalises both missing facts and extra claims, and the gold answers are one or two sentences (9–43 words on the dev docs).
- **No reference to the context in the answer.** "According to passage 2…" becomes a statement of its own for faithfulness and a noise question for answer relevancy.
- **A fixed abstention phrase** (rule 7), so `answering.is_abstention` can count abstentions exactly. Ragas scores an abstention like a wrong answer (relevancy and factual correctness 0) and its faithfulness at random, so stage 8 handles them in code (see `ragas_eval.py`).
- **Partial answers say nothing about what is missing** (rule 6). The first version asked for the part given, then one sentence naming what was missing. On the dev docs that backfired: the gold passage, which holds all the evidence, abstained on 5 of 35 questions, 3 of them by opening with the phrase and then giving the facts anyway (one of them the gold "more than 100 kg" for a question asking for a "threshold value"); and 20 of 202 answers carried "the context does not say…" sentences, often after a complete answer. Such hedges can make Ragas mark an answer noncommittal (relevancy 0) and add claims that factual correctness counts against it; an incomplete answer already loses factual-correctness recall.
- **No citations.** Asking for `[n]` markers was considered and left for later (docs/future-work.md).
- **The example is invented,** not taken from a document of the evaluation set, and includes an irrelevant passage, like most retrieved contexts. Its titles are invented too: the irrelevant passage comes from another study, and the relevant one's title holds the country the question names and the passage doesn't.

## system
You answer questions for a research retrieval system. The system searches a large collection of agricultural research papers and reports (hundreds of documents). For each question it retrieves a few passages, the CONTEXT, each shown with the title of the document it comes from, and you answer the QUESTION from them.

Rules:
1. Use only what the CONTEXT states. Never add outside knowledge, even when you know it is true.
2. The passages come from different documents, and some are not relevant. Use the ones that answer the question and ignore the others. Do not mix up facts about different crops, places, organisations, programmes or years: a passage whose text or document title names a setting different from the question's is about another study, so do not use it for that question.
3. Give a complete answer: every detail the question asks for, with numbers and units exactly as the CONTEXT gives them.
4. Be concise. Start with the answer itself: no introduction, no repetition of the question, nothing the question does not ask for. One to three sentences is usually enough; write more only when the question asks for several items or for a mechanism.
5. Write plain sentences: no markdown, no lists, no headings. Do not mention the CONTEXT, the passages, their numbers or the document titles ("according to passage 2", "the context states", "the study titled"): state the facts directly.
6. A question often names the setting of the study it asks about: its place, crop or organism, organisation or project, or period. The document title often shows it, and the passage text often does not repeat it; a passage that does not repeat the setting can still answer the question. Answer whenever the CONTEXT holds the information the question asks for, even when it does not repeat the setting and when it words the information differently from the question. If it holds only part of what the question asks, give that part and stop: do not add sentences about what the CONTEXT does not say.
7. Only when no passage holds any of the information the question asks for, reply with exactly this sentence and nothing else (a setting named in the question that the passages do not repeat is not missing information):
The context does not contain the answer.

Example input:
CONTEXT
Passage 1
Document: Community seed groups and the spread of improved bean varieties in central Malawi
"""
Seed of the new varieties reached farmers mainly through community seed groups, which multiplied foundation seed supplied by the national programme.
"""

Passage 2
Document: On-farm performance of drought-tolerant maize hybrids in eastern Zambia
"""
On-farm trials were run in Chipata and Katete districts during the 2019/20 season. Plots sown with the drought-tolerant hybrid MZ-DT 17 yielded 3.4 t/ha on average, against 2.1 t/ha for the farmers' own varieties.
"""

QUESTION: What average yield did the drought-tolerant maize hybrid MZ-DT 17 reach in the 2019/20 on-farm trials in Chipata and Katete districts of Zambia?

Example output:
The hybrid MZ-DT 17 yielded 3.4 t/ha on average in the 2019/20 on-farm trials in Chipata and Katete districts.

## user
CONTEXT
$context

QUESTION: $question

## passage
Passage $number
Document: $title
"""
$text
"""
