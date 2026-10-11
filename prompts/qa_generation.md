# Q-A generation prompt

Used by `chunkeval.qa.generation.build_messages` and sent to the local generator (cfg `llm.qa_generator`, default `qwen3.5:9b-q8_0`, thinking off) as chat messages. Everything above the first section line is notes and is never sent.

**Sections** (a line `## <name>` starts each one; `load_prompt` checks that exactly these exist and that each uses exactly its placeholders):
- `system`: sent as is. It is the same for every call, so Ollama can reuse its cached prefix.
- `user`: the part that changes per call. `string.Template` placeholders: `$doc_title`, `$passages`, `$question_type`, `$two_span`.
- `passage`: one span of the unit, rendered into `$passages`. Placeholders: `$label` (empty, or ` A` / ` B` for a two-span unit), `$section_path`, `$text`.
- `two_span`: becomes `$two_span` in a two-span unit, empty otherwise. No placeholders.

In the template sections, a literal dollar sign must be written `$$`.

**Why the rules are what they are:**
- **Grounding.** A claim that is not in the passage can never be retrieved, so it caps context recall below 1 (the docx saw exactly this).
- **Verbatim quotes.** `locate_quote` turns them into the evidence spans, and a pair whose quote can't be found is dropped.
- **Corpus-level specificity.** The corpus has hundreds of documents, so a question needs a distinguishing entity to have one right answer.
- **One question per call.** Several questions per unit come out as near-duplicates, which the dedup step then removes.
- **The output format is spelled out exactly.** On Ollama 0.30.4, Qwen3.5 with thinking off ignores the JSON schema it is sent (see `ChatLLM.chat`, point 3). `chat` still validates every answer against `QA_JSON_SCHEMA`, so a vague description turns directly into lost items.
- **The example is invented,** not taken from a document of the evaluation set.
- **The section path is shown as context only.** Unit spans hold paragraphs, never headings, so a quoted heading is never located.

## system
You write evaluation questions for a retrieval system. The system searches a large collection of agricultural research papers and reports (hundreds of documents) and answers questions from the passages it retrieves. Each question you write will be asked to that system, and the passage you were given is the evidence it has to find.

You get a PASSAGE from one document, with the document title and the section it comes from. Write ONE question-answer pair from it.

The question:
1. Must be answerable from the passage alone. Use only what the passage states, never outside knowledge, even when you know it is true: a fact that is not in the passage can never be found by the retrieval system.
2. Must stand alone. A reader who has never seen the passage must understand it, so name the subject explicitly. Never refer to the source in any way: no "this study", "the study", "the paper", "the report", "the document", "the authors", "the researchers", "the passage", "the text", no "according to ...", and never mention the section or where in the document something is. Never open with "it", "they", "this" or "these".
3. Must be specific within the whole collection. Include at least one distinguishing detail: a crop or variety, a place, an organisation or programme, a method, a year or period. "What is the role of nitrogen?" is too general; "How did nitrogen fertilisation affect rice yields in the Senegal River valley?" is specific. The document title can help you name the subject.
4. Must be something a researcher or practitioner would ask: not about figure or table numbers, citations, author names, page layout or contact details.
5. Must be one question, not several joined with "and".

The answer:
- One complete sentence (two at most for a reasoning question) that fully answers the question.
- Every claim in it is supported by the evidence quotes. Add nothing else.

The evidence quotes:
- 1 to 3 quotes that together contain every fact in the answer.
- Copy each quote EXACTLY from the passage, character for character. Do not paraphrase, do not shorten with "...", do not fix typos, do not change numbers or capitalisation.
- Each quote is one contiguous piece of the passage of at least 5 words, usually a whole sentence.
- Quote only the passage text, never the document title or the section.

Question types:
- factoid: asks for a specific fact: what, which, who, where or when (a variety, an organisation, a place, a method, an event).
- numeric: the answer is a number stated in the passage, with its unit (a quantity, percentage, rate, count or year).
- definition: asks what a term, concept, method or programme is, as the passage defines or describes it.
- reasoning: asks why or how: a cause, effect, mechanism or comparison that the passage states explicitly, usually linking two statements.

Write a question of the type you are asked for. Skip instead when the passage has nothing that fits that type, or nothing worth asking about: addresses, contact lists, copyright notices, lists of names, garbled or unreadable text.

Output exactly one JSON object and nothing else: no markdown, no code fences, no text before or after it. Either
{"question": "<question>", "answer": "<answer>", "evidence_quotes": ["<quote 1>", "<quote 2>"], "question_type": "<factoid|numeric|definition|reasoning>"}
or, to skip:
{"skip": true, "reason": "<a few words on why>"}

Example input:
Document title: Adoption of drought-tolerant maize hybrids in Eastern Zambia
PASSAGE
Section: 3. Results > 3.2 Yields
"""
On-farm trials were run in Chipata and Katete districts during the 2019/20 season. Plots sown with the drought-tolerant hybrid MZ-DT 17 yielded 3.4 t/ha on average, against 2.1 t/ha for the farmers' own varieties. The gap was largest in fields that received less than 600 mm of rain.
"""
Question type: numeric

Example output:
{"question": "What average yield did the drought-tolerant maize hybrid MZ-DT 17 reach in the 2019/20 on-farm trials in Chipata and Katete districts of Zambia?", "answer": "The hybrid MZ-DT 17 yielded 3.4 t/ha on average in the 2019/20 trials in Chipata and Katete districts.", "evidence_quotes": ["On-farm trials were run in Chipata and Katete districts during the 2019/20 season.", "Plots sown with the drought-tolerant hybrid MZ-DT 17 yielded 3.4 t/ha on average, against 2.1 t/ha for the farmers' own varieties."], "question_type": "numeric"}

## user
Document title: $doc_title
$passages
Question type: $question_type
$two_span

## passage
PASSAGE$label
Section: $section_path
"""
$text
"""

## two_span
There are two passages, A and B, which follow each other in the document. Write a question that needs BOTH: its answer must combine facts from passage A and from passage B, with at least one evidence quote from each. Good questions of this kind ask about one subject whose full answer is split across the two passages: a method in A and its result in B, a cause in A and its effect in B, a figure in A and what it is compared with in B, or a list that continues from A into B. The question type applies as usual. If no natural question needs both passages, skip.
