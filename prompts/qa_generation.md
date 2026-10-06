# Q-A generation prompt — OUTLINE (write the final prompt yourself)

Used by `chunkeval.qa.generation`. The model is the local Ollama generator (cfg `llm.qa_generator`, default `qwen3.5:9b-q8_0`, thinking off). It is sent as **chat messages** (system + user). The old script sent a raw string with no chat template.

## System message — what it must establish
- TODO(you): the role. Who is writing these questions, and for what purpose? Say the questions evaluate a *retrieval* system over a large mixed corpus.
- TODO(you): the grounding rule. Use only the PASSAGE. Explain why: if the model brings in outside knowledge, that claim can never be found by retrieval, which caps context recall below 1. The docx observed exactly this.

## User message — required sections
1. **PASSAGE**: the generation unit text. TODO(you): decide whether to include the section heading path as extra context. It helps the model write specific questions, but the heading text must **not** be quotable evidence unless it is part of the canonical text span.
2. **Task rules** (carry these over from `build_prompt` in ragas_evaluation.py:186 and tighten them):
   - The question must stand alone: name the specific subject (crop, region, method, year). Ban openings like "this study / it / they".
   - **Corpus-level specificity.** The corpus has hundreds of documents, so "What is the role of nitrogen?" is a bad question. Hint: ask for at least one distinguishing entity, such as a place, crop, dataset, number or programme name.
   - The answer must be complete but concise. Think about whether you want 1–2 sentences, and remember factual correctness will be scored later.
   - `evidence_quotes`: 1–3 **verbatim** substrings of the PASSAGE that together support the answer. Say "copy exactly, do not paraphrase, do not fix typos". The locator depends on this.
   - `question_type`: one of factoid / numeric / definition / reasoning. TODO(you): decide whether you ask for a specific type per call (to control the mix) or let the model choose.
   - **Two-span variant:** the passage holds two adjacent paragraphs, and the question must need a quote from *each*. TODO(you): write this as a separate instruction block that is switched on by a flag.
   - Escape hatch: if no good question exists, return `{"skip": true}`. This replaces the old "N/A / I don't know" convention.
3. **Output format:** JSON only, matching the schema in `chunkeval.qa.generation.QA_JSON_SCHEMA`. The prompt must spell the format out **exactly**: every field name, its type, and "no markdown, no code fences". The schema is still sent as `response_format`, but **it is not enforced for this model**: on Ollama 0.30.4, Qwen3.5 with thinking off (`reasoning_effort: "none"`) ignores the schema and can answer in markdown (see `ChatLLM.chat`, point 3). `chat` validates every answer against the schema and raises `LLMOutputError` on a mismatch, so a vague format description turns directly into lost items.
   - Showing the exact JSON shape (field names with placeholder values) is the most reliable way to get it right.
   - The escape hatch `{"skip": true}` must also be valid under `QA_JSON_SCHEMA` (e.g. `anyOf` of the full object and `{"skip": true}`). Otherwise every skip is rejected as a schema mismatch.

## Tips
- One or two few-shot examples from *your* domain (agriculture / CGIAR) help a lot. Keep them short, and never take them from documents in the evaluation set.
- Use temperature around 0.7 for diversity. The critic and the quote locator catch the failures.
- If you ask for N questions per unit, the questions within a unit tend to be near-duplicates, which the dedup step then removes. One question per call is cleaner.
