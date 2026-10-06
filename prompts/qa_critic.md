# Q-A critic prompt — OUTLINE (write the final prompt yourself)

Used by `chunkeval.qa.filtering`. The model is `gpt-oss:20b` (local, via Ollama), which comes from a **different family** from the generator, so it is less likely to approve the generator's own mistakes.

The main fix over the old judge: the critic **sees the evidence passage**. The old `build_evaluation_prompt` (ragas_evaluation.py:424) saw only the Q and A, so it could not detect a hallucinated answer.

## Inputs to show
- PASSAGE: the generation unit, or both paragraphs for a two-span question
- QUESTION
- ANSWER
- EVIDENCE QUOTES. Hint: the locator has already confirmed they exist, so the critic only judges whether they *support* the answer.

## Criteria (score each from 1 to 5, with a one-line reason)
- TODO(you): **groundedness.** Is every claim in the answer supported by the passage?
- TODO(you): **answerability.** Does the passage answer the question completely?
- TODO(you): **standalone.** Does it make sense without the passage, with no "this study" or dangling pronouns?
- TODO(you): **specificity.** Would the question have one clear answer across a corpus of hundreds of agricultural documents?
- TODO(you): **usefulness.** Is it something a real user of the RAG system might ask, rather than trivia about figure numbers or author names?

## Output
- JSON with one integer score and one reason per criterion. Enforce it with structured outputs (JSON schema).
- Keep the decision rule (thresholds) in **code**, not in the prompt. Then you can change the thresholds without regenerating anything, and log *why* each pair was rejected.

## Tips
- Ask for the reason **before** the score in the JSON field order. The model then reasons first, which tends to make the score more reliable.
- gpt-oss has a reasoning-effort setting. Low or medium is enough here; check how Ollama 0.30 exposes it (`reasoning_effort` on /v1 or `think` on the native API).
- Calibrate: hand-label 50 pairs yourself, then compare them with the critic's keep/drop decisions and adjust the thresholds.
