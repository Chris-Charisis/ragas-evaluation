# RAG answer prompt — OUTLINE (write the final prompt yourself)

Used by `chunkeval.evaluation.answering`. The generator is fixed across all 12 chunking configs, so differences in faithfulness and relevancy come from retrieval and not from the generator.

The docx recommendations this prompt implements:
- Relax the "1–3 sentences" constraint. Relevancy was flat at about 0.54 everywhere, and a judge that penalises incomplete answers cost us points.
- Add an **abstention** instruction. Low recall combined with low faithfulness means the model answered when the evidence wasn't there.
- Optionally ask for **citations** to chunk numbers. They make answers easier to debug, but think about whether citation markers affect Ragas faithfulness or relevancy scoring. Hint: you can strip them before evaluation and keep the raw version separately.

## Structure
- System: TODO(you). Answer only from the CONTEXT. If the context is not enough, say so with a **fixed abstention phrase**, so that `answering.is_abstention` can detect it reliably.
- User:
  - TODO(you): the numbered context blocks `[1] ... [k]`
  - then the QUESTION
  - TODO(you): decide whether to show the chunk metadata (document title or section). Whatever you choose, keep it the same for every chunker. Otherwise the recursive method gets an unfair advantage from having section paths.

## Tips
- Use temperature 0 and a fixed max_tokens, for reproducibility.
- Keep the chunk order exactly as retrieved (rank order).
- Log the prompt token count for each sample. It feeds the context-tokens and cost metrics.
