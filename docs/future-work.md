# Future work

Extensions considered and left out for now, with what each would need. Ideas deferred during the work
are added here.

## Chunk-size sweep (`chunking.size_sweep`)

An optional extension for after the whole pipeline is complete, built only if we decide we want it. The config key exists and is validated, but no stage reads it yet.
- **What:** rechunk with all 3 methods × overlap on/off at other target sizes (e.g. `[128, 256, 512, 1024]`), index them, retrieve with the same questions, and score with the span metrics only: hit@k, span recall@k, MRR, IoU/precision, context tokens. No RAG answers and no Ragas, so no API cost.
- **Why:**
  - It checks whether the method ranking found at 256 holds at other sizes.
  - It tests whether 256 is a good size: 256 is inherited from the previous round, not tuned (see the comment in `configs/default.yaml`).
  - It measures the precision/recall trade-off against chunk size on a fixed corpus and a fixed metric. That is the controlled ablation the docx says its own size comparison lacked, since chunk size, judge and corpus all changed at once.
- **Open design points:**
  - **Compare at equal context.** At a fixed top-k, bigger chunks win recall almost automatically. Use the budget-matched view (`retrieval.budget_tokens`, scaled per size) as the main comparison.
  - **A cap per size.** 1024 is above `max_tokens: 512`, and the schema rejects a target above the cap. Use something like 2× the size per sweep size.
  - **Naming.** In one data dir, the extra chunk files and Weaviate collections need the size in their names, which `weaviate.collection_template` can't express yet. One data dir per size works today: an `extends` config with `paths.data_dir: data/size512` gets its own vectors and collections (see [configs for other experiments](development.md#configs-for-other-experiments)), but the report then covers one size per run.
  - **Weaviate memory.** Raise `GOMEMLIMIT` (see [setup.md](setup.md#weaviate-docker)).

## Citations in the RAG answers

Considered for `answer` and left out for now (decided 2026-10-11).
- **What:** label the passages `[C1]`…`[C5]` and ask the generator to end each sentence with the passages it used. Keep the raw answer, give Ragas a copy with the markers stripped (they would be noise in faithfulness' statements and relevancy's generated questions), and store the cited chunk ids.
- **Why `C`:** 25 of 191 papers chunks and 17 of 420 reports chunks on the dev docs already contain reference markers like `[12]`, `[5–11]` or `[1, 2]`, so a plain `[3]` copied from a chunk would be read as a citation.
- **What it would give, without the judge:** whether the answer cites a chunk that overlaps the evidence; combined with `evidence_shown`, which questions had the evidence but were answered from other chunks (distraction), and which were answered from a wrong chunk (e.g. report_16 for a report_15 question, the near-duplicate case of the pilot); the rank of the chunks actually used.
- **Limits:** a 9B model's citations are approximate (wrong block, missing ones), so they are a diagnostic, not a headline metric; check ~20 by hand first. A prompt change re-asks every answer.

## Random-context check of the abstention rule

Not a stage (decided 2026-10-11). Run once by hand on 2026-10-21, in the trial that added the document titles and the setting rule (see [evaluation](evaluation.md#the-title-and-the-setting-rule-2026-10-21)): 250 questions answered from 5 random chunks of other documents abstained every time, under all three prompts tried.
- **The sharper check** is the trial's other set: the top 5 bge-m3 chunks from other documents only, on-topic but from another study. The generator should refuse them unless another document holds the fact. It answered 17 of 250, 2 of them with another study's facts.
- **As a stage option** (e.g. `chunkeval answer --check`, ~500 calls, ~10 min), both checks would test any later prompt change the same way. It would need its own output file next to the answers, and questions whose top `k_max` hold at least `top_k` chunks of other documents (about 2 in 3 on the full corpus).

## A stratified Ragas sample

Not implemented (2026-10-12). When a corpus has more questions than `ragas.sample_per_corpus`, `eval` scores a seeded random subset (`ragas_sample`), so by chance it can over- or under-represent a document, a question type or the two-span pairs. A stratified draw would keep the question set's mix, e.g. proportionally per document and question type, while staying nested (a pilot is still the start of the full sample). It matters now: the full corpus has more questions than `ragas.sample_per_corpus` (500) in both corpora. The alternative is to score every question: about $0.04 each, ~$40 per 1,000 questions.

## Models on our own servers

Not implemented (2026-10-12; to revisit once the server details are known). Any role (generator, critic, judge) could use a model served behind the OpenAI API on one of our servers (vLLM, TGI, …). Today only two providers exist: `openai` always goes to api.openai.com, and `ollama` also calls Ollama-only endpoints for its checks and model digest. It needs:
- a third `provider` with its `base_url` in the config and an API key in `.env` (in `Secrets` and `require_secrets`), with no Ollama checks;
- `ChatLLM` (`core/llm.py`) and `ragas_eval.judge_client` / `judge_settings` built from them, and `ragas.prices` optional for it (cost 0);
- a provenance record instead of the Ollama digest: the weights can't be fingerprinted, so the server's model version must be pinned on its side and recorded in the config;
- checking what the server supports: token usage in its replies (the cost guard and token counts need it), `seed`, `reasoning_effort` and JSON-schema output.

As a judge, avoid the Qwen family, which wrote the answers, and the Gemma family, which wrote the gold answers.

## Models partly on the CPU

At the moment, `check_ollama` only logs a warning when a model isn't 100% on the GPU. `test_live_check_ollama` hard-codes that both generators must be 100%. There are two situations to handle:
- **Deliberate:** a MoE generator (only about 3B parameters active per token) is partly on the CPU by design and stays usable.
- **Accidental:** a dense model is pushed partly onto the CPU and gets several times slower. Typical causes: another model still loaded, embedding models running on the GPU, or a higher `OLLAMA_NUM_PARALLEL` or context length. Each parallel slot reserves its own KV cache.

Planned handling:
1. Add a per-role `min_gpu_share` to `LLMModel`, defaulting to `1.0`. Lower it explicitly in the YAML when a MoE model is chosen, so the decision is recorded in the config.
2. Make `check_ollama` raise an error below that threshold, with a diagnosis: other loaded models, and the KV-cache size from context length × parallel slots.
3. Make `check_ollama` refuse to measure while unrelated models are loaded. It should name them and suggest `ollama stop <model>`, not unload them silently.
4. Make the live test read the threshold from the config instead of hard-coding the generator roles.

If a dense model is offloaded by accident, fix it in this order:
1. Free the GPU (check with `ollama ps` and `nvidia-smi`).
2. Lower `OLLAMA_NUM_PARALLEL`.
3. Use a smaller quantization.
4. Lower the context length, but only if the longest prompts still fit. A window that's too small truncates prompts silently.

Offload that starts in the middle of a run shows up as a drop in the tokens/s that `chat_many` logs for each batch.
