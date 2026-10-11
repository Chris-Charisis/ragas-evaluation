"""Stage 8 — Ragas evaluation with an OpenAI judge, per-question scores, plus cheap text metrics.

Correct column mapping (the main bug of the old script, ragas_evaluation.py:470-486, where
`answer` = gold answer and `reference` = whole source chunk):
    user_input          <- question
    retrieved_contexts  <- top-k chunk texts (rank order) of THIS config
    response            <- RAG answer of THIS config (answering.py)
    reference           <- gold answer (QAPair.answer)
    reference_contexts  <- gold evidence passage text (optional; some metrics use it)

Metrics:
  Ragas (judge): context precision (with reference), context recall, faithfulness,
                 response/answer relevancy, factual correctness (response vs reference).
  Non-LLM:       abstention rate, response length, embedding similarity / ROUGE-L / token-F1 vs reference.

Scope: primary embedder (bge_m3) only — see answering.py. Every setup, plus the "gold_passage"
setup: each question answered from its own passage, scored like a setup and reported as the ceiling
next to them (setup − gold = what retrieval costs; 1 − gold = the generator's and judge's own error;
a metric on which gold doesn't beat the setups doesn't respond to retrieval). Its context recall
should be ~1 by construction, so the shortfall measures the judge's error rate.

Abstentions (decided 2026-10-11; `Answer.abstained`, see answering.is_abstention). Ragas 0.2.15 scores
an abstention as relevancy 0 (noncommittal) and factual correctness 0, and its faithfulness at random
(NaN, 0 or 1, depending on which statements the judge extracts). So:
  - don't send abstentions to the judge for the response metrics: set answer relevancy and factual
    correctness to 0 in code (what Ragas gives, without the noise); context precision and recall
    don't read the response, so they are scored as usual;
  - faithfulness: leave abstentions out (NaN) and count them; report it next to the abstention rate,
    plus a sensitivity view with abstentions = 1 (an abstention makes no unsupported claim);
  - never drop abstentions from every metric: a setup that abstains on its hard questions would look
    better on the rest, and the paired comparison needs the same questions in every setup.
Answers with `error` set (cut off, context full) have no response: count them per setup, as NaN.

Reads:  runs/<run_id>/answers/<corpus>_<setup>.jsonl (answering.load_answers; `contexts` are the
        retrieved_contexts as the generator read them), data/qa/filtered.jsonl
Writes: runs/<run_id>/ragas/per_question.parquet, runs/<run_id>/ragas/cost.json
"""
from __future__ import annotations

from typing import Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config


def build_samples(cfg: Config, corpus: str, method: str, overlap: bool,
                  question_ids: list[str]) -> list[Any]:
    """Build Ragas samples for one config.

    TODO(you):
      - ragas >= 0.2: `from ragas import EvaluationDataset, SingleTurnSample` — one
        SingleTurnSample per question with the fields listed in the module docstring.
      - Check the field names against the ragas version you install (0.2.15 now; newer versions
        reorganised metrics — read the migration guide before upgrading).
      - Use the same `question_ids` (the stratified Ragas subset) for all 12 configs → paired data.
    """
    raise NotImplementedError


def build_judge(cfg: Config, secrets: Any) -> tuple[Any, Any]:
    """Return (evaluator_llm, evaluator_embeddings).

    TODO(you):
      - LLM: the old code used `LangchainLLMWrapper(ChatOpenAI(model=...))` (ragas_evaluation.py:766);
        newer ragas also offers `ragas.llms.llm_factory`. Pin the model SNAPSHOT id (not an alias)
        so re-runs use the same judge. temperature=0 where the model supports it.
      - Embeddings (used by answer relevancy): cfg.ragas.embeddings_model = bge-m3. With
        ragas 0.2.15 the old pattern still works:
        `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=..., encode_kwargs={"normalize_embeddings": True}))`
        (ragas_evaluation.py:767). Note `langchain.embeddings.HuggingFaceEmbeddings` is deprecated —
        import it from `langchain_community.embeddings` (installed) or `langchain_huggingface`.
        This is independent of the retrieval embedders; just never change it between configs.
      - Answer relevancy compares embeddings of generated vs original questions; it needs no
        query instruction, so bge-m3's "no prompt" convention is convenient here.
      - The API key comes from secrets, never from the code (the old key at line 764 must be rotated).
    """
    raise NotImplementedError


def build_metrics(evaluator_llm: Any, evaluator_embeddings: Any) -> list[Any]:
    """Instantiate the Ragas metric objects.

    TODO(you): class names in ragas 0.2.x: LLMContextPrecisionWithReference, LLMContextRecall,
    Faithfulness, ResponseRelevancy, FactualCorrectness. FactualCorrectness has a `mode`
    (precision/recall/f1) — f1 is the usual choice; document it.
    """
    raise NotImplementedError


def evaluate_config(samples: list[Any], metrics: list[Any], cfg: Config) -> list[dict[str, Any]]:
    """Run ragas.evaluate and return PER-QUESTION rows (not means).

    TODO(you):
      - `ragas.evaluate(dataset=EvaluationDataset(samples), metrics=..., run_config=RunConfig(...))`
        — reuse the old RunConfig values (ragas_evaluation.py:775) from cfg.ragas.
      - `result.to_pandas()` gives one row per sample: keep them all with the question_id. The old
        code kept only `.mean()`, which made CIs impossible (docx: differences of 0.008 couldn't be
        told apart from noise).
      - NaNs: Ragas returns NaN when a judge call fails to parse. Count them per metric and config —
        never let `mean()` silently drop them without reporting how many.
      - Cost: `evaluate(..., token_usage_parser=...)` can report token usage — use it in the pilot to
        size cfg.ragas.sample_per_corpus.
    """
    raise NotImplementedError


def text_metrics(response: str, reference: str, encoder) -> dict[str, float]:
    """Cheap, judge-free generation metrics.

    TODO(you): response length (tokens), cosine similarity of embeddings (reuse the primary
    embedder, bge-m3, via embedding.Embedder.encode_documents — both texts are answers, not
    queries, so no query instruction), ROUGE-L (e.g. the
    `rouge-score` package) or a simple token-F1 (SQuAD-style: normalise, split, count overlap).
    These are weak alone but useful as a sanity check against the judge's factual correctness.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets, pilot: int | None = None, bridge: bool = False) -> None:
    """Evaluate all configs (or `pilot` questions of one config; or the gpt-4o bridge run).

    TODO(you):
      - Pilot first (≈20 questions, 1 config) → measure tokens/cost per sample → set the subset size.
      - Bridge run: ONE config scored with cfg.llm.bridge_judge (gpt-4o) — the difference vs
        the main judge on the same samples quantifies the "judge effect" the docx had to guess at.
      - Resumable per config; write per-question rows to the long table.
    """
    raise NotImplementedError
