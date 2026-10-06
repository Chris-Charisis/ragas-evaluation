"""Stage 7 — generate the RAG answer (Ragas `response`) from the retrieved chunks.

Why this stage exists: in the old script the Ragas `answer` column was the Q-A generator's own
answer, so faithfulness and answer relevancy did not depend on the chunker at all. A real RAG
answer — from a FIXED generator, conditioned on each config's top-k chunks — makes the generation
metrics meaningful.

Uses the PRIMARY embedder's retrieval only (cfg.embedding.primary = bge_m3). The secondary
embedder (Qwen3) is a retrieval-only robustness check; answering with it would double the LLM cost.
If the robustness table shows a large difference for one config, answer that ONE config with the
secondary as a targeted follow-up.

Reads:  runs/<run_id>/retrieval/<primary embedder>/*.jsonl, data/qa/filtered.jsonl
Writes: runs/<run_id>/answers/<corpus>_<method>_<ovl>.jsonl
Prompt: prompts/rag_answer.md
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config

if TYPE_CHECKING:
    from chunkeval.core.llm import ChatLLM
    from chunkeval.evaluation.retrieval import Hit

# TODO(you): the fixed abstention phrase your prompt instructs the model to use.
ABSTENTION_PHRASE = ""


def build_messages(question: str, hits: list["Hit"]) -> list[dict[str, str]]:
    """Numbered context blocks in rank order + question (see prompts/rag_answer.md)."""
    raise NotImplementedError


def is_abstention(response: str) -> bool:
    """Detect abstentions (fixed phrase + a few variants).

    Why: abstention rate is a metric in itself (an honest miss vs a hallucination), and you may
    want to report faithfulness both with and without abstentions.
    """
    raise NotImplementedError


def answer_config(llm: "ChatLLM", questions: dict[str, Any], retrieval_file: str, k: int,
                  out_path: str) -> None:
    """Generate answers for one config.

    TODO(you):
      - Use the SAME k (and the same prompt) for every config — or run both fixed-k and
        budget-matched variants, but then label them clearly.
      - Temperature 0; log prompt_tokens / completion_tokens per answer.
      - Store the raw response AND a cleaned one (citations stripped) — Ragas gets the cleaned one.
      - Resumable via store.done_ids — 12 configs × N questions is many local calls.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """Loop over all configs (+ the oracle baseline: answer from the gold evidence passage — an
    upper bound for generation metrics that isolates the generator's own error)."""
    raise NotImplementedError
