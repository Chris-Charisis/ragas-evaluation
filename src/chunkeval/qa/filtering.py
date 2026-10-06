"""Stage 5 — layered Q-A filtering: deterministic checks -> LLM critic -> de-duplication.

Why: the old judge (`build_evaluation_prompt`, ragas_evaluation.py:424) only saw the question and
answer, so it could check format but not truth. Here:
  1. cheap deterministic checks remove obvious junk before paying for any LLM call;
  2. a critic from a DIFFERENT model family (gpt-oss-20b) sees the evidence passage;
  3. near-duplicate questions are removed so no topic is over-weighted.
Every rejected pair is written with its reason → you can audit the filter.

Reads:  data/qa/raw.jsonl
Writes: data/qa/filtered.jsonl, data/qa/rejected.jsonl, data/qa/critic_scores.jsonl
Prompt: prompts/qa_critic.md
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config

if TYPE_CHECKING:
    from chunkeval.core.llm import ChatLLM
    from chunkeval.qa.generation import QAPair

# Reuse the old `phrase_blacklist` (ragas_evaluation.py:337) as a starting point.
# TODO(you): turn it into case-insensitive regexes (the old `phrase in q` check was case-sensitive,
# so "In this study ..." slipped through) and add pronoun openings ("^(it|they|this|these)\\b").
BLACKLIST_PATTERNS: list[str] = []

# TODO(you): JSON schema for the critic output (reason + score per criterion, see prompt outline).
CRITIC_JSON_SCHEMA: dict[str, Any] = {}


def deterministic_checks(qa: "QAPair", cfg: Config) -> list[str]:
    """Return a list of failure reasons ([] = pass).

    TODO(you) — ideas, pick what proves useful on your data:
      - blacklist / pronoun-opening regexes on the question;
      - question ends with "?", length bounds (too short = trivial, too long = compound question);
      - answer is not an abstention ("I don't know", "not mentioned") — replaces the old
        "N/A"/"I don't know" filtering;
      - evidence located (should already hold) and answer not just a copy of the whole quote;
      - the answer's key numbers/entities appear in the evidence text (simple regex on digits is a
        surprisingly strong hallucination check for numeric questions).
    """
    raise NotImplementedError


def critic_scores(pairs: list["QAPair"], llm: "ChatLLM", docs: dict[str, Any]) -> list[dict[str, Any]]:
    """Ask the critic to score each pair given its evidence passage.

    TODO(you): build messages from prompts/qa_critic.md, run `llm.chat_many`, store the raw scores
    + reasons per pair id. Keep the keep/drop decision OUT of this function (see `decide`).
    """
    raise NotImplementedError


def decide(scores: dict[str, Any], cfg: Config) -> list[str]:
    """Apply thresholds (cfg.qa.critic_min_score) → list of failure reasons."""
    raise NotImplementedError


def deduplicate(pairs: list["QAPair"], cfg: Config) -> tuple[list["QAPair"], list["QAPair"]]:
    """Remove near-duplicate questions; return (kept, removed).

    TODO(you): embed questions with the primary embedder (bge-m3, `embedding.Embedder`; use
    encode_documents here — this is question-vs-question similarity, symmetric, so no query
    instruction), compute cosine similarity (normalised
    vectors → dot product), greedily keep a question unless it is above
    cfg.qa.dedup_cosine_threshold to an already kept one. Compare within the same corpus.
    Hint: `sentence_transformers.util.community_detection` or a plain numpy matrix for a few
    thousand questions are both fine.
    """
    raise NotImplementedError


def final_sample(pairs: list["QAPair"], cfg: Config, rng) -> list["QAPair"]:
    """Stratified down-sampling to the target size (by corpus, question_type, kind).

    Hint: fix the final set ONCE and evaluate all 12 configs on exactly the same question ids —
    the paired statistics in analysis.py depend on it.
    """
    raise NotImplementedError


def export_for_labelling(pairs: list["QAPair"], n: int, path: str, rng) -> None:
    """Write a random sample of KEPT and REJECTED pairs to CSV for your manual check.

    Why: 50 hand labels give you the filter's precision (are kept pairs good?) and a rough recall
    (are rejected pairs really bad?). Include the evidence text so labelling is quick.
    """
    raise NotImplementedError


def run(cfg: Config, secrets: Secrets) -> None:
    """raw.jsonl → checks → critic → decide → dedup → final_sample → filtered.jsonl (+ rejected)."""
    raise NotImplementedError
