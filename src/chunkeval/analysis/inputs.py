"""Stage 9's inputs, loaded and checked to belong together.

The two long tables carry no provenance of their own, so a table left over from before a re-run of
an earlier stage would silently mix two experiments. The checks here compare what can be compared
and name the stage to re-run:
  - the retrieval table holds exactly the questions of filtered.jsonl (else `retrieve`);
  - the primary embedder's retrieval files are up to date with the questions and chunk files (else
    `retrieve`): the case file and the distractor table read their hits;
  - the Ragas table holds every setup on the same questions, its Ragas sample (unless it was a
    pilot), and each retrieved row's evidence_shown equals span_hit at top_k in the retrieval table
    (else `answer`, then `eval`); details.jsonl has exactly its rows (else `eval`).
The Ragas table is optional: before `eval` has run, the report covers retrieval only.

Names (decided 2026-10-22): the report calls the chunkers fixed-size, sentence and section
(METHOD_LABELS). "character" and "recursive" read as LangChain's CharacterTextSplitter and
RecursiveCharacterTextSplitter, which cut differently. The config, the file names and the stored tables
keep the keys; `load_inputs` checks everything under the keys, then renames (`relabel`).
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from chunkeval.core.schema import Config
from chunkeval.core.store import read_jsonl
from chunkeval.evaluation.answering import GOLD_PASSAGE
from chunkeval.evaluation.retrieval import (Hit, file_provenance, load_hits, load_questions, retrieval_path,
                                            stored_provenance, variant_chunks, variants)
from chunkeval.qa.generation import QAPair

log = logging.getLogger(__name__)


@dataclass
class Inputs:
    retrieval: pd.DataFrame                      # stage 6's long table, with a `setup` column added
    ragas: pd.DataFrame | None                   # stage 8's long table; None before `eval`
    questions: dict[str, QAPair]                 # filtered.jsonl, by question id
    corpora: list[str]                           # the corpora with questions, in config order
    setups: list[str]                            # the variant names, in config order
    hits: dict[tuple[str, str], dict[str, list[Hit]]] = field(default_factory=dict)
    """(corpus, setup) -> question_id -> the primary embedder's hits, in rank order (k_max of them)"""
    chunks_per_doc: dict[tuple[str, str], Counter] = field(default_factory=dict)
    """(corpus, setup) -> doc_id -> number of chunks"""
    details: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    """(question_id, setup) -> its line of details.jsonl"""
    pilot: int | None = None                     # the `eval --pilot N` the Ragas table came from


def variant_name(method: str, overlap: bool) -> str:
    return f"{method}_{'overlap' if overlap else 'no_overlap'}"


METHOD_LABELS = {"character": "fixed-size", "recursive": "section"}


def label(name: str) -> str:
    """A method or setup as the report names it: character_overlap -> fixed-size_overlap; others unchanged."""
    method, sep, rest = name.partition("_")
    return METHOD_LABELS.get(method, method) + sep + rest


def relabel(inputs: Inputs) -> Inputs:
    """Every method and setup in the loaded inputs under its report name (module docstring)."""
    for df in (inputs.retrieval, inputs.ragas):
        if df is not None:
            df["method"] = df["method"].map(label, na_action="ignore")
            df["setup"] = df["setup"].map(label)
    inputs.setups = [label(s) for s in inputs.setups]
    inputs.hits = {(c, label(s)): v for (c, s), v in inputs.hits.items()}
    inputs.chunks_per_doc = {(c, label(s)): v for (c, s), v in inputs.chunks_per_doc.items()}
    inputs.details = {(q, label(s)): v for (q, s), v in inputs.details.items()}
    return inputs


def run_path(cfg: Config, *parts: str) -> Path:
    """runs/<run_id>/<parts>, without creating it (run_dir creates folders)."""
    return cfg.paths.runs_dir.joinpath(cfg.run_id, *parts)


def load_inputs(cfg: Config) -> Inputs:
    """Both long tables, the questions, the primary embedder's hits and details.jsonl, checked, under the
    report's names (module docstring)."""
    by_corpus = load_questions(cfg)
    corpora = [c for c in cfg.paths.corpora if by_corpus.get(c)]
    questions = {qa.id: qa for c in corpora for qa in by_corpus[c]}
    setups = [name for _, _, name in variants(cfg)]

    path = run_path(cfg, "metrics", "retrieval_per_question.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `chunkeval retrieve` first")
    retrieval = pd.read_parquet(path)
    retrieval["setup"] = [variant_name(m, o) for m, o in zip(retrieval["method"], retrieval["overlap"])]
    check_retrieval(cfg, retrieval, by_corpus, corpora, setups, path)

    inputs = Inputs(retrieval=retrieval, ragas=None, questions=questions, corpora=corpora, setups=setups)
    load_hits_and_chunks(cfg, inputs, by_corpus)

    ragas_path = run_path(cfg, "ragas", "per_question.parquet")
    if not ragas_path.exists():
        log.warning("%s not found: the report covers retrieval only (run `chunkeval eval` for the generation "
                    "metrics)", ragas_path)
        return relabel(inputs)
    summary_path = run_path(cfg, "ragas_summary.json")
    if summary_path.exists():
        inputs.pilot = json.loads(summary_path.read_text(encoding="utf-8")).get("pilot")
    ragas = pd.read_parquet(ragas_path)
    check_ragas(cfg, ragas, retrieval, by_corpus, setups, inputs.pilot, ragas_path)
    inputs.ragas = ragas
    inputs.details = load_details(run_path(cfg, "ragas", "details.jsonl"), ragas)
    return relabel(inputs)


def check_retrieval(cfg: Config, retrieval: pd.DataFrame, by_corpus: dict[str, list[QAPair]], corpora: list[str],
                    setups: list[str], path: Path) -> None:
    """The retrieval table holds exactly filtered.jsonl's questions, and the primary embedder at top_k
    for every setup of every corpus."""
    for corpus in corpora:
        expected = {qa.id for qa in by_corpus[corpus]}
        found = set(retrieval.loc[retrieval["corpus"] == corpus, "question_id"])
        if found != expected:
            raise RuntimeError(f"{path} holds other {corpus} questions than data/qa/filtered.jsonl "
                               f"({len(found)} vs {len(expected)}): re-run `chunkeval retrieve`")
    top = retrieval[(retrieval["embedder"] == cfg.embedding.primary)
                    & (retrieval["k_label"] == str(cfg.retrieval.top_k))]
    missing = [f"{c}/{s}" for c in corpora for s in setups
               if not ((top["corpus"] == c) & (top["setup"] == s)).any()]
    if missing:
        raise RuntimeError(f"{path} has no {cfg.embedding.primary} rows at k={cfg.retrieval.top_k} for "
                           f"{', '.join(missing)}: run `chunkeval retrieve`")


def load_hits_and_chunks(cfg: Config, inputs: Inputs, by_corpus: dict[str, list[QAPair]]) -> None:
    """The primary embedder's hits per (corpus, setup), refused when stale, and the chunks per document."""
    primary = cfg.embedding.primary
    for corpus in inputs.corpora:
        for setup in inputs.setups:
            chunks, index_provenance = variant_chunks(cfg, corpus, setup)
            inputs.chunks_per_doc[(corpus, setup)] = Counter(c.doc_id for c in chunks)
            path = retrieval_path(cfg, primary, corpus, setup)
            expected = file_provenance(cfg, primary, corpus, setup, index_provenance["chunks_sha256"],
                                       by_corpus[corpus])
            if stored_provenance(path) != expected:
                raise RuntimeError(f"{path} is missing or older than the current questions, chunks or query "
                                   f"settings: re-run `chunkeval retrieve`")
            inputs.hits[(corpus, setup)] = load_hits(path)


def check_ragas(cfg: Config, ragas: pd.DataFrame, retrieval: pd.DataFrame, by_corpus: dict[str, list[QAPair]],
                setups: list[str], pilot: int | None, path: Path) -> None:
    """Every setup on the same questions, the Ragas sample's questions, and the answers' evidence flag
    equal to the retrieval table's span_hit at top_k (so both tables describe the same retrieval)."""
    from chunkeval.evaluation.ragas_eval import ragas_sample      # imports ragas: only when there is a table

    expected_setups = {*setups, GOLD_PASSAGE}
    for corpus, group in ragas.groupby("corpus", sort=False):
        pairs = by_corpus.get(corpus, [])
        if set(group["setup"]) != expected_setups:
            raise RuntimeError(f"{path}: {corpus} has setups {sorted(set(group['setup']))}, expected "
                               f"{sorted(expected_setups)}: re-run `chunkeval eval`")
        per_setup = group.groupby("setup")["question_id"].apply(frozenset)
        if per_setup.nunique() != 1:
            raise RuntimeError(f"{path}: the {corpus} setups were scored on different questions: re-run `chunkeval eval`")
        ids = per_setup.iloc[0]
        if not ids <= {qa.id for qa in pairs}:
            raise RuntimeError(f"{path} holds {corpus} questions that are no longer in data/qa/filtered.jsonl: "
                               f"re-run `chunkeval answer`, then `chunkeval eval`")
        if not pilot and ids != {qa.id for qa in ragas_sample(pairs, cfg.ragas.sample_per_corpus, cfg.seed)}:
            raise RuntimeError(f"{path}: the {corpus} questions aren't the current Ragas sample "
                               f"(ragas.sample_per_corpus {cfg.ragas.sample_per_corpus}): re-run `chunkeval eval`")
    if pilot:
        log.warning("the Ragas table is from `eval --pilot %d`: the generation metrics cover only those questions", pilot)

    top = retrieval[(retrieval["embedder"] == cfg.embedding.primary) & (retrieval["k_label"] == str(cfg.retrieval.top_k))]
    joined = ragas[ragas["setup"] != GOLD_PASSAGE].merge(
        top[["question_id", "corpus", "setup", "span_hit"]], on=["question_id", "corpus", "setup"], how="left")
    stale = joined["span_hit"].isna() | (joined["evidence_shown"] != (joined["span_hit"] > 0))
    if stale.any():
        raise RuntimeError(f"{path}: {int(stale.sum())} row(s) were answered from other retrieval results than "
                           f"the retrieval table holds: re-run `chunkeval answer`, then `chunkeval eval`")


def load_details(path: Path, ragas: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    """details.jsonl by (question_id, setup), refused unless it has exactly the table's rows."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: re-run `chunkeval eval`")
    details = {(d["question_id"], d["setup"]): d for d in read_jsonl(path)}
    if set(details) != set(zip(ragas["question_id"], ragas["setup"])):
        raise RuntimeError(f"{path} doesn't match ragas/per_question.parquet: re-run `chunkeval eval`")
    return details
