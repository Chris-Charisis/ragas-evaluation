"""Stage 9 (`report`): the statistics on hand-computed cases, the ranking rule, the outcome of an
answer, the input checks, and the whole stage on the offline pipeline of test_ragas_eval.py (tiny_doc,
3 questions, one variant, fake generator and judge). When runs/<run_id>/ holds a real Ragas table
(the dev docs), the stage also runs on a copy of it, with invariants that hold on any data.
"""
import json
import shutil
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chunkeval import analysis
from chunkeval.analysis import plots, views
from chunkeval.analysis import report as report_module
from chunkeval.analysis.inputs import METHOD_LABELS, label, load_inputs
from chunkeval.analysis.stats import (bootstrap_ci, detectable_difference, holm, paired_diff, paired_family, rng_for,
                                      verdict)
from chunkeval.core.config import Secrets, load_config
from chunkeval.core.store import run_dir, write_json
from chunkeval.evaluation.answering import GOLD_PASSAGE
from chunkeval.qa.filtering import filter_paths
from chunkeval.qa.generation import qa_paths
from test_answering import answers  # noqa: F401  (fixtures used below)
from test_ragas_eval import evaluate, evaluated  # noqa: F401
from test_retrieval import VARIANT, offline, score, stage  # noqa: F401
from conftest import DEFAULT_CONFIG

SECRETS = Secrets()


# --- statistics -----------------------------------------------------------------------------------

def test_bootstrap_ci_drops_and_counts_nan_and_is_seeded():
    r = bootstrap_ci([0.5, 0.5, np.nan, 0.5], 500, 0.95, rng_for(1, "x"))
    assert (r["n"], r["nan"], r["mean"], r["low"], r["high"]) == (3, 1, 0.5, 0.5, 0.5)
    values = np.linspace(0, 1, 40)
    a, b = (bootstrap_ci(values, 500, 0.95, rng_for(1, "y")) for _ in range(2))
    assert a == b and a["low"] < a["mean"] < a["high"] and a["mean"] == pytest.approx(0.5)
    assert bootstrap_ci([np.nan], 10, 0.95, rng_for(1))["n"] == 0


def test_document_bootstrap_widens_when_one_document_carries_the_effect():
    """40 documents of 5 questions; every non-zero difference is in one of them. Redrawing questions
    treats its 5 as independent; redrawing documents moves them together, so the interval is wider."""
    values = np.zeros(200)
    values[:5] = 1.0
    docs = np.repeat([f"d{i}" for i in range(40)], 5)
    by_question = bootstrap_ci(values, 2000, 0.95, rng_for(0, "q"))
    by_document = bootstrap_ci(values, 2000, 0.95, rng_for(0, "d"), clusters=docs)
    assert by_document["clusters"] == 40
    assert by_document["high"] - by_document["low"] > by_question["high"] - by_question["low"]


def test_paired_diff_aligns_by_question_and_drops_nan_pairs():
    a = pd.Series({"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": np.nan, "q5": 0.5})
    b = pd.Series({"q5": 0.5, "q3": 0.0, "q2": 0.0, "q1": 0.0, "q4": 1.0, "q9": 1.0})     # other order, extra id
    r = paired_diff(a, b, 500, 0.95, rng_for(0))
    assert (r["n"], r["dropped"]) == (4, 1)                                # q4 dropped, q9 never paired
    assert (r["better"], r["equal"], r["worse"]) == (2, 2, 0)
    assert r["diff"] == pytest.approx(0.5) and r["low"] >= 0
    same = paired_diff(a, a, 100, 0.95, rng_for(0))
    assert (same["diff"], same["low"], same["high"], same["p"]) == (0.0, 0.0, 0.0, 1.0)


def test_holm_by_hand():
    """Sorted p: 0.005×4 = 0.02, 0.01×3 = 0.03, 0.03×2 = 0.06, 0.04×1 = 0.04 -> kept non-decreasing: 0.06."""
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("low, high, p_holm, expected", [
    (0.033, 0.317, 0.33, ""),        # dev: reports fc_recall, sentence_overlap: CI excludes 0, Holm doesn't pass
    (0.01, 0.2, 0.01, "higher"),
    (-0.2, -0.01, 0.01, "lower"),
    (-0.01, 0.2, 0.001, ""),         # p passes, the CI doesn't
])
def test_ranking_rule_needs_the_interval_and_holm(low, high, p_holm, expected):
    assert verdict(low, high, p_holm, 0.05) == expected


def test_paired_family_applies_holm_and_flags_few_documents():
    """Setup "a" is better on 6 of 90 questions, all from one of 29 documents: redrawing questions almost
    never misses all 6 (p ~0.002), redrawing documents misses that one a third of the time. So the
    question-level interval excludes 0 and the document-level one doesn't. Setup "b" equals the baseline."""
    ids = [f"q{i}" for i in range(90)]
    docs = pd.Series(["d0"] * 6 + np.repeat([f"d{i}" for i in range(1, 29)], 3).tolist(), index=ids)
    base = pd.Series(0.0, index=ids)
    a = base.copy()
    a[ids[:6]] = 1.0                                                     # the questions of d0
    wide = pd.DataFrame({"base": base, "a": a, "b": base})
    rows = {r["setup"]: r for r in paired_family(wide, "base", ["a", "b"], docs, n_iter=2000, ci=0.95, seed=0, key=("t",))}
    assert rows["a"]["p"] == pytest.approx(2 / 2 ** 6)                   # exact Wilcoxon: 6 positive ties
    assert rows["a"]["p_holm"] == pytest.approx(2 * rows["a"]["p"]) and rows["b"]["p_holm"] == 1.0
    assert rows["a"]["low"] > 0 and rows["a"]["doc_low"] <= 0 and rows["a"]["few_docs"]
    assert rows["a"]["doc_ci_reliable"] and rows["a"]["docs"] == 29


def test_detectable_difference():
    assert detectable_difference(0.5, 100, 0.05) == pytest.approx((1.95996 + 0.84162) * 0.05, rel=1e-4)
    assert np.isnan(detectable_difference(0.5, 0, 0.05))


# --- outcomes and labels --------------------------------------------------------------------------

@pytest.mark.parametrize("shown, abstained, error, fc_recall, expected", [
    (True, False, False, 1.0, "correct"),
    (True, False, False, 0.5, "partial"),
    (True, False, False, 0.0, "wrong"),
    (True, True, False, 0.0, "abstained_shown"),
    (False, True, False, 0.0, "abstained_missed"),
    (False, False, False, 0.4, "elsewhere"),
    (False, False, False, 0.0, "unsupported"),
    (True, False, True, np.nan, "error"),
    (True, False, False, np.nan, "error"),                               # answered, but no usable score
])
def test_outcome(shown, abstained, error, fc_recall, expected):
    assert views.outcome(shown, abstained, error, fc_recall) == expected


def test_size_bins_are_terciles_in_order():
    values = pd.Series([10, 20, 30, 40, 50, 60])
    labels = views.size_bins(values, views.BIN_NAMES, "chars")
    assert list(labels) == ["short (10–20 chars)"] * 2 + ["medium (30–40 chars)"] * 2 + ["long (50–60 chars)"] * 2
    assert sorted(set(labels), key=views.bin_order)[0].startswith("short")
    assert set(views.size_bins(pd.Series([7, 7]), views.BIN_NAMES, "words")) == {"all (7 words)"}


def test_report_names_replace_the_method_keys():
    keys = ["character_no_overlap", "recursive_overlap", "sentence_overlap", GOLD_PASSAGE, "character", "recursive"]
    assert list(map(label, keys)) == ["fixed-size_no_overlap", "section_overlap", "sentence_overlap", GOLD_PASSAGE,
                                      "fixed-size", "section"]
    assert plots.method_of(label("character_no_overlap")) == plots.method_of(label("character_overlap")) == "fixed-size"


# --- the stage on the offline pipeline ------------------------------------------------------------

@pytest.fixture
def reported(evaluated):
    """The offline pipeline with its span metrics, answered and scored by Ragas (3 questions, one variant,
    which is then the baseline);
    a cheap bootstrap. Returns cfg."""
    cfg, _ = evaluated
    score(cfg)                                                           # the span-metrics table
    evaluate(cfg)
    cfg.analysis.baseline = VARIANT
    cfg.analysis.bootstrap_iterations = 100
    return cfg


def report(cfg):
    """Run the stage; the newest report folder, through reports/latest."""
    analysis.run(cfg, SECRETS)
    return (run_dir(cfg, "reports") / "latest").resolve()


def test_run_writes_the_report_folder(reported):
    out = report(reported)
    assert {p.name for p in out.iterdir()} == {"report.md", "cases.md", "run_record.json", "tables", "figures", "secondary"}
    tables = {p.stem for p in (out / "tables").glob("*.csv")}
    assert {"retrieval_means", "retrieval_vs_baseline", "generation_means", "failure_funnel", "abstentions",
            "metric_response", "judge_vs_spans", "breakdowns", "audit", "questions_by_setup", "per_document",
            "distractors", "detectable_differences", "question_attributes", "generation_rows"} <= tables
    assert {p.stem for p in (out / "secondary").glob("*.csv")} == set(analysis.SECONDARY)
    assert {p.name for p in (out / "figures").glob("*.png")} >= {"failure_funnel.png", "k_curves.png", "retrieval_top_k.png"}

    text = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## How to read this report", "## 1. Run record", "## 3. Retrieval", "## 4. Generation",
                    "### 4.1 Where the answers end up", "## 5. Breakdowns", "## Secondary views", "Small question set"):
        assert heading in text
    record = json.loads((out / "run_record.json").read_text(encoding="utf-8"))
    assert record["corpora"]["papers"]["questions"] == 3 and record["analysis"]["baseline"] == VARIANT

    # every row of a setup lands in exactly one outcome; the gold passage always shows the evidence
    funnel = pd.read_csv(out / "tables" / "failure_funnel.csv")
    assert (funnel[list(views.OUTCOMES)].sum(axis=1) == funnel["rows"]).all() and (funnel["rows"] == 3).all()
    gold = funnel[funnel["setup"] == GOLD_PASSAGE].iloc[0]
    assert gold["abstained_missed"] == gold["elsewhere"] == gold["unsupported"] == 0
    rows = pd.read_csv(out / "tables" / "generation_rows.csv")
    retrieved = rows[rows["setup"] != GOLD_PASSAGE]
    assert set(retrieved.loc[retrieved["span_hit"] == 0, "outcome"]) <= {"abstained_missed", "elsewhere", "unsupported"}


def test_each_run_writes_a_new_folder(reported, monkeypatch):
    """Named by the time (the logs' format); a second run in the same second gets -2. The earlier
    report is left as it was, latest points to the new one, and no temporary folder is left."""
    fixed = datetime(2026, 10, 12, 23, 38, 58)
    monkeypatch.setattr(analysis, "datetime", SimpleNamespace(now=lambda: fixed))
    first = report(reported)
    (first / "note.txt").write_text("mine\n", encoding="utf-8")
    second = report(reported)
    assert (first.name, second.name) == ("20261012-233858", "20261012-233858-2")
    assert (first / "note.txt").exists() and (first / "report.md").exists()
    reports = run_dir(reported, "reports")
    assert sorted(p.name for p in reports.iterdir()) == [first.name, second.name, "latest"]
    record = json.loads((second / "run_record.json").read_text(encoding="utf-8"))
    assert (record["report"], record["generated"]) == (second.name, "2026-10-12T23:38:58")


def test_without_a_ragas_table_the_report_covers_retrieval(reported):
    (run_dir(reported, "ragas") / "per_question.parquet").unlink()
    out = report(reported)
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "No Ragas table yet" in text and "### 3.1" in text
    assert not (out / "tables" / "generation_means.csv").exists()


def test_question_settings_name_what_differs_between_corpora(cfg, tmp_path):
    """Each corpus's gen-qa and filter-qa settings, and the keys that differ between corpora; a corpus
    filtered before the settings were recorded has none and differs in nothing."""
    cfg.paths.data_dir = tmp_path / "data"
    gen = {"generator": {"model": "g", "digest": "sha256:abc"}, "prompt_sha256": "p" * 64, "seed": 42}
    write_json(qa_paths(cfg)["provenance"], {"settings_by_corpus": {"papers": gen, "reports": {**gen, "seed": 7}},
                                             "docs": {}})
    filt = {"critic": {"model": "c", "digest": None}, "critic_prompt_sha256": "q" * 64, "critic_min_score": 4,
            "dedup_embedder": {}, "dedup_cosine_threshold": 0.92}
    write_json(filter_paths(cfg)["provenance"], {"settings_by_corpus": {"papers": filt}})
    made = report_module.question_settings(cfg, ["papers", "reports"])
    assert made["differ"] == {"gen-qa": ["seed"], "filter-qa": []}
    assert made["by_corpus"]["papers"]["generation"]["digest"] == "sha256:abc" and made["by_corpus"]["reports"]["filter"] is None


def test_report_warns_when_corpora_were_made_differently(reported, monkeypatch):
    made = {"by_corpus": {"papers": {"generation": None, "filter": None}},
            "differ": {"gen-qa": ["generator", "seed"], "filter-qa": []}}
    monkeypatch.setattr(report_module, "question_settings", lambda cfg, corpora: made)
    text = (report(reported) / "report.md").read_text(encoding="utf-8")
    assert "questions were made with different settings** (gen-qa: generator, seed; see the run record)" in text
    assert "gen-qa not recorded; filter-qa not recorded" in text


def test_stale_tables_are_refused(reported):
    """A Ragas row whose evidence flag disagrees with the retrieval table, a retrieval table for other
    questions, and a details.jsonl that doesn't match: each names the stage to re-run."""
    path = run_dir(reported, "ragas") / "per_question.parquet"
    original = pd.read_parquet(path)
    changed = original.copy()
    row = changed.index[changed["setup"] == VARIANT][0]
    changed.loc[row, "evidence_shown"] = not changed.loc[row, "evidence_shown"]
    changed.to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="chunkeval answer"):
        load_inputs(reported)
    original.to_parquet(path, index=False)

    details = run_dir(reported, "ragas") / "details.jsonl"
    lines = details.read_text(encoding="utf-8").splitlines()
    details.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="chunkeval eval"):
        load_inputs(reported)

    metrics = run_dir(reported, "metrics") / "retrieval_per_question.parquet"
    table = pd.read_parquet(metrics)
    table[table["question_id"] != table["question_id"].iloc[0]].to_parquet(metrics, index=False)
    with pytest.raises(RuntimeError, match="chunkeval retrieve"):
        load_inputs(reported)


# --- the stage on the dev docs ---------------------------------------------------------------------

def test_report_on_the_dev_run(tmp_path):
    """The real run's tables (copied: nothing is written to runs/), a cheap bootstrap. Invariants only,
    so the test holds on any corpus."""
    real = load_config(DEFAULT_CONFIG)
    source = real.paths.runs_dir / real.run_id
    if not (source / "ragas" / "per_question.parquet").exists():
        pytest.skip(f"no Ragas table in {source}")
    cfg = load_config(DEFAULT_CONFIG)
    cfg.paths.runs_dir = tmp_path / "runs"
    shutil.copytree(source, cfg.paths.runs_dir / cfg.run_id,
                    ignore=shutil.ignore_patterns("logs", "report", "reports", "*_cache.jsonl"))
    cfg.analysis.bootstrap_iterations = 200
    try:
        load_inputs(cfg)
    except (RuntimeError, FileNotFoundError) as e:
        pytest.skip(f"the run's tables are out of date: {e}")
    out = report(cfg)

    rows = pd.read_csv(out / "tables" / "generation_rows.csv")
    funnel = pd.read_csv(out / "tables" / "failure_funnel.csv")
    per_setup = rows.groupby(["corpus", "setup"]).size()
    assert (funnel.set_index(["corpus", "setup"])["rows"] == per_setup.reindex(funnel.set_index(["corpus", "setup"]).index)).all()
    retrieved = rows[rows["setup"] != GOLD_PASSAGE]
    missed = funnel[funnel["setup"] != GOLD_PASSAGE][["abstained_missed", "elsewhere", "unsupported"]].sum().sum()
    assert missed == (retrieved["span_hit"] == 0).sum()
    vb = pd.read_csv(out / "tables" / "retrieval_vs_baseline.csv")
    assert (vb["p_holm"] >= vb["p"] - 1e-12).all()
    assert ((vb["verdict"].fillna("") != "") <= ((vb["low"] > 0) | (vb["high"] < 0))).all()
    assert (out / "cases.md").read_text(encoding="utf-8").startswith("# Cases to read")
    text = (out / "report.md").read_text(encoding="utf-8")
    assert not any(f"{key}_" in text for key in METHOD_LABELS)                # every setup under its report name
