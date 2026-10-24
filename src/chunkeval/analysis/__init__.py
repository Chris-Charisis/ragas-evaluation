"""Stage 9 (`chunkeval report`) — every per-question score turned into tables with uncertainty, figures,
a case file and one Markdown report, for a reviewer of the whole RAG pipeline.

Why: in the previous report several differences were <= 0.008 and there was no way to tell them from
noise (docx, last recommendation). With per-question scores on the SAME questions for every setup,
the comparisons are paired, which is far more sensitive than comparing two means.

Reads:  runs/<run_id>/metrics/retrieval_per_question.parquet  (stage 6) one row per (question_id, embedder,
            corpus, method, overlap, k_label); the baselines are embedder "oracle" / "random"
        runs/<run_id>/retrieval/<primary embedder>/*.jsonl    (stage 6) the hits, for distractors and cases
        runs/<run_id>/ragas/per_question.parquet  (stage 8, optional) one row per (question_id, corpus, setup)
        runs/<run_id>/ragas/details.jsonl, ragas/cost.json, answer_summary.json, gen_qa_summary.json,
        filter_qa_summary.json, data/qa/filtered.jsonl, data/chunks/<corpus>/_size_stats.json
Writes: runs/<run_id>/reports/<YYYYMMDD-HHMMSS>/  (a new folder per run, earlier reports kept;
            runs/<run_id>/reports/latest links to the newest)
        report.md          the report, in reading order (report.py)
        cases.md           a card per question worth reading by hand (cases.py)
        run_record.json    what was measured: data, model pins, settings, cost
        tables/*.csv       every table, with every metric (the Markdown shows the headline ones)
        figures/*.png      the figures (plots.py); each has a table with the same numbers
        secondary/         the secondary views (diagnostics, not results): tables and figures

Modules: inputs.py loads and cross-checks the inputs (a stale table names the stage to re-run);
stats.py has the bootstrap, the paired comparisons and THE RANKING RULE (its docstring explains it);
views.py computes the tables; plots.py, cases.py and report.py present them.

Decisions (2026-10-12):
  * Ranking rule: a setup is higher / lower than another only when the paired bootstrap CI excludes 0
    AND the Wilcoxon p-value, Holm-adjusted within its family (one corpus, metric and k), is below
    1 − analysis.ci. Don't rank chunkers otherwise (the dev docs: no comparison passes).
  * Document-level intervals next to every paired comparison; "few documents" flags a difference that
    holds by question but not by document. Unreliable below stats.MIN_CLUSTER_DOCS documents.
  * Breakdowns by question type, number of evidence quotes (1 vs 2+), evidence length (terciles) and
    kind (only when two-span pairs exist); no breakdown by section (nearly one value per question).
  * Outputs a reviewer needs beyond the scaffold's tables: where the answers end up (outcomes per
    setup), the case file, the run record, span metrics as k grows, the equal-budget view, the
    questions that separate the setups, documents and distractors, score distributions, the
    question set's make-up, and the smallest detectable difference (to size the final run).
    Secondary (diagnostics): context tokens vs span recall, evidence rank vs answer quality, metric
    correlations, answer length vs scores.
  * The dev set (35 questions) is a rehearsal: every output is produced, with n shown and low-n
    cells flagged, and report.md opens with a warning below report.SMALL_SET questions per corpus.

The generation table (stage 8; ragas_eval.py's docstring has the details):
  * Rows: `setup` is one of the 6 variants or "gold_passage" (each question answered from its own
    passage); `method` / `overlap` are null for gold_passage. Within a corpus every setup has the same
    questions, so every comparison between setups is paired.
  * Factual correctness is reported both ways (decided 2026-10-12): factual_correctness (Ragas' f1,
    comparable with other Ragas reports; it counts any true claim the gold answer doesn't state as
    wrong, so complete correct answers can score 0) and fc_recall (the share of the gold answer's
    claims the answer contains: "is the right answer in it?"), with fc_precision ("does it say only
    that?"). fc_recall_ragas is a diagnostic only: CSV, not the Markdown.
  * Abstentions (decided 2026-10-11): an abstained row has answer_relevancy, every factual-correctness
    score and the text metrics at 0 (as Ragas would score it) and faithfulness NaN. Abstained rows are
    never dropped from the other metrics; faithfulness is reported over the answered rows WITH the
    abstention rate next to it, plus the view where abstentions count as faithful (faithfulness_abst1).
  * NaN otherwise means an unusable judge output, an answer error, or faithfulness without
    statements: every mean reports the NaN count it dropped, every comparison the pairs it dropped.
  * Joining with retrieval: the primary embedder's rows at k_label == str(retrieval.top_k) match the
    ragas rows on (question_id, corpus, setup); gold_passage rows have no retrieval row. inputs.py
    refuses the tables if a row's evidence_shown differs from that span_hit.
  * On the dev docs, "some chunk useful" (context precision) agreed with span_hit in 91% of the rows,
    and most disagreements were chunks holding the fact OUTSIDE the gold evidence (elsewhere in the
    document, or in report_16, the near-duplicate of report_15): views.judge_vs_spans lists them with
    where the useful chunks come from, rather than calling them errors.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from chunkeval.analysis import cases, plots, report, views
from chunkeval.analysis.inputs import Inputs, label, load_inputs
from chunkeval.core.config import Secrets
from chunkeval.core.schema import Config
from chunkeval.core.store import run_dir
from chunkeval.evaluation.answering import GOLD_PASSAGE

log = logging.getLogger(__name__)

# The secondary views' tables, written to secondary/ instead of tables/.
SECONDARY = ("context_vs_recall", "evidence_rank", "metric_correlations", "length_vs_scores")
DISTRIBUTED = ["span_recall", "fc_recall", "factual_correctness", "fc_precision", "faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]


def compute(cfg: Config, inputs: Inputs) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None, dict, dict]:
    """Every table (name -> DataFrame), the generation frame, the stage summaries and the costs."""
    questions = views.question_frame(inputs)
    docs = views.doc_of(questions)
    g = views.generation_frame(inputs, cfg, questions) if inputs.ragas is not None else None
    top = views.top_k_rows(inputs.retrieval, cfg)

    t: dict[str, pd.DataFrame] = {"question_attributes": questions}
    t["question_set"], summaries = views.question_set(questions, cfg)
    t["retrieval_means"] = views.retrieval_means(inputs, cfg)
    t["retrieval_vs_baseline"] = views.retrieval_vs_baseline(inputs, cfg, docs)
    t["questions_by_setup"] = views.discriminating(inputs, cfg, questions, g)
    t["per_document"] = views.per_document(inputs, cfg, questions, g)
    t["distractors"] = views.distractors(inputs, cfg)
    t["embedder_robustness"], t["embedder_levers"] = views.embedder_robustness(inputs, cfg, docs,
                                                                               t["retrieval_vs_baseline"])
    sources = {"retrieval": top}
    if g is not None:
        t["generation_rows"] = g
        t["generation_means"] = views.generation_means(g, cfg)
        t["generation_vs_baseline"] = views.generation_vs_baseline(g, inputs, cfg, docs)
        t["generation_vs_gold"] = views.generation_vs_gold(g, inputs, cfg, docs)
        t["metric_response"] = views.metric_response(g, cfg, docs)
        t["abstentions"] = views.abstentions(g, t["generation_means"])
        t["failure_funnel"] = views.failure_funnel(g)
        t["judge_vs_spans"], t["judge_span_disagreements"] = views.judge_vs_spans(g, inputs, cfg)
        t["text_vs_judge"] = views.text_vs_judge(g)
        t["by_evidence"] = views.by_evidence(g, cfg)
        t["audit"] = views.audit_lists(g, inputs)
        sources["generation"] = g[g["setup"] != GOLD_PASSAGE]
    t["breakdowns"] = views.breakdowns(inputs, cfg, questions, g)
    t["score_distributions"] = views.score_distributions(g, top)
    t["detectable_differences"] = views.detectable_differences(sources, inputs, cfg)
    t["chunk_sizes"] = views.chunk_sizes(cfg, inputs.corpora)
    costs = views.costs(cfg)
    if "judge_by_prompt" in costs:
        t["judge_cost_by_prompt"] = costs["judge_by_prompt"]
    if "generator" in costs:
        t["generator_tokens"] = costs["generator"]

    t["context_vs_recall"] = views.context_vs_recall(t["retrieval_means"], cfg)
    if g is not None:
        t["evidence_rank"] = views.evidence_rank(g, cfg)
        t["metric_correlations"] = views.metric_correlations(g)
        t["length_vs_scores"] = views.length_vs_scores(g, cfg)
    return t, g, summaries, costs


def figures(cfg: Config, inputs: Inputs, t: dict[str, pd.DataFrame], g: pd.DataFrame | None, out: Path) -> dict[str, Path]:
    """Every figure, by name (a figure with nothing to show is left out)."""
    main, second = out / "figures", out / "secondary"
    primary, top, corpora, setups = cfg.embedding.primary, str(cfg.retrieval.top_k), inputs.corpora, inputs.setups
    rm = t["retrieval_means"]
    largest_k = str(max(cfg.retrieval.k_values))
    oracle = rm[(rm["embedder"] == "oracle") & (rm["k_label"] == largest_k)]
    figs = {
        "retrieval_top_k": plots.dot_plot(
            rm[(rm["embedder"] == primary) & (rm["k_label"] == top)], [*views.HEADLINE_SPAN, "precision"], corpora,
            setups, main / "retrieval_top_k.png", f"Span metrics at k = {top} ({primary})", marks=oracle,
            mark_label="oracle: the chunker's own ceiling"),
        "retrieval_budget": plots.dot_plot(
            rm[(rm["embedder"] == primary) & (rm["k_label"] == "budget")], [*views.HEADLINE_SPAN, "k"], corpora, setups,
            main / "retrieval_budget.png", f"Span metrics at an equal budget of {cfg.retrieval.budget_tokens} tokens"),
        "retrieval_vs_baseline": plots.diff_plot(
            t["retrieval_vs_baseline"][t["retrieval_vs_baseline"]["k_label"] == top], views.HEADLINE_SPAN, corpora,
            setups, main / "retrieval_vs_baseline.png",
            f"Setup − {label(cfg.analysis.baseline)}, paired over questions (k = {top})"),
        "k_curves": plots.k_curves(rm, primary, corpora, setups, cfg.retrieval.top_k, main / "k_curves.png"),
        "context_vs_recall": plots.context_vs_recall(t["context_vs_recall"], primary, corpora, setups,
                                                     second / "context_vs_recall.png"),
    }
    if g is not None:
        shown = ["fc_recall", "factual_correctness", "faithfulness", "abstention", "answer_relevancy", "context_recall"]
        figs["generation_means"] = plots.dot_plot(t["generation_means"], shown, corpora, setups,
                                                  main / "generation_means.png",
                                                  "Generation metrics per setup, the gold passage as the ceiling")
        figs["generation_vs_baseline"] = plots.diff_plot(t["generation_vs_baseline"], shown, corpora, setups,
                                                         main / "generation_vs_baseline.png",
                                                         f"Setup − {label(cfg.analysis.baseline)}, paired over questions")
        figs["generation_vs_gold"] = plots.diff_plot(t["generation_vs_gold"], shown, corpora, setups,
                                                     main / "generation_vs_gold.png",
                                                     "Setup − gold passage: what retrieval with each chunker costs")
        figs["failure_funnel"] = plots.funnel(t["failure_funnel"], corpora, [*setups, GOLD_PASSAGE],
                                              main / "failure_funnel.png")
        values = pd.concat([g[g["setup"] != GOLD_PASSAGE].melt(id_vars=["corpus"], value_vars=DISTRIBUTED[1:],
                                                                var_name="metric"),
                            views.top_k_rows(inputs.retrieval, cfg).melt(id_vars=["corpus"], value_vars=["span_recall"],
                                                                         var_name="metric")])
        figs["score_distributions"] = plots.distributions(values, DISTRIBUTED, corpora, main / "score_distributions.png")
        figs["evidence_rank"] = plots.evidence_rank(t["evidence_rank"], corpora, second / "evidence_rank.png")
        figs["metric_correlations"] = plots.correlations(t["metric_correlations"], second / "metric_correlations.png")
        figs["length_vs_scores"] = plots.length_vs_scores(t["length_vs_scores"], corpora, views.LENGTH_METRICS,
                                                          second / "length_vs_scores.png")
    return {name: path for name, path in figs.items() if path is not None}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.5g")


def report_folder(reports: Path, generated: datetime) -> Path:
    """reports/<YYYYMMDD-HHMMSS> (the logs' format), with -2, -3, ... if a report of that second exists."""
    stamp = f"{generated:%Y%m%d-%H%M%S}"
    final, n = reports / stamp, 1
    while final.exists():
        n += 1
        final = reports / f"{stamp}-{n}"
    return final


def point_latest(reports: Path, final: Path) -> None:
    """reports/latest -> the newest report (a relative symlink, swapped in atomically), so one path
    always opens the last report. Where symlinks aren't allowed, only a warning."""
    tmp = reports / ".latest.tmp"
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(final.name, target_is_directory=True)
        os.replace(tmp, reports / "latest")
    except OSError as e:
        log.warning("could not point %s to %s: %s", reports / "latest", final.name, e)


def run(cfg: Config, secrets: Secrets) -> None:
    """CLI entry (`chunkeval report`): load and check the inputs, compute every table, draw the figures,
    and write a new report folder (module docstring); earlier reports are kept. No LLM, no API call:
    free, and safe to re-run."""
    inputs = load_inputs(cfg)
    t, g, summaries, costs = compute(cfg, inputs)

    generated = datetime.now()
    reports = run_dir(cfg, "reports")
    final = report_folder(reports, generated)
    out = final.with_name(final.name + ".tmp")       # renamed once complete: a crash leaves no half report
    shutil.rmtree(out, ignore_errors=True)
    for folder in ("tables", "figures", "secondary"):
        (out / folder).mkdir(parents=True)
    for name, df in t.items():
        write_csv(df, out / ("secondary" if name in SECONDARY else "tables") / f"{name}.csv")
    figs = figures(cfg, inputs, t, g, out)

    audit = t.get("audit", pd.DataFrame(columns=["question_id", "reason"]))
    reasons = cases.card_reasons(inputs, audit, t["questions_by_setup"], t.get("judge_span_disagreements"))
    (out / "cases.md").write_text(cases.casebook(inputs, cfg, reasons, g), encoding="utf-8")
    record = report.run_record(cfg, inputs, t["question_attributes"], costs, generated, final.name)
    report.write_record(out / "run_record.json", record)
    (out / "report.md").write_text(report.render(cfg, inputs, t, figs, record, costs, summaries, out), encoding="utf-8")

    out.rename(final)
    point_latest(reports, final)
    log.info("report written to %s (also reports/latest): %d tables, %d figures, %d case(s)", final / "report.md",
             len(t), len(figs), len(reasons))
    for name in ("retrieval_vs_baseline", "generation_vs_baseline"):
        if name in t and len(t[name]):
            shown = t[name][t[name]["verdict"] != ""]
            log.info("%s: %d of %d comparisons pass the ranking rule%s", name, len(shown), len(t[name]),
                     "".join(f"; {r.corpus} {r.setup} {r.verdict} on {r.metric}" for r in shown.head(10).itertuples()))
