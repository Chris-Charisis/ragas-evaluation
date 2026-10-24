"""The statistics of the report: bootstrap intervals, paired comparisons, the ranking rule.

Every comparison is PAIRED: within a corpus, every setup is scored on the same questions, so the
difference is taken per question and only then averaged. That removes the spread between easy and
hard questions, which is most of the spread, and is why paired differences are far more sensitive
than two means side by side.

The ranking rule (decided 2026-10-12). A setup is called higher (or lower) than the one it is
compared with only when BOTH hold:
  1. the paired bootstrap CI of the mean difference excludes 0 (the size of the difference), and
  2. the Wilcoxon signed-rank p-value, Holm-adjusted within its family, is below 1 - analysis.ci.
Why both: a report compares 5 setups with the baseline for each corpus, metric and k, well over 100
comparisons in all. A 95% CI excludes 0 by chance about 1 time in 20, so with nothing to find, ~5
"winners" would still appear. The Wilcoxon test asks whether the per-question differences lean one
way more than chance would give (it uses only the questions that changed). Holm adjusts the p-values
of one family (the 5 comparisons of one corpus, metric and k) so the chance of even one false winner
among them stays below alpha: sort the p-values, multiply the smallest by 5, the next by 4, ...,
keeping them non-decreasing. Dev-set example: reports, fc_recall, sentence_overlap − baseline =
+0.16, CI [0.03, 0.32] excludes 0, but it rests on 4 of 20 questions and its p 0.066 becomes 0.33
after Holm: no difference shown.

Document-level intervals (decided 2026-10-12). The bootstrap above redraws QUESTIONS, as if each were
an independent draw. Questions of one document share its quirks (garbled OCR, a near-duplicate
document), so they fail or succeed together, and an effect carried by one or two documents looks
more certain than it is (dev: 4 of the 7 papers questions behind section_no_overlap's +0.17
fc_recall are from paper_4). The paired comparisons therefore also get a CI that redraws whole
DOCUMENTS with all their questions; a difference whose question-level CI excludes 0 but whose
document-level CI doesn't is flagged `few_docs` ("rests on a few documents"). It needs many
documents: below MIN_CLUSTER_DOCS it is computed but marked unreliable.
"""
from __future__ import annotations

import math
import warnings
import zlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

# Below this many documents, a document-level interval is too unstable to read (dev: 5 per corpus).
MIN_CLUSTER_DOCS = 20
# Below this many questions, a cell's interval is flagged `low_n` (computed all the same).
LOW_N = 10
# Differences smaller than this are ties, and a score this close to 1 is 1 (float noise: Ragas' context
# precision of all-useful chunks is 0.9999999999).
TIE = 1e-6
# For the smallest detectable difference: the chance of finding a real difference of that size.
POWER = 0.8


def rng_for(seed: int, *key: Any) -> np.random.Generator:
    """A generator for one table cell, seeded by the run's seed and the cell's key: a cell's interval
    doesn't change when other cells are added, removed or computed in another order."""
    return np.random.default_rng([seed, zlib.crc32("|".join(map(str, key)).encode("utf-8"))])


def bootstrap_ci(values, n_iter: int, ci: float, rng: np.random.Generator, clusters=None) -> dict[str, Any]:
    """Mean and percentile-bootstrap CI of `values`, NaNs dropped (and counted in "nan").

    Without `clusters`, it redraws values (questions) with replacement. With `clusters` (one label per
    value, e.g. its doc_id), it redraws whole clusters and takes the mean of all their values, so
    values of one cluster always move together; "clusters" is then their number.
    Returns {"n", "nan", "mean", "low", "high"} (+ "clusters"); the interval of one value is that value.
    """
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    v = values[keep]
    out: dict[str, Any] = {"n": int(keep.sum()), "nan": int((~keep).sum()),
                           "mean": math.nan, "low": math.nan, "high": math.nan}
    if clusters is not None:
        labels, which = np.unique(np.asarray(clusters)[keep], return_inverse=True)
        out["clusters"] = len(labels)
    if not len(v):
        return out
    out["mean"] = float(v.mean())
    if clusters is None:
        means = v[rng.integers(0, len(v), size=(n_iter, len(v)))].mean(axis=1)
    else:
        sums, counts = np.bincount(which, weights=v), np.bincount(which)
        drawn = rng.integers(0, len(labels), size=(n_iter, len(labels)))
        means = sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)
    tail = (1 - ci) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    # the percentiles of identical means can differ from the mean by float rounding
    out["low"], out["high"] = min(float(low), out["mean"]), max(float(high), out["mean"])
    return out


def wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired differences, ties (0) dropped; 1.0 when every
    difference is a tie (nothing to test)."""
    nonzero = diffs[np.abs(diffs) > TIE]
    if not len(nonzero):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # scipy warns on tiny samples; the p-value is still right
        return float(wilcoxon(nonzero, zero_method="wilcox").pvalue)


def paired_diff(a: pd.Series, b: pd.Series, n_iter: int, ci: float, rng: np.random.Generator,
                clusters: pd.Series | None = None) -> dict[str, Any]:
    """Mean difference a − b over the SAME questions, with its paired bootstrap CI.

    `a`, `b`: scores indexed by question_id, aligned here (inner join). A question with NaN on either
    side (an abstention's faithfulness, an unusable judge output) is dropped from this pair only and
    counted in "dropped", so the n behind each difference is visible. `clusters`: doc_id by
    question_id, for the document-level CI ("doc_low", "doc_high", "docs").
    Also returned: how many questions are better / equal / worse (more readable than a mean for
    binary-ish metrics like span_hit), and the Wilcoxon p-value.
    """
    both = pd.concat({"a": a, "b": b}, axis=1, join="inner")
    complete = both.dropna()
    d = (complete["a"] - complete["b"]).to_numpy(dtype=float)
    d[np.abs(d) < TIE] = 0.0
    q = bootstrap_ci(d, n_iter, ci, rng)
    out: dict[str, Any] = {"n": q["n"], "dropped": len(both) - len(complete), "diff": q["mean"],
                           "low": q["low"], "high": q["high"],
                           "better": int((d > 0).sum()), "equal": int((d == 0).sum()), "worse": int((d < 0).sum()),
                           "p": wilcoxon_p(d)}
    if clusters is not None:
        docs = bootstrap_ci(d, n_iter, ci, rng, clusters=clusters.reindex(complete.index).to_numpy())
        out.update(doc_low=docs["low"], doc_high=docs["high"], docs=docs["clusters"])
    return out


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values of one family, in the input order: the i-th smallest p is multiplied by
    (m − i), and the results are made non-decreasing in that order (capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - i) * p[idx]))
        adjusted[idx] = running
    return adjusted


def verdict(low: float, high: float, p_holm: float, alpha: float) -> str:
    """The ranking rule (module docstring): "higher" / "lower" when the CI excludes 0 AND the
    Holm-adjusted p is below alpha; "" (no difference shown) otherwise."""
    if p_holm < alpha and low > 0:
        return "higher"
    if p_holm < alpha and high < 0:
        return "lower"
    return ""


def paired_family(wide: pd.DataFrame, against: str, others: list[str], docs: pd.Series, *, n_iter: int,
                  ci: float, seed: int, key: tuple) -> list[dict[str, Any]]:
    """Each column of `others` − column `against` of `wide` (question_id × setup), paired, as one
    Holm family. `docs`: doc_id by question_id. `key` names the family (corpus, metric, ...) for the
    per-cell seeds. One dict per compared column, with the verdict and the `few_docs` flag."""
    rows = []
    for other in others:
        if other not in wide or against not in wide:
            continue
        r = paired_diff(wide[other], wide[against], n_iter, ci, rng_for(seed, *key, other, against), clusters=docs)
        rows.append({"setup": other, "against": against, **r})
    if not rows:
        return rows
    alpha = 1 - ci
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(p)
        r["verdict"] = verdict(r["low"], r["high"], r["p_holm"], alpha)
        excludes_zero = r["low"] > 0 or r["high"] < 0
        r["doc_ci_reliable"] = r["docs"] >= MIN_CLUSTER_DOCS
        r["few_docs"] = bool(excludes_zero and not (r["doc_low"] > 0 or r["doc_high"] < 0))
    return rows


def summary_table(df: pd.DataFrame, metrics: list[str], group_by: list[str], *, n_iter: int, ci: float,
                  seed: int) -> pd.DataFrame:
    """Mean and question-level bootstrap CI of each metric per group: one row per (group, metric) with
    n (values used), nan (dropped), mean, low, high and `low_n` (n < LOW_N)."""
    rows = []
    for key, group in df.groupby(group_by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            if m not in group:
                continue
            b = bootstrap_ci(group[m].to_numpy(dtype=float), n_iter, ci, rng_for(seed, *key, m))
            rows.append({**dict(zip(group_by, key)), "metric": m, **b, "low_n": b["n"] < LOW_N})
    return pd.DataFrame(rows)


def detectable_difference(sd: float, n: int, alpha: float, power: float = POWER) -> float:
    """The smallest true mean difference a paired test at level `alpha` finds with probability `power`,
    from the standard deviation of the per-question differences and the number of questions (normal
    approximation)."""
    if n < 1 or not np.isfinite(sd):
        return math.nan
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / math.sqrt(n))


def questions_needed(sd: float, delta: float, alpha: float, power: float = POWER) -> float:
    """Questions needed for `detectable_difference` to reach `delta`."""
    if not np.isfinite(sd) or sd == 0:
        return math.nan
    return float(math.ceil(((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / delta) ** 2))
