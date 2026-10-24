"""The report's figures (PNG, matplotlib). Each one has a table with the same numbers (CSV, and
report.md), so no value lives in a figure only.

Colours (checked with the dataviz skill's palette validator, 2026-10-12):
  - a chunking method keeps its hue in every figure: the first three categorical slots (blue,
    orange, aqua), which stay distinguishable for colour-blind readers even all against all; overlap
    is the marker fill (filled = with overlap, hollow = without), never a fourth hue;
  - references (gold passage, oracle) are ink diamonds, chance (random) is a muted gray cross;
  - the outcome bars use two one-hue ramps: blue for "evidence shown" (darkest = correct) and orange
    for "evidence not shown" (darkest = answered without a gold claim, the hallucination risk).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # files only, no display

import matplotlib.pyplot as plt          # noqa: E402  (after the backend is chosen)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from chunkeval.analysis.views import OUTCOMES, bin_order   # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_HUES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1–3
SINGLE = "#2a78d6"                                         # single-series marks
OUTCOME_COLOURS = {
    "correct": "#0d366b", "partial": "#1c5cab", "wrong": "#3987e5", "abstained_shown": "#86b6ef",
    "abstained_missed": "#f19a74", "elsewhere": "#d95926", "unsupported": "#8c3511", "error": AXIS,
}
DARK_OUTCOMES = {"correct", "partial", "elsewhere", "unsupported"}     # white text inside
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#ec8f8e", "#a92a2a"]   # −1 … 0 (neutral gray) … +1
REFERENCES = ("gold_passage", "oracle")
DPI = 150


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def method_of(setup: str) -> str:
    """fixed-size_no_overlap, fixed-size_overlap -> fixed-size."""
    return setup.rsplit("_overlap", 1)[0].removesuffix("_no")


def method_colours(setups: list[str]) -> dict[str, str]:
    """Hue by chunking method, in the order the methods first appear (config order): a method keeps
    its hue in every figure of a report."""
    methods = list(dict.fromkeys(method_of(s) for s in setups if s not in REFERENCES))
    return {m: METHOD_HUES[i % len(METHOD_HUES)] for i, m in enumerate(methods)}


def marker_style(setup: str, colours: dict[str, str]) -> dict:
    """Filled marker with overlap, hollow without; references as ink diamonds, random as a gray cross."""
    if setup in REFERENCES:
        return {"marker": "D", "color": INK, "markerfacecolor": INK, "markeredgecolor": SURFACE, "markersize": 6}
    if setup == "random":
        return {"marker": "X", "color": MUTED, "markerfacecolor": MUTED, "markeredgecolor": SURFACE, "markersize": 7}
    colour = colours[method_of(setup)]
    return {"marker": "o", "color": colour, "markerfacecolor": SURFACE if setup.endswith("no_overlap") else colour,
            "markeredgecolor": colour, "markeredgewidth": 1.8, "markersize": 7}


def legend_handles(keys: list[str], colours: dict[str, str], labels: dict[str, str] | None = None) -> list[Line2D]:
    labels = labels or {}
    return [Line2D([], [], linestyle="none", label=labels.get(k, k), **marker_style(k, colours)) for k in keys]


def error_bars(d: pd.DataFrame) -> np.ndarray:
    """The (below, above) lengths of the CI bars around each mean, as matplotlib wants them."""
    return np.vstack([(d["mean"] - d["low"]).to_numpy(dtype=float), (d["high"] - d["mean"]).to_numpy(dtype=float)])


def save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _row_panels(n_rows: int, n_cols: int, n_items: int, width: float = 2.6):
    return plt.subplots(n_rows, n_cols, figsize=(width * n_cols + 1.8, 0.32 * n_items * n_rows + 1.5),
                        squeeze=False, sharey=True)


def _setup_axis(ax, rows: list[str], first_column: bool) -> None:
    """Setup names on the first column only: with a shared y axis, blanking the labels of one panel
    would blank them all, so the other panels hide theirs instead."""
    if first_column:
        ax.set_yticks(range(len(rows)), rows)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def dot_plot(means: pd.DataFrame, metrics: list[str], corpora: list[str], setups: list[str], path: Path, title: str,
             marks: pd.DataFrame | None = None, mark_label: str = "") -> Path:
    """Mean and CI per row (setups, then any reference setup in `means`, e.g. the gold passage) for each
    metric (columns), one row of panels per corpus. `means`: corpus, setup, metric, mean, low, high.
    `marks` (same columns): an ink diamond on each setup's row, without its CI (the oracle's ceiling
    for that chunker)."""
    style()
    colours = method_colours(setups)
    rows = [*setups, *[r for r in REFERENCES if r in set(means["setup"])]]
    fig, axes = _row_panels(len(corpora), len(metrics), len(rows))
    for i, corpus in enumerate(corpora):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            here = means[(means["corpus"] == corpus) & (means["metric"] == m)].set_index("setup")
            dots = None if marks is None else marks[(marks["corpus"] == corpus) & (marks["metric"] == m)].set_index("setup")
            for y, setup in enumerate(rows):
                if setup in here.index:
                    r = here.loc[setup]
                    st = marker_style(setup, colours)
                    ax.plot([r["low"], r["high"]], [y, y], color=st["color"], linewidth=1.4)
                    ax.plot([r["mean"]], [y], linestyle="none", zorder=3, **st)
                if dots is not None and setup in dots.index:
                    ax.plot([dots.loc[setup, "mean"]], [y], linestyle="none", zorder=4, **marker_style("oracle", colours))
            _setup_axis(ax, rows, j == 0)
            if m not in ("context_tokens", "k"):
                ax.set_xlim(-0.03, 1.03)
            ax.set_title(f"{corpus}: {m}", loc="left", fontsize=9)
    keys = [*setups, *[r for r in rows if r in REFERENCES]] + (["oracle"] if marks is not None else [])
    fig.legend(handles=legend_handles(keys, colours, {"oracle": mark_label} if marks is not None else None),
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def diff_plot(comparisons: pd.DataFrame, metrics: list[str], corpora: list[str], setups: list[str], path: Path,
              title: str) -> Path | None:
    """Paired difference and its CI per setup and metric, one row of panels per corpus, with the zero
    line; the verdict of the ranking rule, when there is one, is written beside its row."""
    rows = [s for s in setups if s in set(comparisons["setup"])]
    if not rows:
        return None
    style()
    colours = method_colours(setups)
    fig, axes = _row_panels(len(corpora), len(metrics), len(rows))
    for i, corpus in enumerate(corpora):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            ax.axvline(0, color=AXIS, linewidth=1)
            here = comparisons[(comparisons["corpus"] == corpus) & (comparisons["metric"] == m)].set_index("setup")
            for y, setup in enumerate(rows):
                if setup not in here.index:
                    continue
                r = here.loc[setup]
                st = marker_style(setup, colours)
                ax.plot([r["low"], r["high"]], [y, y], color=st["color"], linewidth=1.4)
                ax.plot([r["diff"]], [y], linestyle="none", zorder=3, **st)
                if r["verdict"]:
                    ax.annotate(r["verdict"], (r["high"], y), xytext=(4, 0), textcoords="offset points",
                                va="center", fontsize=8, color=INK_2)
            _setup_axis(ax, rows, j == 0)
            ax.set_title(f"{corpus}: {m}", loc="left", fontsize=9)
    fig.legend(handles=legend_handles(rows, colours), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def k_curves(r_means: pd.DataFrame, primary: str, corpora: list[str], setups: list[str], top_k: int, path: Path) -> Path:
    """span_hit and span_recall as k grows, per setup, with random retrieval as chance; the gray
    vertical line is top_k, what the generator is shown."""
    style()
    colours = method_colours(setups)
    metrics = ["span_hit", "span_recall"]
    df = r_means[(r_means["k_label"] != "budget") & r_means["metric"].isin(metrics)].assign(
        k=lambda d: d["k_label"].astype(int))
    fig, axes = plt.subplots(len(metrics), len(corpora), figsize=(4.2 * len(corpora), 3.0 * len(metrics)), squeeze=False)
    for i, m in enumerate(metrics):
        for j, corpus in enumerate(corpora):
            ax = axes[i][j]
            ax.axvline(top_k, color=AXIS, linewidth=1)
            here = df[(df["corpus"] == corpus) & (df["metric"] == m)]
            lines = [(s, here[(here["embedder"] == primary) & (here["setup"] == s)]) for s in setups]
            lines.append(("random", here[(here["embedder"] == "random") & (here["setup"] == setups[0])]))
            for key, d in lines:
                d = d.sort_values("k")
                st = marker_style(key, colours)
                ax.plot(d["k"], d["mean"], color=st["color"], linewidth=1.4 if key == "random" else 2, zorder=2)
                ax.plot(d["k"], d["mean"], linestyle="none", zorder=3, **st)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(sorted(df["k"].unique()))
            ax.set_title(f"{corpus}: {m}", loc="left")
            ax.set_xlabel("k" if i == len(metrics) - 1 else "")
    fig.legend(handles=legend_handles([*setups, "random"], colours, {"random": "random (chance)"}),
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"Span metrics as k grows ({primary}; the line at k = {top_k} is what the generator is shown)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def funnel(table: pd.DataFrame, corpora: list[str], rows: list[str], path: Path) -> Path:
    """Where the answers end up: one stacked bar per setup (shares of its rows), one panel per corpus.
    A share is written inside its segment only when it fits."""
    style()
    present = [o for o in OUTCOMES if table[o].sum() > 0] or ["correct"]
    fig, axes = plt.subplots(len(corpora), 1, figsize=(9.5, 0.42 * len(rows) * len(corpora) + 1.6), squeeze=False)
    for i, corpus in enumerate(corpora):
        ax = axes[i][0]
        t = table[table["corpus"] == corpus].set_index("setup")
        for y, setup in enumerate(rows):
            if setup not in t.index:
                continue
            left = 0.0
            for o in present:
                share = float(t.loc[setup, f"{o}_share"])
                if share <= 0:
                    continue
                ax.barh(y, share, left=left, height=0.62, color=OUTCOME_COLOURS[o], edgecolor=SURFACE, linewidth=2)
                if share >= 0.07:
                    ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=7.5,
                            color="white" if o in DARK_OUTCOMES else INK)
                left += share
        _setup_axis(ax, rows, True)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(corpus, loc="left")
    fig.legend(handles=[Patch(color=OUTCOME_COLOURS[o], label=OUTCOMES[o]) for o in present],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Where the answers end up (share of each setup's questions)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def distributions(values: pd.DataFrame, metrics: list[str], corpora: list[str], path: Path) -> Path:
    """Histogram of each metric's per-question scores over the retrieved setups (pooled), per corpus:
    the lumps at 0 and 1 a mean hides. `values`: corpus, metric, value."""
    style()
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(len(corpora), len(metrics), figsize=(2.3 * len(metrics), 2.2 * len(corpora)),
                             squeeze=False, sharex=True)
    for i, corpus in enumerate(corpora):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            v = values[(values["corpus"] == corpus) & (values["metric"] == m)]["value"].dropna().clip(0, 1)
            if len(v):
                ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), color=SINGLE, edgecolor=SURFACE, linewidth=2)
            ax.set_title(m if i == 0 else "", loc="left", fontsize=9)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            if j == 0:
                ax.set_ylabel(corpus, color=INK)
    fig.suptitle("Per-question scores over the retrieved setups (share of rows per bin of 0.1)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# --- secondary views -------------------------------------------------------------------------------

def context_vs_recall(table: pd.DataFrame, primary: str, corpora: list[str], setups: list[str], path: Path) -> Path:
    """Span recall against the context tokens handed over, per setup across k (the budget row left
    out), with each chunker's oracle as a diamond in its colour: the fewest tokens holding the evidence."""
    style()
    colours = method_colours(setups)
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.6 * len(corpora), 3.6), squeeze=False)
    for j, corpus in enumerate(corpora):
        ax = axes[0][j]
        here = table[(table["corpus"] == corpus) & (table["k_label"] != "budget")]
        for setup in setups:
            st = marker_style(setup, colours)
            d = here[(here["embedder"] == primary) & (here["setup"] == setup)].sort_values("context_tokens")
            ax.plot(d["context_tokens"], d["span_recall"], color=st["color"], zorder=2)
            ax.plot(d["context_tokens"], d["span_recall"], linestyle="none", zorder=3, **st)
            o = here[(here["embedder"] == "oracle") & (here["setup"] == setup)]
            if len(o):
                o = o.loc[[o["k_label"].astype(int).idxmax()]]            # the oracle stops once it has the evidence
                ax.plot(o["context_tokens"], o["span_recall"], linestyle="none", zorder=4,
                        **{**marker_style("oracle", colours), "markerfacecolor": st["color"]})
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("context tokens handed over (mean)")
        ax.set_title(f"{corpus}: span_recall", loc="left")
    fig.legend(handles=legend_handles([*setups, "oracle"], colours,
                                      {"oracle": "oracle (fewest tokens holding the evidence)"}),
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Secondary: what each setup pays in context for the evidence it finds", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def evidence_rank(table: pd.DataFrame, corpora: list[str], path: Path) -> Path:
    """Abstention rate and fc_recall (with CI and n) by the rank of the first chunk overlapping the
    evidence ("none": no top-k chunk does)."""
    style()
    metrics = ["abstention", "fc_recall"]
    fig, axes = plt.subplots(len(corpora), len(metrics), figsize=(4.0 * len(metrics), 2.6 * len(corpora)), squeeze=False)
    for i, corpus in enumerate(corpora):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            d = table[(table["corpus"] == corpus) & (table["metric"] == m)].copy()
            d["order"] = d["evidence_rank"].replace(0, 99)                # "none" last
            d = d.sort_values("order")
            x = np.arange(len(d))
            ax.bar(x, d["mean"], width=0.55, color=SINGLE)
            ax.errorbar(x, d["mean"], yerr=error_bars(d), fmt="none", ecolor=INK_2,
                        elinewidth=1, capsize=0)
            for xi, n in zip(x, d["n"]):
                ax.text(xi, 1.03, f"n={n}", ha="center", fontsize=7, color=INK_2)
            ax.set_xticks(x, ["none" if r == 0 else str(int(r)) for r in d["evidence_rank"]])
            ax.set_ylim(0, 1.12)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{corpus}: {m}", loc="left")
            ax.set_xlabel("rank of the first chunk with evidence" if i == len(corpora) - 1 else "")
    fig.suptitle("Secondary: is evidence at a low rank used as well as at rank 1?", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save(fig, path)


def correlations(table: pd.DataFrame, path: Path) -> Path:
    """Spearman's correlation matrix: diverging blue–gray–red, with the value written in each cell."""
    style()
    order = list(dict.fromkeys(table["metric_a"]))
    m = table.pivot(index="metric_a", columns="metric_b", values="rho").reindex(index=order, columns=order)
    values = m.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.55 * len(order) + 2.5, 0.5 * len(order) + 1.6))
    im = ax.imshow(values, cmap=LinearSegmentedColormap.from_list("diverging", DIVERGING), vmin=-1, vmax=1)
    for (a, b), v in np.ndenumerate(values):
        if not np.isnan(v):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman's rho")
    ax.set_title("Secondary: metric correlations (answered rows of the retrieved setups)", loc="left")
    fig.tight_layout()
    return save(fig, path)


def length_vs_scores(table: pd.DataFrame, corpora: list[str], metrics: list[str], path: Path) -> Path:
    """Mean score (with CI) per response-length tercile: one panel per metric, one row per corpus."""
    style()
    fig, axes = plt.subplots(len(corpora), len(metrics), figsize=(2.9 * len(metrics), 2.6 * len(corpora)),
                             squeeze=False, sharey=True)
    for i, corpus in enumerate(corpora):
        for j, m in enumerate(metrics):
            ax = axes[i][j]
            d = table[(table["corpus"] == corpus) & (table["metric"] == m)]
            d = d.iloc[np.argsort([bin_order(v) for v in d["length"]], kind="stable")]
            x = np.arange(len(d))
            ax.errorbar(x, d["mean"], yerr=error_bars(d), color=SINGLE, marker="o",
                        markersize=6, markeredgecolor=SURFACE, elinewidth=1)
            ax.set_xticks(x, [v.split(" (")[0] for v in d["length"]])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{corpus}: {m}", loc="left", fontsize=9)
    fig.suptitle("Secondary: scores by answer length (terciles of response words, answered rows)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    return save(fig, path)
