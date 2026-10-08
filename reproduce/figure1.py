"""Figure 1 of the paper: the 2x2 Delta / talent-rate panel.

Panels, in the paper's ordering:

    (A) Delta_1   — gap when only the single top-ranked architecture is trained
    (B) Delta_10  — gap when the top-10 ranked architectures are trained
    (C) Talent rate at k = 10% of the pool
    (D) Delta_10% — gap when the top-10% ranked architectures are trained

Each panel shows one bar per proxy plus a random-search bar whose error bar is
one standard deviation over 1000 Monte-Carlo draws.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import PercentFormatter

from reproduce.metrics import (
    ALL_PROXIES,
    DATASET_LABELS,
    DatasetStats,
    ORDERED_DATASETS,
)

# Tableau 10, matching the figure in the paper.
PALETTE10 = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]
RANDOM_SEARCH_LABEL = "Random search"
RANDOM_SEARCH_COLOR = "#66CCEE"

PANEL_SYMBOL = {"delta": "Δ", "talent": "τ"}

# Bars are drawn in alphabetical order so that each proxy keeps the colour it has
# in the published figure.
PLOT_ORDER = sorted(ALL_PROXIES)


def _style_ax(ax) -> None:
    import matplotlib.pyplot as plt

    ax.set_facecolor(plt.rcParams["axes.facecolor"])
    ax.grid(True, axis="y", color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", length=0)


def _draw_panel(ax, stats, quantity, panel_label, datasets, percent_xmax):
    """Draw one grouped bar panel. `quantity` is 'delta' or 'talent'."""
    x = np.arange(len(datasets))
    n_groups = len(PLOT_ORDER) + 1  # proxies + random search
    width = 0.8 / n_groups
    lo, hi = np.inf, -np.inf

    for j, proxy in enumerate(PLOT_ORDER):
        values = [getattr(stats[ds], quantity)[proxy] for ds in datasets]
        ax.bar(
            x + (j - n_groups / 2) * width + width / 2,
            values,
            width=width,
            label=proxy,
            color=PALETTE10[j % len(PALETTE10)],
            edgecolor="black",
            linewidth=0.8,
        )
        lo, hi = min(lo, min(values)), max(hi, max(values))

    means = [getattr(stats[ds], f"random_{quantity}") for ds in datasets]
    errs = [getattr(stats[ds], f"random_{quantity}_std") for ds in datasets]
    ax.bar(
        x + (len(PLOT_ORDER) - n_groups / 2) * width + width / 2,
        means,
        width=width,
        label=RANDOM_SEARCH_LABEL,
        color=RANDOM_SEARCH_COLOR,
        edgecolor="black",
        linewidth=0.8,
        yerr=errs,
        error_kw=dict(elinewidth=1.2, ecolor="black", capsize=4, capthick=1.2),
    )
    lo = min(lo, min(m - e for m, e in zip(means, errs)))
    hi = max(hi, max(m + e for m, e in zip(means, errs)))

    if percent_xmax is not None:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=percent_xmax, decimals=0))
    ax.text(
        0.01,
        0.95,
        f"{panel_label} — {PANEL_SYMBOL[quantity]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in datasets], rotation=0)
    ax.margins(x=0.02)
    _style_ax(ax)

    span = hi - lo if hi > lo else 1.0
    pad = 0.12 * span
    ax.set_ylim(lo - 0.2 * pad, hi + pad)
    return ax.get_legend_handles_labels()


def plot_figure1(
    stats_k1: dict[str, DatasetStats],
    stats_k10: dict[str, DatasetStats],
    stats_k_large: dict[str, DatasetStats],
    outpath: Path,
    *,
    datasets: list[str] | None = None,
    figsize: tuple[float, float] = (16, 10),
    legend_ncols: int = 5,
) -> list[Path]:
    """Render Figure 1 and write it as both PDF and PNG. Returns the paths."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = datasets or ORDERED_DATASETS
    # k for the 10% panels is resolved per dataset, so it varies by a few models
    # depending on how many architectures dropped out of each pool.
    k_large_values = sorted({stats_k_large[ds].k for ds in datasets})
    k_large_label = (
        f"{k_large_values[0]}"
        if len(k_large_values) == 1
        else f"10% ({k_large_values[0]}–{k_large_values[-1]})"
    )

    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, dpi=200)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0, 0], stats_k1, "delta", "(A) k = 1", 1.0),
        (axes[0, 1], stats_k10, "delta", "(B) k = 10", 1.0),
        (axes[1, 0], stats_k_large, "talent", f"(C) k = {k_large_label}", 100.0),
        (axes[1, 1], stats_k_large, "delta", f"(D) k = {k_large_label}", 1.0),
    ]
    handles_all, labels_all = [], []
    for ax, stats, quantity, label, percent_xmax in panels:
        handles, labels = _draw_panel(ax, stats, quantity, label, datasets, percent_xmax)
        handles_all.extend(handles)
        labels_all.extend(labels)

    for ax in (axes[0, 0], axes[0, 1]):
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    seen, handles, labels = set(), [], []
    for handle, label in zip(handles_all, labels_all):
        if label not in seen:
            seen.add(label)
            handles.append(handle)
            labels.append(label)

    fig.subplots_adjust(bottom=0.16)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(labels), legend_ncols),
        frameon=True,
        edgecolor="black",
        facecolor="white",
    )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = outpath.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
