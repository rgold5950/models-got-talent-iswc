from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

from utils.path_utils import PROJECT_ROOT
# Default OUT_DIR, but can be overridden by passing out_dir parameter to functions
DEFAULT_OUT_DIR = Path(PROJECT_ROOT) / "analysis" / "figures" / "zero_cost"
# Multiple ID patterns to support different run_dir formats
# Pattern 1: arch_seed=X,combo_idx=Y (old format)
# Pattern 2: arch_seed=X,init_name=Y,init_seed=Z (new format)
# Pattern 3: train_tune_XXXXX_NNNNN_N (trial identifier fallback)
ID_REGEX = r"(arch_seed=\d+,combo_idx=\d+|arch_seed=\d+,init_name=\w+,init_seed=\d+|train_tune_\w+_\d+_\d+)"
METRIC_COLUMNS = [
    "initial_val_f1_macro",
    "best_test_f1_macro",  # use test metric -> val metric = (best_val_f1_macro)
    # "measures_sec",  # don't keep time but its useful for other analysis such as understanding the computation time
    "grad_norm",
    "snip",
    "grasp",
    "fisher",
    "jacob_cov",
    "plain",
    "synflow_bn",
    "synflow",
]
ZERO_COST_METRICS = [
    "grad_norm",
    "snip",
    "grasp",
    "fisher",
    "jacob_cov",
    "plain",
    # "synflow_bn",  # we aren't using batch-norm
    "synflow",
]
# Higher-is-better = +1,  lower-is-better = –1
# ─────────────────────────────────────────────────────────────────────────
# • SNIP: saliency ∑|∂L/∂w · w| – larger score ⇒ weight judged more “critical”
#         to keeping capacity.  Higher values correlate with accuracy. :contentReference[oaicite:0]{index=0}
# • GraSP: negative second-order Taylor estimate of loss change – *more negative*
#         means pruning those weights hurts less / improves curvature. :contentReference[oaicite:1]{index=1}
# • Fisher: fisher information potential (blockswap)
# • Jacob-Cov: score = –Σ[log λ + 1/λ]; closer to 0 (less negative) ⇒ smoother,
#              better-conditioned network.  So “higher” (less-negative) is better. :contentReference[oaicite:3]{index=3}
# • Plain / Grad-norm: ‖∇θ f(x)‖ – smaller norm often favours flatter minima
#                      and better generalisation. :contentReference[oaicite:4]{index=4}
# • SynFlow / SynFlow-BN: product of positive “flow” through weights;
#                         larger flow ⇒ greater expressive capacity. :contentReference[oaicite:5]{index=5}
# • initial_val_f1_macro: obvious – higher validation F1 is better.
#
METRIC_DIRECTION = {
    "snip": +1,  # higher score → better  :contentReference[oaicite:6]{index=6}
    "grasp": -1,  # more negative → better :contentReference[oaicite:7]{index=7}
    "fisher": +1,  # higher → better       :contentReference[oaicite:8]{index=8}
    "jacob_cov": +1,  # less negative → better :contentReference[oaicite:9]{index=9}
    "plain": +1,  # smaller grad-norm → flatter minima :contentReference[oaicite:10]{index=10}
    "grad_norm": +1,  # alias of ‘plain’      :contentReference[oaicite:11]{index=11}
    "synflow_bn": +1,  # higher flow → better  :contentReference[oaicite:12]{index=12}
    "synflow": +1,  # higher flow → better  :contentReference[oaicite:13]{index=13}
    "initial_val_f1_macro": +1,  # validation metric – higher is better
    "ensemble": +1,
}

METRIC_COLORS = {
    "snip": mcolors.to_rgb("#4C78A8"),  # blue
    "grasp": mcolors.to_rgb("#F58518"),  # orange
    "fisher": mcolors.to_rgb("#54A24B"),  # green
    "jacob_cov": mcolors.to_rgb("#E45756"),  # red
    "plain": mcolors.to_rgb("#B279A2"),  # purple
    "synflow_bn": mcolors.to_rgb("#FF9DA7"),  # pink
    "synflow": mcolors.to_rgb("#9D755D"),  # brown
    "grad_norm": mcolors.to_rgb("#72B7B2"),  # teal
    "ensemble": mcolors.to_rgb("#333333"),
    "initial_val_f1_macro": mcolors.to_rgb("#C81486"),  # grey
}
ORDERED_DATASETS = ["myogym", "hhar", "mobiactv2", "motionsense", "pamap2", "rwhar"]
ORDERED_METRICS = ["snip", "grasp", "fisher", "jacob_cov", "plain", "grad_norm", "synflow", "synflow_bn", "ensemble", "initial_val_f1_macro"]

def _load_and_align(
    cfg: dict, id_regex: str, use_abs: bool = False, min_cluster_size: int = 10, keep_run_dir: bool = False
) -> pd.DataFrame:
    """Load full-results and zero-cost pickles, align by short_id, and return merged metric subset."""
    full_df = pd.read_pickle(Path(cfg["full_results_pkl"]))
    zc_df = pd.read_pickle(Path(cfg["zero_cost_pkl"]))
    full_df["short_id"] = full_df["run_dir"].str.extract(id_regex)
    zc_df["short_id"] = zc_df["run_dir"].str.extract(id_regex)
    if keep_run_dir:
        zc_df = zc_df.drop(columns=['run_dir'])
        cols = METRIC_COLUMNS + ["run_dir"]
    else:
        cols = METRIC_COLUMNS
    full_df = full_df.merge(zc_df, on="short_id", how="left")[cols].dropna(
        how="any"
    )
    if use_abs:
        for m in ZERO_COST_METRICS:
            full_df[m] = full_df[m].abs()
    else:
        for m, sign in METRIC_DIRECTION.items():
            if sign == -1 and m in full_df.columns:
                full_df[m] *= -1

    # if drop_low_signal:
    #     counts = full_df["initial_val_f1_macro"].value_counts()
    #     low_signal_values = counts[counts > min_cluster_size].index
    #     full_df = full_df[~full_df["initial_val_f1_macro"].isin(low_signal_values)]

    # compute an ensemble score
    full_df['ensemble'] = full_df[ZERO_COST_METRICS].rank(pct=True).mean(axis=1)
    return full_df


def _corr_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return Spearman and Pearson correlation matrices."""
    return {
        "spearman": df.corr(method="spearman"),
        "pearson": df.corr(method="pearson"),
    }


def corr_for_dataset(
    name: str,
    cfg: dict,
    id_regex: str = ID_REGEX,
    use_abs: bool = False,
    max_cluster_size: int = 10,
) -> dict[str, pd.DataFrame]:
    """Compute correlations for one dataset."""
    subset = _load_and_align(
        cfg,
        id_regex,
        use_abs=use_abs,
        min_cluster_size=max_cluster_size,
    )
    if subset.empty:
        raise ValueError(f"No overlapping runs for '{name}'")
    return _corr_matrices(subset)


def run_all_correlations(
    use_abs: bool = False, min_cluster_size: int = 10
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run correlations for every dataset in dataset_configs and return nested dict."""
    results = {}
    for name, cfg in DATASET_CONFIGS.items():
        try:
            results[name] = corr_for_dataset(
                name,
                cfg,
                use_abs=use_abs,
                max_cluster_size=min_cluster_size,
            )
            print(f"{name}: done")
        except Exception as exc:
            print(f"Failed to run dataset {name}: {exc}")
    return results


def plot_correlations(
    group_by_dataset: bool = True,
    save: bool = True,
    show: bool = False,
    drop_low_signal: bool = True,
    min_cluster_size: int = 10,
    filetype: str = "png",
    out_dir: Path = None,
    *,
    style: str = "bar",                  # "bar" or "heatmap"
    figsize: tuple[int, int] = (14, 6),
    label_threshold: float = 0.20,       # show numeric labels only if |ρ| >= threshold
    show_strength_bands: bool = True,    # shade weak/medium/strong ranges for bars
) -> None:
    """
    Plot Spearman correlations between zero-cost proxies and best_val_f1_macro.

    style="bar"     -> grouped bars (your original, but cleaned up)
    style="heatmap" -> datasets x metrics heatmap with annotations
    """
    # ---- collect ---------------------------------------------------------
    corrs = run_all_correlations(
        drop_low_signal=drop_low_signal,
        min_cluster_size=min_cluster_size
    )
    records = []
    for ds, corr_dict in corrs.items():
        spearman = corr_dict["spearman"]
        for met in ZERO_COST_METRICS + ["ensemble", "initial_val_f1_macro"]:
            if met in spearman.index and "best_test_f1_macro" in spearman.columns:
                records.append({
                    "dataset":     ds,
                    "metric":      met,
                    "correlation": float(spearman.loc[met, "best_test_f1_macro"])
                })
    df = pd.DataFrame(records)

    # ---- pivot -----------------------------------------------------------
    if group_by_dataset:
        pivot = df.pivot(index="dataset", columns="metric", values="correlation")
        pivot = pivot[[m for m in ORDERED_METRICS if m in pivot.columns]]
        xlabel = "Dataset"
    else:
        pivot = df.pivot(index="metric", columns="dataset", values="correlation")
        pivot = pivot[[d for d in ORDERED_DATASETS if d in pivot.columns]]
        xlabel = "Zero-Cost Metric"

    # consistent ordering for rows, too
    pivot = pivot.reindex(
        ORDERED_DATASETS if group_by_dataset else ORDERED_METRICS
    ).dropna(how="all")

    # ---- common title ----------------------------------------------------
    title = "Spearman Correlation with best test f1 macro"
    if drop_low_signal:
        title += f" (no low-signal, thresh={min_cluster_size})"

    # =====================================================================
    # STYLE: GROUPED BAR
    # =====================================================================
    if style.lower() == "bar":
        fig, ax = plt.subplots(figsize=figsize)

        metrics = list(pivot.columns)
        # fallback color for metrics missing from palette
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = []
        for i, m in enumerate(metrics):
            c = METRIC_COLORS.get(m, default_colors[i % len(default_colors)] if default_colors else None)
            colors.append(c)

        pivot.plot(kind="bar", ax=ax, width=0.78, color=colors, legend=False, alpha=0.9)

        # zero line + grid
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        # optional strength bands
        if show_strength_bands:
            for lo, hi, a in [(0.0, 0.2, 0.04), (0.2, 0.5, 0.06), (0.5, 1.0, 0.08)]:
                ax.axhspan(lo, hi, facecolor="grey", alpha=a, zorder=0)
                ax.axhspan(-hi, -lo, facecolor="grey", alpha=a, zorder=0)

        # y-lims symmetric with a little headroom for labels
        ymax = float(np.nanmax(np.abs(pivot.values)))
        ylim = max(0.6, np.ceil((ymax + 0.05) * 10) / 10)   # to the nearest 0.1
        ax.set_ylim(-ylim, ylim)

        # annotate (only for |ρ| >= label_threshold)
        for rect in ax.patches:
            h = rect.get_height()
            if np.isfinite(h) and abs(h) >= label_threshold:
                x = rect.get_x() + rect.get_width() / 2
                y = h + (0.02 * np.sign(h))
                ax.text(
                    x, y, f"{h:.2f}",
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=9,
                    color="black",
                    clip_on=False,
                )

        # legend below, with smart wrapping
        ncol = min(len(metrics), 6)
        legend_handles = [Patch(facecolor=colors[i], label=metrics[i]) for i in range(len(metrics))]
        ax.legend(
            handles=legend_handles,
            title="Metric",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=ncol,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()

        fig_to_save = fig

    # =====================================================================
    # STYLE: HEATMAP
    # =====================================================================
    elif style.lower() == "heatmap":
        # order: rows=index, cols=columns already set
        data = pivot.values.astype(float)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)

        # ticks & labels
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=9)

        # gridlines between cells
        ax.set_xticks(np.arange(-.5, pivot.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, pivot.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="both", bottom=False, left=False)

        # annotations (suppress tiny values to reduce clutter)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = data[i, j]
                if np.isfinite(val) and abs(val) >= label_threshold:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Spearman ρ", rotation=270, labelpad=12)

        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("" if group_by_dataset else "Dataset", fontsize=11)
        ax.set_ylabel("Dataset" if group_by_dataset else "Zero-Cost Metric", fontsize=11)
        plt.tight_layout()

        fig_to_save = fig

    else:
        raise ValueError(f"Unknown style='{style}'. Use 'bar' or 'heatmap'.")

    # ---- save/show -------------------------------------------------------
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    fn_root = f"zero_cost_proxies_correlations_{style}"
    if drop_low_signal:
        fn_root += f"_no_low_signal_{min_cluster_size}"
    out_path = out_dir / f"{fn_root}.{filetype}"

    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig_to_save.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation plot → {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig_to_save)


def _prep_df_for_final_plot(cfg: dict, *, id_regex: str = ID_REGEX) -> pd.DataFrame:
    """
    Load + align, then rename `best_val_f1_macro` → `final_val_f1_macro`
    so it matches the signature expected by `plot_final_bars_with_metrics`.
    """
    df = _load_and_align(cfg, id_regex, drop_low_signal=False, min_cluster_size=0)

    # If your pickles already have `final_val_f1_macro`, this is a no-op.
    if "final_val_f1_macro" not in df.columns and "best_val_f1_macro" in df.columns:
        df = df.rename(columns={"best_val_f1_macro": "final_val_f1_macro"})

    # Keep only runs that have the (renamed) final metric
    df = df.dropna(subset=["final_val_f1_macro"])
    return df

def plot_final_bars_with_metrics(
    df: pd.DataFrame,
    save: bool = True,
    show: bool = False,
    out_path: Path = DEFAULT_OUT_DIR / "zero_cost_proxies_against_all_runs.pdf",
):
    """
    df : DataFrame that contains at least
         'final_val_f1_macro' + all keys in METRIC_DIRECTION
    """
    df = df.copy()

    # 1.  direction-flip, then min-max normalise every metric
    for m, sign in METRIC_DIRECTION.items():
        if m not in df.columns:
            continue

        # flip sign so "higher-is-better"
        col = df[m] * sign

        # min-max scale to [0, 1]
        vmin, vmax = col.min(), col.max()
        if np.isclose(vmax, vmin):
            col = 0.5                          # flat metric → middle of the scale
        else:
            col = (col - vmin) / (vmax - vmin)

        df[m] = col

    #  add 20-run moving-average columns 
    runs = 100
    for m in METRIC_DIRECTION:
        if m in df.columns:
            df[f"{m}_ma{runs}"] = df[m].rolling(window=runs, min_periods=runs).mean()

    # ─── 2. sort by final score  ─────────────────────────────────────────
    df = df.sort_values("final_val_f1_macro", ascending=False).reset_index(drop=True)

    # ─── 3. plot ─────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 4))
    x = np.arange(len(df))

    # bars
    ax1.bar(x, df["final_val_f1_macro"], 0.8, color="#CCCCCC", label="final F1")
    ax1.set_ylabel("Final val F1")
    ax1.set_xlabel("Runs (sorted)")

    # smoothed metric curves
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"Metric (min–max scaled, {runs}-run MA)")
    for m in METRIC_DIRECTION:
        ma_col = f"{m}_ma{runs}"
        if ma_col in df.columns:
            ax2.plot(x, df[ma_col], color=METRIC_COLORS[m], label=f"{m} (MA-{runs})", lw=1.5)

    # tidy
    ax1.set_xlim(-0.5, len(df) - 0.5)
    ax1.margins(x=0)
    ax1.tick_params(axis="x", labelbottom=False)  # too many runs to label

    # combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8,
        frameon=False,
        ncol=2,
    )

    plt.tight_layout()
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"all_datasets_all_metrics: saved plot to {out_path}")

def plot_final_bars_all_datasets(
    *,
    save: bool = True,
    show_each: bool = False,
    id_regex: str = ID_REGEX,
) -> None:
    """
    Iterate over every dataset in DATASET_CONFIGS and call
    `plot_final_bars_with_metrics` once per dataset.
    """
    for name, cfg in DATASET_CONFIGS.items():
        try:
            df = _prep_df_for_final_plot(cfg, id_regex=id_regex)
            if df.empty:
                print(f"[{name}] no runs with aligned metrics – skipped.")
                continue

            # Use a dataset-specific output file
            suffix = f"_final_bars_{name}.pdf"
            out_path = Path(DEFAULT_OUT_DIR) / suffix

            plot_final_bars_with_metrics(
                df,
                save=save,
                show=show_each,
                out_path=out_path,          # new kw-arg added below
            )
            print(f"[{name}] saved → {out_path}")
        except Exception as exc:
            print(f"[{name}] FAILED: {exc}")


def plot_zero_cost_scatter(
    configs: dict,
    *,
    id_regex: str         = ID_REGEX,
    metrics: list[str]    = ZERO_COST_METRICS,
    top_k: int            = 4,
    drop_low_signal: bool = False,
    min_cluster_size: int = 0,
    figsize: tuple        = (16, 4),
    save_path: Path       = None,
    out_dir: Path         = None,
    show: bool            = True,
):
    """
    For each dataset in `configs`, picks its top_k zero-cost proxies by |Spearman|
    vs. best_val_f1_macro, and lays them out as a grid of scatterplots with ρ annotated.
    """
    if save_path is None:
        if out_dir is None:
            out_dir = DEFAULT_OUT_DIR
        save_path = out_dir / "zcp_scatter_all_datasets_w_spearman.pdf"
    n = len(configs)
    fig, axes = plt.subplots(
        n, top_k,
        figsize=(figsize[0], figsize[1] * n),
        squeeze=False
    )

    for row_idx, (ds_name, cfg) in enumerate(configs.items()):
        # load + align all metrics into a single df
        df = _load_and_align(
            cfg, id_regex,
            drop_low_signal=drop_low_signal,
            min_cluster_size=min_cluster_size
        )

        # compute |ρ| with best_val_f1_macro
        corrs = {
            m: abs(spearmanr(df[m], df["best_val_f1_macro"], nan_policy="omit").correlation)
            for m in metrics
        }
        top_metrics = sorted(corrs, key=corrs.get, reverse=True)[:top_k]

        for col_idx, m in enumerate(top_metrics):
            ax = axes[row_idx][col_idx]

            sns.scatterplot(
                data=df,
                x=m,
                y="best_val_f1_macro",
                ax=ax,
                s=20,
                alpha=0.6,
            )

            rho, _ = spearmanr(df[m], df["best_val_f1_macro"], nan_policy="omit")
            ax.set_title(f"{ds_name} | {m}\nρ = {rho:.2f}", fontsize=10)
            ax.set_xlabel(m, fontsize=9)
            ax.set_ylabel("F1 Macro", fontsize=9)
            ax.tick_params(labelsize=8)

    plt.tight_layout()

    # save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {save_path}")

    # show / close
    if show:
        plt.show()
    else:
        plt.close(fig)

# Standard palette used across figures
PALETTE10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def _norm_key(ds: str) -> str:
    """Normalize dataset names (e.g., 'mobiactv2' -> 'mobiact')."""
    ds = str(ds).strip().lower()
    return "mobiact" if ds == "mobiactv2" else ds

def _fmt_label(ds_norm: str) -> str:
    """Format dataset labels for display."""
    return (ds_norm.capitalize()
            if ds_norm in {"motionsense", "mobiact", "myogym"}
            else ds_norm.upper())

def _tight_ylim(ax, arrays, pad_frac=0.03):
    """Adjust y-axis limits with padding."""
    all_vals = []
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            all_vals.extend(arr[np.isfinite(arr)])
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        pad = pad_frac * (ymax - ymin) if ymax > ymin else pad_frac
        ax.set_ylim(ymin - pad, ymax + pad)

def _style_axes(ax, border: bool):
    """Style matplotlib axes, including grid, axis visibility, and tick parameters."""
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    if border:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(1.0)
            sp.set_edgecolor("black")
        ax.tick_params(axis="both", length=3)
    else:
        # No border around the plot
        for sp in ax.spines.values():
            sp.set_visible(True)
        # Optional: remove tick marks but keep labels
        ax.tick_params(axis="both", length=0)

def plot_delta_two_panels(
    df,                               # long df with: dataset, metric, delta, k
    outpath,                          # Path or str to save (single figure with 2 panels)
    *,
    k_top=1,
    k_bottom=None,                    # if None -> max(df['k'])
    dataset_order=None,               # list[str]; if None -> sorted unique in df
    expected_df=None,                 # DataFrame: dataset, expected_delta, expected_delta_var [, k]
    figsize=(18, 10),
    legend_ncols=5,
    legend_loc="lower center",
    zero_label_tol=1e-5,
    rs_label="Random search",
    rs_color="#66CCEE",
    rs_ecolor="#000000",
    rs_err_capsize=4,
):
    """
    Plot two stacked panels showing the "Delta from Best Model" for various zero-cost metrics
    across different datasets. Includes bars for individual metrics and random search.
    """
    def _norm_ds(ds: str) -> str:
        ds = ds.lower()
        return "mobiact" if ds == "mobiactv2" else ds

    def _exp_for_k(exp_df, k_val):
        if exp_df is None:
            return {}
        e = exp_df.copy()
        if "k" in e.columns:
            e = e[e["k"] == k_val]
        out = {}
        for _, r in e.iterrows():
            ds = _norm_ds(str(r["dataset"]))
            mu = float(r["expected_delta"])
            std = float(np.sqrt(r["expected_delta_var"])) if "expected_delta_var" in e.columns else np.nan
            out[ds] = (mu, std)
        return out

    def _draw_panel(ax, dfi, exp_map, k_label):
        """Draw a single panel of the delta plot."""
        metrics = sorted(dfi["metric"].unique())
        n_groups = len(metrics) + 1     # +1 slot for Random search
        x = np.arange(len(dataset_order))
        width = 0.8 / max(n_groups, 1)

        max_val, min_val = -np.inf, np.inf

        # regular metric bars (now with black outlines)
        for j, m in enumerate(metrics):
            offsets = x + (j - n_groups / 2) * width + width / 2
            sub = dfi[dfi["metric"] == m].set_index("dataset")
            vals = [sub.loc[ds, "delta"] if ds in sub.index else np.nan for ds in dataset_order]

            bars = ax.bar(
                offsets, vals, width=width, label=m,
                color=PALETTE10[j % len(PALETTE10)],
                edgecolor="black", linewidth=0.9
            )

            for rect, val in zip(bars, vals):
                if np.isfinite(val) and abs(val) <= zero_label_tol:
                    ax.annotate(
                        "0",
                        (rect.get_x() + rect.get_width()/2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", va="bottom", fontsize=9
                    )

            fin = [v for v in vals if np.isfinite(v)]
            if fin:
                max_val = max(max_val, max(fin))
                min_val = min(min_val, min(fin))

        # random search bars (mean with SD error bars) — outlined
        j_rs = len(metrics)
        rs_offsets = x + (j_rs - n_groups / 2) * width + width / 2
        rs_vals, rs_yerr = [], []
        for ds in dataset_order:
            key = _norm_ds(ds)
            if key in exp_map:
                mu, std = exp_map[key]
                rs_vals.append(mu)
                rs_yerr.append(std if np.isfinite(std) else 0.0)
                if np.isfinite(mu):
                    max_val = max(max_val, mu + (std if np.isfinite(std) else 0.0))
                    min_val = min(min_val, mu - (std if np.isfinite(std) else 0.0))
            else:
                rs_vals.append(np.nan)
                rs_yerr.append(0.0)

        ax.bar(
            rs_offsets, rs_vals, width=width, label=rs_label,
            color=rs_color, edgecolor="black", linewidth=0.9,
            yerr=rs_yerr,
            error_kw=dict(elinewidth=1.4, ecolor=rs_ecolor, capsize=rs_err_capsize, capthick=1.4)
        )

        # ticks / labels
        tick_labels = [("mobiact" if d == "mobiactv2" else d) for d in dataset_order]
        tick_labels = [d.upper() if d.lower() not in {"motionsense", "mobiact", "myogym"}
                       else d.capitalize() for d in tick_labels]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_ylabel("Delta from Best Model")

        # annotate k (no titles)
        ax.text(0.01, 0.98, f"k = {k_label}", transform=ax.transAxes,
                ha="left", va="top", fontsize=14)

        # y-lims with headroom
        if np.isfinite(max_val) and np.isfinite(min_val):
            pad = 0.1 * (max_val - min_val if max_val > min_val else 1.0)
            ax.set_ylim(min_val - 0.2 * pad, max_val + pad)

        _style_axes(ax, border=False)
        return ax.get_legend_handles_labels()

    if dataset_order is None:
        dataset_order = sorted(df["dataset"].unique(), key=str)
    if k_bottom is None:
        k_bottom = int(df["k"].max())

    top_df = df[df["k"] == k_top].copy()
    bot_df = df[df["k"] == k_bottom].copy()
    exp_top = _exp_for_k(expected_df, k_top)
    exp_bot = _exp_for_k(expected_df, k_bottom)

    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    h1, l1 = _draw_panel(ax_top, top_df, exp_top, k_top)
    h2, l2 = _draw_panel(ax_bot, bot_df, exp_bot, k_bottom if k_bottom is not None else "N/A")

    # shared legend (dedupe)
    handles_all = h1 + h2
    labels_all  = l1 + l2
    seen, H, L = set(), [], []
    for h, l in zip(handles_all, labels_all):
        if l and l not in seen:
            seen.add(l); H.append(h); L.append(l)

    # spacing & legend placement
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    fig.subplots_adjust(hspace=0.05, bottom=0.18)
    fig.legend(
        H, L,
        loc="upper center",
        bbox_to_anchor=(0.5, -.15),
        bbox_transform=ax_bot.transAxes,
        ncol=min(len(L), legend_ncols),
        frameon=True, edgecolor="black", facecolor="white"
    )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[plot saved → {outpath}]")
    plt.show()

def plot_ensemble_delta_grid(
    df,         # columns: dataset, metric, delta, k
    exp_df,     # columns: dataset, expected_delta, expected_delta_var, k
    out=None,   # directory OR full path, e.g. "./figs" or "./figs/delta_grid.pdf"
    filetype="png",
    dpi=300,
    xscale="log",        # "log" or "linear"
    show=True,
    band_alpha=0.18,
    sigma_mult=1.0,      # std-band multiplier (±sigma_mult·std). No CIs.
    dataset_order=None,  # e.g. ORDERED_DATASETS from your other plots
):
    """
    Plot a grid of ensemble delta values.
    """
    # --- normalize dataset keys on both frames ---
    dfe = df[df["metric"] == "ensemble"].copy()
    dfe["dataset_norm"] = dfe["dataset"].map(_norm_key)

    exf = exp_df.copy()
    exf["dataset_norm"] = exf["dataset"].map(_norm_key)

    # union of datasets present
    union_norm = list(sorted(set(dfe["dataset_norm"].unique()) | set(exf["dataset_norm"].unique())))

    # apply requested ordering (normalized), falling back to union order
    if dataset_order is not None:
        order_norm = [_norm_key(ds) for ds in dataset_order]
        datasets = [ds for ds in order_norm if ds in union_norm]
        # append any extras not in the provided order
        datasets += [ds for ds in union_norm if ds not in datasets]
    else:
        datasets = union_norm

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    axes = axes.ravel()

    c_delta = "#ff7f0e"  # orange

    for i, ds_norm in enumerate(datasets[:6]):  # expect 6 datasets
        ax = axes[i]
        sub = dfe[dfe["dataset_norm"] == ds_norm].sort_values("k")
        ex  = exf[exf["dataset_norm"] == ds_norm].sort_values("k")

        # Actual delta
        if not sub.empty:
            ax.plot(sub["k"], sub["delta"], color=c_delta, linewidth=1.8, label="Delta")

        # Expected delta (dashed line + ±sigma std band)
        if not ex.empty:
            x = ex["k"].to_numpy()
            y = ex["expected_delta"].to_numpy()
            std = np.sqrt(ex["expected_delta_var"].to_numpy()) if "expected_delta_var" in ex.columns else None

            ax.plot(x, y, linestyle="--", linewidth=1.2, color=c_delta, label="Delta from Random Selection")

            if std is not None and np.all(np.isfinite(std)):
                ylo = y - sigma_mult * std
                yhi = y + sigma_mult * std
                ax.fill_between(x, ylo, yhi, color=c_delta, alpha=band_alpha, linewidth=0)

        if xscale == "log":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(_fmt_label(ds_norm))

        if i % 3 == 0:
            ax.set_ylabel("Delta")
        if i // 3 == 1:
            ax.set_xlabel("K (log scale)" if xscale == "log" else "K")

    # Hide unused panes if fewer than 6 datasets
    for j in range(len(datasets), 6):
        fig.delaxes(axes[j])

    # Shared legend (2 entries)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color=c_delta, lw=1.8, label="Ensemble ZCP Delta"),
        Line2D([0], [0], color=c_delta, lw=1.2, ls="--", label="Delta from random selection"),
    ]
    fig.legend(legend_lines, [l.get_label() for l in legend_lines],
               loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    # Save (dir or full path)
    saved_path = None
    if out is not None:
        out = Path(out)
        saved_path = out if out.suffix else (out / f"ensemble_delta_grid.{filetype}")
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved → {saved_path}]")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path

def plot_spearman_from_df_single(
    df,                 # columns: dataset, metric, correlation
    outpath,
    figsize=(18, 5),
    dataset_order: list[str] | None = None,
):
    """
    Plot Spearman correlation from a DataFrame in a single panel.
    """
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    # dataset order + label normalization
    def _norm_label(d: str) -> str:
        d = "mobiact" if d == "mobiactv2" else d
        return d.capitalize() if d.lower() in {"motionsense", "mobiact", "myogym"} else d.upper()

    datasets = dataset_order or list(df["dataset"].unique())
    metrics = sorted(df["metric"].unique())
    n_metrics = max(len(metrics), 1)

    x = np.arange(len(datasets))
    width = 0.8 / n_metrics  # group width = 0.8

    fig, ax = plt.subplots(figsize=figsize)

    # remove plot borders (spines) & ticks (keep labels)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
    ax.tick_params(axis="both", length=0)

    # y-grid only
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # grouped bars by metric
    for j, m in enumerate(metrics):
        offsets = x + (j - n_metrics / 2) * width + width / 2
        sub = df[df["metric"] == m].set_index("dataset")
        vals = [float(sub.loc[ds, "correlation"]) if ds in sub.index else np.nan for ds in datasets]
        ax.bar(
            offsets, vals, width=width, label=m,
            color=PALETTE10[j % len(PALETTE10)],
            edgecolor="black", linewidth=0.9
        )

    # y label & limits (Spearman ρ ∈ [-1, 1], with padding)
    ax.set_ylabel("Spearman ρ")
    yvals = df["correlation"].to_numpy(dtype=float)
    yfinite = yvals[np.isfinite(yvals)]
    ymin, ymax = -.4, 1.0
    pad = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin, ymax)

    # x ticks
    ax.set_xticks(x)
    ax.set_xticklabels([_norm_label(d) for d in datasets], rotation=0, ha="right")
    ax.margins(x=0.02)

    # single, bottom legend (no frame)
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.2)
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(6, len(labels)),
        frameon=True, edgecolor="black", facecolor="white"
    )

    # save & show
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[plot saved → {outpath}]")
    plt.show()

def plot_spearman_from_df(
    df,                 # columns: dataset, metric, correlation
    outpath=None,
    figsize=(18, 5),
    dataset_order: list[str] | None = None,
    style: str = "bar",  # "bar" or "heatmap"
    show: bool = True,
    save: bool = True,
):
    """
    Plot Spearman correlation from a DataFrame, with options for bar or heatmap style.
    """
    if style.lower() == "bar":
        plot_spearman_from_df_single(df, outpath, figsize, dataset_order)
    elif style.lower() == "heatmap":
        # Heatmap implementation can be added here if needed
        raise NotImplementedError("Heatmap style not yet implemented")
    else:
        raise ValueError(f"Unknown style='{style}'. Use 'bar' or 'heatmap'.")

if __name__ == "__main__":
    # plot_correlations(show=True, save=True, drop_low_signal=False)
    df = _load_and_align(DATASET_CONFIGS["motionsense"], id_regex=ID_REGEX, keep_run_dir=True)
    # plot_final_bars_all_datasets(
    #     save=True,      # write the files
    #     show_each=False # don't pop up an interactive window for every plot
    # )