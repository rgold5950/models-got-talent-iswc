import json
import math
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS
from analysis.model_testing.zero_cost_metric_analysis import (
    METRIC_COLORS,
    METRIC_DIRECTION,
    ORDERED_DATASETS,
    ORDERED_METRICS,
)
from analysis.model_testing.zero_cost_metric_analysis import (
    ZERO_COST_METRICS as ZC_METRICS,
)
from analysis.post_processing_pipeline.run_processing import process_dataset

plt.style.use("ggplot")
plt.rcParams.update(
    {  # bump every default font
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

# directory to save the figures from the talented models analysis
from utils.path_utils import PROJECT_ROOT
# Default OUT_DIR, but can be overridden by passing out_dir parameter to functions
DEFAULT_OUT_DIR = Path(PROJECT_ROOT) / "analysis" / "figures" / "current" / "talented_models"
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure the directory exists
OUT_DIR = DEFAULT_OUT_DIR  # alias for backwards compatibility with notebooks


def read_first_record(path: Path):
    """Return the first non-blank JSON object in result.json or None."""
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:  # iterate until we hit a non-blank line
            line = line.strip()
            if line:  # non-empty
                return json.loads(line)
    return None  # file was empty


class ExperimentVisualizer:
    def __init__(
        self,
        dataset_name: str,
        top_final_metric: str = "best_val_f1_macro",
        top_initial_metric: str = "initial_val_f1_macro",
        drop_low_signal: bool = False,
        min_cluster_size: int = 10,
    ):
        """
        Initialize visualizer with the base directory containing all experiment data.

        Args:
            base_dir: Base directory containing all experiment folders
        """
        self.cfg = DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.experiments = {}
        self.experiments_df = None
        self.load_all_experiments()
        self.top_final_metric = top_final_metric
        self.top_initial_metric = top_initial_metric
        self.drop_low_signal = drop_low_signal
        self.min_cluster_size = int(min_cluster_size)

        # Pre-compute exclusion sets (lazy: compute on first use if preferred)
        self._low_signal_exclude_cache: dict[str, set[str]] = {}

    def _low_signal_exclude(self, metric: str) -> set[str]:
        """
        Return set of model_ids whose value in `metric` belongs to a cluster
        of size >= self.min_cluster_size. Cached per-metric.
        """
        if not self.drop_low_signal:
            return set()
        if metric in self._low_signal_exclude_cache:
            return self._low_signal_exclude_cache[metric]

        if metric not in self.experiments_df.columns:
            self._low_signal_exclude_cache[metric] = set()
            return set()

        counts = self.experiments_df[metric].value_counts(dropna=False)
        clustered_values = counts[counts >= self.min_cluster_size].index
        low_signal_model_ids = set(
            self.experiments_df.loc[
                self.experiments_df[metric].isin(clustered_values), "model_id"
            ].tolist()
        )
        self._low_signal_exclude_cache[metric] = low_signal_model_ids
        if self.drop_low_signal:
            print(
                f"[low-signal] {metric}: dropping {len(low_signal_model_ids)} model(s) "
                f"in {len(low_signal_model_ids)} cluster(s) ≥{self.min_cluster_size}."
            )
        return low_signal_model_ids

    def load_all_experiments(self) -> None:
        full_pkl = Path(self.cfg["full_results_pkl"])

        if not full_pkl.exists():
            print(f"[pipeline] {full_pkl.name} missing – building now …")
            process_dataset(Path(self.cfg["results_dir"]), full_pkl)
            if not full_pkl.exists():
                raise FileNotFoundError(f"Failed to create {full_pkl}")

        full_df = pd.read_pickle(full_pkl)
        req_cols = {
            "run_dir",
            "initial_val_f1_macro",
            "best_val_f1_macro",
            "best_val_epoch",
            "best_test_f1_macro",
            "best_test_epoch",
        }
        missing = req_cols - set(full_df.columns)
        if missing:
            raise KeyError(f"{full_pkl} is missing: {', '.join(missing)}")

        # Extract model_id from run_dir - try multiple patterns
        # Pattern 1: arch_seed=X,combo_idx=Y (old format)
        # Pattern 2: arch_seed=X,init_name=Y,init_seed=Z (new format)
        # Pattern 3: Use the full trial identifier (train_tune_XXXXX_NNNNN_N)
        model_id = full_df["run_dir"].str.extract(r"(arch_seed=\d+,combo_idx=\d+)")
        if model_id[0].isna().all():
            # Try new format: extract arch_seed and init_seed
            model_id = full_df["run_dir"].str.extract(r"(arch_seed=\d+,init_name=\w+,init_seed=\d+)")
        if model_id[0].isna().all():
            # Fall back to trial identifier
            model_id = full_df["run_dir"].str.extract(r"(train_tune_\w+_\d+_\d+)")
        full_df["model_id"] = model_id
        before = len(full_df)
        full_df = full_df.dropna(subset=["initial_val_f1_macro", "best_val_f1_macro"])
        dropped = before - len(full_df)
        if dropped > 0:
            print(
                f"Warning: dropped {dropped} rows from experiment df for {self.dataset_name}"
            )

        self.experiments_df = full_df.drop_duplicates(subset=["model_id"]).copy()
        self.experiments = (
            full_df.drop_duplicates(subset=["model_id"])
            .set_index("model_id")
            .to_dict(orient="index")
            .copy()
        )

        print(f"[load] {len(self.experiments_df)} experiments from {full_pkl}")

        zc_path = Path(self.cfg["zero_cost_pkl"])

        if not zc_path.exists():
            print(f"[warn] {zc_path} not found – skipping zero-cost merge")
            return
        zc = pd.read_pickle(zc_path)  # must contain 'run_dir'
        # keep only the metrics we need
        zc = zc[["run_dir", *ZC_METRICS]]
        # Extract model_id using same patterns as full_df
        zc_model_id = zc["run_dir"].str.extract(r"(arch_seed=\d+,combo_idx=\d+)")
        if zc_model_id[0].isna().all():
            zc_model_id = zc["run_dir"].str.extract(r"(arch_seed=\d+,init_name=\w+,init_seed=\d+)")
        if zc_model_id[0].isna().all():
            zc_model_id = zc["run_dir"].str.extract(r"(train_tune_\w+_\d+_\d+)")
        zc["model_id"] = zc_model_id
        zc = zc.drop(columns=["run_dir"])
        # calculate ZC ensemble score
        zc["ensemble"] = zc[ZC_METRICS].rank(pct=True).mean(axis=1)
        self.experiments_df = self.experiments_df.merge(zc, on="model_id", how="left")
        for _, row in zc.iterrows():
            mid = row["model_id"]
            if mid in self.experiments:
                self.experiments[mid].update({m: row[m] for m in ZC_METRICS + ["ensemble"]})
        for m, sign in METRIC_DIRECTION.items():
            if sign == 1:
                continue
            if m in self.experiments_df.columns:
                self.experiments_df[m] *= (
                    sign  # flip the column to ensure that the metric is pointing the right direction (more positive is better)
                )
            for mid in self.experiments:
                if m in self.experiments[mid]:
                    self.experiments[mid][m] *= sign

    def get_exp_details(self) -> tuple[int]:
        """
        Return exp details
        For now just return the total number of runs
        """
        return (len(self.experiments),)

    def get_top_k_models(
        self,
        metric,
        k: int | None = None,
    ):
        """
        Get top-k models based on specified metric.

        Args:
            metric: Which metric to use ('initial_val_f1_macro' or 'best_val_f1_macro')
            k: Number of top models to return

        Returns:
            List of (model_id, metric_value) tuples for top-k models
        """
        valid_models = []
        for model_id, data in self.experiments.items():
            metric_val = data.get(metric)
            test = data.get("best_test_f1_macro")
            if metric_val is not None:
                valid_models.append((model_id, metric_val, test))

        # Sort descending (higher=better; metrics were flipped earlier if needed)
        valid_models.sort(key=lambda x: x[1], reverse=True)
        if k is not None:
            valid_models = valid_models[: int(k)]
        return valid_models  # top k valid models

    def find_talented_models(self, k):
        """
        Find the talent rate for the top-k models.

        Args:
            dataset_name: Name of the dataset
            k: Number of top models to consider

        Returns:
            float: Talent rate for the top-k models
        """
        # Get top-k models for each metric
        top_initial = self.get_top_k_models(self.top_initial_metric, k)
        top_best = self.get_top_k_models(
            "best_test_f1_macro", # we want to get the best globally by test f1 macro score
            k,
        )
        # Identify models in both lists
        initial_ids = set(model_id for model_id, _, _ in top_initial)
        best_ids = set(model_id for model_id, _, _ in top_best)
        common_ids = initial_ids.intersection(best_ids)
        return top_initial, top_best, common_ids

    def get_top_k_talent_delta(self, k):
        """
        Get the delta between the top model and the top talented model.

        Args:
            k: Number of top models to consider
        Returns:
            tuple: (talented_model_id, delta, top_model_score, talented_model_score)
        """
        top_initial, top_best, common_ids = self.find_talented_models(k)

        # Get the top model from the best list
        _, top_best_model_val_score, top_best_model_test_score = top_best[0]

        # Get the talented model val scores
        talented_models = []

        for model_id, _, _ in top_initial:  # Correctly iterate over model IDs
            model_details = self.get_model_details(model_id)
            val_score = model_details[self.top_final_metric]
            test_score = model_details["best_test_f1_macro"]
            talented_models.append((model_id, val_score, test_score))

        talented_models.sort(key=lambda x: x[1], reverse=True)
        talented_model_id, talented_model_val_score, talented_model_test_score = talented_models[0]
        delta = top_best_model_test_score - talented_model_test_score
        return talented_model_id, delta, top_best_model_test_score, talented_model_val_score, talented_model_test_score

    def compute_talent_stats(self, k):
        # Calculate talent rate (number of common models divided by k)
        top_initial, top_best, common_ids = self.find_talented_models(k)
        talent_rate = len(common_ids) / k
        return top_initial, top_best, common_ids, talent_rate

    def visualize_top_models(
        self, dataset_name, k=5, show=False, save=True, filetype: str = "png", out_dir: Path = None
    ):
        """
        Visualize top-k models based on both metrics side by side and print test scores.

        Args:
            k: Number of top models to display

        Returns:
            tuple: (common_ids, talent_rate) - Set of common model IDs and the talent rate
        """
        if out_dir is None:
            out_dir = DEFAULT_OUT_DIR
        suffix = ""
        if self.drop_low_signal:
            suffix = f"_no-low-signal_{self.min_cluster_size}"
        top_initial, top_best, common_ids, talent_rate = self.compute_talent_stats(k)
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Add a title that includes K and talent rate
        filt_note = ""
        if self.drop_low_signal:
            filt_note = (
                f" (low-signal clusters ≥{self.min_cluster_size} dropped from initial)"
            )
        fig.suptitle(
            f"{dataset_name}: Top-{k} Models Comparison (Talent Rate: {talent_rate:.2f}){filt_note}",
            fontsize=16,
        )

        blue = (137 / 255, 207 / 255, 240 / 255)
        green = (9 / 255, 121 / 255, 105 / 255)
        mustard = (255 / 255, 219 / 255, 88 / 255)

        # Plot top initial models
        # Store count of model types
        # get_top_k_models returns (model_id, metric_val, test_score) tuples
        model_ids, scores, _ = zip(*top_initial)
        initial_models_frequency = {}
        model_labels = []
        for model_id in model_ids:
            model_type = self.experiments[model_id].get("model_type", None)
            assert model_type is not None
            model_label = f"{model_id} ({model_type})"

            if model_type not in initial_models_frequency.keys():
                initial_models_frequency[model_type] = 1
            else:
                initial_models_frequency[model_type] += 1

            model_labels.append(model_label)

        bars1 = ax1.barh(
            model_labels,
            scores,
            color=[green if model_id in common_ids else blue for model_id in model_ids],
        )

        # Remove the y-axis ticks
        ax1.set_yticks([])
        ax1.invert_yaxis()  # This will show the highest score at the top
        ax1.set_xlabel(f"{self.top_initial_metric}")
        ax1.set_title(f"Top {k} Models by {self.top_initial_metric}")

        # Plot top best models
        # get_top_k_models returns (model_id, metric_val, test_score) tuples
        model_ids, scores, _ = zip(*top_best)
        model_labels = []
        best_models_frequency = {}

        for model_id in model_ids:
            model_type = self.experiments[model_id].get("model_type", None)
            assert model_type is not None
            model_label = f"{model_id} ({model_type})"

            if model_type not in best_models_frequency.keys():
                best_models_frequency[model_type] = 1
            else:
                best_models_frequency[model_type] += 1

            model_labels.append(model_label)

        print("Top-initial model type frequencies:")
        print(best_models_frequency)

        # print("\nInitial Validation F1 Macro scores for displayed models:")
        # for model_id, _ in top_best:
        #     test_score = self.experiments[model_id]["initial_val_f1_macro"]
        # print(f"Model {model_id}: {test_score:.4f}")

        # print("\nBest Validation F1 Macro scores for displayed models:")
        # for model_id, score in top_best:
        # print(f"Model {model_id}: {score:.4f}")

        # print("\nBest Test F1 Macro scores for displayed models:")
        # for model_id, _ in top_best:
        #     test_score = self.experiments[model_id]["best_test_f1_macro"]
        # print(f"Model {model_id}: {test_score:.4f}")

        bars2 = ax2.barh(
            model_labels,
            scores,
            color=[
                green if model_id in common_ids else mustard for model_id in model_ids
            ],
        )
        ax2.set_yticks([])
        ax2.invert_yaxis()  # This will show the highest score at the top
        ax2.set_xlabel("Best Val F1 Macro")
        ax2.set_title(f"Top {k} Models by Best Validation F1 Macro")
        legend_elements = [
            Patch(facecolor=green, label="Common in both lists"),
            Patch(facecolor=blue, label="Top initial only"),
            Patch(facecolor=mustard, label="Top best only"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=3,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        out_path = (
            out_dir
            / f"{dataset_name}_{self.top_initial_metric}_talented_model_analysis{suffix}.{filetype}"
        )
        if save:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        print(f"all_datasets_all_metrics: saved plot to {out_path}")
        return common_ids, talent_rate, out_path

    def find_first_talented_model(self, max_k=20):
        """
        Find the first "talented" model and calculate its delta from the top model.

        Args:
            max_k: Maximum number of top models to consider (default 20)

        Returns:
            tuple: (k, talented_model_id, delta, top_model_score, talented_model_score)
        """
        top_model_score = float("-inf")
        for k in range(1, max_k + 1):
            top_initial = self.get_top_k_models(self.top_initial_metric, k)
            top_best = self.get_top_k_models(self.top_final_metric, k)

            initial_ids = set(model_id for model_id, _, _ in top_initial)
            best_ids = set(model_id for model_id, _, _ in top_best)
            common_ids = initial_ids.intersection(best_ids)

            if common_ids:
                print(f"Found talented model at k={k}")
                top_model_id, top_model_score = top_best[0]
                talented_model_id = next(iter(common_ids)) if common_ids else None

                if talented_model_id:
                    talented_model_score = next(
                        score
                        for model_id, score in top_best
                        if model_id == talented_model_id
                    )
                    delta = top_model_score - talented_model_score
                    return (
                        k,
                        talented_model_id,
                        delta,
                        top_model_score,
                        talented_model_score,
                    )
                else:
                    return k, None, None, top_model_score, None

        return max_k, None, None, top_model_score, None

    def find_minimum_k_for_talent_rate(self, target_talent_rate):
        """
        Find the k value that gives the target talent rate.

        Args:
            target_talent_rate: Desired talent rate (between 0 and 1)

        Returns:
            int: k value that gives the target talent rate
        """
        for k in range(1, len(self.experiments) + 1):
            _, _, common_ids = self.find_talented_models(k)
            current_talent_rate = len(common_ids) / k

            if current_talent_rate >= target_talent_rate:
                return k

        return None

    def get_model_details(self, model_id):
        """
        Get detailed information about a specific model.

        Args:
            model_id: ID of the model to inspect

        Returns:
            Dictionary with model details
        """
        if model_id not in self.experiments:
            return f"Model {model_id} not found"

        exp_data = self.experiments[model_id]
        details = {
            "model_id": model_id,
            "initial_val_f1_macro": exp_data["initial_val_f1_macro"],
            "best_val_f1_macro": exp_data["best_val_f1_macro"],
            "best_val_epoch": exp_data["best_val_epoch"],
            # "config": exp_data["config"],
            "best_test_f1_macro": exp_data["best_test_f1_macro"],
        }
        return details


def main():
    # Use the user's home directory
    dataset_name = "rwhar"
    top_initial_metric = "initial_val_f1_macro"
    top_final_metric = "best_val_f1_macro"
    visualizer = ExperimentVisualizer(
        dataset_name,
        top_final_metric=top_final_metric,
        top_initial_metric=top_initial_metric,
    )

    (num_experiments,) = visualizer.get_exp_details()
    # Set k value
    top_percentile = 0.10
    k = int(num_experiments * top_percentile)

    # Visualize top k models
    common_models, talent_rate, _ = visualizer.visualize_top_models(dataset_name, k)
    print(f"\nK: {k}")
    print(f"Talent Rate: {talent_rate:.2f} ({len(common_models)}/{k} models)")
    print(f"Models in both top-k lists: {common_models}")

    # Get details for common models
    # if common_models:
    #     for model_id in common_models:
    #         details = visualizer.get_model_details(model_id)
    #         print(f"\nDetails for model {model_id}:")
    #         print(f"Initial val F1 macro: {details['initial_val_f1_macro']:.4f}")
    #         print(f"Best val F1 macro: {details['best_val_f1_macro']:.4f} (at epoch {details['best_val_epoch']})")
    #         print(f"Best test F1 macro: {details['best_test_f1_macro']:.4f}")
    #         print(f"Learning rate: {details['config']['learning_rate']}")
    #         print(f"Batch size: {details['config']['batch_size']}")
    #         print(f"Model architecture: {details['config']['model']}")

    # Find difference between top valid model and talented model with highest val score
    talented_model_id, delta, top_score, talented_val_score, talented_test_score = (
        visualizer.get_top_k_talent_delta(k)
    )
    top_initial, top_best, common_ids = visualizer.find_talented_models(k)
    ordered_models = visualizer.get_top_k_models(top_final_metric, k=None)
    talented_model_position_from_top = (
        ordered_models.index((talented_model_id, talented_val_score)) + 1
    )
    print(f"\nTop model ID: {talented_model_id}")
    print(f"Delta from top model: {delta:.4f}")
    print(f"Top model score: {top_score:.4f}")
    print(f"Talented model score: {talented_val_score:.4f}")
    print(f"Talented model index: {talented_model_position_from_top}")

    # Find first talented model
    k, talented_model_id, delta, top_score, talented_val_score = (
        visualizer.find_first_talented_model(k)
    )

    print("\nFirst talented model analysis:")
    print(f"K value where talented model found: {k}")
    if talented_model_id:
        print(f"Talented model ID: {talented_model_id}")
        print(f"Delta from top model: {delta:.4f}")
        print(f"Top model score: {top_score:.4f}")
        print(f"Talented model score: {talented_val_score:.4f}")
    else:
        print("No talented model found within the specified range.")
        print(f"Top model score: {top_score:.4f}")


def build_summary_bar_chart(
    results, metrics, out_path=None, show=False, save=True, filetype: str = "png"
):
    """
    Draw a grouped bar chart with dual y-axes:
        • Talent Rate  (left axis, teal)
        • Δ from Top   (right axis, orange)

    Outer groups = datasets; inside each group we repeat (talent, delta) for every
    `metric` in `metrics`.
    """
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / f"summary_grouped_bar.{filetype}"

    datasets = list(results.keys())
    n_datasets = len(datasets)
    n_metrics = len(metrics)
    if n_datasets == 0 or n_metrics == 0:
        print("[plot_summary] Nothing to plot.")
        return out_path

    # ── layout parameters ────────────────────────────────────────────────────
    bar_w = 0.35  # width of a single bar
    inner_gap = 0.05  # gap between talent & delta bars
    metric_gap = 0.40  # gap between metric clusters
    group_pad = 0.80  # gap between dataset groups

    cluster_w = 2 * bar_w + inner_gap
    group_w = n_metrics * cluster_w + (n_metrics - 1) * metric_gap

    # Figure size heuristic: width grows with number of datasets
    fig_w = max(10, n_datasets * (group_w + group_pad) * 0.25)
    fig_h = 6
    fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    # Twin axis for Δ bars
    ax2 = ax1.twinx()

    # Colours
    talent_col = (26 / 255, 158 / 255, 148 / 255)  # teal-green
    delta_col = (242 / 255, 142 / 255, 43 / 255)  # orange

    # ── place bars ───────────────────────────────────────────────────────────
    tick_positions = []  # x for metric tick labels
    tick_labels = []  # metric names (possibly shortened)
    ds_label_pos = []  # x for dataset labels

    for di, ds in enumerate(datasets):
        group_start = di * (group_w + group_pad)

        for mi, metric in enumerate(metrics):
            t_rate, delta = results[ds].get(metric, (None, None))
            if t_rate is None:
                continue

            base = group_start + mi * (cluster_w + metric_gap)
            # Talent-rate bar (left axis)
            ax1.bar(
                base,
                t_rate,
                width=bar_w,
                color=talent_col,
                label="Talent Rate" if di == mi == 0 else "_nolegend_",
            )
            # Δ-from-top bar (right axis)
            ax2.bar(
                base + bar_w + inner_gap,
                delta,
                width=bar_w,
                color=delta_col,
                label="Δ from Top" if di == mi == 0 else "_nolegend_",
            )

            tick_positions.append(base + bar_w / 2)  # centre of talent bar
            tick_labels.append(metric)

        ds_label_pos.append(group_start + group_w / 2)

    # ── x-axis: metric labels on the main axis ───────────────────────────────
    ax1.set_xticks(tick_positions, tick_labels, rotation=45, ha="right", fontsize=8)

    # ── add dataset names centred under each group (2nd x-axis line) ─────────
    for x, ds in zip(ds_label_pos, datasets):
        ax1.text(
            x,
            -0.12,  # move below tick labels
            ds,
            ha="center",
            va="top",
            transform=ax1.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
        )

    # ── axis labels, limits, legend ──────────────────────────────────────────
    ax1.set_ylabel("Talent Rate", color=talent_col)
    ax1.set_ylim(0, 1)  # talent rate is a proportion
    ax2.set_ylabel("Δ from Top", color=delta_col)
    ax2.tick_params(axis="y", labelcolor=delta_col)

    ax1.set_title("Talent Rate and Δ-from-Top per Dataset / Initial Metric")
    # Combine legends from both axes
    handles, labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax1.legend(handles, labels, loc="upper left")

    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"all_datasets_all_metrics: saved plot to {out_path}")
    return out_path


def _get_talent_rate_df(
    results: dict[str, dict[str, tuple[float, float]]],
    metrics: list[str],
):
    rows = []
    for ds, dd in results.items():
        for m in metrics:
            if (vals := dd.get(m)) is None:
                continue
            t_rate, delta = vals
            rows.append({"Dataset": ds, "Metric": m, "Talent": t_rate, "Delta": delta})

    if not rows:
        print("[plot] nothing to plot")
        return

    df = pd.DataFrame(rows)
    df["Dataset"] = pd.Categorical(
        df["Dataset"], categories=ORDERED_DATASETS, ordered=True
    )
    df["Metric"] = pd.Categorical(
        df["Metric"], categories=ORDERED_METRICS, ordered=True
    )
    df = df.sort_values(["Dataset", "Metric"]).dropna(subset=["Dataset"])
    return df


def bar_talent_rate_with_delta(
    results: dict[str, dict[str, tuple[float, float]]],
    metrics: list[str],
    figsize: tuple[int, int] = (14, 5),
    *,
    save: bool = True,
    show: bool = True,
    out_path: Path = None,
    out_dir: Path = None,
    filetype: str = "png",
):
    """
    Grouped bars  = Talent Rate  (0‒1, left y-axis)
    Black dots    = Δ-from-Best  (right y-axis)

    `results[dataset][metric] = (talent_rate, delta)`
    """
    if out_path is None:
        if out_dir is None:
            out_dir = DEFAULT_OUT_DIR
        out_path = out_dir / f"talent_rate_with_delta.{filetype}"
    # ------------- assemble tidy DataFrame -----------------
    df = _get_talent_rate_df(results=results, metrics=metrics)
    # pivot tables
    p_talent = df.pivot(index="Dataset", columns="Metric", values="Talent").fillna(0)
    p_delta = df.pivot(index="Dataset", columns="Metric", values="Delta").fillna(0)
    p_talent = p_talent[[m for m in ORDERED_METRICS if m in p_talent.columns]]
    p_delta = p_delta[[m for m in ORDERED_METRICS if m in p_delta.columns]]

    # ------------- plot -----------------------------------
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    bar_colors = [METRIC_COLORS.get(m, "#999999") for m in p_talent.columns]
    bar_containers = p_talent.plot(
        kind="bar",
        ax=ax1,
        width=0.75,
        alpha=0.9,
        color=bar_colors,
        legend=False,
    )

    for metric_idx, (container, metric) in enumerate(
        zip(bar_containers.containers, p_talent.columns)
    ):
        for dataset_idx, bar in enumerate(container):
            height = bar.get_height()
            if height < 1e-12:
                continue
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                metric,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color=METRIC_COLORS.get(metric, "black"),
                clip_on=False,
            )

    # colours already taken from pandas default / rcParams – keep
    # ---- add ∆ dots --------------------------------------
    for metric_idx, (container, metric) in enumerate(
        zip(bar_containers.containers, p_talent.columns)
    ):
        for dataset_idx, bar in enumerate(container):
            x = bar.get_x() + bar.get_width() / 2
            ds = p_talent.index[dataset_idx]

            talent_val = p_talent.loc[ds, metric]
            delta_val = p_delta.loc[ds, metric]

            if pd.isna(talent_val) or abs(talent_val) < 1e-12:
                continue
            if pd.isna(delta_val):
                continue
            ax2.scatter(
                x,
                delta_val,
                color="black",
                marker="x",
                s=30,
                zorder=5,
                label="Δ from Top"
                if metric_idx == 0 and dataset_idx == 0
                else "_nolegend_",
            )
            ax2.annotate(
                f"{delta_val:.1f}",   # ✅ no percent sign
                xy=(x, delta_val),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=90,
                color="black",
                zorder=6,
            )

    # ---- axes limits & labels ----------------------------
    ax1.set_ylabel("Talent Rate (0-1)")
    ax1.set_ylim(0, p_talent.to_numpy().max() * 1.10 or 1)

    ax2.set_ylabel("Δ from Best Model")
    ax2.tick_params(axis="y", labelcolor="black")

    ax1.set_xlabel("Dataset")
    ax1.set_title("Talent Rate and Delta")

    ax1.tick_params(axis="x", rotation=45, labelsize=9)

    # ---- legend (bars use colours, dots black) ------------
    handles_bars, labels_bars = ax1.get_legend_handles_labels()
    handles_bars, labels_bars = ax1.get_legend_handles_labels()

    ax1.legend(
        handles_bars,
        labels_bars ,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),  # put legend outside plot if you prefer
        ncol=2,
        frameon=False,
        fontsize=9,
        title="Metric / Marker",
        title_fontsize=10,
    )

    plt.tight_layout()

    # ---- save / show -------------------------------------
    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[plot] saved → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_all_datasets():
    return run_some_datasets(datasets=DATASET_CONFIGS.keys())


def run_some_datasets(
    datasets: list[str],
    show=False,
    save=True,
    min_cluster_size: int = 1,
    visualize_all: bool = False,
    filetype: str = "png",
    top_percentile = 0.05,
    out_dir: Path = None
):
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    png_paths = []
    top_final_metric = "best_val_f1_macro"
    initial_metrics = ["initial_val_f1_macro"] + ZC_METRICS + ["ensemble"]

    # collect for summary chart
    results: dict[str, dict[str, tuple[float, float]]] = {
        ds: {} for ds in DATASET_CONFIGS if ds in datasets
    }
    k_used = {}

    for dataset_name in DATASET_CONFIGS:
        if dataset_name not in datasets:
            continue
        print(f"\n{'#' * 40}\nDataset: {dataset_name}\n{'#' * 40}")
        k_used[dataset_name] = {}

        for initial_metric in initial_metrics:
            print(f"\n--- Running with initial metric: {initial_metric} ---")

            visualizer = ExperimentVisualizer(
                dataset_name,
                top_final_metric=top_final_metric,
                top_initial_metric=initial_metric,
                drop_low_signal=False,
                min_cluster_size=min_cluster_size,
            )

            (num_experiments,) = visualizer.get_exp_details()
            if num_experiments == 0:
                print(
                    f"Skipping {dataset_name}/{initial_metric}: no experiments found."
                )
                continue

            k = max(1, int(num_experiments * top_percentile))
            k_used[dataset_name][initial_metric] = k
            try:
                # Visualize + save per-dataset plot (already writes metric name in filename)
                if visualize_all:
                    common_models, talent_rate, output_path = (
                        visualizer.visualize_top_models(
                            dataset_name, k, save=save, show=show, filetype=filetype, out_dir=out_dir
                        )
                    )
                    png_paths.append(output_path)
                else:
                    top_initial, top_best, common_models, talent_rate = (
                        visualizer.compute_talent_stats(k)
                    )
                print(f"K: {k}")
                print(
                    f"Talent Rate: {talent_rate:.2f} ({len(common_models)}/{k} models)"
                )
                # Grab delta (talented vs. top)
                talented_model_id, delta, top_score, talented_val_score, talented_test_score = (
                    visualizer.get_top_k_talent_delta(k)
                )

                # store for summary
                results[dataset_name][initial_metric] = (talent_rate, delta)
            except Exception as e:
                raise e
    return results, k_used


def make_grid(
    png_paths: list[Path],
    *,
    cols: int = 4,  # 4 metrics per dataset row
    scale: float = 0.55,  # ↑ enlarge / ↓ shrink
    out_path: Path = DEFAULT_OUT_DIR / "all_datasets_grid.pdf",
    save: bool = True,
    show: bool = False,
):
    """
    Tile PNGs into a tight grid with ggplot style.

    `scale` multiplies the size of every thumbnail **on the canvas**:
        • 1.0  = original pixel-for-pixel
        • 0.55 = (default) shrink 45 % so 4×N fits on most monitors / PDFs
    """
    if not png_paths:
        print("[grid] No PNGs to tile.")
        return None

    # read one image to know native w×h  (assumes all same size)
    sample_img = mpimg.imread(png_paths[0])
    h_nat, w_nat = sample_img.shape[:2]

    rows = math.ceil(len(png_paths) / cols)

    # -------- figure size (in inches) ------------------------------------
    # dpi = 100  →  inches = pixels / 100
    dpi = 100
    fig_w = cols * (w_nat / dpi) * scale
    fig_h = rows * (h_nat / dpi) * scale
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # -------- grid spec with almost no padding ---------------------------
    gs = GridSpec(
        rows,
        cols,
        figure=fig,
        wspace=0.01,  # 1 % gap horizontally
        hspace=0.0,  # 2 % gap vertically
    )

    for i, img_path in enumerate(png_paths):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(mpimg.imread(img_path))
        ax.axis("off")

    legend_patches = [
        Patch(facecolor=(9 / 255, 121 / 255, 105 / 255), label="Common in both lists"),
        Patch(facecolor=(137 / 255, 207 / 255, 240 / 255), label="Top initial only"),
        Patch(facecolor=(255 / 255, 219 / 255, 88 / 255), label="Top best only"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),  # nudges legend just below the grid
    )

    plt.tight_layout(pad=0)

    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[grid] saved → {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_talent_delta_summary(
    results: dict[str, dict[str, tuple[float, float]]],
    metrics: list[str],
    save: bool = True,
    show: bool = False,
    out_path: Path = DEFAULT_OUT_DIR / "talent_delta_summary.pdf",
):
    """
    Creates a grouped bar chart showing Talent Rate and Delta for each metric grouped by dataset.

    Args:
        results: dict[dataset][metric] = (talent_rate, delta)
        metrics: list of metrics to plot
    """
    datasets = list(results.keys())
    n_datasets = len(datasets)
    n_metrics = len(metrics)

    bar_w = 0.35
    group_w = n_metrics * bar_w * 2 + 0.5  # 2 bars (talent, delta) per metric + padding
    x_positions = []

    fig, ax1 = plt.subplots(figsize=(max(10, n_datasets * (group_w + 0.5) * 0.25), 6))
    ax2 = ax1.twinx()

    # Colors
    talent_col = (26 / 255, 158 / 255, 148 / 255)  # teal
    delta_col = (242 / 255, 142 / 255, 43 / 255)  # orange

    # Iterate over datasets and metrics
    for di, ds in enumerate(datasets):
        ds_start = di * (group_w + 0.8)
        for mi, metric in enumerate(metrics):
            talent_rate, delta = results[ds].get(metric, (None, None))
            if talent_rate is None:
                continue

            base = ds_start + mi * (2 * bar_w + 0.15)
            # Talent bar
            talent_bar = ax1.bar(
                base,
                talent_rate,
                width=bar_w,
                color=talent_col,
                label="Talent Rate" if di == mi == 0 else "",
            )
            # Delta bar
            delta_bar = ax2.bar(
                base + bar_w,
                delta,
                width=bar_w,
                color=delta_col,
                label="Δ from Best" if di == mi == 0 else "",
            )

            # Add labels
            ax1.bar_label(talent_bar, fmt="%.2f", padding=2, fontsize=7)
            ax2.bar_label(delta_bar, fmt="%.2f", padding=2, fontsize=7)

            x_positions.append(base + bar_w / 2)

        # Add dataset label under group
        group_center = ds_start + (n_metrics * (2 * bar_w + 0.15)) / 2
        ax1.text(
            group_center,
            -0.07,
            ds,
            ha="center",
            va="top",
            transform=ax1.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
        )

    # X-axis ticks
    tick_positions = [
        ds * (group_w + 0.8) + (n_metrics - 1) * bar_w for ds in range(n_datasets)
    ]
    ax1.set_xticks([])
    ax1.set_xlim(-0.5, n_datasets * (group_w + 0.8))

    # Axis labels
    ax1.set_ylabel("Talent Rate", color=talent_col)
    ax1.set_ylim(0, 1)
    ax2.set_ylabel("Δ from Best", color=delta_col)

    ax1.set_title("Talent Rate and Δ from Best by Metric and Dataset")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()

    if save:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[plot] Saved bar chart to {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def grid_best_metric_per_dataset(
    *,
    k_percentile: float = 0.05,  # k = max(1, round(p * #runs))
    drop_low_signal: bool = False,
    min_cluster_size: int = 10,
    cols: int = 3,  # thumbnails per row in final grid
    grid_out: Path = DEFAULT_OUT_DIR / "best_metric_grid.pdf",
    show: bool = True,
    save: bool = True,
):
    """
    For each dataset:
        1.  Compute talent-rate for *all* initial metrics.
        2.  Pick the metric with the highest talent-rate.
        3.  Call `visualize_top_models()` once => PNG.
    Finally tile those PNGs into a tight grid (no extra white-space).
    """
    initial_metrics = ["initial_val_f1_macro"] + list(ZC_METRICS)
    pngs: list[Path] = []

    for ds in DATASET_CONFIGS:
        print(f"\n### {ds} ###")
        best_metric, best_rate = None, -1.0

        # --- find metric with highest talent-rate -----------------------
        for m in initial_metrics:
            vis = ExperimentVisualizer(
                ds,
                top_initial_metric=m,
                top_final_metric="best_val_f1_macro",
                drop_low_signal=drop_low_signal,
                min_cluster_size=min_cluster_size,
            )
            (n_runs,) = vis.get_exp_details()
            if n_runs == 0:
                continue
            k = max(1, round(n_runs * k_percentile))
            _, _, common = vis.find_talented_models(k)
            rate = len(common) / k
            if rate > best_rate:
                best_metric, best_rate = m, rate

        if best_metric is None:
            print("  – no experiments; skipping")
            continue

        print(f"  best metric = {best_metric}  (talent-rate {best_rate:.2f})")

        # --- generate final PNG for that dataset -----------------------
        vis = ExperimentVisualizer(
            ds,
            top_initial_metric=best_metric,
            top_final_metric="best_val_f1_macro",
            drop_low_signal=drop_low_signal,
            min_cluster_size=min_cluster_size,
        )
        k = max(1, round(vis.get_exp_details()[0] * k_percentile))
        _, _, png_path = vis.visualize_top_models(ds, k=k, save=True, show=False)
        pngs.append(png_path)

    # --- tile into grid (tight layout) ---------------------------------
    if pngs:
        make_grid(
            png_paths=pngs,
            cols=cols,
            out_path=grid_out,
            save=save,
            show=show,
        )
    else:
        print("[grid] nothing to plot")


if __name__ == "__main__":
    # results, k = run_all_datasets()

    # results, k = run_some_datasets(datasets=ORDERED_DATASETS)
    results, k = run_some_datasets(datasets=['mobiactv2'])

    df = _get_talent_rate_df(results, ZC_METRICS + ["ensemble"] + ["initial_val_f1_macro"])

    from IPython import embed; embed()
    # embed
    # bar_talent_rate_with_delta(
    #     results, ZC_METRICS + ["ensemble"] + ["initial_val_f1_macro"], save=True, show=True
    # )
    # check to ensure that metrics have been properly flipped
    # vis = ExperimentVisualizer("pamap2")
    # raw = vis.experiments_df["grasp"].head()
    # print("raw grasp values:\n", raw)

    # # After flip the largest (least-negative) should now be top
    # top = vis.get_top_k_models("grasp", k=5)
    # print("top 5 by flipped grasp:", top[:5])
    # grid_best_metric_per_dataset(
    #     k_percentile=0.05,     # same 5 % rule you used before
    #     drop_low_signal=False,
    #     cols=3,                # 3 plots per row; change as you like
    #     show=True,
    #     save=True,
    # )
