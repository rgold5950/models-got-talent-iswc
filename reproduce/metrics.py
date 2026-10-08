"""Canonical definitions of the paper's evaluation metrics.

Section 3.4 of the paper defines two metrics over a pool of fully-trained
architectures:

Performance gap (Delta_k)
    Take the top-k architectures ranked by a zero-cost proxy, train only those
    k, and pick the one with the highest *validation* F1. Delta_k is the test F1
    of the best architecture found by full-scale training minus the test F1 of
    that ZCP-selected architecture. Lower is better; negative means the ZCP
    found a model that generalises better than the one full-scale training
    selected.

Talent rate
    The fraction of the top-k proxy-ranked architectures that also fall in the
    top-k after full training, reported as a percentage. The paper uses
    k = 10% of the pool for this metric.

Both metrics use the *validation* ranking as the ground truth top-k, because
that is the model-selection signal a practitioner actually has access to; test
F1 is only ever read off once a model has been selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Dataset order used in every figure and table of the paper.
ORDERED_DATASETS = ["myogym", "hhar", "mobiactv2", "motionsense", "pamap2", "rwhar"]

# Display names as they appear in the paper.
DATASET_LABELS = {
    "myogym": "Myogym",
    "hhar": "HHAR",
    "mobiactv2": "Mobiact",
    "motionsense": "Motionsense",
    "pamap2": "PAMAP2",
    "rwhar": "RWHAR",
}

# The seven zero-cost proxies from Abdelfattah et al. that make up the ensemble.
# synflow_bn is computed by the pipeline but excluded: none of the sampled
# architectures use batch norm, so it is identical to synflow.
ZC_METRICS = [
    "grad_norm",
    "snip",
    "grasp",
    "fisher",
    "jacob_cov",
    "plain",
    "synflow",
]

# The eighth proxy is initial_val_f1_macro (validation F1 before any training),
# introduced in this paper. "ensemble" is the mean of the percentile ranks of
# the seven ZC_METRICS.
ALL_PROXIES = ["ensemble"] + ZC_METRICS + ["initial_val_f1_macro"]

# Proxies where a more negative raw score indicates a better architecture. Their
# sign is flipped so that "higher is better" holds uniformly before ranking.
LOWER_IS_BETTER = ("grasp",)

# Number of Monte-Carlo draws used for the random-search baseline.
N_RANDOM_DRAWS = 1000

# Fraction of the pool used for the top-10% panels.
K_PERCENTAGE = 0.1


def add_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Add the rank-averaged ensemble proxy column."""
    df = df.copy()
    for m in LOWER_IS_BETTER:
        if m in df.columns:
            df[m] = -df[m]
    df["ensemble"] = df[ZC_METRICS].rank(pct=True).mean(axis=1)
    return df


@dataclass
class DatasetStats:
    """Delta_k and talent rate for every proxy on one dataset at one k."""

    dataset: str
    k: int
    pool_size: int
    best_test_f1: float
    delta: dict[str, float] = field(default_factory=dict)
    talent: dict[str, float] = field(default_factory=dict)
    random_delta: float = float("nan")
    random_delta_std: float = float("nan")
    random_talent: float = float("nan")
    random_talent_std: float = float("nan")


def compute_stats(df: pd.DataFrame, dataset: str, k: int, *, seed: int = 0) -> DatasetStats:
    """Compute Delta_k and talent rate for each proxy on a single dataset.

    `df` must already contain the ensemble column (see `add_ensemble`) plus
    `model_id`, `best_val_f1_macro` and `best_test_f1_macro`.
    """
    if k < 1 or k > len(df):
        raise ValueError(f"k={k} out of range for pool of {len(df)} architectures")

    top_by_val = df.sort_values("best_val_f1_macro", ascending=False).head(k)
    # Reference point: the model full-scale training would have shipped, i.e.
    # the highest-validation model in the whole pool.
    best_test_f1 = float(top_by_val["best_test_f1_macro"].iloc[0])
    actual_ids = set(top_by_val["model_id"])

    stats = DatasetStats(
        dataset=dataset, k=k, pool_size=len(df), best_test_f1=best_test_f1
    )
    for proxy in ALL_PROXIES:
        top_by_proxy = df.sort_values(proxy, ascending=False).head(k)
        selected = top_by_proxy.sort_values("best_val_f1_macro", ascending=False)
        stats.delta[proxy] = best_test_f1 - float(selected["best_test_f1_macro"].iloc[0])
        predicted_ids = set(top_by_proxy["model_id"])
        stats.talent[proxy] = len(predicted_ids & actual_ids) / len(actual_ids) * 100

    rng = np.random.RandomState(seed)
    deltas, talents = [], []
    for _ in range(N_RANDOM_DRAWS):
        draw = df.sample(n=k, replace=False, random_state=rng)
        picked = draw.sort_values("best_val_f1_macro", ascending=False)
        deltas.append(best_test_f1 - float(picked["best_test_f1_macro"].iloc[0]))
        talents.append(len(set(draw["model_id"]) & actual_ids) / len(actual_ids) * 100)
    stats.random_delta = float(np.mean(deltas))
    stats.random_delta_std = float(np.std(deltas))
    stats.random_talent = float(np.mean(talents))
    stats.random_talent_std = float(np.std(talents))
    return stats


def compute_all_stats(
    pools: dict[str, pd.DataFrame],
    k: int | None = None,
    *,
    k_percentage: float | None = None,
    seed: int = 0,
) -> dict[str, DatasetStats]:
    """Compute stats for every dataset at a fixed k or at a fraction of the pool.

    Exactly one of `k` and `k_percentage` must be given. `k_percentage` resolves
    k per dataset, so pools of slightly different sizes each get their own 10%.
    """
    if (k is None) == (k_percentage is None):
        raise ValueError("pass exactly one of k or k_percentage")
    out = {}
    for dataset, df in pools.items():
        k_ds = k if k is not None else int(round(len(df) * k_percentage))
        out[dataset] = compute_stats(df, dataset, k_ds, seed=seed)
    return out


def stats_to_frame(stats: dict[str, DatasetStats]) -> pd.DataFrame:
    """Flatten stats into a long dataframe: dataset, metric, delta, talent, k."""
    rows = []
    for dataset, s in stats.items():
        for proxy in ALL_PROXIES:
            rows.append(
                {
                    "dataset": dataset,
                    "metric": proxy,
                    "delta": s.delta[proxy],
                    "talent": s.talent[proxy],
                    "k": s.k,
                }
            )
    return pd.DataFrame(rows)


def random_baseline_to_frame(stats: dict[str, DatasetStats]) -> pd.DataFrame:
    """Random-search baseline as a dataframe, one row per dataset."""
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "expected_delta": s.random_delta,
                "expected_delta_std": s.random_delta_std,
                "expected_talent_rate": s.random_talent,
                "expected_talent_rate_std": s.random_talent_std,
                "k": s.k,
            }
            for dataset, s in stats.items()
        ]
    )


def best_proxy_per_dataset(
    stats: dict[str, DatasetStats], metric: str = "delta"
) -> dict[str, tuple[str, float]]:
    """For each dataset, the proxy with the best (lowest delta / highest talent)."""
    if metric not in ("delta", "talent"):
        raise ValueError("metric must be 'delta' or 'talent'")
    out = {}
    for dataset, s in stats.items():
        values = getattr(s, metric)
        if metric == "delta":
            proxy = min(values, key=values.get)
        else:
            proxy = max(values, key=values.get)
        out[dataset] = (proxy, values[proxy])
    return out
