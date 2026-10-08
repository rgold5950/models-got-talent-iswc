"""Every quantitative claim the paper makes, expressed as an executable check.

Each `Claim` pairs the sentence from the paper with a predicate over the
recomputed results, so `python -m reproduce.run` and
`tests/test_paper_reproduction.py` are checking exactly the same thing. If a
future re-run of the pipeline changes a number enough to break a claim, the
failing claim names the section of the paper that would need revisiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from reproduce.metrics import ALL_PROXIES, ORDERED_DATASETS, DatasetStats

# Delta is stored as a fraction of F1; the paper quotes percentages.
PP = 100.0


@dataclass
class Results:
    """Everything the claims are evaluated against."""

    stats_k1: dict[str, DatasetStats]
    stats_k10: dict[str, DatasetStats]
    stats_k_large: dict[str, DatasetStats]
    pool_sizes: dict[str, int]
    model_type_counts: dict[str, dict[str, int]]
    flops_by_selection: dict[str, dict[str, float]]

    def best_delta(self, stats: dict[str, DatasetStats], dataset: str) -> float:
        """Lowest Delta over all proxies, in percentage points."""
        return min(stats[dataset].delta.values()) * PP

    def best_talent(self, dataset: str) -> float:
        return max(self.stats_k_large[dataset].talent.values())


@dataclass
class Claim:
    id: str
    section: str
    quote: str
    check: Callable[[Results], tuple[bool, str]]


def _c_delta1_within_7(r: Results) -> tuple[bool, str]:
    worst_ds, worst = max(
        ((ds, r.best_delta(r.stats_k1, ds)) for ds in ORDERED_DATASETS), key=lambda t: t[1]
    )
    return worst < 7.0, f"largest best-proxy Δ₁ is {worst:.2f}pp on {worst_ds}"


def _c_delta10_within_2(r: Results) -> tuple[bool, str]:
    worst_ds, worst = max(
        ((ds, r.best_delta(r.stats_k10, ds)) for ds in ORDERED_DATASETS), key=lambda t: t[1]
    )
    return worst < 2.0, f"largest best-proxy Δ₁₀ is {worst:.2f}pp on {worst_ds}"


def _c_delta1_under_5_most(r: Results) -> tuple[bool, str]:
    under = [ds for ds in ORDERED_DATASETS if r.best_delta(r.stats_k1, ds) < 5.0]
    return len(under) >= 5, f"{len(under)}/6 datasets have best Δ₁ < 5%: {', '.join(under)}"


def _c_pamap2_negative_delta1(r: Results) -> tuple[bool, str]:
    expected = {"ensemble", "fisher", "jacob_cov", "initial_val_f1_macro"}
    actual = {p for p, v in r.stats_k1["pamap2"].delta.items() if v < 0}
    return actual == expected, f"proxies with Δ₁ < 0 on PAMAP2: {sorted(actual)}"


def _c_delta10_negative_except_myogym(r: Results) -> tuple[bool, str]:
    others = [ds for ds in ORDERED_DATASETS if ds != "myogym"]
    negative = [ds for ds in others if r.best_delta(r.stats_k10, ds) < 0]
    myogym = r.best_delta(r.stats_k10, "myogym")
    ok = len(negative) == len(others) and 0 <= myogym < 2.0
    return ok, f"{len(negative)}/5 non-Myogym datasets have best Δ₁₀ < 0; Myogym is {myogym:.2f}pp"


def _c_pool_size(r: Results) -> tuple[bool, str]:
    # A handful of CNNs fail to produce finite zero-cost scores and drop out of
    # the pool, so allow a small shortfall against the 2,000 that were sampled.
    bad = {ds: n for ds, n in r.pool_sizes.items() if not 1950 <= n <= 2000}
    return not bad, f"pool sizes: {r.pool_sizes}"


def _c_model_type_mix(r: Results) -> tuple[bool, str]:
    problems = []
    for ds, counts in r.model_type_counts.items():
        if counts.get("TRANSFORMER") != 500 or counts.get("RNN") != 300:
            problems.append(f"{ds}={counts}")
        if not 1150 <= counts.get("CNN", 0) <= 1200:
            problems.append(f"{ds} CNN={counts.get('CNN')}")
    cnn = {ds: c.get("CNN") for ds, c in r.model_type_counts.items()}
    return not problems, f"500 Transformer / 300 RNN everywhere; CNN counts {cnn}"


def _c_talent_about_20(r: Results) -> tuple[bool, str]:
    at_least_20 = [ds for ds in ORDERED_DATASETS if r.best_talent(ds) >= 20.0]
    return (
        len(at_least_20) >= 4,
        f"{len(at_least_20)}/6 datasets reach a talent rate ≥ 20%: {', '.join(at_least_20)}",
    )


def _c_talent_over_15_four_datasets(r: Results) -> tuple[bool, str]:
    over = [ds for ds in ORDERED_DATASETS if r.best_talent(ds) > 15.0]
    return len(over) >= 4, f"{len(over)}/6 datasets have a talent rate > 15%: {', '.join(over)}"


def _c_myogym_highest_38(r: Results) -> tuple[bool, str]:
    ensemble = r.stats_k_large["myogym"].talent["ensemble"]
    return (
        abs(ensemble - 38.0) < 1.0,
        f"Myogym ensemble talent rate is {ensemble:.2f}%",
    )


def _c_gradient_proxies_beat_ensemble(r: Results) -> tuple[bool, str]:
    gradient = ("grasp", "snip", "grad_norm")
    datasets = [ds for ds in ORDERED_DATASETS if r.best_talent(ds) > 15.0]
    beaten = [
        ds
        for ds in datasets
        if max(r.stats_k_large[ds].talent[p] for p in gradient)
        > r.stats_k_large[ds].talent["ensemble"]
    ]
    return (
        len(beaten) >= 3,
        f"grasp/snip/grad_norm beat ensemble on {len(beaten)}/{len(datasets)} "
        f"high-talent datasets: {', '.join(beaten)}",
    )


def _c_synflow_only_for_hhar(r: Results) -> tuple[bool, str]:
    talent = r.stats_k_large["hhar"].talent
    synflow = talent["synflow"]
    runner_up = max(v for p, v in talent.items() if p != "synflow")
    return (
        synflow > 15.0 and synflow > 5 * max(runner_up, 1e-9),
        f"HHAR: synflow {synflow:.2f}%, next best {runner_up:.2f}%",
    )


def _c_talent_beats_random(r: Results) -> tuple[bool, str]:
    datasets = ["myogym", "mobiactv2", "pamap2", "rwhar"]
    ok = [ds for ds in datasets if r.best_talent(ds) >= r.stats_k_large[ds].random_talent]
    return (
        len(ok) == len(datasets),
        "best talent ≥ random search on "
        + ", ".join(
            f"{ds} ({r.best_talent(ds):.1f}% vs {r.stats_k_large[ds].random_talent:.1f}%)"
            for ds in datasets
        ),
    )


def _c_ensemble_delta_large_k_beats_delta1(r: Results) -> tuple[bool, str]:
    better = [
        ds
        for ds in ORDERED_DATASETS
        if r.stats_k_large[ds].delta["ensemble"] < r.stats_k1[ds].delta["ensemble"]
    ]
    return (
        len(better) == 4,
        f"ensemble has Δ₁₀% < Δ₁ on {len(better)}/6 datasets: {', '.join(better)}",
    )


def _c_flops_orders_of_magnitude(r: Results) -> tuple[bool, str]:
    expected = {"top 1": 10, "top 10": 11, "top 10%": 12, "full-scale training": 13}
    problems, seen = [], {}
    for selection, exponent in expected.items():
        # Median across datasets, since the claim is per-dataset order of magnitude.
        values = [r.flops_by_selection[ds][selection] for ds in ORDERED_DATASETS]
        order = int(round(np.log10(np.median(values))))
        seen[selection] = order
        if order != exponent:
            problems.append(f"{selection}: 1e{order} (paper says 1e{exponent})")
    return not problems, "orders of magnitude " + ", ".join(
        f"{k}=1e{v}" for k, v in seen.items()
    )


def _c_200_fold_saving(r: Results) -> tuple[bool, str]:
    ratios = [r.pool_sizes[ds] / 10 for ds in ORDERED_DATASETS]
    return (
        all(abs(ratio - 200) <= 5 for ratio in ratios),
        f"training 10 of ~{int(np.mean([r.pool_sizes[d] for d in ORDERED_DATASETS]))} "
        f"candidates is a {np.mean(ratios):.0f}-fold reduction",
    )


PAPER_CLAIMS: list[Claim] = [
    Claim(
        "sampling_pool_size",
        "§3.1 / §4",
        "we sample 2,000 architectures per dataset",
        _c_pool_size,
    ),
    Claim(
        "sampling_model_mix",
        "§3.1",
        "2,000 architectures per dataset (1,200 CNNs, 500 Transformers, and 300 RNNs)",
        _c_model_type_mix,
    ),
    Claim(
        "delta1_within_7pct",
        "Abstract / Fig. 1 caption",
        "the top-predicted architectures obtain performance within 7% of that "
        "attained by full-scale training of 2,000 randomly sampled architectures",
        _c_delta1_within_7,
    ),
    Claim(
        "delta1_under_5pct_for_most",
        "§5.1",
        "For most datasets, the best-performing ZCPs have Δ₁ < 5%",
        _c_delta1_under_5_most,
    ),
    Claim(
        "pamap2_negative_delta1",
        "§5.1",
        "For PAMAP2, ensemble, fisher, jacob_cov, and initial_val_f1_macro have Δ₁ < 0%",
        _c_pamap2_negative_delta1,
    ),
    Claim(
        "delta10_within_2pct",
        "Abstract / Fig. 1 caption",
        "training the top-10 predicted architectures results in performance "
        "within 2% of full-scale training",
        _c_delta10_within_2,
    ),
    Claim(
        "delta10_negative_except_myogym",
        "§5.1",
        "the best ZCPs have Δ₁₀ < 0% for all datasets except Myogym which has Δ₁₀ < 2%",
        _c_delta10_negative_except_myogym,
    ),
    Claim(
        "talent_rate_about_20pct",
        "Fig. 1 caption",
        "approx. 20% of the top ZCP-predicted models appear within the best "
        "fully-trained models",
        _c_talent_about_20,
    ),
    Claim(
        "talent_rate_over_15_on_four",
        "§5.2",
        "Four out of six datasets have talent rates > 15%",
        _c_talent_over_15_four_datasets,
    ),
    Claim(
        "myogym_talent_38pct",
        "§5.2",
        "with the highest, Myogym, at 38%",
        _c_myogym_highest_38,
    ),
    Claim(
        "gradient_proxies_beat_ensemble",
        "§5.2",
        "grasp, snip, and grad_norm perform better than ensemble on these datasets",
        _c_gradient_proxies_beat_ensemble,
    ),
    Claim(
        "synflow_only_high_talent_for_hhar",
        "§5.2",
        "synflow is the only ZCP with a high talent rate for HHAR",
        _c_synflow_only_for_hhar,
    ),
    Claim(
        "talent_beats_random_search",
        "§5.2",
        "For Myogym, Mobiact, PAMAP2, and RWHAR, the ZCP talent rate is "
        "comparable to or better than random search",
        _c_talent_beats_random,
    ),
    Claim(
        "ensemble_delta_10pct_beats_delta1",
        "§5.2",
        "ensemble has a Δ₁₀% < Δ₁ in four out of six datasets",
        _c_ensemble_delta_large_k_beats_delta1,
    ),
    Claim(
        "flops_orders_of_magnitude",
        "§5.2",
        "Top predicted architecture requires on the order of 10¹⁰ FLOPs for "
        "training, while the top-10 and top-10% require around 10¹¹ and 10¹² "
        "respectively, compared to 10¹³ for training all 2,000 candidates",
        _c_flops_orders_of_magnitude,
    ),
    Claim(
        "two_hundred_fold_saving",
        "§5.1",
        "Training the top-10 ZCP-ranked candidates instead of all 2,000 reduces "
        "the number of fully trained models by 200-fold",
        _c_200_fold_saving,
    ),
]


def evaluate_claims(results: Results) -> list[tuple[Claim, bool, str]]:
    out = []
    for claim in PAPER_CLAIMS:
        try:
            ok, detail = claim.check(results)
        except Exception as exc:  # a missing column should read as a failure, not a crash
            ok, detail = False, f"check raised {type(exc).__name__}: {exc}"
        out.append((claim, ok, detail))
    return out


def proxy_order() -> list[str]:
    return list(ALL_PROXIES)
