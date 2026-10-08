"""Verify that the committed artifacts still reproduce the paper.

These tests need nothing but pandas, numpy and matplotlib — no datasets, no GPU,
no Ray — and run in a few seconds:

    pytest tests/test_paper_reproduction.py -v

Every claim the paper makes quantitatively is checked as its own test case, so a
failure names the sentence in the paper that no longer holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reproduce.claims import PAPER_CLAIMS, Results, evaluate_claims
from reproduce.data import ARTIFACT_COLUMNS, load_manifest, load_pools
from reproduce.metrics import (
    ALL_PROXIES,
    K_PERCENTAGE,
    ORDERED_DATASETS,
    ZC_METRICS,
    compute_all_stats,
    compute_stats,
)
from reproduce.run import _flops_by_selection


@pytest.fixture(scope="module")
def pools():
    return load_pools()


@pytest.fixture(scope="module")
def results(pools) -> Results:
    return Results(
        stats_k1=compute_all_stats(pools, k=1),
        stats_k10=compute_all_stats(pools, k=10),
        stats_k_large=compute_all_stats(pools, k_percentage=K_PERCENTAGE),
        pool_sizes={ds: len(df) for ds, df in pools.items()},
        model_type_counts={
            ds: {str(k): int(v) for k, v in df["model_type"].value_counts().items()}
            for ds, df in pools.items()
        },
        flops_by_selection=_flops_by_selection(pools),
    )


@pytest.fixture(scope="module")
def claim_results(results) -> dict[str, tuple[bool, str]]:
    return {claim.id: (ok, detail) for claim, ok, detail in evaluate_claims(results)}


# --- the artifacts themselves ------------------------------------------------


def test_all_datasets_present(pools):
    assert sorted(pools) == sorted(ORDERED_DATASETS)


@pytest.mark.parametrize("dataset", ORDERED_DATASETS)
def test_artifact_schema(pools, dataset):
    df = pools[dataset]
    missing = set(ARTIFACT_COLUMNS) - set(df.columns)
    assert not missing, f"{dataset} is missing columns {sorted(missing)}"
    assert df["model_id"].is_unique, f"{dataset} has duplicate architecture ids"
    assert not df[["best_val_f1_macro", "best_test_f1_macro", *ZC_METRICS]].isna().any().any()


@pytest.mark.parametrize("dataset", ORDERED_DATASETS)
def test_f1_scores_in_range(pools, dataset):
    for column in ("initial_val_f1_macro", "best_val_f1_macro", "best_test_f1_macro"):
        values = pools[dataset][column]
        assert values.between(0, 1).all(), f"{dataset}.{column} outside [0, 1]"


@pytest.mark.parametrize("dataset", ORDERED_DATASETS)
def test_manifest_matches_artifact(pools, dataset):
    entry = load_manifest()["datasets"][dataset]
    assert entry["architectures"] == len(pools[dataset])


def test_untrained_models_are_near_chance(pools):
    """initial_val_f1_macro is measured before training, so it must be low."""
    for dataset, df in pools.items():
        assert df["initial_val_f1_macro"].mean() < 0.2, (
            f"{dataset}: mean F1 at initialisation is "
            f"{df['initial_val_f1_macro'].mean():.3f}, which is too high to be untrained"
        )


# --- metric definitions ------------------------------------------------------


def test_delta_is_zero_for_a_perfect_proxy(pools):
    """A proxy that ranks exactly like validation F1 must have Delta = 0."""
    df = pools["pamap2"].copy()
    df["ensemble"] = df["best_val_f1_macro"]
    for k in (1, 10, 100):
        stats = compute_stats(df, "pamap2", k)
        assert stats.delta["ensemble"] == pytest.approx(0.0)
        assert stats.talent["ensemble"] == pytest.approx(100.0)


def test_delta_at_full_pool_is_zero(pools):
    """With k = pool size every proxy selects from the whole pool."""
    df = pools["rwhar"]
    stats = compute_stats(df, "rwhar", len(df))
    for proxy in ALL_PROXIES:
        assert stats.delta[proxy] == pytest.approx(0.0)
        assert stats.talent[proxy] == pytest.approx(100.0)


def test_ensemble_is_mean_of_percentile_ranks(pools):
    df = pools["hhar"]
    expected = df[ZC_METRICS].rank(pct=True).mean(axis=1)
    pd.testing.assert_series_equal(df["ensemble"], expected, check_names=False)


def test_random_baseline_is_deterministic_given_a_seed(pools):
    a = compute_stats(pools["myogym"], "myogym", 10, seed=7)
    b = compute_stats(pools["myogym"], "myogym", 10, seed=7)
    assert a.random_delta == b.random_delta
    assert a.random_talent == b.random_talent


def test_random_talent_rate_is_k_over_pool(results):
    """Random search should recover k/n of the top-k, i.e. ~10% at k = 10%."""
    for dataset in ORDERED_DATASETS:
        stats = results.stats_k_large[dataset]
        expected = stats.k / stats.pool_size * 100
        assert stats.random_talent == pytest.approx(expected, abs=0.5)


def test_k_out_of_range_is_rejected(pools):
    with pytest.raises(ValueError):
        compute_stats(pools["hhar"], "hhar", 0)
    with pytest.raises(ValueError):
        compute_stats(pools["hhar"], "hhar", len(pools["hhar"]) + 1)


@pytest.mark.parametrize("dataset", ORDERED_DATASETS)
def test_selected_validation_score_improves_with_k(pools, dataset):
    """The top-k proxy sets are nested, so the best validation F1 inside them
    can only improve as k grows. Delta itself is *not* monotone in k: it is
    measured on test F1 while selection happens on validation F1, so a larger k
    can surface a model that validates better but tests worse."""
    df = pools[dataset]
    best_val = [
        df.sort_values("ensemble", ascending=False).head(k)["best_val_f1_macro"].max()
        for k in (1, 10, 100, 200)
    ]
    assert best_val == sorted(best_val)


def test_best_proxy_delta_improves_from_k1_to_k10(results):
    """The paper's headline result: going from top-1 to top-10 closes the gap."""
    mean_best_delta = [
        np.mean([min(stats[ds].delta.values()) for ds in ORDERED_DATASETS])
        for stats in (results.stats_k1, results.stats_k10)
    ]
    assert mean_best_delta[0] > mean_best_delta[1]


# --- the paper's claims ------------------------------------------------------


@pytest.mark.parametrize("claim", PAPER_CLAIMS, ids=lambda c: c.id)
def test_paper_claim(claim, claim_results):
    ok, detail = claim_results[claim.id]
    assert ok, f"{claim.section}: “{claim.quote}” — but found: {detail}"


# --- the figure --------------------------------------------------------------


def test_figure1_renders(results, tmp_path):
    from reproduce.figure1 import plot_figure1

    written = plot_figure1(
        results.stats_k1,
        results.stats_k10,
        results.stats_k_large,
        tmp_path / "figure1.pdf",
    )
    assert {p.suffix for p in written} == {".pdf", ".png"}
    for path in written:
        assert path.stat().st_size > 10_000, f"{path} looks empty"
