"""Regenerate every number and the figure in the ISWC'26 paper.

    python -m reproduce.run

Reads `paper_artifacts/*.csv`, recomputes Delta_1, Delta_10, Delta_10% and the
talent rate for all eight proxies plus the rank-averaged ensemble and the
random-search baseline, prints them as tables, checks them against the claims
made in the paper, and writes Figure 1.

Options:
    --from-pickles   recompute from the raw post-processing output instead of the
                     committed artifacts (authors only; see reproduce/README.md)
    --out-dir DIR    where to write the figure and results.json
    --no-figure      skip rendering (useful on headless machines without a font
                     cache)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce.claims import Results, evaluate_claims
from reproduce.data import load_pools, load_pools_from_pickles
from reproduce.metrics import (
    ALL_PROXIES,
    DATASET_LABELS,
    K_PERCENTAGE,
    N_RANDOM_DRAWS,
    ORDERED_DATASETS,
    compute_all_stats,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "figures" / "paper"

FLOPS_SELECTIONS = {
    "top 1": lambda n: 1,
    "top 10": lambda n: 10,
    "top 10%": lambda n: int(np.ceil(n * K_PERCENTAGE)),
    "full-scale training": lambda n: n,
}


def _table(stats, quantity: str, *, scale: float = 1.0) -> pd.DataFrame:
    """Proxies as rows, datasets as columns, random search in the last row."""
    rows = {}
    for proxy in ALL_PROXIES:
        rows[proxy] = {
            DATASET_LABELS[ds]: getattr(stats[ds], quantity)[proxy] * scale
            for ds in ORDERED_DATASETS
        }
    rows["random search"] = {
        DATASET_LABELS[ds]: getattr(stats[ds], f"random_{quantity}") * scale
        for ds in ORDERED_DATASETS
    }
    return pd.DataFrame(rows).T


def _flops_by_selection(pools) -> dict[str, dict[str, float]]:
    """Total training FLOPs for the top-k architectures ranked by the ensemble."""
    out = {}
    for dataset, df in pools.items():
        ranked = df.sort_values("ensemble", ascending=False)
        out[dataset] = {
            label: float(ranked.head(min(k_fn(len(ranked)), len(ranked)))["train_flops"].sum())
            for label, k_fn in FLOPS_SELECTIONS.items()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-pickles", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=ORDERED_DATASETS)
    parser.add_argument("--seed", type=int, default=0, help="seed for the random-search baseline")
    args = parser.parse_args(argv)

    pools = (
        load_pools_from_pickles(args.datasets)
        if args.from_pickles
        else load_pools(args.datasets)
    )

    print("Architecture pools")
    print("-" * 78)
    for ds in args.datasets:
        counts = pools[ds]["model_type"].value_counts().to_dict()
        mix = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        print(f"  {DATASET_LABELS[ds]:<13} {len(pools[ds]):>5} architectures   ({mix})")

    stats_k1 = compute_all_stats(pools, k=1, seed=args.seed)
    stats_k10 = compute_all_stats(pools, k=10, seed=args.seed)
    stats_k_large = compute_all_stats(pools, k_percentage=K_PERCENTAGE, seed=args.seed)

    k_large = {ds: stats_k_large[ds].k for ds in args.datasets}
    print(f"\nk for the 10% panels: {k_large}")
    print(f"Random-search baseline: {N_RANDOM_DRAWS} draws, seed {args.seed}")

    for title, stats, quantity, scale in [
        ("Δ₁  — performance gap, top-1 predicted architecture (pp of F1)", stats_k1, "delta", 100),
        ("Δ₁₀ — performance gap, top-10 predicted architectures (pp of F1)", stats_k10, "delta", 100),
        ("Δ₁₀% — performance gap, top-10% predicted architectures (pp of F1)", stats_k_large, "delta", 100),
        ("Talent rate at k = 10% of the pool (%)", stats_k_large, "talent", 1),
    ]:
        print()
        print(title)
        print("-" * 78)
        print(_table(stats, quantity, scale=scale).round(2).to_string())

    print("\nBest proxy per dataset")
    print("-" * 78)
    header = f"  {'dataset':<13} {'Δ₁':>26} {'Δ₁₀':>26} {'talent @ 10%':>26}"
    print(header)
    for ds in args.datasets:
        def best(stats, quantity, pick):
            values = getattr(stats[ds], quantity)
            proxy = pick(values, key=values.get)
            value = values[proxy] * (100 if quantity == "delta" else 1)
            return f"{proxy} ({value:+.2f})"

        print(
            f"  {DATASET_LABELS[ds]:<13} {best(stats_k1, 'delta', min):>26} "
            f"{best(stats_k10, 'delta', min):>26} {best(stats_k_large, 'talent', max):>26}"
        )

    flops = _flops_by_selection(pools)
    print("\nTotal training FLOPs by ZCP-ranked selection (ensemble ranking)")
    print("-" * 78)
    flops_df = pd.DataFrame(
        {DATASET_LABELS[ds]: flops[ds] for ds in args.datasets}
    ).T[list(FLOPS_SELECTIONS)]
    print(flops_df.map(lambda v: f"{v:.2e}").to_string())

    results = Results(
        stats_k1=stats_k1,
        stats_k10=stats_k10,
        stats_k_large=stats_k_large,
        pool_sizes={ds: len(pools[ds]) for ds in args.datasets},
        model_type_counts={
            ds: {str(k): int(v) for k, v in pools[ds]["model_type"].value_counts().items()}
            for ds in args.datasets
        },
        flops_by_selection=flops,
    )

    print("\nClaims made in the paper")
    print("-" * 78)
    evaluated = evaluate_claims(results)
    for claim, ok, detail in evaluated:
        print(f"  [{'PASS' if ok else 'FAIL'}] {claim.id}  ({claim.section})")
        print(f"         paper: {claim.quote}")
        print(f"         found: {detail}")
    n_passed = sum(1 for _, ok, _ in evaluated if ok)
    print("-" * 78)
    print(f"  {n_passed}/{len(evaluated)} claims reproduced")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "post-processing pickles" if args.from_pickles else "paper_artifacts",
        "seed": args.seed,
        "pool_sizes": results.pool_sizes,
        "k_large": k_large,
        "delta_1": {ds: stats_k1[ds].delta for ds in args.datasets},
        "delta_10": {ds: stats_k10[ds].delta for ds in args.datasets},
        "delta_10pct": {ds: stats_k_large[ds].delta for ds in args.datasets},
        "talent_rate_10pct": {ds: stats_k_large[ds].talent for ds in args.datasets},
        "random_search": {
            ds: {
                "delta_1": stats_k1[ds].random_delta,
                "delta_10": stats_k10[ds].random_delta,
                "delta_10pct": stats_k_large[ds].random_delta,
                "talent_rate_10pct": stats_k_large[ds].random_talent,
            }
            for ds in args.datasets
        },
        "train_flops": flops,
        "claims": [
            {"id": c.id, "section": c.section, "quote": c.quote, "passed": ok, "found": detail}
            for c, ok, detail in evaluated
        ],
    }
    summary_path = args.out_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {summary_path}")

    if not args.no_figure:
        from reproduce.figure1 import plot_figure1

        for path in plot_figure1(
            stats_k1, stats_k10, stats_k_large, args.out_dir / "figure1.pdf", datasets=args.datasets
        ):
            print(f"wrote {path}")

    return 0 if n_passed == len(evaluated) else 1


if __name__ == "__main__":
    raise SystemExit(main())
