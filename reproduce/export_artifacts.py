"""Distil the post-processing output into the committed paper artifacts.

The full post-processing output is ~220 MB of pickles per run, most of which is
per-epoch training curves. Every number and the figure in the paper only need
one row per architecture, so this script writes a small CSV per dataset plus a
manifest recording where the numbers came from.

Run this on a machine that has the post-processed results:

    python -m reproduce.export_artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from reproduce.data import ARTIFACT_COLUMNS, ARTIFACT_DIR, load_raw_from_pickles
from reproduce.metrics import ORDERED_DATASETS

REPO_ROOT = ARTIFACT_DIR.parent


def _source_path(path) -> str:
    """Record sources relative to the repo when they live inside it, so the
    manifest does not pin the numbers to one machine's directory layout."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ARTIFACT_DIR, help="where to write the CSVs"
    )
    parser.add_argument("--datasets", nargs="+", default=ORDERED_DATASETS)
    args = parser.parse_args()

    from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pools = load_raw_from_pickles(args.datasets)

    manifest = {
        "description": (
            "One row per sampled architecture: its raw zero-cost proxy scores at "
            "initialisation and its F1 after full-scale training. Source of every "
            "number in the ISWC'26 paper. Scores are as measured: the sign "
            "conventions and the rank-averaged ensemble are applied at load time "
            "by reproduce.metrics.add_ensemble."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "columns": ARTIFACT_COLUMNS,
        "datasets": {},
    }

    for dataset, df in pools.items():
        out = args.out_dir / f"{dataset}.csv"
        df[ARTIFACT_COLUMNS].to_csv(out, index=False)
        manifest["datasets"][dataset] = {
            "file": out.name,
            "architectures": int(len(df)),
            "model_type_counts": {
                str(k): int(v) for k, v in df["model_type"].value_counts().items()
            },
            "sha256": _sha256(out),
            "source_experiment": Path(DATASET_CONFIGS[dataset]["results_dir"]).parent.name,
            "source_full_results_pkl": _source_path(DATASET_CONFIGS[dataset]["full_results_pkl"]),
            "source_zero_cost_pkl": _source_path(DATASET_CONFIGS[dataset]["zero_cost_pkl"]),
            "source_model_flops_pkl": _source_path(DATASET_CONFIGS[dataset]["model_flops_pkl"]),
        }
        print(f"wrote {out}  ({len(df)} architectures, {out.stat().st_size / 1e6:.1f} MB)")

    manifest_path = args.out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
