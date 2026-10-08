"""Loading the per-architecture scores the paper's results are computed from.

Two sources are supported:

`load_pools()` (default)
    Reads the committed CSVs in `paper_artifacts/`. This is what a reader of the
    paper should use: it needs nothing but pandas and reproduces every number in
    the paper exactly.

`load_pools_from_pickles()`
    Rebuilds the same table from the raw post-processing output
    (`full_results.pkl` + `zero_cost_scores.pkl` per dataset) resolved through
    `DATASET_CONFIGS`. Use this after re-running training and post-processing;
    `reproduce/export_artifacts.py` then refreshes `paper_artifacts/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reproduce.metrics import ORDERED_DATASETS, ZC_METRICS, add_ensemble

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "paper_artifacts"

# Columns stored in the committed CSVs.
ARTIFACT_COLUMNS = [
    "model_id",
    "model_type",
    "initial_val_f1_macro",
    "best_val_f1_macro",
    "best_test_f1_macro",
    *ZC_METRICS,
    "synflow_bn",
    "train_flops",
]

# run_dir naming schemes emitted by the different generations of the Ray Tune
# pipeline. The first group that matches is used as the architecture identifier.
ID_REGEX = (
    r"(arch_seed=\d+,combo_idx=\d+"
    r"|arch_seed=\d+,init_name=\w+,init_seed=\d+"
    r"|train_tune_\w+_\d+_\d+)"
)


def artifact_path(dataset: str) -> Path:
    return ARTIFACT_DIR / f"{dataset}.csv"


def load_pools(datasets: list[str] | None = None, *, artifact_dir: Path | None = None):
    """Load the committed per-architecture scores, ensemble column included."""
    datasets = datasets or ORDERED_DATASETS
    base = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    pools = {}
    for dataset in datasets:
        path = base / f"{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing artifact {path}.\n"
                "Run `python -m reproduce.export_artifacts` on a machine that has "
                "the post-processed results, or fetch paper_artifacts/ from the "
                "release described in reproduce/README.md."
            )
        pools[dataset] = add_ensemble(pd.read_csv(path))
    return pools


def load_manifest(artifact_dir: Path | None = None) -> dict:
    base = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    return json.loads((base / "MANIFEST.json").read_text())


def load_raw_from_pickles(datasets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Rebuild the score table from the post-processing pickles, scores unmodified.

    This is what gets written to `paper_artifacts/`: raw proxy scores exactly as
    the pipeline measured them, with no sign flipping and no ensemble column, so
    that the conventions live in one place (`add_ensemble`) and are applied once.
    """
    # Imported lazily so the artifact path stays free of project imports.
    from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

    datasets = datasets or ORDERED_DATASETS
    return {dataset: _merge_one(DATASET_CONFIGS[dataset]) for dataset in datasets}


def load_pools_from_pickles(datasets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Rebuild the analysis-ready score table from the post-processing pickles."""
    return {
        dataset: add_ensemble(df)
        for dataset, df in load_raw_from_pickles(datasets).items()
    }


def _merge_one(cfg: dict) -> pd.DataFrame:
    """Join training outcomes and zero-cost scores for one dataset."""
    full_df = pd.read_pickle(Path(cfg["full_results_pkl"]))
    zc_df = pd.read_pickle(Path(cfg["zero_cost_pkl"]))

    full_df["model_id"] = full_df["run_dir"].str.extract(ID_REGEX)
    zc_df["model_id"] = zc_df["run_dir"].str.extract(ID_REGEX)

    training_cols = [
        "model_id",
        "model_type",
        "initial_val_f1_macro",
        "best_val_f1_macro",
        "best_test_f1_macro",
    ]
    score_cols = ["model_id", *ZC_METRICS, "synflow_bn"]

    merged = full_df[training_cols].merge(
        zc_df[score_cols], on="model_id", how="inner", validate="one_to_one"
    )

    # Training FLOPs are measured in a separate pass and live in their own pickle.
    flops_df = pd.read_pickle(Path(cfg["model_flops_pkl"]))
    flops_df["model_id"] = flops_df["run_dir"].str.extract(ID_REGEX)
    merged = merged.merge(
        flops_df[["model_id", "flops"]].rename(columns={"flops": "train_flops"}),
        on="model_id",
        how="left",
        validate="one_to_one",
    )

    # An architecture only enters the pool if it both trained to completion and
    # produced finite zero-cost scores. FLOPs are reported separately and are not
    # part of that requirement.
    required = [c for c in merged.columns if c != "train_flops"]
    return merged.dropna(subset=required).reset_index(drop=True)
