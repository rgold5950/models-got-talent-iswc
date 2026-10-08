import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS


def process_dataset(results_dir: Path, output_pkl: Path) -> None:
    RUN_GLOB = "train_tune_*"
    PARAMFILE = "params.json"

    def infer_model_type(cfg_raw: dict[str, Any]) -> Any:
        if "model_type" in cfg_raw:
            return cfg_raw["model_type"]

        mapping = cfg_raw.get("_model_type_mapping")
        arch_seed = cfg_raw.get("arch_seed")
        if not mapping or arch_seed is None:
            return None

        for model_type, seed_range in zip(
            mapping.get("ordered_types", []), mapping.get("seed_ranges", [])
        ):
            start_seed, end_seed = seed_range
            if start_seed <= int(arch_seed) <= end_seed:
                return model_type
        return None

    def flatten(obj: Any, parent: str = "", sep: str = ".") -> Dict[str, Any]:
        """Recursively flatten nested dict / list structures."""
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent}{sep}{k}" if parent else k
                items.update(
                    flatten(v, new_key, sep)
                    if isinstance(v, (dict, list))
                    else {new_key: v}
                )
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                new_key = f"{parent}{sep}{idx}" if parent else str(idx)
                items.update(
                    flatten(v, new_key, sep)
                    if isinstance(v, (dict, list))
                    else {new_key: v}
                )
        else:
            items[parent] = obj
        return items

    rows, all_keys = [], set()
    for run_dir in tqdm(
        sorted(results_dir.glob(RUN_GLOB)), desc=f"{results_dir.name}: params"
    ):
        param_path = run_dir / PARAMFILE
        if not param_path.exists():
            continue
        try:
            cfg_raw = json.load(param_path.open())
        except Exception:
            continue
        flat_cfg = flatten(cfg_raw)
        inferred_model_type = infer_model_type(cfg_raw)
        if inferred_model_type is not None:
            flat_cfg["model_type"] = inferred_model_type
        flat_cfg["run_dir"] = run_dir.name
        flat_cfg["arch_config.n_layers"] = len(cfg_raw.get("arch_config", []))
        rows.append(flat_cfg)
        all_keys.update(flat_cfg.keys())

    if not rows:
        print(f"[WARN] No parameter files found in {results_dir}")
        return
    params_df = pd.DataFrame(rows).reindex(columns=sorted(all_keys))

    metric_rows = []
    for run_dir in tqdm(
        sorted(results_dir.glob(RUN_GLOB)), desc=f"{results_dir.name}: metrics"
    ):
        logs_dir = run_dir / "logs"
        if not logs_dir.exists():
            continue
        try:
            metrics_path = next(logs_dir.glob("*_metrics.csv"))
            mdf = pd.read_csv(metrics_path)
        except (StopIteration, Exception):
            continue

        class_cols = [c for c in mdf.columns if re.fullmatch(r"class_\d+", c)]
        row = {"run_dir": run_dir.name}

        def capture(prefix: str, metric_row: pd.Series):
            """Copy every metric (incl. per-class) with the chosen prefix."""
            for col in class_cols:
                row[f"{prefix}_{col}"] = metric_row[col]
            for col in metric_row.index.difference(["phase", "epoch"] + class_cols):
                row[f"{prefix}_{col}"] = metric_row[col]
            row[f"{prefix}_epoch"] = metric_row["epoch"]

        # initial_train  (first = epoch 0)
        init_train = mdf[mdf["phase"] == "initial_train"]
        if not init_train.empty:
            capture("initial_train", init_train.iloc[0])

        # initial_val
        init_val = mdf[mdf["phase"] == "initial_val"]
        if not init_val.empty:
            capture("initial_val", init_val.iloc[0])

        # best_val        (row with max best_f1_macro)
        val_rows = mdf[mdf["phase"] == "val"]
        if not val_rows.empty and "best_f1_macro" in val_rows.columns:
            best_val_row = val_rows.loc[val_rows["best_f1_macro"].idxmax()]
            capture("best_val", best_val_row)

        # best_test       (row with max f1_macro on test)
        test_rows = mdf[mdf["phase"] == "test"]
        if not test_rows.empty and "f1_macro" in test_rows.columns:
            best_test_row = test_rows.loc[test_rows["f1_macro"].idxmax()]
            capture("best_test", best_test_row)

        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df["run_config_id"] = metrics_df["run_dir"].str.extract(
        r"(arch_seed=\d+,combo_idx=\d+)"
    )
    params_df["run_config_id"] = params_df["run_dir"].str.extract(
        r"(arch_seed=\d+,combo_idx=\d+)"
    )

    DROP_COLS = {
        "run_dir",
        "save_dir",
        "dataset",
        "combo_idx",
        "arch_seed",
        "seed",
        "run_config_id",
    }
    wanted_cols = [c for c in params_df.columns if c not in DROP_COLS]

    # create a unique arch_id hash for convenience
    params_df["arch_id"] = params_df[wanted_cols].apply(
        lambda r: "_".join(f"{c}={r[c]}" for c in wanted_cols), axis=1
    )

    params_df = params_df.drop(columns=["save_dir"], errors="ignore")  # Keep run_dir for merge

    # Merge on run_dir (unique per run) instead of run_config_id (can have duplicates)
    full_df = params_df.merge(metrics_df, on="run_dir", how="left")

    print(f"✅ {results_dir.name}: processed {len(full_df)} runs")
    full_df.to_pickle(output_pkl)
    print(f"💾 saved → {output_pkl}")


if __name__ == "__main__":
    for ds, cfg in DATASET_CONFIGS.items():
        # uncomment to process a subset:  if ds != "hhar": continue
        print(f"\n=== Processing {ds} ===")
        process_dataset(cfg["results_dir"], cfg["full_results_pkl"])
