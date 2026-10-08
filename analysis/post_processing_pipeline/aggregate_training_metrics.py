from pathlib import Path
import pandas as pd
import traceback
from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS

METRICS_GLOB = "**/*_metrics.csv"


def _read_one(metrics_path: Path, dataset: str) -> pd.DataFrame:
    """Return a DataFrame with an added run and dataset identifier."""
    df = pd.read_csv(metrics_path)
    df["run_dir"] = metrics_path.parent.parent.name
    df["exp_dir"] = df["run_dir"].str.extract(r"^(train_tune_[^_]+)")
    df["dataset"] = dataset
    return df


def build_dataset_frame(dataset: str, cfg: dict) -> pd.DataFrame:
    """Concatenate every *_metrics.csv file under cfg['results_dir']."""
    csv_files = list(Path(cfg["results_dir"]).glob(METRICS_GLOB))
    if not csv_files:
        # Return empty DataFrame instead of raising error - allows pipeline to continue
        print(f"⚠️  {dataset}: No metrics files found - creating empty DataFrame")
        return pd.DataFrame()
    parts = []
    for f in csv_files:
        try:
            parts.append(_read_one(f, dataset))
        except Exception as exc:
            print(f"{dataset}: {f} -> {exc}")
            traceback.print_exc()
    if not parts:
        # All files failed to read - return empty DataFrame
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_and_save_all(dataset_configs: dict[str, dict]) -> None:
    """Build full training-metrics DataFrame for each dataset and pickle it."""
    for name, cfg in dataset_configs.items():
        try:
            df = build_dataset_frame(name, cfg)
            out_path = cfg["full_training_metrics_pkl"]
            # Convert to Path if it's a string
            if isinstance(out_path, str):
                out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(out_path)
            if len(df) > 0:
                print(f"{name}: saved {len(df)} rows to {out_path}")
            else:
                print(f"{name}: no metrics found - saved empty DataFrame to {out_path}")
        except Exception as exc:
            print(f"{name}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    build_and_save_all(DATASET_CONFIGS)
