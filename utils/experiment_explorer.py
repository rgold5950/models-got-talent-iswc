"""
Experiment explorer functionality for viewing and analyzing experiment collections.
"""
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from utils.pipeline_status import get_pipeline_status


def get_all_collections(base_dir: Path) -> List[Path]:
    """Find all experiment collections in the given base directory."""
    collections = []

    # Check production collections
    prod_experiments_dir = base_dir / "experiments"
    if prod_experiments_dir.exists():
        for item in prod_experiments_dir.iterdir():
            if item.is_dir() and (item / "MANIFEST.yaml").exists():
                collections.append(item)

    # Check test collections
    test_experiments_dir = base_dir / "experiments" / "test"
    if test_experiments_dir.exists():
        for item in test_experiments_dir.iterdir():
            if item.is_dir() and (item / "MANIFEST.yaml").exists():
                collections.append(item)

    return sorted(collections, key=lambda p: p.stat().st_mtime, reverse=True)


def get_collection_stats(collection_dir: Path) -> Dict[str, Any]:
    """Compute statistics for a collection."""
    # Read manifest
    manifest_path = collection_dir / "MANIFEST.yaml"
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f) or {}
    except Exception:
        manifest = {}

    # Get pipeline status
    pipeline_status = get_pipeline_status(collection_dir)

    # Find all dataset directories
    dataset_dirs = []
    for item in collection_dir.iterdir():
        if item.is_dir() and item.name not in ["__pycache__"]:
            # Check if it looks like a dataset directory (has trial subdirs or experiment.parquet)
            trial_dirs = list(item.glob("train_tune_*"))
            if trial_dirs or (item / "experiment.parquet").exists():
                dataset_dirs.append(item)

    # Compute stats
    stats = {
        "collection_id": collection_dir.name,
        "collection_path": collection_dir,
        "exp_tag": manifest.get("exp_tag"),
        "exp_note": manifest.get("exp_note"),
        "timestamp": manifest.get("timestamp"),
        "test_mode": manifest.get("test_mode", False),
        "git_commit": manifest.get("git_commit"),
        "num_datasets": len(dataset_dirs),
        "datasets": [d.name for d in dataset_dirs],
        "total_models": 0,
        "completed_trials": 0,
        "failed_trials": 0,
        "date_range": None,
        "aggregation_status": pipeline_status.get("aggregation", {}).get("status", "not_started"),
        "zcp_status": pipeline_status.get("zero_cost_proxies", {}).get("status", "not_started"),
        "analysis_status": pipeline_status.get("analysis", {}).get("status", "not_started"),
    }

    # Compute model counts and date range
    earliest_date = None
    latest_date = None

    for dataset_dir in dataset_dirs:
        # Count trial directories
        trial_dirs = list(dataset_dir.glob("train_tune_*"))
        stats["total_models"] += len(trial_dirs)

        # Check for parquet file (indicates aggregation completed)
        parquet_path = dataset_dir / "experiment.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                stats["completed_trials"] += len(df)
                # Try to get date range from parquet if available
                if "timestamp" in df.columns and not df.empty:
                    timestamps = pd.to_datetime(df["timestamp"], errors='coerce')
                    valid_timestamps = timestamps.dropna()
                    if not valid_timestamps.empty:
                        if earliest_date is None or valid_timestamps.min() < earliest_date:
                            earliest_date = valid_timestamps.min()
                        if latest_date is None or valid_timestamps.max() > latest_date:
                            latest_date = valid_timestamps.max()
            except Exception:
                # If we can't read parquet, count trial dirs as completed
                stats["completed_trials"] += len(trial_dirs)

    # Set date range if we found timestamps
    if earliest_date and latest_date:
        if earliest_date.date() == latest_date.date():
            stats["date_range"] = earliest_date.strftime("%Y-%m-%d")
        else:
            stats["date_range"] = f"{earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}"
    elif manifest.get("timestamp"):
        # Fallback to manifest timestamp
        try:
            dt = datetime.fromisoformat(manifest["timestamp"])
            stats["date_range"] = dt.strftime("%Y-%m-%d")
        except:
            stats["date_range"] = manifest["timestamp"][:10] if manifest["timestamp"] else None

    return stats


def get_experiment_details(collection_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific experiment in a collection."""
    dataset_dir = collection_dir / dataset_name
    if not dataset_dir.exists():
        return {}

    # Get pipeline status for this dataset
    pipeline_status = get_pipeline_status(collection_dir)

    # Find trial directories
    trial_dirs = list(dataset_dir.glob("train_tune_*"))

    # Check for aggregated results
    parquet_path = dataset_dir / "experiment.parquet"
    aggregated_data = None
    if parquet_path.exists():
        try:
            aggregated_data = pd.read_parquet(parquet_path)
        except Exception:
            pass

    # Compute stats
    details = {
        "dataset": dataset_name,
        "dataset_path": dataset_dir,
        "num_trials": len(trial_dirs),
        "model_types": set(),
        "date_range": None,
        "best_val_f1": None,
        "best_test_f1": None,
        "aggregation_status": pipeline_status.get("aggregation", {}).get("status", "not_started"),
        "zcp_status": pipeline_status.get("zero_cost_proxies", {}).get("status", "not_started"),
    }

    # Extract model types and metrics from aggregated data if available
    if aggregated_data is not None and not aggregated_data.empty:
        if "model_type" in aggregated_data.columns:
            details["model_types"] = sorted(aggregated_data["model_type"].unique())

        # Get best metrics
        if "best_val_f1_macro" in aggregated_data.columns:
            best_val = aggregated_data["best_val_f1_macro"].max()
            if not pd.isna(best_val):
                details["best_val_f1"] = f"{best_val:.4f}"

        if "best_test_f1_macro" in aggregated_data.columns:
            best_test = aggregated_data["best_test_f1_macro"].max()
            if not pd.isna(best_test):
                details["best_test_f1"] = f"{best_test:.4f}"

        # Get date range from timestamps if available
        if "timestamp" in aggregated_data.columns:
            timestamps = pd.to_datetime(aggregated_data["timestamp"], errors='coerce')
            valid_timestamps = timestamps.dropna()
            if not valid_timestamps.empty:
                earliest = valid_timestamps.min()
                latest = valid_timestamps.max()
                if earliest.date() == latest.date():
                    details["date_range"] = earliest.strftime("%Y-%m-%d")
                else:
                    details["date_range"] = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"

    return details


def format_collections_table(collections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format collections data as a pandas DataFrame for display."""
    if not collections:
        return pd.DataFrame()

    # Define columns in desired order
    columns = [
        "Collection ID", "Tag", "Notes", "Date", "Test Mode", "# Datasets",
        "Datasets", "Total Models", "Git Commit", "Aggregation", "ZCP", "Analysis"
    ]

    data = []
    for collection in collections:
        row = {
            "Collection ID": collection["collection_id"],
            "Tag": collection.get("exp_tag") or "",
            "Notes": collection.get("exp_note") or "",
            "Date": collection.get("date_range") or (collection.get("timestamp") or "")[:10],
            "Test Mode": "✓" if collection.get("test_mode") else "",
            "# Datasets": collection["num_datasets"],
            "Datasets": ", ".join(collection["datasets"]) if collection["datasets"] else "",
            "Total Models": collection["total_models"],
            "Git Commit": (collection.get("git_commit") or "")[:7],  # Short commit hash
            "Aggregation": collection["aggregation_status"],
            "ZCP": collection["zcp_status"],
            "Analysis": collection["analysis_status"],
        }
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df


def format_experiments_table(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format experiments data as a pandas DataFrame for display."""
    if not experiments:
        return pd.DataFrame()

    # Define columns in desired order
    columns = [
        "Dataset", "# Trials", "Model Types", "Date Range",
        "Best Val F1", "Best Test F1", "Aggregation", "ZCP"
    ]

    data = []
    for exp in experiments:
        row = {
            "Dataset": exp["dataset"],
            "# Trials": exp["num_trials"],
            "Model Types": ", ".join(exp["model_types"]) if exp["model_types"] else "",
            "Date Range": exp.get("date_range") or "",
            "Best Val F1": exp.get("best_val_f1") or "",
            "Best Test F1": exp.get("best_test_f1") or "",
            "Aggregation": exp["aggregation_status"],
            "ZCP": exp["zcp_status"],
        }
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df