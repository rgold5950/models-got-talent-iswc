"""
Aggregate experiment results from Ray Tune trial directories into a single parquet file.

Each trial directory should contain:
- summary.json (required): Contains trial metrics and metadata
- params.json (optional): Contains architecture configuration
- config_resolved.yaml (optional): Full resolved config

Output: experiment.parquet with one row per trial
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_trial_data(trial_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Read data from a trial directory.
    
    Returns:
        Dictionary with trial data, or None if summary.json is missing
    """
    summary_file = trial_dir / "summary.json"
    if not summary_file.exists():
        # Missing summary.json - will be summarized at end
        return None
    
    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Skipping {trial_dir.name}: Invalid JSON in summary.json: {e}")
        return None
    except Exception as e:
        logger.error(f"Skipping {trial_dir.name}: Error reading summary.json: {e}")
        return None
    
    # Start with summary data
    row = {
        "trial_dir": str(trial_dir),
        **summary
    }
    
    # Read params.json if present (contains arch_config)
    params_file = trial_dir / "params.json"
    if params_file.exists():
        try:
            with open(params_file, "r") as f:
                params = json.load(f)
            
            # Flatten arch_config if present
            if "arch_config" in params:
                arch_config_flat = flatten_dict(params["arch_config"], "arch_config")
                row.update(arch_config_flat)
            
            # Add other params fields (excluding arch_config which we already flattened)
            for k, v in params.items():
                if k != "arch_config":
                    row[f"params_{k}"] = v
        except Exception as e:
            logger.warning(f"{trial_dir.name}: Error reading params.json: {e}")
    
    # Optionally read config_resolved.yaml for full config
    config_file = trial_dir / "config_resolved.yaml"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            # Flatten and add with prefix
            config_flat = flatten_dict(config, "config")
            row.update(config_flat)
        except Exception as e:
            logger.warning(f"{trial_dir.name}: Error reading config_resolved.yaml: {e}")
    
    return row


def aggregate_experiment_index(exp_dir: Path, output_path: Optional[Path] = None) -> Path:
    """
    Aggregate all trial results into a single parquet file.
    
    Args:
        exp_dir: Directory containing Ray trial subdirectories
        output_path: Output parquet path (default: exp_dir/experiment.parquet)
    
    Returns:
        Path to created parquet file
    """
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {exp_dir}")
    
    if output_path is None:
        output_path = exp_dir / "experiment.parquet"
    
    logger.info(f"Aggregating trials from: {exp_dir}")
    logger.info(f"Output will be written to: {output_path}")
    
    # Find all trial subdirectories (filter out known non-trial directories)
    all_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    # Filter to only Ray Tune trial directories (start with train_tune_)
    # and exclude known non-trial directories
    excluded_dirs = {"post-processed-results"}
    trial_dirs = [
        d for d in all_dirs 
        if d.name.startswith("train_tune_") and d.name not in excluded_dirs
    ]
    logger.info(f"Found {len(trial_dirs)} trial directories (filtered from {len(all_dirs)} total directories)")
    
    rows = []
    successful = 0
    failed = 0
    
    for trial_dir in sorted(trial_dirs):
        row = read_trial_data(trial_dir)
        if row is not None:
            rows.append(row)
            successful += 1
        else:
            failed += 1
    
    if not rows:
        warning_msg = (
            f"No valid trial data found in {exp_dir}. "
            f"Found {len(trial_dirs)} trial directories, but none had valid summary.json files. "
            f"This may indicate all trials failed or were incomplete. "
            f"Creating empty parquet file to allow pipeline to continue."
        )
        logger.warning(warning_msg)
        # Create empty DataFrame with expected structure
        df = pd.DataFrame()
        # Write empty parquet file so pipeline can continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"📊 Empty parquet file written: {output_path}")
        return output_path
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Write to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Successfully aggregated {successful} trials")
    if failed > 0:
        pct = 100 * failed / len(trial_dirs) if len(trial_dirs) > 0 else 0
        logger.info(f"ℹ️  Skipped {failed}/{len(trial_dirs)} trials ({pct:.1f}%) - missing or invalid summary.json")
    logger.info(f"📊 Parquet file written: {output_path}")
    logger.info(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Ray Tune experiment results into a single parquet file"
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Directory containing Ray trial subdirectories"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path (default: <exp_dir>/experiment.parquet)"
    )

    args = parser.parse_args()

    try:
        output_path = aggregate_experiment_index(args.exp_dir, args.out)
        print(f"\n✅ Aggregation complete: {output_path}")

        # Update pipeline status if this is part of a collection
        # exp_dir is like: results/experiments/collection_name/dataset_name
        # collection_dir is: results/experiments/collection_name
        collection_dir = args.exp_dir.parent
        if (collection_dir / "MANIFEST.yaml").exists():
            try:
                from utils.pipeline_status import mark_aggregation_complete
                mark_aggregation_complete(collection_dir, output_path)
                print("📊 Pipeline status updated: aggregation completed")
            except Exception as status_error:
                logger.warning(f"Could not update pipeline status: {status_error}")

        return 0
    except Exception as e:
        logger.error(f"❌ Aggregation failed: {e}")

        # Update pipeline status on failure if this is part of a collection
        collection_dir = args.exp_dir.parent
        if (collection_dir / "MANIFEST.yaml").exists():
            try:
                from utils.pipeline_status import mark_aggregation_failed
                mark_aggregation_failed(collection_dir)
                print("📊 Pipeline status updated: aggregation failed")
            except Exception as status_error:
                logger.warning(f"Could not update pipeline status: {status_error}")

        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
