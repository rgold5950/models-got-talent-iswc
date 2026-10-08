"""
Utility to extract architecture configurations from trial results into a DataFrame.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle lists by creating indexed keys
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}.{idx}", sep=sep).items())
                else:
                    items.append((f"{new_key}.{idx}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def extract_arch_configs_from_trials(
    experiment_dir: Path,
    include_trial_metadata: bool = True
) -> Optional[pd.DataFrame]:
    """
    Extract architecture configurations from all completed trials.
    
    Args:
        experiment_dir: Directory containing trial subdirectories (train_tune_*)
        include_trial_metadata: If True, include trial directory name and other metadata
    
    Returns:
        DataFrame with flattened architecture configurations, or None if no trials found
    """
    trial_dirs = sorted(experiment_dir.glob("train_tune_*"))
    
    if not trial_dirs:
        logging.warning(f"No trial directories found in {experiment_dir}")
        return None
    
    rows = []
    all_keys = set()
    
    for trial_dir in trial_dirs:
        if not trial_dir.is_dir():
            continue
        
        params_file = trial_dir / "params.json"
        if not params_file.exists():
            logging.debug(f"Skipping {trial_dir.name}: params.json not found")
            continue
        
        try:
            with open(params_file, "r") as f:
                params = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Skipping {trial_dir.name}: Error reading params.json: {e}")
            continue
        
        # Extract arch_config
        arch_config = params.get("arch_config")
        if arch_config is None:
            logging.debug(f"Skipping {trial_dir.name}: No arch_config in params.json")
            continue
        
        # Flatten arch_config
        if isinstance(arch_config, dict):
            arch_flat = flatten_dict(arch_config, "arch_config")
        elif isinstance(arch_config, list):
            # Handle list of dicts (common for CNN/RNN architectures)
            arch_flat = {}
            for idx, item in enumerate(arch_config):
                if isinstance(item, dict):
                    arch_flat.update(flatten_dict(item, f"arch_config.{idx}"))
                else:
                    arch_flat[f"arch_config.{idx}"] = item
            # Add number of layers as a summary
            arch_flat["arch_config.n_layers"] = len(arch_config)
        else:
            # Simple value, just store it
            arch_flat = {"arch_config": arch_config}
        
        # Add trial metadata if requested
        if include_trial_metadata:
            arch_flat["trial_dir"] = trial_dir.name
            # Add other useful params
            for key in ["arch_seed", "model_type", "init_seed", "init_name"]:
                if key in params:
                    arch_flat[f"params_{key}"] = params[key]
        
        rows.append(arch_flat)
        all_keys.update(arch_flat.keys())
    
    if not rows:
        logging.warning(f"No valid architecture configurations found in {experiment_dir}")
        return None
    
    # Create DataFrame with all columns aligned
    df = pd.DataFrame(rows)
    
    # Reindex to include all keys (fill missing with NaN)
    df = df.reindex(columns=sorted(all_keys))
    
    logging.info(f"Extracted {len(df)} architecture configurations with {len(df.columns)} columns")
    
    return df


def save_arch_configs_df(
    df: pd.DataFrame,
    output_path: Path,
    format: str = "parquet"
) -> None:
    """
    Save architecture configurations DataFrame to file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path (without extension)
        format: File format ("parquet", "csv", or "pickle")
    """
    if format == "parquet":
        file_path = output_path.with_suffix(".parquet")
        df.to_parquet(file_path, index=False)
    elif format == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format == "pickle":
        file_path = output_path.with_suffix(".pkl")
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'csv', or 'pickle'")
    
    logging.info(f"Saved architecture configurations to {file_path}")
