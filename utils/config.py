import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries. Override values take precedence.
    For lists, replace entirely (don't append).
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override takes precedence (including for lists)
            result[key] = value
    
    return result


def load_unified_config(config_path: str, dataset_name: str) -> Dict[str, Any]:
    """
    Load a unified config file and select a specific dataset configuration.
    
    The unified config should have:
    - Global defaults (seed, device, default_* paths, default_experiments)
    - A 'datasets' section; each enabled dataset entry must include 'gpu_per_trial'
    
    Args:
        config_path: Path to the unified config file
        dataset_name: Name of the dataset to select from the 'datasets' section
    
    Returns:
        Merged configuration dictionary with dataset-specific config applied
    """
    config_path_obj = Path(config_path)
    
    # Load the unified config
    with open(config_path_obj, 'r') as f:
        unified_config = yaml.safe_load(f) or {}

    # Extract global defaults
    global_config = {
        "seed": unified_config.get("seed", 42),
        "device": unified_config.get("device", "cuda"),
    }

    # Get project root for base_path substitution
    from utils.path_utils import PROJECT_ROOT
    base_path = str(PROJECT_ROOT)

    # Extract default paths (relative to project root)
    default_save_dir = unified_config.get("default_save_dir", "results/")
    default_data_dir = unified_config.get("default_data_dir", "data/all_data/")
    # Accept either name at the top level; per-dataset temp_dir still overrides it.
    default_temp_dir = unified_config.get("default_temp_dir") or unified_config.get("temp_dir", ".")

    # Substitute {base_path} in default paths
    default_save_dir = default_save_dir.replace("{base_path}", base_path)
    default_data_dir = default_data_dir.replace("{base_path}", base_path)
    default_temp_dir = default_temp_dir.replace("{base_path}", base_path)
    
    # Extract default experiments
    default_experiments = unified_config.get("default_experiments", {})
    
    # Get dataset-specific config
    datasets = unified_config.get("datasets", {})
    
    # Filter out disabled datasets
    available_datasets = {
        name: config 
        for name, config in datasets.items() 
        if not config.get("disabled", False)
    }
    
    if dataset_name not in datasets:
        available_names = list(available_datasets.keys())
        all_names = list(datasets.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in unified config. "
            f"Available (enabled) datasets: {available_names}. "
            f"All datasets (including disabled): {all_names}"
        )
    
    dataset_config = datasets[dataset_name]

    # Check if the requested dataset is disabled
    if dataset_config.get("disabled", False):
        available_names = list(available_datasets.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' is disabled in the unified config. "
            f"Set 'disabled: false' to enable it. "
            f"Available (enabled) datasets: {available_names}"
        )

    if "gpu_per_trial" not in dataset_config or dataset_config["gpu_per_trial"] is None:
        raise ValueError(
            f"datasets.{dataset_name} must define 'gpu_per_trial' (no top-level default in unified config)"
        )
    global_config["gpu_per_trial"] = float(dataset_config["gpu_per_trial"])
    
    # Paths are now relative to project root (no formatting needed)
    
    # Merge defaults with dataset-specific config
    # Paths: use dataset-specific if provided, otherwise use defaults
    global_config["save_dir"] = dataset_config.get("save_dir", default_save_dir).replace("{base_path}", base_path)
    global_config["data_dir"] = dataset_config.get("data_dir", default_data_dir).replace("{base_path}", base_path)
    global_config["temp_dir"] = dataset_config.get("temp_dir", default_temp_dir).replace("{base_path}", base_path)
    
    # Add rho if present
    if "rho" in dataset_config:
        global_config["rho"] = dataset_config["rho"]
    
    # Merge experiments: start with defaults, then apply dataset-specific overrides
    experiments = _deep_merge(default_experiments.copy(), dataset_config.get("experiments", {}))
    
    # Ensure dataset name is set
    experiments["dataset"] = dataset_config.get("dataset", dataset_name)
    
    global_config["experiments"] = experiments
    
    # Add optional exp_tag and exp_note if present
    if "exp_tag" in dataset_config:
        global_config["exp_tag"] = dataset_config["exp_tag"]
    if "exp_note" in dataset_config:
        global_config["exp_note"] = dataset_config["exp_note"]
    
    return global_config


def list_available_datasets(config_path: str, variant: Optional[str] = None) -> List[str]:
    """
    List all available (non-disabled) datasets from a unified config file.
    
    Args:
        config_path: Path to the unified config file
        variant: Optional variant filter ("original", "imbalance", or None for all)
    
    Returns:
        List of available dataset names matching the variant filter
    """
    config_path_obj = Path(config_path)
    
    with open(config_path_obj, 'r') as f:
        unified_config = yaml.safe_load(f) or {}
    
    datasets = unified_config.get("datasets", {})
    
    available_datasets = [
        name 
        for name, config in datasets.items() 
        if not config.get("disabled", False)
    ]
    
    # Filter by variant if specified
    if variant is not None:
        available_datasets = [
            name
            for name in available_datasets
            if datasets[name].get("variant") == variant
        ]
    
    return sorted(available_datasets)


def load_config(config_path: str, dataset_name: str) -> Dict[str, Any]:
    """
    Load a unified config file and select a specific dataset configuration.
    
    Args:
        config_path: Path to the unified config file
        dataset_name: Name of the dataset to select from the 'datasets' section
    
    Returns:
        Configuration dictionary
    """
    if dataset_name is None:
        raise ValueError(
            "dataset_name is required. Use unified_config.yml format: "
            "--config configs/experiments/unified_config.yml --dataset <dataset_name>"
        )
    return load_unified_config(config_path, dataset_name)