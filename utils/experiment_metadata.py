"""
Experiment metadata utilities for generating experiment IDs, git commit tracking,
and creating experiment manifests.
"""
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def generate_experiment_id(test_mode: bool = False) -> str:
    """
    Generate a unique experiment ID in format: YYYY-MM-DD_HH-MM-SS__<6-char-hash>
    For test runs, uses format: test_YYYY-MM-DD_HH-MM-SS__<6-char-hash>

    Args:
        test_mode: Whether this is a test run

    Returns:
        Experiment ID string
    """
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create hash from timestamp + random component
    timestamp_str = now.isoformat()
    hash_input = f"{timestamp_str}_{now.microsecond}"
    hash_obj = hashlib.md5(hash_input.encode())
    short_hash = hash_obj.hexdigest()[:6]

    if test_mode:
        return f"test_{datetime_str}__{short_hash}"
    else:
        return f"{datetime_str}__{short_hash}"


def get_git_commit() -> Optional[str]:
    """
    Get the current git commit hash.
    
    Returns:
        Git commit hash string, or None if not in a git repo or git is unavailable
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd()
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_collection_dir(
    base_results_dir: Path,
    exp_id: str,
    exp_tag: Optional[str] = None,
    test_mode: bool = False
) -> Path:
    """
    Get the collection directory path for an experiment.
    
    Args:
        base_results_dir: Base results directory (e.g., from config save_dir)
        exp_id: Experiment ID (e.g., "2025-01-15__a3f2b1")
        exp_tag: Optional experiment tag (e.g., "zcp_v2")
        test_mode: If True, route to test subdirectory
    
    Returns:
        Path to collection directory
    """
    # Build collection name: <exp_id>__<exp_tag> or just <exp_id>
    if exp_tag:
        collection_name = f"{exp_id}__{exp_tag}"
    else:
        collection_name = exp_id
    
    # Route to test or production
    if test_mode:
        collection_dir = base_results_dir / "experiments" / "test" / collection_name
    else:
        collection_dir = base_results_dir / "experiments" / collection_name
    
    return collection_dir


def save_collection_config(collection_dir: Path, config_path: str) -> Path:
    """
    Save a copy of the unified config to the collection directory for reproducibility.

    Args:
        collection_dir: Collection directory path
        config_path: Path to the unified config file

    Returns:
        Path to the saved config file
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Save as unified_config.yml in collection directory
    saved_config_path = collection_dir / "unified_config.yml"

    # Copy the config file
    import shutil
    shutil.copy2(config_path_obj, saved_config_path)

    return saved_config_path


def create_manifest(
    collection_dir: Path,
    exp_id: str,
    exp_tag: Optional[str],
    exp_note: Optional[str],
    git_commit: Optional[str],
    test_mode: bool,
    timestamp: Optional[str] = None,
    config_path: Optional[str] = None
) -> Path:
    """
    Create MANIFEST.yaml in the collection directory.

    Only creates if it doesn't already exist (allows multiple datasets in same collection).

    Args:
        collection_dir: Collection directory path
        exp_id: Experiment ID
        exp_tag: Optional experiment tag
        exp_note: Optional experiment note
        git_commit: Optional git commit hash
        test_mode: Whether this is a test run
        timestamp: Optional timestamp (defaults to current time)
        config_path: Optional path to unified config (will be saved for reproducibility)

    Returns:
        Path to MANIFEST.yaml file
    """
    manifest_path = collection_dir / "MANIFEST.yaml"
    
    # Don't overwrite existing manifest (allows multiple datasets in same collection)
    if manifest_path.exists():
        return manifest_path
    
    # Create collection directory if needed
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility if provided
    saved_config_path = None
    if config_path:
        saved_config_path = save_collection_config(collection_dir, config_path)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    manifest_data = {
        "experiment_id": exp_id,
        "exp_tag": exp_tag,
        "exp_note": exp_note,
        "git_commit": git_commit,
        "timestamp": timestamp,
        "test_mode": test_mode,
        "config_path": str(saved_config_path.relative_to(collection_dir)) if saved_config_path else None
    }
    
    with open(manifest_path, "w") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

    return manifest_path


def update_collection_manifest(collection_dir: Path, **updates) -> bool:
    """
    Update the MANIFEST.yaml file with additional runtime statistics.

    Args:
        collection_dir: Collection directory path
        **updates: Key-value pairs to update in the manifest

    Returns:
        True if update was successful, False otherwise
    """
    manifest_path = collection_dir / "MANIFEST.yaml"

    if not manifest_path.exists():
        return False

    try:
        # Read current manifest
        with open(manifest_path, 'r') as f:
            manifest_data = yaml.safe_load(f) or {}

        # Update with new data
        manifest_data.update(updates)

        # Write back
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception:
        return False

