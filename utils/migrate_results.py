"""
Utilities for migrating orphaned experiments into proper collection structure.
"""
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.experiment_metadata import generate_experiment_id, create_manifest


def find_orphaned_experiments(base_dir: Path) -> List[Path]:
    """
    Find experiment directories that are not properly organized into collections.

    Looks for directories that contain trial subdirectories but are not in the
    proper collection structure (results/experiments/collection_name/dataset_name/).

    Args:
        base_dir: Base results directory (e.g., results/)

    Returns:
        List of orphaned experiment directory paths
    """
    orphaned = []

    # Check for experiments directly in results/ that should be in collections
    results_dir = base_dir / "results"
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and item.name not in ["experiments"]:
                # Check if this looks like an experiment directory
                if _is_experiment_directory(item):
                    orphaned.append(item)

    # Also check for experiments in experiments/ that aren't in proper collections
    experiments_dir = base_dir / "experiments"
    if experiments_dir.exists():
        for item in experiments_dir.iterdir():
            if item.is_dir() and not (item / "MANIFEST.yaml").exists():
                # This might be an orphaned collection or experiment
                if _is_experiment_directory(item):
                    orphaned.append(item)

    return orphaned


def _is_experiment_directory(dir_path: Path) -> bool:
    """
    Check if a directory looks like an experiment directory.

    An experiment directory typically contains:
    - trial subdirectories (train_tune_*)
    - summary.json files
    - config files

    Args:
        dir_path: Directory to check

    Returns:
        True if directory looks like an experiment
    """
    # Check for trial directories
    trial_dirs = list(dir_path.glob("train_tune_*"))
    if trial_dirs:
        return True

    # Check for summary.json files
    summary_files = list(dir_path.glob("**/summary.json"))
    if summary_files:
        return True

    return False


def migrate_to_collection(exp_dir: Path, collection_id: Optional[str] = None,
                         test_mode: bool = False) -> Path:
    """
    Migrate an orphaned experiment directory into proper collection structure.

    Args:
        exp_dir: Path to the orphaned experiment directory
        collection_id: Optional collection ID. If None, generates one.
        test_mode: Whether this is a test collection

    Returns:
        Path to the new collection directory
    """
    if collection_id is None:
        collection_id = generate_experiment_id()

    # Determine base results directory
    base_dir = exp_dir.parent.parent if exp_dir.parent.name == "experiments" else exp_dir.parent

    # Create collection structure
    collection_type = "test" if test_mode else ""
    collection_dir = base_dir / "experiments" / collection_type / collection_id
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Extract dataset name from experiment directory
    dataset_name = exp_dir.name

    # Create new experiment directory in collection
    new_exp_dir = collection_dir / dataset_name

    # Move experiment directory
    if exp_dir != new_exp_dir:
        if new_exp_dir.exists():
            # If destination exists, create a backup
            backup_dir = new_exp_dir.parent / f"{new_exp_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            new_exp_dir.rename(backup_dir)

        shutil.move(str(exp_dir), str(new_exp_dir))

    # Try to extract metadata from existing files to create manifest
    manifest_data = _extract_metadata_from_experiment(new_exp_dir)

    # Create manifest
    create_manifest(
        collection_dir,
        collection_id,
        manifest_data.get("exp_tag"),
        manifest_data.get("exp_note"),
        manifest_data.get("git_commit"),
        test_mode,
        config_path=manifest_data.get("config_path")
    )

    return collection_dir


def _extract_metadata_from_experiment(exp_dir: Path) -> Dict[str, Any]:
    """
    Try to extract metadata from an existing experiment directory.

    Args:
        exp_dir: Experiment directory path

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}

    # Try to read from existing summary.json or config files
    summary_file = exp_dir / "summary.json"
    if summary_file.exists():
        try:
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                # Extract any experiment metadata that might be in summary
                metadata.update({
                    k: v for k, v in summary.items()
                    if k in ["experiment_id", "exp_tag", "exp_note", "git_commit"]
                })
        except Exception:
            pass

    # Try to find config file
    config_file = exp_dir / "config_resolved.yaml"
    if config_file.exists():
        metadata["config_path"] = str(config_file)

    return metadata


def ensure_collection_structure(base_dir: Path) -> Dict[str, Any]:
    """
    Ensure all experiments are properly organized into collections.

    Args:
        base_dir: Base results directory

    Returns:
        Dictionary with migration results
    """
    results = {
        "orphaned_found": 0,
        "migrated": 0,
        "errors": 0,
        "collections_created": []
    }

    # Find orphaned experiments
    orphaned = find_orphaned_experiments(base_dir)
    results["orphaned_found"] = len(orphaned)

    if not orphaned:
        print("✅ No orphaned experiments found")
        return results

    print(f"📦 Found {len(orphaned)} orphaned experiment(s)")

    for exp_dir in orphaned:
        try:
            print(f"  Migrating: {exp_dir}")
            collection_dir = migrate_to_collection(exp_dir)
            results["migrated"] += 1
            results["collections_created"].append(str(collection_dir))
            print(f"    → {collection_dir}")
        except Exception as e:
            print(f"    ❌ Error migrating {exp_dir}: {e}")
            results["errors"] += 1

    return results


def main():
    """Command-line interface for migration utility."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate orphaned experiments into proper collection structure"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results").parent,
        help="Base directory containing results/ (default: parent of results/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it"
    )

    args = parser.parse_args()

    if args.dry_run:
        orphaned = find_orphaned_experiments(args.base_dir)
        if orphaned:
            print(f"📦 Would migrate {len(orphaned)} orphaned experiment(s):")
            for exp in orphaned:
                print(f"  {exp}")
        else:
            print("✅ No orphaned experiments found")
        return

    results = ensure_collection_structure(args.base_dir)

    print("
📊 Migration Summary:"    print(f"  Orphaned experiments found: {results['orphaned_found']}")
    print(f"  Successfully migrated: {results['migrated']}")
    print(f"  Errors: {results['errors']}")

    if results['collections_created']:
        print(f"  New collections created: {len(results['collections_created'])}")


if __name__ == "__main__":
    main()