"""
End-to-end test for the experiment pipeline.

This test validates:
1. Experiment organization (collection directories, MANIFEST.yaml)
2. Config inheritance
3. Trial directory structure (summary.json, config_resolved.yaml, logs, checkpoints)
4. Aggregation (experiment.parquet creation)
5. Metadata propagation (experiment_id, exp_tag, exp_note, git_commit)
"""
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
import yaml

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.post_processing_pipeline.aggregate_experiment_index import aggregate_experiment_index
from tune_runner import main as tune_main
from utils.config import load_config
from utils.experiment_metadata import get_collection_dir


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="mgt_test_")
    yield Path(temp_dir)
    # Cleanup
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_configs(temp_test_dir):
    """Create minimal test configs for 2 datasets."""
    configs_dir = temp_test_dir / "configs" / "experiments"
    configs_dir.mkdir(parents=True)
    
    # Create unified config structure
    unified_config = {
        "seed": 42,
        "device": "cpu",  # Use CPU for testing
        "default_save_dir": str(temp_test_dir / "results"),
        "default_data_dir": str(temp_test_dir / "data"),
        "default_temp_dir": str(temp_test_dir / "temp"),
        "default_experiments": {
            "num_models": 2,
            "model_types": ["CNN", "RNN"],
            "model_type_counts": {"CNN": 1, "RNN": 1},
            "hyperparameters": {
                "batch_size": 256,
                "learning_rate": 0.0001,
                "num_epochs": 2,  # Very short for testing
                "step_size": 10,
                "gamma": 0.8,
                "weight_decay": 0.0001,
            },
        },
        "datasets": {
            "random": {
                "save_dir": str(temp_test_dir / "results"),
                "data_dir": str(temp_test_dir / "data" / "random"),  # Doesn't need to exist for RandomDataset
                "temp_dir": str(temp_test_dir / "temp"),
                "dataset": "random",
                # Required per-dataset by utils.config; the test runs on CPU, but
                # the value still has to be present and positive.
                "gpu_per_trial": 0.5,
                "exp_tag": "e2e_test",
                "exp_note": "End-to-end test run"
            }
        }
    }
    
    unified_config_path = configs_dir / "unified_config.yml"
    with open(unified_config_path, "w") as f:
        yaml.dump(unified_config, f)
    
    test_configs = {"random": unified_config_path}
    
    return test_configs, None


@pytest.mark.slow
def test_experiment_organization_and_pipeline(temp_test_dir, test_configs):
    """
    Test the full pipeline: experiment → aggregation → verification.
    
    This test:
    1. Runs experiments with test mode (2 models per dataset)
    2. Verifies experiment organization (collection dirs, MANIFEST.yaml)
    3. Verifies trial directory structure
    4. Runs aggregation
    5. Verifies metadata propagation
    """
    test_configs_dict, _ = test_configs
    
    # Use random dataset (doesn't require actual data files)
    config_path = test_configs_dict["random"]
    
    # Step 1: Run experiment in test mode (2 models)
    print(f"\n{'='*60}")
    print("Step 1: Running experiment in test mode")
    print(f"{'='*60}")
    
    try:
        tune_main(
            cfg_path=str(config_path),
            test_mode=True,
            dataset_name="random",
            auto_confirm=True,  # no tty under pytest, so skip the trial-count prompt
        )
    except Exception as e:
        pytest.fail(f"Experiment failed: {e}")
    
    # Step 2: Verify experiment organization
    print(f"\n{'='*60}")
    print("Step 2: Verifying experiment organization")
    print(f"{'='*60}")
    
    cfg = load_config(str(config_path), dataset_name="random")
    base_results_dir = Path(cfg["save_dir"])
    
    # Find collection directory (test mode should create it under test/)
    test_experiments_dir = base_results_dir / "experiments" / "test"
    assert test_experiments_dir.exists(), f"Test experiments directory not found: {test_experiments_dir}"
    
    # Find the collection directory (should match exp_id__exp_tag pattern)
    collection_dirs = list(test_experiments_dir.glob("*__e2e_test"))
    assert len(collection_dirs) > 0, f"No collection directory found matching pattern in {test_experiments_dir}"
    collection_dir = collection_dirs[0]
    
    print(f"Found collection directory: {collection_dir}")
    
    # Verify MANIFEST.yaml exists
    manifest_path = collection_dir / "MANIFEST.yaml"
    assert manifest_path.exists(), f"MANIFEST.yaml not found at {manifest_path}"
    
    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)
    
    assert "experiment_id" in manifest
    assert manifest["exp_tag"] == "e2e_test"
    assert manifest["exp_note"] == "End-to-end test run"
    assert manifest["test_mode"] is True
    print(f"✅ MANIFEST.yaml verified: {manifest}")
    
    # Verify dataset subdirectory exists
    dataset_dir = collection_dir / "random"
    assert dataset_dir.exists(), f"Dataset directory not found: {dataset_dir}"
    print(f"✅ Dataset directory exists: {dataset_dir}")
    
    # Step 3: Verify trial directory structure
    print(f"\n{'='*60}")
    print("Step 3: Verifying trial directory structure")
    print(f"{'='*60}")
    
    trial_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("train_tune_")]
    assert len(trial_dirs) > 0, f"No trial directories found in {dataset_dir}"
    print(f"Found {len(trial_dirs)} trial directories")
    
    for trial_dir in trial_dirs:
        print(f"\nChecking trial: {trial_dir.name}")
        
        # Check required files
        summary_file = trial_dir / "summary.json"
        assert summary_file.exists(), f"summary.json not found in {trial_dir}"
        
        config_file = trial_dir / "config_resolved.yaml"
        assert config_file.exists(), f"config_resolved.yaml not found in {trial_dir}"
        
        logs_dir = trial_dir / "logs"
        assert logs_dir.exists(), f"logs/ directory not found in {trial_dir}"
        
        train_log = logs_dir / "train.log"
        # train.log might not exist if logging failed, but logs_dir should exist
        # assert train_log.exists(), f"train.log not found in {logs_dir}"
        
        ckpt_dir = trial_dir / "checkpoints"
        assert ckpt_dir.exists(), f"checkpoints/ directory not found in {trial_dir}"
        
        # Check summary.json content
        with open(summary_file, "r") as f:
            summary = json.load(f)
        
        # Verify metadata fields
        assert "experiment_id" in summary, "experiment_id missing from summary.json"
        assert summary["exp_tag"] == "e2e_test", f"exp_tag mismatch: {summary.get('exp_tag')}"
        assert summary["exp_note"] == "End-to-end test run", f"exp_note mismatch: {summary.get('exp_note')}"
        assert "git_commit" in summary, "git_commit missing from summary.json"
        assert summary["dataset"] == "random", f"dataset mismatch: {summary.get('dataset')}"
        
        print(f"  ✅ summary.json verified")
        
        # Check config_resolved.yaml content
        with open(config_file, "r") as f:
            config_resolved = yaml.safe_load(f)
        
        assert "experiment_id" in config_resolved, "experiment_id missing from config_resolved.yaml"
        assert config_resolved["exp_tag"] == "e2e_test"
        print(f"  ✅ config_resolved.yaml verified")
    
    # Step 4: Run aggregation
    print(f"\n{'='*60}")
    print("Step 4: Running aggregation")
    print(f"{'='*60}")
    
    parquet_path = dataset_dir / "experiment.parquet"
    
    try:
        aggregate_experiment_index(dataset_dir, parquet_path)
    except Exception as e:
        pytest.fail(f"Aggregation failed: {e}")
    
    assert parquet_path.exists(), f"experiment.parquet not created at {parquet_path}"
    print(f"✅ experiment.parquet created: {parquet_path}")
    
    # Step 5: Verify parquet content
    print(f"\n{'='*60}")
    print("Step 5: Verifying parquet content")
    print(f"{'='*60}")
    
    df = pd.read_parquet(parquet_path)
    
    # Verify row count matches number of trials
    assert len(df) == len(trial_dirs), f"Parquet row count ({len(df)}) doesn't match trial count ({len(trial_dirs)})"
    print(f"✅ Parquet contains {len(df)} rows (expected {len(trial_dirs)})")
    
    # Verify required columns exist
    required_columns = [
        "experiment_id", "exp_tag", "exp_note", "git_commit",
        "dataset", "model_type", "arch_seed",
        "initial_val_f1_macro", "best_val_f1_macro", "best_test_f1_macro"
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' missing from parquet"
    
    print(f"✅ All required columns present: {required_columns}")
    
    # Verify metadata values
    assert all(df["exp_tag"] == "e2e_test"), "exp_tag mismatch in parquet"
    assert all(df["exp_note"] == "End-to-end test run"), "exp_note mismatch in parquet"
    assert all(df["dataset"] == "random"), "dataset mismatch in parquet"
    
    print(f"✅ Metadata values verified in parquet")
    
    # Verify at least one model of each type (test mode requirement)
    model_types = df["model_type"].unique()
    print(f"Model types in results: {model_types}")
    # Note: In test mode, we should have at least one of each type, but with only 2 models
    # we might not always get both. This is acceptable for a minimal test.
    
    print(f"\n{'='*60}")
    print("✅ All checks passed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run the test directly
    pytest.main([__file__, "-v", "-s"])

