import os
from pathlib import Path

import ray

from analysis.model_testing.dataset_processing_config import DATASET_CONFIGS
from analysis.post_processing_pipeline import aggregate_training_metrics
from analysis.post_processing_pipeline import run_processing
from analysis.post_processing_pipeline import zero_cost_scores_parallel

ONLY_DATASETS: list[str] = [
    # "motionsense",
    # "hhar",
    # "pamap2"
    # "mobiactv2"
    # "rwhar"
]


def _init_ray():
    ray_tmp_dir = Path(os.environ.get("RAY_TMPDIR", Path.cwd() / "ray_tmp_dir"))
    ray_tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔧 Using Ray temp dir: {ray_tmp_dir}")
    ray.init(ignore_reinit_error=True, _temp_dir=str(ray_tmp_dir))


def main(collection_dir=None, skip_zcp=False, output_dir=None):
    """
    Run post-processing pipeline: aggregate training metrics → process params/metrics → zero-cost scores.

    Args:
        collection_dir: Optional collection directory path. If None, uses DATASET_CONFIGS.
        skip_zcp: If True, skip zero-cost proxy computation (still creates full_results_pkl files).
        output_dir: Optional directory for post-processed outputs. With collection_dir, results are
            written to output_dir/<dataset_name>/ instead of collection_dir/<dataset_name>/post-processed-results/.
    """
    # decide which datasets to run
    if ONLY_DATASETS:
        targets = [ds for ds in ONLY_DATASETS if ds in DATASET_CONFIGS]
        if not targets:
            print(f"⚠️  No matching datasets in DATASET_CONFIGS for {ONLY_DATASETS}")
            return
        configs = {ds: DATASET_CONFIGS[ds] for ds in targets}
    else:
        targets = list(DATASET_CONFIGS.keys())
        configs = DATASET_CONFIGS

    # If collection_dir is provided, override paths to use collection-based structure
    # This works for both test and production collections
    if collection_dir is not None:
        print(f"🔧 Using collection directory: {collection_dir}")
        if output_dir is not None:
            print(f"📦 Writing post-processed outputs to: {output_dir}")
        # Find datasets that exist directly in the collection directory
        # Ray Tune stores trials at collection_dir/<dataset_name>/train_tune_*
        collection_path = Path(collection_dir)
        output_path = Path(output_dir) if output_dir is not None else None
        test_datasets = []
        for ds_name in targets:
            dataset_dir = collection_path / ds_name
            if dataset_dir.exists() and dataset_dir.is_dir():
                test_datasets.append(ds_name)

        if not test_datasets:
            print(f"⚠️  No datasets found in test collection {collection_dir}")
            return

        # Override configs to use test collection paths
        test_configs = {}
        for ds_name in test_datasets:
            dataset_dir = collection_path / ds_name
            if output_path is not None:
                post_processed_dir = output_path / ds_name
            else:
                post_processed_dir = dataset_dir / "post-processed-results"
            post_processed_dir.mkdir(parents=True, exist_ok=True)
            test_configs[ds_name] = {
                "results_dir": str(dataset_dir),  # Still point to dataset dir for trial data
                "full_results_pkl": str(post_processed_dir / "full_results.pkl"),
                "zero_cost_pkl": str(post_processed_dir / "zero_cost_scores.pkl"),
                "full_training_metrics_pkl": str(post_processed_dir / "full_training_metrics.pkl"),
                # Keep data_root from original config
                "data_root": DATASET_CONFIGS[ds_name]["data_root"]
            }
        configs = test_configs
        targets = test_datasets

    # Step 1: build & pickle full training‐metrics for every dataset
    print("\n=== Aggregating training metrics ===")
    aggregate_training_metrics.build_and_save_all(configs)

    # Step 2: flatten params + pick best metrics → pickle
    print("\n=== Processing params + metrics for each dataset ===")
    for ds_name, cfg in configs.items():
        results_dir = Path(cfg["results_dir"])
        output_pkl = Path(cfg["full_results_pkl"])
        print(f"\n→ {ds_name}")
        run_processing.process_dataset(results_dir, output_pkl)

    # Step 3: compute zero‐cost scores in parallel via Ray (optional)
    if skip_zcp:
        print("\n⏭️  Skipping zero-cost score computation (skip_zcp=True)")
    else:
        print("\n=== Computing zero‐cost scores ===")
        if ONLY_DATASETS:
            # custom loop so we only run our subset
            try:
                _init_ray()
                for ds_name in targets:
                    # Pass configs override if in test mode (collection_dir provided)
                    config_override = configs if collection_dir is not None else None
                    zero_cost_scores_parallel.run_for_dataset(ds_name, config_override=config_override)
            finally:
                ray.shutdown()
        else:
            # run everything as before
            if collection_dir is not None:
                # Test mode: use overridden configs
                try:
                    _init_ray()
                    for ds_name in targets:
                        zero_cost_scores_parallel.run_for_dataset(ds_name, config_override=configs)
                finally:
                    ray.shutdown()
            else:
                # Normal mode: use DATASET_CONFIGS
                zero_cost_scores_parallel.main()


if __name__ == "__main__":
    main()
