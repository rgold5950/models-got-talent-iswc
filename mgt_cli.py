#!/usr/bin/env python3
"""
Models Got Talent - Unified CLI

Provides commands to run experiments, aggregate results, and run the full pipeline.
"""
import argparse
import logging
import sys
from pathlib import Path

from utils.config import load_config, list_available_datasets
from utils.path_utils import PROJECT_ROOT
from data.dataset_registry import dataset_registry


def run_with_error_handling(step_name: str, func, *args, **kwargs):
    """
    Execute a function with error handling.
    
    Returns:
        Tuple of (success: bool, result: Any)
    """
    try:
        print(f"\n{'='*60}")
        print(f"Starting: {step_name}")
        print(f"{'='*60}")
        result = func(*args, **kwargs)
        print(f"✅ {step_name} completed successfully")
        return True, result
    except Exception as e:
        print(f"❌ {step_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def cmd_run(args):
    """Run experiment command."""
    from tune_runner import main as tune_main
    # --test / --num-test-models only affect layout (experiments/test/); trial counts come from YAML.
    test_mode = bool(
        (hasattr(args, "test") and args.test)
        or (hasattr(args, "num_test_models") and args.num_test_models)
    )
    
    auto_skip_note = (hasattr(args, "yes") and args.yes) or (
        hasattr(args, "no_note") and args.no_note
    )
    # Handle experiment note
    exp_note = None
    if hasattr(args, 'note') and args.note:
        exp_note = args.note
    elif not auto_skip_note:
        # Prompt for note
        print("\n📝 Experiment Note")
        print("Enter a brief note describing this experiment (optional):")
        try:
            exp_note = input("> ").strip()
            if not exp_note:
                exp_note = None
        except (EOFError, KeyboardInterrupt):
            print("\n⏭️  Skipping note input")
            exp_note = None

    if exp_note:
        print(f"📝 Note: {exp_note}")
    else:
        print("📝 No note provided")

    # Determine which datasets to run
    datasets_to_run = []
    if args.variant:
        # Filter by variant
        datasets_to_run = list_available_datasets(args.config, variant=args.variant)
        if not datasets_to_run:
            print(f"❌ No datasets found with variant '{args.variant}'")
            return 1
        print(f"📊 Running {len(datasets_to_run)} datasets with variant '{args.variant}': {', '.join(datasets_to_run)}")
    elif args.dataset:
        # Single dataset specified
        datasets_to_run = [args.dataset]
    else:
        print("❌ Either --dataset or --variant must be specified")
        return 1
     
     
    # Run experiments for each dataset
    all_success = True
    for dataset_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Running experiment for dataset: {dataset_name}")
        print(f"{'='*60}")
        reset = hasattr(args, 'reset') and args.reset
        # Use auto_confirm if -y/--yes flag was passed
        auto_confirm = hasattr(args, 'yes') and args.yes
        retry_errored = bool(getattr(args, "retry_errored", False))
        ray_address = getattr(args, "ray_address", None)
        success, _ = run_with_error_handling(
            f"Experiment ({dataset_name})",
            tune_main,
            args.config,
            test_mode,
            dataset_name,
            reset,
            exp_note,
            None,  # custom_params
            auto_confirm,
            extract_arch_configs=False,
            exp_id=args.exp_id,
            restart_errored=retry_errored,
            ray_address=ray_address,
        )
        if not success:
            all_success = False
    
    return 0 if all_success else 1


def cmd_aggregate(args):
    """Aggregate experiment results command."""
    from analysis.post_processing_pipeline.aggregate_experiment_index import aggregate_experiment_index
    
    exp_dir = Path(args.exp_dir)
    output_path = Path(args.out) if args.out else None
    
    success, _ = run_with_error_handling(
        "Aggregate",
        aggregate_experiment_index,
        exp_dir,
        output_path
    )
    return 0 if success else 1


def find_collections(base_dir: Path, test_mode: bool = False):
    """Find all experiment collections in the given base directory."""
    if test_mode:
        collections_dir = base_dir / "experiments" / "test"
    else:
        collections_dir = base_dir / "experiments"
    
    if not collections_dir.exists():
        return []
    
    collections = []
    for item in collections_dir.iterdir():
        if item.is_dir() and (item / "MANIFEST.yaml").exists():
            collections.append(item)
    
    return sorted(collections, key=lambda p: p.stat().st_mtime, reverse=True)


def find_datasets_in_collection(collection_dir: Path):
    """Find all dataset directories in a collection."""
    datasets = []
    for item in collection_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            # Check if it looks like a dataset directory (has trial subdirs or experiment.parquet)
            trial_dirs = list(item.glob("train_tune_*"))
            if trial_dirs or (item / "experiment.parquet").exists():
                datasets.append(item)
    return sorted(datasets)


def cmd_post_process(args):
    """Post-process experiments: aggregate → zero-cost proxies → analysis."""
    from analysis.post_processing_pipeline.aggregate_experiment_index import aggregate_experiment_index
    from utils.path_utils import PROJECT_ROOT
    from utils.pipeline_status import (
        mark_aggregation_complete, mark_aggregation_failed,
        mark_zcp_complete, mark_zcp_failed,
        mark_analysis_complete
    )

    # Import optional modules (may fail if dependencies aren't available)
    full_pipeline = None
    if not args.skip_zcp:
        try:
            from analysis.post_processing_pipeline import full_pipeline
        except ImportError as e:
            print(f"⚠️  Warning: Could not import full_pipeline: {e}")
            print("   Zero-cost proxy computation will be skipped")
            args.skip_zcp = True

    # Determine base results directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
        # Resolve relative paths against project root
        if not base_dir.is_absolute():
            base_dir = Path(PROJECT_ROOT) / base_dir
    elif args.config:
        # Load config to get save_dir
        from utils.config import list_available_datasets
        datasets = list_available_datasets(args.config)
        if datasets:
            cfg = load_config(args.config, dataset_name=datasets[0])
            # Resolve relative paths against project root
            save_dir = cfg["save_dir"]
            if not Path(save_dir).is_absolute():
                save_dir = str(Path(PROJECT_ROOT) / save_dir)
            base_dir = Path(save_dir)
        else:
            print("❌ Could not determine save_dir from config")
            return 1
    else:
        # Try to auto-detect from current directory
        base_dir = Path(PROJECT_ROOT) / "results"
        if not base_dir.exists():
            print("❌ Could not find results directory. Specify --config or --base-dir")
            return 1

    # Find collections to process
    collections_to_process = []

    if args.collection:
        # Specific collection specified
        collection_path = base_dir / "experiments" / args.collection
        if not collection_path.exists():
            # Try test directory
            collection_path = base_dir / "experiments" / "test" / args.collection
        if collection_path.exists() and (collection_path / "MANIFEST.yaml").exists():
            collections_to_process = [collection_path]
        else:
            print(f"❌ Collection not found: {args.collection}")
            return 1
    else:
        if args.skip_test:
            all_collections = find_collections(base_dir, test_mode=False)
        else:
            test_collections = find_collections(base_dir, test_mode=True)
            prod_collections = find_collections(base_dir, test_mode=False)
            all_collections = test_collections + prod_collections

        if args.latest:
            if all_collections:
                collections_to_process = [all_collections[0]]  # Most recent
            else:
                print("❌ No collections found")
                return 1
        else:
            collections_to_process = all_collections

    if not collections_to_process:
        print("❌ No experiment collections found")
        return 1

    print(f"📊 Found {len(collections_to_process)} collection(s) to process")

    all_success = True

    # Process each collection
    for collection_dir in collections_to_process:
        print(f"\n{'='*60}")
        print(f"Processing collection: {collection_dir.name}")
        print(f"{'='*60}")

        # Find all datasets in this collection
        datasets = find_datasets_in_collection(collection_dir)
        if not datasets:
            print(f"⚠️  No datasets found in collection {collection_dir.name}")
            continue

        print(f"📊 Found {len(datasets)} dataset(s): {[d.name for d in datasets]}")

        # Step 1: Aggregate each dataset
        if not args.skip_aggregate:
            for dataset_dir in datasets:
                print(f"\n  Aggregating {dataset_dir.name}...")
                success, _ = run_with_error_handling(
                    f"Aggregate ({dataset_dir.name})",
                    aggregate_experiment_index,
                    dataset_dir,
                    None  # Use default output path
                )
                if success:
                    # Mark aggregation complete for this dataset
                    parquet_path = dataset_dir / "experiment.parquet"
                    if parquet_path.exists():
                        mark_aggregation_complete(collection_dir, parquet_path)
                    else:
                        mark_aggregation_failed(collection_dir)
                else:
                    mark_aggregation_failed(collection_dir)
                    all_success = False
        else:
            print("\n⏭️  Skipping aggregation step (--skip-aggregate)")

        # Step 2: Zero-cost proxy computation (optional)
        if not args.skip_zcp and full_pipeline is not None:
            print(f"\n  Computing zero-cost proxies...")
            try:
                # Pass collection_dir so it uses collection-based paths instead of legacy DATASET_CONFIGS
                success, _ = run_with_error_handling("Zero-Cost Proxies", full_pipeline.main, str(collection_dir))
                if success:
                    # Mark ZCP complete - for now we don't track specific output paths
                    mark_zcp_complete(collection_dir, [])
                else:
                    mark_zcp_failed(collection_dir)
                    print("⚠️  Zero-cost proxy computation had issues (continuing)")
            except Exception as e:
                mark_zcp_failed(collection_dir)
                print(f"⚠️  Zero-cost proxy computation failed (continuing): {e}")
        elif args.skip_zcp:
            print("\n⏭️  Skipping zero-cost proxy computation (--skip-zcp)")
        else:
            print("\n⏭️  Skipping zero-cost proxy computation (module not available)")

        # Step 3: Analysis. The paper's numbers and Figure 1 are produced by the
        # `reproduce` package from the pickles this pipeline just wrote, so the
        # CLI only points at it rather than plotting anything itself.
        if not args.skip_analysis:
            print("\n  Aggregation and zero-cost proxies are done. To turn them into")
            print("  the paper's tables and Figure 1:")
            print(f"    python -m reproduce.export_artifacts   # refresh paper_artifacts/ from {collection_dir}")
            print("    python -m reproduce.run")
            mark_analysis_complete(collection_dir, [])
        else:
            print("\n⏭️  Skipping analysis step (--skip-analysis)")

    return 0 if all_success else 1


def cmd_explore(args):
    """Explore experiment collections and view statistics."""
    from utils.experiment_explorer import (
        get_all_collections, get_collection_stats, get_experiment_details,
        format_collections_table, format_experiments_table
    )
    from utils.path_utils import PROJECT_ROOT

    # Determine base results directory
    if hasattr(args, 'base_dir') and args.base_dir:
        base_dir = Path(args.base_dir)
        if not base_dir.is_absolute():
            base_dir = Path(PROJECT_ROOT) / base_dir
    else:
        base_dir = Path(PROJECT_ROOT) / "results"

    if not base_dir.exists():
        print(f"❌ Results directory not found: {base_dir}")
        return 1

    # Handle different subcommands
    if args.explore_command == "collections":
        # List all collections with stats
        collections = get_all_collections(base_dir)
        if not collections:
            print("📭 No experiment collections found")
            return 0

        # Get stats for each collection
        collection_stats = []
        for collection_dir in collections:
            stats = get_collection_stats(collection_dir)
            if stats:
                collection_stats.append(stats)

        # Format and display table
        df = format_collections_table(collection_stats)
        if df.empty:
            print("📭 No collection data available")
        else:
            print(f"\n📊 Found {len(collection_stats)} experiment collection(s):\n")
            # Use pandas to display nicely
            with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                                 'display.width', None, 'display.max_colwidth', 50):
                print(df.to_string(index=False))

    elif args.explore_command == "collection":
        # Show details of a specific collection
        collection_name = args.collection_name

        # Find the collection
        collection_path = base_dir / "experiments" / collection_name
        if not collection_path.exists():
            collection_path = base_dir / "experiments" / "test" / collection_name

        if not collection_path.exists() or not (collection_path / "MANIFEST.yaml").exists():
            print(f"❌ Collection not found: {collection_name}")
            return 1

        # Get collection stats
        stats = get_collection_stats(collection_path)
        if not stats:
            print(f"❌ Could not read collection data: {collection_name}")
            return 1

        print(f"\n📁 Collection: {stats['collection_id']}")
        print(f"📍 Path: {stats['collection_path']}")
        print(f"🏷️  Tag: {stats.get('exp_tag', 'None')}")
        print(f"📝 Notes: {stats.get('exp_note', 'None')}")
        print(f"📅 Date: {stats.get('timestamp', 'Unknown')}")
        print(f"🧪 Test Mode: {'Yes' if stats.get('test_mode') else 'No'}")
        print(f"🔧 Git Commit: {stats.get('git_commit', 'Unknown')[:7] if stats.get('git_commit') else 'Unknown'}")
        print(f"📊 Datasets: {stats['num_datasets']} ({', '.join(stats['datasets'])})")
        print(f"🤖 Total Models: {stats['total_models']}")
        print(f"✅ Completed Trials: {stats['completed_trials']}")
        print(f"📅 Date Range: {stats.get('date_range', 'Unknown')}")
        print(f"🔄 Aggregation: {stats['aggregation_status']}")
        print(f"🧮 ZCP: {stats['zcp_status']}")
        print(f"📈 Analysis: {stats['analysis_status']}")

    elif args.explore_command == "experiments":
        # List experiments in a collection
        collection_name = args.collection_name

        # Find the collection
        collection_path = base_dir / "experiments" / collection_name
        if not collection_path.exists():
            collection_path = base_dir / "experiments" / "test" / collection_name

        if not collection_path.exists() or not (collection_path / "MANIFEST.yaml").exists():
            print(f"❌ Collection not found: {collection_name}")
            return 1

        # Get experiments for each dataset in the collection
        experiments = []
        for dataset_name in get_collection_stats(collection_path)["datasets"]:
            exp_details = get_experiment_details(collection_path, dataset_name)
            if exp_details:
                experiments.append(exp_details)

        if not experiments:
            print(f"📭 No experiments found in collection {collection_name}")
            return 0

        # Format and display table
        df = format_experiments_table(experiments)
        print(f"\n📊 Experiments in collection '{collection_name}':\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                             'display.width', None, 'display.max_colwidth', 50):
            print(df.to_string(index=False))

    elif args.explore_command == "status":
        # Show pipeline status for all collections
        collections = get_all_collections(base_dir)
        if not collections:
            print("📭 No experiment collections found")
            return 0

        print(f"\n📊 Pipeline Status for {len(collections)} collection(s):\n")

        for collection_dir in collections:
            from utils.pipeline_status import get_pipeline_status
            status = get_pipeline_status(collection_dir)

            print(f"📁 {collection_dir.name}")
            print(f"  🔄 Aggregation: {status['aggregation']['status']}")
            print(f"  🧮 ZCP: {status['zero_cost_proxies']['status']}")
            print(f"  📈 Analysis: {status['analysis']['status']}")
            print()

    return 0


def cmd_migrate_results(args):
    """Migrate orphaned experiments to proper collection structure."""
    from utils.migrate_results import ensure_collection_structure
    from utils.path_utils import PROJECT_ROOT

    # Determine base results directory
    if hasattr(args, 'base_dir') and args.base_dir:
        base_dir = Path(args.base_dir)
        if not base_dir.is_absolute():
            base_dir = Path(PROJECT_ROOT) / base_dir
    else:
        base_dir = Path(PROJECT_ROOT) / "results"

    if not base_dir.exists():
        print(f"❌ Results directory not found: {base_dir}")
        return 1

    if args.dry_run:
        from utils.migrate_results import find_orphaned_experiments
        orphaned = find_orphaned_experiments(base_dir)
        if orphaned:
            print(f"📦 Would migrate {len(orphaned)} orphaned experiment(s):")
            for exp in orphaned:
                print(f"  {exp}")
        else:
            print("✅ No orphaned experiments found")
        return 0

    # Run migration
    results = ensure_collection_structure(base_dir)

    print("\n📊 Migration Summary:")
    print(f"  Orphaned experiments found: {results['orphaned_found']}")
    print(f"  Successfully migrated: {results['migrated']}")
    print(f"  Errors: {results['errors']}")

    if results['collections_created']:
        print(f"  New collections created: {len(results['collections_created'])}")
        for collection in results['collections_created']:
            print(f"    - {Path(collection).name}")

    return 0 if results['errors'] == 0 else 1


def find_collections_with_same_config(args, datasets_to_run):
    """
    Find existing collections with the same configuration as the current run.

    Args:
        args: CLI arguments
        datasets_to_run: List of dataset names being run

    Returns:
        List of tuples: (collection_path, manifest_data) for matching collections
    """
    import yaml
    from pathlib import Path
    from utils.path_utils import PROJECT_ROOT

    # Get base directory
    base_dir_arg = getattr(args, 'base_dir', None)
    if base_dir_arg:
        base_dir = Path(base_dir_arg)
        if not base_dir.is_absolute():
            base_dir = Path(PROJECT_ROOT) / base_dir
    else:
        base_dir = Path(PROJECT_ROOT) / "results"

    # Load current config to compare against
    try:
        current_cfg = load_config(args.config, dataset_name=datasets_to_run[0] if datasets_to_run else None)
    except Exception as e:
        print(f"⚠️  Could not load config for comparison: {e}")
        return []

    # Find all collections
    matching_collections = []
    for test_mode in [False, True]:
        collections_dir = base_dir / "experiments"
        if test_mode:
            collections_dir = collections_dir / "test"

        if not collections_dir.exists():
            continue

        for collection_dir in collections_dir.iterdir():
            if not collection_dir.is_dir():
                continue

            manifest_path = collection_dir / "MANIFEST.yaml"
            config_path = collection_dir / "unified_config.yml"

            if not manifest_path.exists() or not config_path.exists():
                continue

            try:
                # Read manifest
                with open(manifest_path, 'r') as f:
                    manifest_data = yaml.safe_load(f) or {}

                # Read and compare config
                with open(config_path, 'r') as f:
                    saved_config = yaml.safe_load(f) or {}

                # Compare key configuration elements
                if configs_match(current_cfg, saved_config):
                    matching_collections.append((collection_dir, manifest_data))

            except Exception as e:
                print(f"⚠️  Could not read collection {collection_dir.name}: {e}")
                continue

    return matching_collections


def configs_match(config1, config2, ignore_keys=None):
    """
    Compare two configs to see if they are functionally equivalent.

    Args:
        config1, config2: Config dictionaries to compare
        ignore_keys: Keys to ignore in comparison (e.g., timestamps, notes)

    Returns:
        True if configs match, False otherwise
    """
    if ignore_keys is None:
        ignore_keys = ['timestamp', 'exp_note', 'git_commit', 'experiment_id']

    def normalize_config(config):
        """Normalize config for comparison by removing ignored keys and sorting."""
        if not isinstance(config, dict):
            return config

        normalized = {}
        for key, value in config.items():
            if key not in ignore_keys:
                if isinstance(value, dict):
                    normalized[key] = normalize_config(value)
                elif isinstance(value, list):
                    # Sort lists for consistent comparison
                    try:
                        normalized[key] = sorted(value)
                    except TypeError:
                        # Can't sort mixed types, keep as-is
                        normalized[key] = value
                else:
                    normalized[key] = value
        return normalized

    return normalize_config(config1) == normalize_config(config2)


def cmd_pipeline(args):
    """Run full pipeline: experiment → aggregate → analysis."""
    from tune_runner import main as tune_main
    from analysis.post_processing_pipeline.aggregate_experiment_index import aggregate_experiment_index
    from analysis.post_processing_pipeline import full_pipeline

    # Determine which datasets to run
    datasets_to_run = []
    if args.variant:
        if args.variant == "all":
            datasets_to_run = list_available_datasets(args.config, variant=None)
        else:
            datasets_to_run = list_available_datasets(args.config, variant=args.variant)
        if not datasets_to_run:
            print(f"❌ No datasets found with variant '{args.variant}'")
            return 1
        print(f"📊 Running pipeline for {len(datasets_to_run)} datasets with variant '{args.variant}': {', '.join(datasets_to_run)}")
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        print("❌ Either --dataset or --variant must be specified")
        return 1
    
    auto_skip_note = (hasattr(args, "yes") and args.yes) or (
        hasattr(args, "no_note") and args.no_note
    )
    # Handle experiment note
    exp_note = None
    if hasattr(args, 'note') and args.note:
        exp_note = args.note
    elif not auto_skip_note:
        # Prompt for note
        print("\n📝 Experiment Note")
        print("Enter a brief note describing this experiment (optional):")
        try:
            exp_note = input("> ").strip()
            if not exp_note:
                exp_note = None
        except (EOFError, KeyboardInterrupt):
            print("\n⏭️  Skipping note input")
            exp_note = None

    if exp_note:
        print(f"📝 Note: {exp_note}")
    else:
        print("📝 No note provided")

    # Check for existing collections with same configuration
    existing_collections = find_collections_with_same_config(args, datasets_to_run)
    if existing_collections:
        print(f"\n⚠️  Found {len(existing_collections)} existing collection(s) with same configuration:")
        for i, (collection_path, manifest_data) in enumerate(existing_collections, 1):
            timestamp = manifest_data.get('timestamp', 'Unknown')[:19]  # Show date/time
            note = manifest_data.get('exp_note', 'No note')
            print(f"  {i}. {collection_path.name} ({timestamp}) - {note}")

        print("\nWhat would you like to do?")
        print("  [o] Overwrite existing collection(s)")
        print("  [u] Update existing collection(s) (add new datasets if any)")
        print("  [n] Create new collection")
        print("  [c] Cancel")

        while True:
            try:
                choice = input("> ").strip().lower()
                if choice in ['o', 'overwrite']:
                    print("🗑️  Overwriting existing collection(s)...")
                    # Remove existing collections
                    for collection_path, _ in existing_collections:
                        import shutil
                        shutil.rmtree(collection_path)
                        print(f"  Removed: {collection_path}")
                    break
                elif choice in ['u', 'update']:
                    print("🔄 Updating existing collection(s)...")
                    # For now, we'll just proceed - the logic will handle adding to existing collections
                    break
                elif choice in ['n', 'new']:
                    print("🆕 Creating new collection...")
                    break
                elif choice in ['c', 'cancel']:
                    print("❌ Cancelled")
                    return 1
                else:
                    print("Invalid choice. Please enter 'o', 'u', 'n', or 'c'")
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Cancelled")
                return 1

    # Calculate trial counts for all datasets upfront and get user confirmation once
    user_confirmed = False
    test_mode = False
    custom_params = {}
    shared_exp_id = None  # Shared experiment ID for all datasets in this pipeline run
    
    if not args.skip_experiment:
        from utils.trial_diagnostics import (
            calculate_trial_counts, 
            display_combined_trial_summary, 
            prompt_combined_confirmation
        )
        from utils.experiment_config import ExperimentConfig
        from utils.experiment_metadata import generate_experiment_id
        
        test_mode = bool(
            (hasattr(args, "test") and args.test)
            or (hasattr(args, "num_test_models") and args.num_test_models)
        )
        
        if hasattr(args, 'num_models') and args.num_models is not None:
            custom_params['num_models'] = args.num_models
        if hasattr(args, 'num_epochs') and args.num_epochs is not None:
            custom_params['num_epochs'] = args.num_epochs
        
        # Generate a single experiment ID for all datasets in this pipeline run
        shared_exp_id = generate_experiment_id(test_mode=test_mode)
        logging.info(f"📋 Using shared experiment ID for all datasets: {shared_exp_id}")
        
        # Calculate trial counts for each dataset
        dataset_summaries = []
        for dataset_name in datasets_to_run:
            cfg = load_config(args.config, dataset_name=dataset_name)
            
            if custom_params:
                if 'num_models' in custom_params:
                    cfg["experiments"]["num_models"] = custom_params['num_models']
                if 'num_epochs' in custom_params:
                    cfg["experiments"]["hyperparameters"]["num_epochs"] = custom_params['num_epochs']
            
            exp_cfg = ExperimentConfig(cfg["experiments"], cfg["seed"], test_mode=test_mode)
            ds_cls = dataset_registry.get_dataset(dataset_name)
            param_space = exp_cfg.build_search_space(
                input_size=int(ds_cls.config["input_size"]),
                num_classes=int(ds_cls.config["num_classes"]),
            )
            total_trials, model_type_breakdown = calculate_trial_counts(param_space, exp_cfg)
            dataset_summaries.append((dataset_name, total_trials, model_type_breakdown))
        
        # Display combined summary and get confirmation once
        if dataset_summaries:
            display_combined_trial_summary(dataset_summaries, test_mode=test_mode)
            total_all_trials = sum(trials for _, trials, _ in dataset_summaries)
            
            # Check if -y/--yes flag was passed
            if hasattr(args, 'yes') and args.yes:
                print(f"\n✅ Auto-confirming: Proceeding with {total_all_trials} total trial(s) across {len(datasets_to_run)} dataset(s)")
                user_confirmed = True
            else:
                user_confirmed = prompt_combined_confirmation(total_all_trials, len(datasets_to_run))
            
            if not user_confirmed:
                print("\n❌ Experiments cancelled by user")
                return 1
    
    # For pipeline, we'll run each dataset sequentially
    # Note: This is a simplified approach - in the future, we might want to parallelize
    all_success = True
    for dataset_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Running pipeline for dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load config to extract experiment directory
        cfg = load_config(args.config, dataset_name=dataset_name)
        save_dir = Path(cfg["save_dir"])
        # Ensure save_dir is absolute
        if not save_dir.is_absolute():
            save_dir = Path(PROJECT_ROOT) / save_dir
        
        # With new experiment organization, exp_dir should be provided or we try to find it
        if args.exp_dir:
            exp_dir = Path(args.exp_dir)
            # If explicitly provided and skipping experiment, verify it exists
            if args.skip_experiment and not exp_dir.exists():
                print(
                    f"\n❌ ERROR: Specified experiment directory does not exist: {exp_dir}\n"
                    f"   Cannot run analysis without experiment data.\n"
                    f"   Please either:\n"
                    f"   1. Run the experiment first (remove --skip-experiment)\n"
                    f"   2. Provide a valid --exp-dir pointing to existing experiment directory"
                )
                return 1
        else:
            # Try to find the most recent experiment directory in the new structure
            # Look in both test and production locations
            test_experiments_dir = save_dir / "experiments" / "test"
            prod_experiments_dir = save_dir / "experiments"
            
            # Find most recent collection directory
            exp_dir = None
            for base_dir in [test_experiments_dir, prod_experiments_dir]:
                if base_dir.exists():
                    collections = [d for d in base_dir.iterdir() if d.is_dir()]
                    if collections:
                        # Get most recently modified
                        latest_collection = max(collections, key=lambda p: p.stat().st_mtime)
                        # New structure: collection/dataset_name (Ray Tune stores directly under collection)
                        dataset_exp_dir = latest_collection / dataset_name
                        if dataset_exp_dir.exists():
                            exp_dir = dataset_exp_dir
                            break
            
            if exp_dir is None:
                # Fallback to old structure for backward compatibility
                exp_dir = save_dir / f"{dataset_name}_grid"
                if not exp_dir.exists():
                    # If skipping experiment, we need the directory to exist for analysis
                    if args.skip_experiment:
                        print(
                            f"\n❌ ERROR: Experiment directory not found and --skip-experiment is set.\n"
                            f"   Cannot run analysis without experiment data.\n"
                            f"   Please either:\n"
                            f"   1. Run the experiment first (remove --skip-experiment)\n"
                            f"   2. Specify --exp-dir pointing to existing experiment directory\n"
                            f"   Expected structure: <save_dir>/experiments/<collection_id>/<dataset>/"
                        )
                        return 1
                    else:
                        # Experiment will run first and create the directory - this is expected
                        print(
                            f"ℹ️  Experiment directory not found yet (will be created when experiment runs).\n"
                            f"   Expected structure: <save_dir>/experiments/<collection_id>/<dataset>/"
                        )
        
        exp_dir = Path(exp_dir)
        
        results = {
            "experiment": False,
            "aggregate": False,
            "post_processing": False,
            "analysis": False
        }
        
        # Step 1: Run Experiment
        if not args.skip_experiment:
            # test_mode and custom_params are already set above
            # Just display info if not already shown
            if test_mode and not user_confirmed:
                print("\n🧪 TEST MODE: Writing under experiments/test/ (trial counts unchanged from config)")
            if custom_params and not user_confirmed:  # Only show if we didn't show combined summary
                param_str = ", ".join([f"{k}={v}" for k, v in custom_params.items()])
                print(f"\n⚙️  Custom parameters: {param_str}")

            reset = hasattr(args, 'reset') and args.reset
            retry_errored = bool(getattr(args, "retry_errored", False))
            ray_address = getattr(args, "ray_address", None)
            # Use auto_confirm=True since we already got confirmation upfront
            # Pass shared_exp_id so all datasets use the same collection directory
            success, _ = run_with_error_handling(
                f"Experiment ({dataset_name})",
                tune_main,
                args.config,
                test_mode,
                dataset_name,
                reset,
                exp_note,
                custom_params,
                True if user_confirmed else False,  # auto_confirm - use True if we already confirmed
                False,  # extract_arch_configs
                shared_exp_id,  # exp_id - shared across all datasets
                restart_errored=retry_errored,
                ray_address=ray_address,
            )
            results["experiment"] = success
            
            # Update exp_dir to point to the actual collection directory that was created
            if success and shared_exp_id:
                from utils.experiment_metadata import get_collection_dir
                exp_tag = cfg.get("exp_tag")
                collection_dir_path = get_collection_dir(save_dir, shared_exp_id, exp_tag, test_mode)
                # Ensure absolute path
                if not collection_dir_path.is_absolute():
                    collection_dir_path = Path(PROJECT_ROOT) / collection_dir_path
                exp_dir = collection_dir_path / dataset_name
                logging.info(f"📁 Updated exp_dir to actual collection directory: {exp_dir}")
            
            # Stop pipeline if experiment failed
            if not success:
                print(f"\n❌ Experiment failed for {dataset_name}. Stopping pipeline for this dataset.")
                # Skip remaining steps for this dataset
                results["aggregate"] = None
                results["post_processing"] = None
                results["analysis"] = None
                all_success = False
                continue  # Skip to next dataset
        else:
            print("\n⏭️  Skipping experiment step (--skip-experiment)")
            results["experiment"] = None
        
        # Step 2: Aggregate Results
        if not args.skip_aggregate:
            if not exp_dir.exists():
                print(f"⚠️  Experiment directory not found: {exp_dir}")
                print("   Skipping aggregation step")
                results["aggregate"] = False
            else:
                success, _ = run_with_error_handling(
                    f"Aggregate ({dataset_name})",
                    aggregate_experiment_index,
                    exp_dir,
                    None  # Use default output path
                )
                results["aggregate"] = success
        else:
            print("\n⏭️  Skipping aggregation step (--skip-aggregate)")
            results["aggregate"] = None
        
        # Step 3: Run Post-Processing (optional, for zero-cost metrics)
        # Note: This step may fail if zero-cost dependencies aren't available
        # We'll make it optional and continue on failure
        # Get collection directory - use shared_exp_id if available, otherwise derive from exp_dir
        if shared_exp_id and not args.skip_experiment:
            from utils.experiment_metadata import get_collection_dir
            exp_tag = cfg.get("exp_tag")
            collection_dir_path = get_collection_dir(save_dir, shared_exp_id, exp_tag, test_mode)
            # Ensure absolute path
            if not collection_dir_path.is_absolute():
                collection_dir_path = Path(PROJECT_ROOT) / collection_dir_path
            collection_dir = collection_dir_path
        else:
            # Fallback: derive from exp_dir (should be collection/dataset)
            if exp_dir.exists():
                collection_dir_path = exp_dir.parent  # Go up from dataset dir to collection dir
                # Ensure absolute path
                if not collection_dir_path.is_absolute():
                    collection_dir_path = Path(PROJECT_ROOT) / collection_dir_path
                collection_dir = collection_dir_path
            else:
                # If exp_dir doesn't exist or is old structure, skip post-processing
                print(f"⚠️  Cannot determine collection directory from exp_dir: {exp_dir}")
                print("   Skipping post-processing step")
                results["post_processing"] = False
                collection_dir = None
        
        if collection_dir and collection_dir.exists():
            success, _ = run_with_error_handling("Post-Processing", full_pipeline.main, str(collection_dir))
            results["post_processing"] = success
        else:
            print(f"⚠️  Collection directory not found: {collection_dir}")
            print("   Skipping post-processing step")
            success, _ = run_with_error_handling("Post-Processing", full_pipeline.main, str(collection_dir))
        parquet_path = exp_dir / "experiment.parquet"
        if not parquet_path.exists():
            print(f"⚠️  experiment.parquet not found: {parquet_path}")
            print("   Analysis may fail or produce incomplete results")

        # Get collection directory - use shared_exp_id if available, otherwise derive from exp_dir
        if shared_exp_id and not args.skip_experiment:
            from utils.experiment_metadata import get_collection_dir
            exp_tag = cfg.get("exp_tag")
            collection_dir_path = get_collection_dir(save_dir, shared_exp_id, exp_tag, test_mode)
            # Ensure absolute path
            if not collection_dir_path.is_absolute():
                collection_dir_path = Path(PROJECT_ROOT) / collection_dir_path
            collection_dir = collection_dir_path
        else:
            # Fallback: derive from exp_dir (should be collection/dataset)
            if exp_dir.exists():
                collection_dir_path = exp_dir.parent  # Go up from dataset dir to collection dir
                # Ensure absolute path
                if not collection_dir_path.is_absolute():
                    collection_dir_path = Path(PROJECT_ROOT) / collection_dir_path
                collection_dir = collection_dir_path
            else:
                collection_dir = None
        
        if collection_dir and collection_dir.exists():
            # The paper's tables and Figure 1 come from the `reproduce` package,
            # which reads the pickles post-processing just wrote.
            print("\nTo turn these results into the paper's tables and Figure 1:")
            print(f"    python -m reproduce.export_artifacts   # refresh paper_artifacts/ from {collection_dir}")
            print("    python -m reproduce.run")
            results["analysis"] = True
        else:
            print(f"⚠️  Collection directory not found: {collection_dir}")
            print("   Skipping analysis step")
            results["analysis"] = False
        
        # Print summary for this dataset
        print("\n" + "="*60)
        print(f"Pipeline Summary for {dataset_name}:")
        for step, success in results.items():
            if success is None:
                status = "⏭️  Skipped"
            elif success:
                status = "✅ Completed"
            else:
                status = "❌ Failed"
            print(f"  {status} {step.replace('_', ' ').title()}")
        print("="*60)
        
        # Track if any step failed
        failed_steps = [step for step, success in results.items() 
                       if success is not None and not success]
        if failed_steps:
            all_success = False
    
    return 0 if all_success else 1


def main():
    parser = argparse.ArgumentParser(
        description="Models Got Talent CLI - Unified interface for experiments and analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to unified config (configs/experiments/unified_config.yml)"
    )
    run_parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (required if --variant not specified)"
    )
    run_parser.add_argument(
        "--variant",
        choices=["original", "imbalance", "all"],
        default=None,
        help="Filter datasets by variant: 'original' (normal datasets), 'imbalance' (imbalanced variants), or 'all' (all variants). If specified, runs all matching datasets."
    )
    run_parser.add_argument(
        "--test",
        action="store_true",
        help="Write results under experiments/test/ (trial counts still come from YAML)",
    )
    run_parser.add_argument(
        "--num-test-models",
        type=int,
        default=None,
        help="Alias for test layout: any positive value enables experiments/test/ (does not change num_models)",
    )
    run_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/restart experiment: don't resume from existing checkpoints, start fresh (existing experiment will be backed up)"
    )
    run_parser.add_argument(
        "--retry-errored",
        action="store_true",
        help="On resume only: re-run Tune trials that finished with ERROR",
    )
    run_parser.add_argument(
        "--exp-id",
        default=None,
        metavar="ID",
        help=(
            "Reuse an existing experiment/collection ID (MANIFEST.yaml 'experiment_id') so Ray Tune resumes "
            "instead of starting a new dated folder"
        ),
    )
    run_parser.add_argument(
        "--no-note",
        action="store_true",
        help="Skip prompting for experiment note (redundant if --yes)",
    )
    run_parser.add_argument(
        "--note",
        help="Experiment note (overrides prompting)"
    )
    run_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip interactive prompts where supported (experiment note, trial-count confirm); non-interactive run",
    )
    run_parser.add_argument(
        "--ray-address",
        default=None,
        metavar="ADDR",
        help=(
            "Address of an existing Ray cluster to attach to (e.g. 'auto', "
            "'ray://head:10001', '<head_ip>:6379'). Use this when running on a "
            "multi-node Ray cluster. Falls back to the "
            "RAY_ADDRESS env var; if neither is set, a local Ray instance is started."
        ),
    )

    # aggregate command
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate experiment results")
    agg_parser.add_argument(
        "--exp-dir",
        required=True,
        help="Experiment directory with trial subdirs"
    )
    agg_parser.add_argument(
        "--out",
        help="Output parquet path (default: <exp_dir>/experiment.parquet)"
    )
    
    # post-process command
    post_process_parser = subparsers.add_parser(
        "post-process",
        help="Post-process experiments: aggregate → zero-cost proxies → analysis"
    )
    post_process_parser.add_argument(
        "--config",
        help="Path to unified config (to determine save_dir). Optional if --base-dir is provided."
    )
    post_process_parser.add_argument(
        "--base-dir",
        help="Base results directory (default: results/). Optional if --config is provided."
    )
    post_process_parser.add_argument(
        "--collection",
        help="Specific collection name to process (e.g., '2026-01-10__4cc174'). If not specified, processes all collections."
    )
    post_process_parser.add_argument(
        "--latest",
        action="store_true",
        help="Process only the most recent collection"
    )
    post_process_parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregation step"
    )
    post_process_parser.add_argument(
        "--skip-zcp",
        action="store_true",
        help="Skip zero-cost proxy computation"
    )
    post_process_parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis/plotting step"
    )
    post_process_parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test collections (only process production collections)"
    )
    # explore command
    explore_parser = subparsers.add_parser(
        "explore",
        help="Explore experiment collections and view statistics"
    )
    explore_parser.add_argument(
        "--base-dir",
        help="Base results directory (default: results/)"
    )

    # Subcommands for explore
    explore_subparsers = explore_parser.add_subparsers(dest="explore_command", help="Explore subcommand")

    # explore collections
    explore_subparsers.add_parser("collections", help="List all collections with statistics")

    # explore collection <name>
    collection_parser = explore_subparsers.add_parser("collection", help="Show details of a specific collection")
    collection_parser.add_argument("collection_name", help="Collection name (e.g., '2026-01-10__4cc174')")

    # explore experiments <name>
    experiments_parser = explore_subparsers.add_parser("experiments", help="List experiments in a collection")
    experiments_parser.add_argument("collection_name", help="Collection name")

    # explore status
    explore_subparsers.add_parser("status", help="Show pipeline status for all collections")

    # migrate-results command
    migrate_parser = subparsers.add_parser(
        "migrate-results",
        help="Migrate orphaned experiments into proper collection structure"
    )
    migrate_parser.add_argument(
        "--base-dir",
        help="Base directory containing results/ (default: auto-detect)"
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it"
    )
    
    # pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: experiment → aggregate → analysis"
    )
    pipeline_parser.add_argument(
        "--config",
        required=True,
        help="Path to unified config (configs/experiments/unified_config.yml)"
    )
    pipeline_parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (required if --variant not specified)"
    )
    pipeline_parser.add_argument(
        "--variant",
        choices=["original", "imbalance", "all"],
        default=None,
        help="Filter datasets by variant: 'original' (normal datasets), 'imbalance' (imbalanced variants), or 'all' (all variants). If specified, runs pipeline for all matching datasets."
    )
    pipeline_parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Skip experiment step (assume already run)"
    )
    pipeline_parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregation step"
    )
    pipeline_parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis step"
    )
    pipeline_parser.add_argument(
        "--exp-dir",
        help="Experiment directory (auto-detected from config if not provided)"
    )
    pipeline_parser.add_argument(
        "--test",
        action="store_true",
        help="Write results under experiments/test/ (trial counts still come from YAML)",
    )
    pipeline_parser.add_argument(
        "--num-test-models",
        type=int,
        default=None,
        help="Alias for test layout: any positive value enables experiments/test/ (does not change num_models)",
    )
    pipeline_parser.add_argument(
        "--num-models",
        type=int,
        default=None,
        help="Override experiments.num_models (must still equal sum of model_type_counts)"
    )
    pipeline_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config num_epochs; must exist in hyperparameters)"
    )
    pipeline_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/restart experiment: don't resume from existing checkpoints, start fresh (existing experiment will be backed up)"
    )
    pipeline_parser.add_argument(
        "--retry-errored",
        action="store_true",
        help="On resume only: re-run Tune trials that finished with ERROR",
    )
    pipeline_parser.add_argument(
        "--no-note",
        action="store_true",
        help="Skip prompting for experiment note (redundant if --yes)",
    )
    pipeline_parser.add_argument(
        "--note",
        help="Experiment note (overrides prompting)"
    )
    pipeline_parser.add_argument(
        "--extract-arch-configs",
        action="store_true",
        help="Extract and save architecture configurations DataFrame to arch_configs.parquet"
    )
    pipeline_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip interactive prompts where supported (experiment note, trial-count confirm); non-interactive run",
    )
    pipeline_parser.add_argument(
        "--ray-address",
        default=None,
        metavar="ADDR",
        help=(
            "Address of an existing Ray cluster to attach to (e.g. 'auto', "
            "'ray://head:10001', '<head_ip>:6379'). Use this when running on a "
            "multi-node Ray cluster. Falls back to the "
            "RAY_ADDRESS env var; if neither is set, a local Ray instance is started."
        ),
    )

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Validate that either --dataset or --variant is provided for run and pipeline commands
    if args.command in ["run", "pipeline"]:
        if not args.dataset and not args.variant:
            print("❌ Error: Either --dataset or --variant must be specified")
            if args.command == "run":
                run_parser.print_help()
            else:
                pipeline_parser.print_help()
            return 1

    # Validate that explore subcommand is provided
    if args.command == "explore":
        if not hasattr(args, 'explore_command') or not args.explore_command:
            print("❌ Error: Explore subcommand must be specified")
            explore_parser.print_help()
            return 1
    
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "aggregate":
        return cmd_aggregate(args)
    elif args.command == "pipeline":
        return cmd_pipeline(args)
    elif args.command == "post-process":
        return cmd_post_process(args)
    elif args.command == "explore":
        return cmd_explore(args)
    elif args.command == "migrate-results":
        return cmd_migrate_results(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

