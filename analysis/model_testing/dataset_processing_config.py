"""Where each experiment's results live on disk.

Nothing here is needed to reproduce the paper: the paper's numbers and figure are
regenerated from the committed `paper_artifacts/` by `python -m reproduce.run`.
This module is for re-running the pipeline end to end, and it points at whichever
tree the training runs wrote to.

Paths are resolved from environment variables so the same code works on a
different machine:

    MGT_ROOT              project checkout (default: auto-detected)
    MGT_RESULTS_ROOT      where `mgt_cli run` writes (default: $MGT_ROOT/results)
    MGT_PAPER_RESULTS_ROOT
                          the run the paper reports, containing `results/experiments/`
                          and `post-processed-results/` (default:
                          $MGT_RESULTS_ROOT/paper_run)
"""

import os
from pathlib import Path

from utils.path_utils import get_project_root

# Get base directory from env or auto-detect
MGT_ROOT = Path(os.environ.get("MGT_ROOT", get_project_root()))
MGT_RESULTS_ROOT = Path(os.environ.get("MGT_RESULTS_ROOT", MGT_ROOT / "results"))

# The training run the paper reports on. Set MGT_PAPER_RESULTS_ROOT to the copy of
# that tree on your machine before re-running post-processing.
PAPER_RESULTS_ROOT = Path(
    os.environ.get("MGT_PAPER_RESULTS_ROOT", MGT_RESULTS_ROOT / "paper_run")
)

BASE_DIR = MGT_ROOT
EXPERIMENTS_PATH = PAPER_RESULTS_ROOT / "results" / "experiments"
POST_PROCESSED_PATH = PAPER_RESULTS_ROOT / "post-processed-results"

# Tune collection that trained each dataset's 2,000 sampled architectures.
HHAR_EXP = EXPERIMENTS_PATH / "2026-05-12_21-04-50__3cbd25"
MOTIONSENSE_EXP = EXPERIMENTS_PATH / "2026-05-12_20-29-49__ade8f6"
MOBIACTV2_EXP = EXPERIMENTS_PATH / "2026-05-12_21-04-37__9b8d3b"
PAMAP2_EXP = EXPERIMENTS_PATH / "2026-05-12_20-29-49__303adf"
MYOGYM_EXP = EXPERIMENTS_PATH / "2026-05-12_20-15-35__f3b50c"
RWHAR_EXP = EXPERIMENTS_PATH / "2026-05-12_21-15-52__cbe7ee"

HHAR_RESULTS_PATH = POST_PROCESSED_PATH / "hhar"
MOTIONSENSE_RESULTS_PATH = POST_PROCESSED_PATH / "motionsense"
MOBIACTV2_RESULTS_PATH = POST_PROCESSED_PATH / "mobiactv2"
PAMAP2_RESULTS_PATH = POST_PROCESSED_PATH / "pamap2"
MYOGYM_RESULTS_PATH = POST_PROCESSED_PATH / "myogym"
RWHAR_RESULTS_PATH = POST_PROCESSED_PATH / "rwhar"


DATASET_CONFIGS = {
    "hhar": {
        "results_dir": HHAR_EXP / "hhar",
        "full_results_pkl": HHAR_RESULTS_PATH / "full_results.pkl",
        "zero_cost_pkl": HHAR_RESULTS_PATH / "zero_cost_scores.pkl",
        "model_flops_pkl": HHAR_RESULTS_PATH / "model_flops_hhar.pkl",
        "full_training_metrics_pkl": HHAR_RESULTS_PATH / "full_training_metrics.pkl",
        "data_root": BASE_DIR / "data/all_data/hhar",
    },
    "motionsense": {
        "results_dir": MOTIONSENSE_EXP / "motionsense",
        "full_results_pkl": MOTIONSENSE_RESULTS_PATH / "full_results.pkl",
        "zero_cost_pkl": MOTIONSENSE_RESULTS_PATH / "zero_cost_scores.pkl",
        "model_flops_pkl": MOTIONSENSE_RESULTS_PATH / "model_flops_motionsense.pkl",
        "full_training_metrics_pkl": MOTIONSENSE_RESULTS_PATH / "full_training_metrics.pkl",
        "data_root": BASE_DIR / "data/all_data/motionsense",
    },
    "mobiactv2": {
        "results_dir": MOBIACTV2_EXP / "mobiactv2",
        "full_results_pkl": MOBIACTV2_RESULTS_PATH / "full_results.pkl",
        "zero_cost_pkl": MOBIACTV2_RESULTS_PATH / "zero_cost_scores.pkl",
        "model_flops_pkl": MOBIACTV2_RESULTS_PATH / "model_flops_mobiactv2.pkl",
        "full_training_metrics_pkl": MOBIACTV2_RESULTS_PATH / "full_training_metrics.pkl",
        "data_root": BASE_DIR / "data/all_data/mobiactv2/Jul-27-2025/",
    },
    "myogym": {
        "results_dir": MYOGYM_EXP / "myogym",
        "full_results_pkl": MYOGYM_RESULTS_PATH / "full_results.pkl",
        "zero_cost_pkl": MYOGYM_RESULTS_PATH / "zero_cost_scores.pkl",
        "model_flops_pkl": MYOGYM_RESULTS_PATH / "model_flops_myogym.pkl",
        "full_training_metrics_pkl": MYOGYM_RESULTS_PATH / "full_training_metrics.pkl",
        "data_root": BASE_DIR / "data/all_data/myogym/Jul-27-2025/",
    },
    "pamap2": {
        "results_dir": PAMAP2_EXP / "pamap2",
        "full_results_pkl": PAMAP2_RESULTS_PATH / "full_results.pkl",
        "zero_cost_pkl": PAMAP2_RESULTS_PATH / "zero_cost_scores.pkl",
        "model_flops_pkl": PAMAP2_RESULTS_PATH / "model_flops_pamap2.pkl",
        "full_training_metrics_pkl": PAMAP2_RESULTS_PATH / "full_training_metrics.pkl",
        "data_root": BASE_DIR / "data/all_data/pamap2_full_proc_final/",
    },
    "rwhar": {
        "results_dir": RWHAR_EXP / "rwhar",
        "full_results_pkl": RWHAR_RESULTS_PATH / "full_results.pkl",
        "zero_cost_pkl": RWHAR_RESULTS_PATH / "zero_cost_scores.pkl",
        "model_flops_pkl": RWHAR_RESULTS_PATH / "model_flops_rwhar.pkl",
        "full_training_metrics_pkl": RWHAR_RESULTS_PATH / "full_training_metrics.pkl",
        "data_root": BASE_DIR / "data/all_data/rwhar/Jul-03-2025/",
    },
}

# Label encoding each dataset's preparation script produces. Myogym is omitted:
# its 31 classes are the exercise names carried through from the source release.
LABEL_TO_ACTIVITY = {
    "motionsense": {
        0: "descending stairs",
        1: "ascending stairs",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "jogging",
    },
    "hhar": {
        0: "standing",
        1: "sitting",
        2: "walking",
        3: "ascending stairs",
        4: "descending stairs",
        5: "biking",
    },
    "rwhar": {
        0: "climbing down stairs",
        1: "climbing up stairs",
        2: "jumping",
        3: "lying",
        4: "running",
        5: "sitting",
        6: "standing",
        7: "walking",
    },
    "pamap2": {
        0: "lying",
        1: "sitting",
        2: "standing",
        3: "walking",
        4: "ascending stairs",
        5: "descending stairs",
    },
    "mobiactv2": {
        0: "standing (STD)",
        1: "walking (WAL)",
        2: "jogging (JOG)",
        3: "jumping (JUM)",
        4: "ascending stairs (STU)",
        5: "descending stairs (STN)",
        6: "stand to sit (SCH)",
        7: "sit in chair (SIT)",
        8: "sit to stand (CHU)",
        9: "car step in (CSI)",
        10: "car step out (CSO)",
    },
}
