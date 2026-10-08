# Obtaining and preparing the datasets

**You do not need any of this to reproduce the paper.** The results and Figure 1
are regenerated from the committed `paper_artifacts/` — see
[`reproduce/README.md`](../reproduce/README.md). This document is for re-running
the training pipeline from scratch.

None of the six datasets is redistributed here; each has its own licence and
must be obtained from its authors. Cite the original papers if you use them.

## Where things go

Raw downloads go in `data/raw/<dataset>/` (the default the prep scripts expect,
overridable with `--dataset_loc`). Prepared files go in `data/all_data/`, which
is where `configs/experiments/unified_config.yml` points.

```
data/
├── raw/                                  # downloads, gitignored
│   ├── mobiactv2/MobiAct_Dataset_v2.0/Annotated_Data/
│   ├── myogym/MyoGym.mat
│   ├── pamap2/PAMAP2_Dataset/
│   └── rwhar/
└── all_data/                             # prepared, gitignored
    ├── hhar/hhar_watch_sr_50.pkl
    ├── mobiactv2/Jul-27-2025/mobiactv2_6_sr_50_fold_0.joblib
    ├── motionsense/motionsense.pkl
    ├── myogym/Jul-27-2025/myogym_3_sr_50.joblib
    ├── pamap2_full_proc_final/pamap2_3_sr_50.joblib
    └── rwhar/Jul-03-2025/rwhar_15_sr_50_fold_0.joblib
```

The date-stamped subdirectories are what the prep scripts emit and what the
shipped config expects. If you regenerate them the dates will differ, so update
the matching `data_dir` in `unified_config.yml`.

## Per dataset

| Dataset | Source | Prep |
|---|---|---|
| **HHAR** | [Stisen et al. 2015](https://doi.org/10.1145/2809695.2809718) — [Heterogeneity Activity Recognition, UCI](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | no script in repo, see below |
| **MobiAct v2** | [Chatzaki et al. 2016](https://doi.org/10.1007/978-3-319-62704-5_7) — request from the [BMI Lab, TEI of Crete](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) | `python data/prep_mobiact.py` |
| **MotionSense** | [Malekzadeh et al. 2018](https://doi.org/10.1145/3195258.3195260) — [github.com/mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense) | no script in repo, see below |
| **MyoGym** | [Koskimäki et al. 2017](https://doi.org/10.1145/3123024.3124400) — [MyoGym on Zenodo / author request](http://www.oulu.fi/bisg/node/40364) | `python data/myogym_data_prep.py` |
| **PAMAP2** | [Reiss & Stricker 2012](https://doi.org/10.1109/ISWC.2012.13) — [PAMAP2 Physical Activity Monitoring, UCI](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | `python data/pamap_prep.py` |
| **RealWorld (RWHAR)** | [Sztyler & Stuckenschmidt 2016](https://doi.org/10.1109/PERCOM.2016.7456521) — [RealWorld HAR, Uni Mannheim](https://sensor.informatik.uni-mannheim.de/#dataset_realworld) | `python data/real_world_data_prep.py` |

Each prep script takes `--dataset_loc` (raw input), `--sampling_rate` (50 Hz for
the paper) and writes into a date-stamped directory under `all_data/`. Run
`python data/<script>.py --help` for the full set of options.

RealWorld ships as per-subject zip archives; unzip them first (there is an
`unzip_all()` helper in `data/real_world_data_prep.py`).

### HHAR and MotionSense

These two have no prep script in this repo; the runs used preprocessed files
inherited from earlier work in the group (`hhar_watch_sr_50.pkl` and
`motionsense.pkl`). Both are plain pickles holding a dict of `train` / `val` /
`test` splits, each with windowed accelerometer data and labels — see
`data/hhar_dataset.py` and `data/motionsense_dataset.py` for the exact structure
the loaders expect. To rebuild them from the public downloads you need to apply
the preprocessing described below yourself.

## Preprocessing the paper used

Identical across all six datasets (§4 of the paper):

- **Accelerometer only.** For HHAR, smartwatch data. For RealWorld, five of the
  seven body locations — head, chest, upper arm, waist and shin — because the
  Climbing Up and Jumping activities are missing data at the other two.
- **Downsample to 50 Hz**, the lowest sampling rate across the six datasets.
- **Split by user ID**, randomly assigning users 60% / 20% / 20% to
  train / validation / test. Splits are by participant, never by window, so no
  subject appears in more than one split.
- **Window within each split** after concatenation: 2-second windows with 50%
  overlap.

## Checking a prepared dataset

Each loader has a `__main__` block that loads a split through `DATASET_CONFIGS`
and prints per-class sample counts:

```bash
python -m data.rwhar_dataset
```

The pipeline itself can be smoke-tested without any real data, since the
synthetic `random` dataset stands in for one:

```bash
pytest tests/test_e2e_pipeline.py
```
