# Models Got Talent

Code and data for **Models Got Talent: Identifying High Performing Wearable
Human Activity Recognition Models Without Training**, ISWC '26.
Richard Goldman, Varun Komperla, Thomas Ploetz, Harish Haresamudram.
[arXiv:2511.06157](https://arxiv.org/abs/2511.06157) ·
[doi:10.1145/3830727.3834826](https://doi.org/10.1145/3830727.3834826)

We evaluate eight zero-cost proxies (ZCPs) across six benchmark HAR datasets and
2,000 sampled CNN / RNN / Transformer architectures per dataset, and show that
training only the top-ranked predicted architectures gets within 7% (top-1) and
2% (top-10) of exhaustively training all 2,000.

This is the camera-ready artifact: the reproduction of the paper, the analysis
notebook behind it, the results of the run it reports, and the pipeline that
produced them — and nothing else.

## Reproducing the paper

Every number and the figure in the paper are regenerated from committed data in
about 15 seconds. No datasets, no GPU, no Ray:

```bash
pip install -r requirements-reproduce.txt
python -m reproduce.run
```

This prints Δ₁, Δ₁₀, Δ₁₀% and the talent rate for every proxy and dataset,
checks each of the paper's 16 quantitative claims against the recomputed values,
and writes `analysis/figures/paper/figure1.pdf` (Figure 1 of the paper) alongside
`results.json`. It exits non-zero if any claim fails to reproduce.

Or as a test suite:

```bash
pytest -m "not slow"        # 50 tests, ~20s, reproduces every claim
pytest                      # adds the end-to-end pipeline test (~4 min, trains 2 models)
```

The inputs live in [`paper_artifacts/`](paper_artifacts) — one row per sampled
architecture with its proxy scores at initialisation and its F1 after full
training (576 KB for all six datasets). See
[`reproduce/README.md`](reproduce/README.md) for the metric definitions, sign
conventions, and how to regenerate the artifacts from a full pipeline run.

## The paper's run, as committed

`results/paper_run/` holds the output of the training run the paper reports, so
the analysis works with no configuration and nothing to download:

| Path | What it is |
|---|---|
| `post-processed-results/<dataset>/full_results.pkl` | One row per trained architecture: validation and test F1 at every checkpoint |
| `post-processed-results/<dataset>/zero_cost_scores.pkl` | The eight proxies, scored on each model at initialisation |
| `post-processed-results/<dataset>/model_flops_<dataset>.pkl` | Forward-pass FLOPs per architecture |
| `results/experiments/<collection>/<dataset>/<trial>/params.json` | The sampled architecture config, seeds and initialiser for all 2,000 trials per dataset |

That is 61 MB of pickles and 12,000 trial configs. Two things are deliberately
left out: `full_training_metrics.pkl` (695 MB of per-epoch curves that no result
in the paper depends on) and the noise-robustness results, which are follow-up
work that is not in the paper.

Because this tree sits at the path `MGT_PAPER_RESULTS_ROOT` defaults to,
`python -m reproduce.run --from-pickles` recomputes everything from these
pickles rather than from `paper_artifacts/`, which is the check that the
committed CSVs still match their source.

## The analysis notebook

[`analysis/notebooks/final_paper_cleaned_up.ipynb`](analysis/notebooks/final_paper_cleaned_up.ipynb)
is the notebook the paper's analysis was developed in. It reads the same
`results/paper_run/` pickles and goes beyond `reproduce/`: per-model-type
Spearman correlations, the transformer ablation, parameter counts of the selected
architectures, and the FLOPs breakdown.

Outputs are stripped, so run it top to bottom to populate it:

```bash
pip install -r requirements.txt
jupyter lab analysis/notebooks/final_paper_cleaned_up.ipynb
```

`reproduce/` is the authoritative, tested path to the paper's numbers; the
notebook is the wider exploration around them. One deliberate difference: at the
"top 10%" budget the notebook uses k = 200 for every dataset, while `reproduce`
takes 10% of each dataset's own pool, so k runs 196–200. Both pick the same best
proxy on all six datasets and the talent rates differ by at most 0.55 percentage
points (the notebook's 15.5–49.5% range against 15.6–49.2%).

Everything below concerns re-running the pipeline itself, which needs GPUs and on
the order of a thousand GPU-hours per dataset.

## Project structure

```
models_got_talent/
├── reproduce/                # Self-contained reproduction of the paper
│   ├── metrics.py            #   Δ_k, talent rate, ensemble, random baseline
│   ├── claims.py             #   the paper's 16 quantitative claims as predicates
│   ├── figure1.py            #   Figure 1
│   ├── data.py               #   loads paper_artifacts/, or rebuilds from pipeline pickles
│   ├── export_artifacts.py   #   distils a pipeline run into paper_artifacts/
│   └── run.py                #   entry point: python -m reproduce.run
├── paper_artifacts/          # Per-architecture scores the paper is computed from
├── results/paper_run/        # The run the paper reports (pickles + trial configs)
├── tests/                    # pytest suite (reproduction + end-to-end pipeline)
├── docs/DATASETS.md          # Where to get the six datasets and how to prepare them
├── configs/
│   ├── experiments/unified_config.yml   # All datasets, search space, trial counts
│   └── environments/environment.yml    # Conda environment
├── data/                     # Dataset classes and preparation scripts
├── models/                   # CNN / RNN / Transformer generators
├── utils/                    # Config resolution, paths, experiment bookkeeping
├── analysis/
│   ├── notebooks/            #   The paper's analysis notebook
│   ├── model_testing/        #   Dataset/result path config, FLOPs counting
│   └── post_processing_pipeline/  # Aggregation and zero-cost proxy scoring
├── train.py                  # Training loop
├── trainable.py              # Ray Tune trainable wrapper
├── tune_runner.py            # Launches the Tune experiment
└── mgt_cli.py                # Unified CLI (run / aggregate / post-process / pipeline)
```

## Inventory

The whole repository is 12,094 files and clones in about 10 MB. 76 of those files
are code and documentation (13,859 lines of Python); the other 12,018 are the
paper's run — 18 pickles of results, and one architecture config per trial.

| Path | Files | Size | What it is |
|---|---:|---:|---|
| `reproduce/` | 8 | 46 KB | Recomputes the paper's numbers and Figure 1, and checks its 16 claims |
| `paper_artifacts/` | 7 | 3.0 MB | One row per sampled architecture; the input `reproduce.run` reads |
| `tests/` | 3 | 17 KB | 50 reproduction tests plus one end-to-end training test |
| `results/paper_run/post-processed-results/` | 18 | 61.4 MB | Training results, the eight proxies at initialisation, and FLOPs |
| `results/paper_run/results/experiments/` | 12,000 | 14.3 MB | `params.json` per trial: sampled architecture, seeds, initialiser |
| `analysis/notebooks/` | 1 | 99 KB | The notebook the paper's analysis was developed in |
| `analysis/talented_models_v3.py` | 1 | 43 KB | Analysis helpers the notebook imports |
| `analysis/model_testing/` | 3 | 50 KB | Result-path config, FLOPs counting, proxy correlation helpers |
| `analysis/post_processing_pipeline/` | 5 | 36 KB | Aggregation and zero-cost proxy scoring |
| `data/` | 13 | 76 KB | Seven dataset classes, the loader, and four preparation scripts |
| `models/` | 8 | 34 KB | CNN, RNN, Transformer and TinyHAR generators, and the registry |
| `utils/` | 14 | 98 KB | Config resolution, path handling, experiment bookkeeping |
| `configs/` | 2 | 14 KB | `unified_config.yml` (datasets, search space, trial counts) and the conda environment |
| `docs/DATASETS.md` | 1 | 5 KB | Where to get the six datasets and how to prepare them |
| root | 10 | 107 KB | `mgt_cli.py`, `train.py`, `trainable.py`, `tune_runner.py`, requirements, `pytest.ini`, `project_settings.example`, `.gitignore`, this file |

Deliberately not included:

- `full_training_metrics.pkl` — 695 MB of per-epoch training curves. No result in
  the paper depends on them.
- The noise-robustness results, and the notebook cell that read them. That is
  follow-up work which is not in the paper.
- The raw datasets. They are third-party releases with their own licences;
  [`docs/DATASETS.md`](docs/DATASETS.md) has the download sources and the exact
  preprocessing used.
- Model checkpoints. Recomputing the proxies from scratch needs them, but every
  number in the paper is derived from the scores already in
  `results/paper_run/`.

## Architecture search space

We sample candidate models from CNN, LSTM, and Transformer families commonly used in HAR. Parameter ranges and approximate search-space sizes are below.

| Parameter | CNN | LSTM | Transformer |
|---|---|---|---|
| # Layers | 1–7 | 2–4 | 2–4 |
| Hidden dim.‡ | 8–1024 (step 8) | 8–504 (step 8) | {64, 128, 256} |
| Kernel size | 2–9 | — | — |
| Stride | 1 (fixed) | — | — |
| Attn. heads | — | — | {2, 4, 8} |
| FFN multiplier | — | — | {2, 4, 8}† |
| Dropout | 0.1–0.5 | 0.1–0.5* | 0.0–0.2 |
| Attn. dropout | — | — | 0.0–0.1 |
| Activation | ReLU | ReLU | ReLU, GELU |
| Pooling | max | — | mean, last, CLS |
| Input LayerNorm | — | — | on/off |
| Max pos. length | — | — | {512, …, 4096} |
| **# Architectures** | **≈ 9.23×10²⁵** | **≈ 9.88×10⁹** | **≈ 2.33×10⁴** |

**Notes**
- † FFN hidden dimension is computed as a multiplier of the Transformer model dimension.
- * LSTM dropout is applied between recurrent layers.
- ‡ Hidden dimension, dropout, and (where applicable) kernel size are selected independently for each layer. For CNNs, the count is computed as ∑_{ℓ=1}^{7} (128 · 8 · 5)^ℓ ≈ 9.23×10²⁵.

**Sampling used in the paper:** 2,000 architectures per dataset (1,200 CNNs, 500 Transformers, 300 RNNs).

## Setting up the environment

Versions are pinned to what the reported runs and the test suite were verified
against (Python 3.12, torch 2.6, Ray 2.58):

```bash
pip install -r requirements.txt            # full pipeline
pip install -r requirements-reproduce.txt  # paper results only
```

A conda environment file is also provided, though it predates the pinned
requirements and specifies an older torch build (2.2.1, CUDA 12.1):

```bash
conda env create -f configs/environments/environment.yml
```

Recomputing zero-cost proxies from checkpoints additionally needs the `foresight`
package from [zero-cost-nas](https://github.com/SamsungLabs/zero-cost-nas), which
is not on PyPI under that name; install it from source.

## Configuring project paths

The code detects the project root by looking for `train.py` in the current
directory or its parents, so usually there is nothing to configure. To point it
elsewhere, either export the variable:

```bash
export PROJECT_ROOT="/path/to/models_got_talent"
```

or copy the example settings file and source it:

```bash
cp project_settings.example project_settings
# edit PROJECT_ROOT, then
source project_settings
```

`project_settings` is gitignored so each user keeps their own. It also documents
the optional variables that redirect results — `MGT_RESULTS_ROOT`,
`MGT_PAPER_RESULTS_ROOT`, `RAY_TMPDIR` — which are resolved in
`analysis/model_testing/dataset_processing_config.py`.

## Re-running the pipeline from scratch

### 1. Prepare the datasets

See [`docs/DATASETS.md`](docs/DATASETS.md) for download links, the expected
directory layout, the per-dataset preparation commands, and the exact
preprocessing (accelerometer only, 50 Hz, user-level 60/20/20 split, 2 s windows
with 50% overlap). Then point `configs/experiments/unified_config.yml` at wherever
you put them.

### 2. Train the candidate architectures

Start with a short smoke run before committing GPU time:

```bash
python mgt_cli.py run --config configs/experiments/unified_config.yml \
    --dataset hhar --test --num-test-models 3 -y
```

The production run trains all 2,000 sampled architectures for one dataset:

```bash
python mgt_cli.py run --config configs/experiments/unified_config.yml --dataset hhar
```

Results land in `results/experiments/<exp_id>__<tag>/<dataset>/`. Re-running the
same command resumes; `--reset` starts over (backing up the old directory), and
`--retry-errored` re-runs only the trials that errored. To attach to an existing
Ray cluster instead of starting a local one, pass `--ray-address <addr>` or set
`RAY_ADDRESS`.

Use `--variant original` in place of `--dataset` to sweep all six datasets, and
`python mgt_cli.py explore collections` to inspect what has been run.

### 3. Aggregate and score

```bash
python mgt_cli.py post-process --base-dir results --latest
```

This aggregates each dataset's trials into `experiment.parquet`, then writes
`post-processed-results/full_results.pkl` (training metrics) and
`zero_cost_scores.pkl` (the proxies, scored from each model's initial checkpoint).

`python mgt_cli.py pipeline --config ... --dataset hhar` runs steps 2 and 3 back
to back.

### 4. Count FLOPs

The FLOPs column the paper's cost analysis uses is produced separately, since it
needs `fvcore` and a forward pass per architecture:

```bash
python -m analysis.model_testing.calculate_flops
```

### 5. Regenerate the artifacts and the results

```bash
python -m reproduce.export_artifacts   # pipeline pickles -> paper_artifacts/*.csv
python -m reproduce.run                # tables, claim checks, Figure 1
```

Point `MGT_PAPER_RESULTS_ROOT` at your own run first, otherwise both commands
read the committed `results/paper_run/` described above.

## Citation

```bibtex
@inproceedings{goldman2026models,
  title     = {Models Got Talent: Identifying High Performing Wearable Human
               Activity Recognition Models Without Training},
  author    = {Goldman, Richard and Komperla, Varun and Ploetz, Thomas and
               Haresamudram, Harish},
  booktitle = {Proceedings of the 2026 ACM International Symposium on Wearable
               Computers (ISWC '26)},
  year      = {2026},
  doi       = {10.1145/3830727.3834826}
}
```
