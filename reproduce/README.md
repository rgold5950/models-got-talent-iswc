# Reproducing the paper

> **Models Got Talent: Identifying High Performing Wearable Human Activity
> Recognition Models Without Training**
> Richard Goldman, Varun Komperla, Thomas Ploetz, Harish Haresamudram.
> ISWC '26. [arXiv:2511.06157](https://arxiv.org/abs/2511.06157) ·
> [doi:10.1145/3830727.3834826](https://doi.org/10.1145/3830727.3834826)

Everything the paper reports comes from one table per dataset: for each of the
~2,000 sampled architectures, its eight zero-cost proxy scores measured at
initialisation and its F1 after full-scale training. Those tables are committed
under [`paper_artifacts/`](../paper_artifacts) (576 KB total), so the results can
be checked without the datasets, a GPU, or the 220 MB of raw pipeline output.

## Quick start

```bash
pip install pandas numpy matplotlib     # nothing else is needed
python -m reproduce.run
```

This prints Δ₁, Δ₁₀, Δ₁₀%, and the talent rate for every proxy and dataset,
checks each against the corresponding sentence in the paper, and writes
`analysis/figures/paper/figure1.pdf` (Figure 1) plus `results.json`. It takes
about 15 seconds and exits non-zero if any claim fails to reproduce.

To verify as a test suite instead:

```bash
pip install pytest
pytest tests/test_paper_reproduction.py -v
```

## What gets checked

The 16 quantitative claims in [`claims.py`](claims.py) each pair a sentence from
the paper with a predicate over the recomputed numbers, so a failure names the
section that would need revisiting. They cover the sampling setup (2,000
architectures split 1,200 CNN / 500 Transformer / 300 RNN), the headline gaps
(Δ₁ within 7%, Δ₁₀ within 2%), the per-dataset details (PAMAP2's four proxies
with Δ₁ < 0%, Myogym's 38% talent rate, synflow being the only proxy that works
on HHAR), the comparison against random search, and the FLOPs savings.

All 16 currently reproduce exactly.

## The metrics

Both metrics are defined in [`metrics.py`](metrics.py) and treat the
**validation** ranking as ground truth, because that is the model-selection
signal a practitioner actually has; test F1 is only read off once a model has
been selected.

**Performance gap, Δ_k.** Take the top-k architectures ranked by a proxy, train
only those k, keep the one with the highest validation F1, and subtract its test
F1 from the test F1 of the architecture that full-scale training would have
shipped. Lower is better, and negative means the proxy found a model that
generalises better than the one exhaustive training selected.

Δ_k is *not* monotone in k. The top-k proxy sets are nested, so a larger k can
only improve the validation F1 of the selected model — but Δ is measured on test
F1, so a model that validates better can still test worse.

**Talent rate.** The percentage of the top-k proxy-ranked architectures that also
land in the top-k after full training, with k = 10% of the pool.

**Random-search baseline.** 1000 draws of k architectures without replacement,
scored with the same two formulas. Reported as a mean with a one-standard-deviation
error bar. Seeded (`--seed`, default 0) so runs are bit-reproducible.

## Sign conventions and the ensemble

Proxy scores are stored raw, exactly as the pipeline measured them. `grasp` is
negated at load time because more-negative scores indicate better architectures
there; every other proxy is already higher-is-better. The `ensemble` proxy is the
mean of the percentile ranks of the seven proxies from Abdelfattah et al.; it does
not include `initial_val_f1_macro`, which the paper introduces as a separate
eighth proxy. `synflow_bn` is measured and stored but excluded from the ensemble,
since none of the sampled architectures use batch norm and it is therefore
identical to `synflow`.

These conventions live only in `metrics.add_ensemble`, so they are applied exactly
once.

## Reading the pickles instead of the CSVs

The paper's run is committed under `results/paper_run/`, which is exactly where
`MGT_PAPER_RESULTS_ROOT` points by default, so this works with no setup:

```bash
python -m reproduce.run --from-pickles     # recompute from the run, not the CSVs
```

Use it to confirm the committed CSVs still match their source. The two paths
agree on every number; only the `source` field of `results.json` differs.

## Regenerating the artifacts from your own run

Only needed if you have re-run training and post-processing yourself:

```bash
export MGT_PAPER_RESULTS_ROOT=/path/to/your/run   # holds post-processed-results/
python -m reproduce.export_artifacts              # refreshes paper_artifacts/
python -m reproduce.run --from-pickles
```

`paper_artifacts/MANIFEST.json` records the source pickle paths, a SHA-256 per
CSV, and the commit the export ran at. Reading the committed CSVs and reading the
source pickles give bit-identical results.

To reproduce the pipeline that produced those pickles in the first place — data
preparation, architecture sampling, 2,000 training runs per dataset, proxy
computation — see [`docs/DATASETS.md`](../docs/DATASETS.md) for obtaining the
data and the main [`README.md`](../README.md) for the `mgt_cli` workflow. That
path needs GPUs and on the order of a thousand GPU-hours per dataset.

## Files

| File | Purpose |
|---|---|
| `metrics.py` | Δ_k, talent rate, random-search baseline |
| `claims.py` | the paper's claims as executable checks |
| `data.py` | loads `paper_artifacts/`, or rebuilds from pipeline pickles |
| `figure1.py` | the 2×2 panel figure |
| `run.py` | CLI: tables, claim checks, `results.json`, figure |
| `export_artifacts.py` | distils pipeline output into `paper_artifacts/` |
