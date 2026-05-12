# `results/` — Figures and tables

Index of the numbers behind the poster and 2-page report. Each row links
a result file to the section/figure it backs in the write-up, the
configurations it sweeps, and the script that produces it (where one
exists).

## Figures

| File | What it shows | Backs |
|---|---|---|
| [`figures/fig_train_loss_curves.png`](figures/fig_train_loss_curves.png) | Training-loss trajectories: LD4LG AE val loss, LD4LG diffusion train loss, DPLM train loss. | Methodology / training stability |
| [`figures/fig_ld4lg_cfg_sweep.png`](figures/fig_ld4lg_cfg_sweep.png) | LD4LG CFG-weight sweep (w ∈ {1, 1.5, 2, 3, 5}); 4-gram diversity, validity, linker recovery per weight. | Pareto-frontier discussion |
| [`figures/fig_latency_benchmark.png`](figures/fig_latency_benchmark.png) | Sampling latency for the five (model, decoder) configurations on the same hardware: LD4LG-250, LD4LG-50, DPLM-default, DPLM-tuned, DPLM-greedy. | Reflections / cost analysis |

## Tables (raw JSON)

| File | What it holds |
|---|---|
| [`tables/train_loss_curves.json`](tables/train_loss_curves.json) | Step-indexed loss arrays for `ae_val_loss`, `diff_train_loss`, `dplm_train_loss`. Plots in `fig_train_loss_curves.png` are derived from this. |
| [`tables/ld4lg_cfg_sweep.json`](tables/ld4lg_cfg_sweep.json) | Per-(weight, cell) metrics for the LD4LG CFG sweep: 4-gram diversity, validity, linker recovery; n=128 per (weight, cell), 3 cells. Produced by [`code/diffusion/LD4LG/sampling_cfg_sweep.py`](../code/diffusion/LD4LG/sampling_cfg_sweep.py). |
| [`tables/cfg_ablation.json`](tables/cfg_ablation.json) | Independent-CFG vs joint-CFG dropout ablation for DPLM. Four regimes (`full`, `drop_iso`, `drop_vfam`, `drop_loc`) × 3 cells × n=64 per cell. Produced by the DPLM sampler with the `--joint-cfg` flag (see `code/diffusion/DPLM/sample.py`). |
| [`tables/latency_benchmark.json`](tables/latency_benchmark.json) | Per-config sampling throughput: `elapsed_s`, `samples_per_sec`, `ms_per_sample`; batch=32, 8 batches per config. |

## Headline numbers

The summary table in the top-level [`README.md`](../README.md#6-results--insights)
(7 metrics × 3 configurations across 9,216 generated sequences) is computed
from FASTA evaluations under [`code/common/`](../code/common/) and is not
mirrored here as JSON; the per-cell breakdown lives in those eval-report
JSONs (one per cell), which exceed GitHub's UI-friendly size and are kept
out of this directory.

## Regenerating

- LD4LG CFG sweep:
  `python -m code.diffusion.LD4LG.sampling_cfg_sweep --ckpt … --out results/tables/ld4lg_cfg_sweep.json`
- DPLM CFG-dropout ablation:
  rerun `code/diffusion/DPLM/sample.py` over the four regimes and aggregate
  (the ablation JSON is post-processed from the per-regime FASTAs).
- Loss curves and latency benchmark were assembled from the training and
  sampling runs reported in [`README.md`](../README.md). The plot scripts
  are not packaged; the underlying JSON is the authoritative artifact.
