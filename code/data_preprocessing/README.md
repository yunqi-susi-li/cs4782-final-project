# Data Preprocessing: Paired-Chain Sequence Similarity Reduction

A portable, MMseqs2-based pipeline that takes a paired-chain immune-receptor
dataset (BCR heavy + light, TCR β + α, or any user-defined chain pair) and
produces a leakage-free `train / val / test` split at a configurable identity
threshold (default 95%).

This directory holds the upstream similarity-reduction pipeline for the
diffusion models in this project. The downstream OAS-export → token-memmap
step that prepares model input lives in
[`../diffusion/LD4LG/preprocess.py`](../diffusion/LD4LG/preprocess.py) and is
shared by the LD4LG and DPLM tracks.

- Runs on any Linux box or HPC node with MMseqs2 v18+ on `$PATH`
- CPU-only by default, with an opt-in GPU mode for the two search steps
- Self-resuming: every step writes a marker file; rerun with `--resume`
  to skip what has already finished
- Receptor-agnostic: pick `--receptor bcr` (heavy + light) or
  `--receptor tcr` (β + α), or supply custom chain column suffixes

---

## Quick start

```bash
cd code/data_preprocessing

# BCR at 95% identity, CPU
bash run_pipeline.sh \
    --input    /path/to/paired_bcr.pkl \
    --output   /path/to/output/bcr_095 \
    --receptor bcr \
    --threshold 0.95

# TCR at 95% identity, CPU
bash run_pipeline.sh \
    --input    /path/to/paired_tcr.pkl \
    --output   /path/to/output/tcr_095 \
    --receptor tcr \
    --threshold 0.95
```

The wrapper auto-detects `mmseqs` and `python3` on `$PATH`, runs all 7 steps,
and writes the final data into `<output>/06_export/`.

For more aggressive deduplication (e.g. data containing many germinal-center
clones from the same donor) lower the threshold to `0.90`.

---

## Pipeline at a glance

```
Input pickle (paired chains)
   │
   ▼
[1] Extract       pkl -> per-chain FASTA + metadata.tsv
[2] Exact dedup   drop byte-identical (chain1, chain2) pairs by SHA-1
[3] NN search     MMseqs2 all-vs-all, top non-self hit per sequence  (GPU optional)
[4] linclust      MMseqs2 linclust, per chain, at the chosen threshold (CPU only)
[5] Group + split group rows by (chain1_cluster, chain2_cluster) tuples,
                  assign each tuple atomically to train / val / test
[6] Export        write train/val/test FASTA + metadata.tsv + .pkl
[7] Leakage audit cross-search val/test -> train; expect 0% leakage    (GPU optional)
```

The split is tuple-cluster-aware: a paired sample belongs to the
`(chain1_cluster, chain2_cluster)` tuple, and an entire tuple is assigned to
exactly one of train / val / test. This is what guarantees 0% leakage on
both chains simultaneously.

---

## Output layout

```
<output>/
├── 01_raw/           per-chain FASTA + metadata.tsv
├── 02_exact/         after SHA-1 pair dedup
├── 03_nn/            top non-self hit per sequence (per chain)
├── 04_clusters/      MMseqs2 cluster maps (per chain)
├── 05_split/         scheme_c_<TAG>.assignments.tsv (full metadata + split label)
├── 06_export/
│   ├── train/        train.{chain1,chain2,pair}.fasta + train.metadata.tsv + train.pkl
│   ├── val/          (same)
│   └── test/         (same)
├── 07_audit/         {val,test}_vs_train_{chain1,chain2}.summary.tsv   # leakage report
└── logs/             one log per step
```

For TCR, the chain file names internally use `heavy/light` aliases
(chain1 -> "heavy", chain2 -> "light"); the content is the β and α
sequences as specified.

---

## Loading a prepared split (Python)

```python
import pandas as pd

ROOT = "/path/to/output/bcr_095/06_export"
train = pd.read_pickle(f"{ROOT}/train/train.pkl")
val   = pd.read_pickle(f"{ROOT}/val/val.pkl")
test  = pd.read_pickle(f"{ROOT}/test/test.pkl")

print(train[["sequence_alignment_aa_heavy", "sequence_alignment_aa_light"]].head())
```

The pickles preserve all original metadata columns (V/J calls, isotype,
SHM rate, species, ...) plus the cluster representatives and split label.
Filter on whatever you need downstream.

---

## GPU acceleration (optional)

The two MMseqs2 search steps (Step 3: NN self-search; Step 7: leakage audit)
support GPU acceleration with `--gpu 1`. linclust (Step 4) does not: this is
an MMseqs2 design choice (linclust is already linear-time on CPU and has not
been ported to GPU as of v18).

### Enable GPU mode

```bash
bash run_pipeline.sh \
    --input    /path/to/paired_bcr.pkl \
    --output   /path/to/output/bcr_095_gpu \
    --receptor bcr \
    --threshold 0.95 \
    --gpu \
    --mmseqs   /path/to/mmseqs-gpu/bin/mmseqs
```

When `--gpu` is set, the wrapper:

1. Builds a GPU-padded sequence DB via `mmseqs makepaddedseqdb` before each
   search call (required by the GPU prefilter).
2. Passes `--gpu 1` to `mmseqs search` in steps 3 and 7.
3. Runs step 4 (linclust) on CPU as usual.

### Caveats

- Use a dedicated GPU build of MMseqs2. Some clusters ship both an AVX2 CPU
  build and a separate GPU build (e.g. a `mmseqs-avx2` binary alongside a
  `mmseqs-gpu` binary). Pass the GPU one explicitly via `--mmseqs PATH`.
- Verify identity enforcement on a tiny sample first. At least one MMseqs2
  v18 GPU build has been observed silently not enforcing `--min-seq-id`
  during alignment validation, clustering all sequences into one giant
  cluster regardless of similarity. The CPU build of the same version on
  the same data behaves correctly. Check the cluster size distribution
  (`scripts/06_summarize_clusters.py`) and the leakage audit (Step 7) on a
  known-easy case before trusting a full GPU run.
- Memory. The GPU prefilter holds the padded target DB on the GPU; for
  ~2M paired antibody sequences, plan for >= 40 GB GPU memory at default
  settings. If OOM, fall back to CPU or lower `--max-seqs`.
- Throughput. When the GPU build behaves, expect roughly an order-of-magnitude
  speedup on the search steps. Total wall-clock improvement is smaller
  because steps 1, 2, 4, 5, 6 run on CPU regardless.

If a GPU run produces a single mega-cluster or non-zero leakage, that is a
build-specific bug. Rerun the search steps on CPU. The CPU path is the
authoritative one.

---

## Command-line reference

```
Usage: run_pipeline.sh [options]

Required:
  --input PATH         Input pickle with paired-chain sequences
  --output DIR         Output directory (created if missing)

Receptor selection:
  --receptor bcr       BCR mode -> chain1=heavy, chain2=light  (default)
  --receptor tcr       TCR mode -> chain1=beta,  chain2=alpha
  --chain1 NAME        Custom chain-1 suffix (overrides --receptor)
  --chain2 NAME        Custom chain-2 suffix (overrides --receptor)

Optional:
  --threshold FLOAT    Identity threshold in (0,1] (default 0.95)
  --train-frac FLOAT   Train fraction (default 0.96)
  --val-frac FLOAT     Val fraction   (default 0.02)
  --test-frac FLOAT    Test fraction  (default 0.02)
  --threads INT        CPU threads (default: all available)
  --gpu                Use --gpu 1 for the two search steps (3 and 7)
  --resume             Skip steps whose output already exists
  --skip-audit         Skip the leakage audit (Step 7)
  --mmseqs PATH        Path to mmseqs binary (default: auto-detect on $PATH)
  --python PATH        Path to python (default: auto-detect on $PATH)
  --help               Show usage
```

Working launch scripts are in [`examples/`](examples/):
[`run_bcr_95.sh`](examples/run_bcr_95.sh),
[`run_bcr_90.sh`](examples/run_bcr_90.sh),
[`run_tcr_95.sh`](examples/run_tcr_95.sh),
[`run_tcr_90.sh`](examples/run_tcr_90.sh),
[`run_bcr_95_gpu.sh`](examples/run_bcr_95_gpu.sh),
and [`slurm_submit.sh`](examples/slurm_submit.sh).

For the full input-pickle column requirements, output schema, and per-step
internals, see [`PIPELINE.md`](PIPELINE.md). For a TCR-specific walkthrough,
see [`TCR_USAGE.md`](TCR_USAGE.md).

---

## Software requirements

- MMseqs2 v18 or newer (https://github.com/soedinglab/MMseqs2)
  - For GPU mode, additionally a GPU build of the same version
- Python 3.9+ with `pandas`, `numpy`
- A POSIX shell (`bash`) and standard GNU coreutils

No CUDA needed in CPU mode. In GPU mode, MMseqs2 handles the CUDA runtime
itself, with no extra Python deps.

---

## Reference run statistics (paired human BCR, 95% identity)

For calibration, these are the numbers from a CPU run on the human paired-BCR
pickle that fed the diffusion models in this project:

| Stage                          | Pairs           |
|--------------------------------|-----------------|
| Raw input (paired human BCR)   | 2,090,939       |
| After exact-pair dedup (SHA-1) | 1,811,977       |
| Train (96%)                    | 1,690,549       |
| Val (2%)                       | 35,215          |
| Test (2%)                      | 35,170          |

Leakage audit at 95% AA identity (cross-search val/test vs train):

| Pair                  | n queries with any hit | frac >= 95% identity |
|-----------------------|-----------------------:|---------------------:|
| val   vs train, heavy | 35,168                 | 0.00%                |
| val   vs train, light | 35,134                 | 0.00%                |
| test  vs train, heavy | 35,168                 | 0.00%                |
| test  vs train, light | 35,133                 | 0.00%                |

Both chains audited, both at 0.00%. Numbers will differ for your own input;
treat the table as a sanity check, not a target.

---

## File map

| File / dir | Purpose |
|------------|---------|
| `run_pipeline.sh`                      | Single entry point; orchestrates all 7 steps |
| `scripts/01_extract.py`                | pkl -> per-chain FASTA + metadata |
| `scripts/02_exact_dedup.py`            | SHA-1 pair-level exact dedup |
| `scripts/03_nn_search.sh`              | MMseqs2 all-vs-all NN search (CPU or GPU) |
| `scripts/04_pick_top_nonself_hit.py`   | Pick top non-self hit per query |
| `scripts/05_cluster.sh`                | MMseqs2 linclust (CPU only) |
| `scripts/06_summarize_clusters.py`     | Cluster size histogram + purity |
| `scripts/07_build_groups_and_split.py` | Tuple-aware train/val/test assignment |
| `scripts/08_export_splits.py`          | Write FASTA + metadata + pkl per split |
| `scripts/09_audit_leakage.sh`          | Cross-search val/test -> train (CPU or GPU) |
| `scripts/10_summarize_leakage.py`      | Leakage identity-percentile summary |
| `examples/`                            | Ready-to-run BCR/TCR launch scripts |
| `PIPELINE.md`                          | Detailed pipeline reference |
| `TCR_USAGE.md`                         | TCR (β + α) usage notes |
