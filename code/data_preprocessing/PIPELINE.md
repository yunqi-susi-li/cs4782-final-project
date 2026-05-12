# Paired-Chain Similarity Reduction Pipeline (Detailed Reference)

A pipeline that takes a paired-chain immune-receptor dataset (BCR or TCR),
removes near-duplicates at a configurable identity threshold (95% or 90%),
and produces leakage-free train / val / test splits suitable for protein
language model training (ESM, AbLang, or the diffusion models in this
project).

Powered by MMseqs2 v18+ (https://github.com/soedinglab/MMseqs2). CPU-only
by default; the two search steps can run on GPU. Runs on any Linux node and
automatically uses all available CPU cores.

---

## Quick Start

```bash
cd code/data_preprocessing

# BCR at 95% identity (default)
bash run_pipeline.sh \
    --input    /path/to/paired_bcr.pkl \
    --output   /path/to/output/bcr_095 \
    --receptor bcr \
    --threshold 0.95

# BCR at 90% identity (more aggressive deduplication)
bash run_pipeline.sh \
    --input    /path/to/paired_bcr.pkl \
    --output   /path/to/output/bcr_090 \
    --receptor bcr \
    --threshold 0.90

# TCR at 95% identity
bash run_pipeline.sh \
    --input    /path/to/paired_tcr.pkl \
    --output   /path/to/output/tcr_095 \
    --receptor tcr \
    --threshold 0.95

# TCR at 90% identity
bash run_pipeline.sh \
    --input    /path/to/paired_tcr.pkl \
    --output   /path/to/output/tcr_090 \
    --receptor tcr \
    --threshold 0.90
```

The pipeline runs all 7 steps and writes the final train/val/test data into
`<output>/06_export/`. Steps that have already completed are skipped on
re-run, so interruption and resume are safe.

---

## What the Pipeline Does

```
Input pickle (paired chains)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: Extract                                                  │
│   Reads pkl, writes per-chain FASTA + metadata.tsv               │
├──────────────────────────────────────────────────────────────────┤
│ Step 2: Exact Dedup                                              │
│   Removes byte-identical (chain1, chain2) pairs by SHA-1 hash    │
├──────────────────────────────────────────────────────────────────┤
│ Step 3: Nearest-Neighbor Search   (CPU, all cores)               │
│   All-vs-all MMseqs2 search at the chosen threshold              │
│   Reports the top non-self hit for every sequence                │
├──────────────────────────────────────────────────────────────────┤
│ Step 4: Clustering   (linclust, CPU)                             │
│   Independent clustering of each chain at the chosen threshold   │
│   linclust is linear-time and scales to billions of sequences    │
├──────────────────────────────────────────────────────────────────┤
│ Step 5: Group + Split                                            │
│   Groups by (chain1_cluster, chain2_cluster) tuples              │
│   Assigns each tuple-group to train / val / test (96/2/2)        │
│   Guarantees no group is split across sets                       │
├──────────────────────────────────────────────────────────────────┤
│ Step 6: Export                                                   │
│   Writes train/val/test as FASTA + metadata.tsv + pickle         │
├──────────────────────────────────────────────────────────────────┤
│ Step 7: Leakage Audit   (CPU)                                    │
│   Cross-searches val/test against train at the chosen threshold  │
│   Reports identity histogram and leakage fraction (target: 0%)   │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
06_export/
  train/  train.{heavy,light,pair}.fasta + train.metadata.tsv + train.pkl
  val/    val.{heavy,light,pair}.fasta   + val.metadata.tsv   + val.pkl
  test/   test.{heavy,light,pair}.fasta  + test.metadata.tsv  + test.pkl
07_audit/
  {val,test}_vs_train_{heavy,light}.summary.tsv   # leakage report
```

For TCR the file names use `{heavy,light}` internally (chain1 -> "heavy",
chain2 -> "light"); the content is your beta and alpha chains as specified.

---

## Why Two Thresholds?

The choice of threshold trades dataset size against how aggressively
near-duplicates are removed:

| Threshold | Cluster reduction | When to use |
|-----------|-------------------|-------------|
| 95% | Conservative | Default. Keeps almost all biologically distinct sequences. Recommended starting point. |
| 90% | More aggressive | Use when training data contains many germinal-center clones from the same donor and you want to ensure the model never sees clonally-related sequences. |

Running both thresholds against the same input is straightforward: change
the `--threshold` and `--output` flags. Since Steps 1-2 are deterministic,
run 95% first then 90% with `--resume` to reuse the extract/dedup output.

---

## Command-Line Reference

```
Usage: run_pipeline.sh [options]

Required:
  --input PATH         Input pickle with paired-chain sequences
  --output DIR         Output directory (created if missing)

Receptor selection:
  --receptor bcr       chain1=heavy, chain2=light  (default)
  --receptor tcr       chain1=beta,  chain2=alpha
  --chain1 NAME        Custom chain-1 suffix (overrides --receptor)
  --chain2 NAME        Custom chain-2 suffix (overrides --receptor)

Optional:
  --threshold FLOAT    Identity threshold in (0,1] (default 0.95)
  --train-frac FLOAT   Train fraction (default 0.96)
  --val-frac FLOAT     Val fraction   (default 0.02)
  --test-frac FLOAT    Test fraction  (default 0.02)
  --threads INT        CPU threads (default: all available)
  --resume             Skip steps whose output already exists
  --skip-audit         Skip the leakage audit (Step 7)
  --mmseqs PATH        Override path to mmseqs binary (default: auto-detect)
  --python PATH        Override path to python (default: auto-detect)
  --help               Show this message
```

Examples:

```bash
# Resume an interrupted run, using 16 threads
bash run_pipeline.sh \
    --input data.pkl --output myrun --receptor bcr --threshold 0.95 \
    --resume --threads 16

# Custom column names (e.g. paired beta+alpha TCR with non-standard suffixes)
bash run_pipeline.sh \
    --input tcr.pkl --output myrun \
    --chain1 chainA --chain2 chainB --threshold 0.90
```

---

## Input Pickle Format

The pipeline expects a pandas DataFrame saved as a pickle, with one row per
paired-chain cell.

### BCR (`--receptor bcr`)

| Column | Required? | Description |
|--------|-----------|-------------|
| `sequence_id_heavy` | yes | Unique identifier for heavy chain |
| `sequence_id_light` | yes | Unique identifier for light chain |
| `productive_heavy` | recommended | `T`/`F` |
| `productive_light` | recommended | `T`/`F` |
| `sequence_alignment_aa_heavy` | yes | Heavy chain amino acid sequence (full V region) |
| `sequence_alignment_aa_light` | yes | Light chain amino acid sequence |
| `cdr3_aa_heavy` | yes | Heavy chain CDR3 amino acid sequence |
| `cdr3_aa_light` | yes | Light chain CDR3 amino acid sequence |
| `v_call_heavy`, `j_call_heavy`, `v_call_light`, `j_call_light` | recommended | Gene calls (allele suffix `*01` is OK) |
| `Isotype_heavy`, `locus_light` | optional | Antibody class, kappa/lambda |
| `shm_rate_heavy`, `shm_rate_light` | optional | SHM rate (float) |
| `species` | recommended | e.g. `human`, `mouse` |

### TCR (`--receptor tcr`)

| Column | Required? | Description |
|--------|-----------|-------------|
| `sequence_id_beta` | yes | TRB unique id |
| `sequence_id_alpha` | yes | TRA unique id |
| `productive_beta` | recommended | `T`/`F` |
| `productive_alpha` | recommended | `T`/`F` |
| `sequence_alignment_aa_beta` | yes | TRB amino acid sequence |
| `sequence_alignment_aa_alpha` | yes | TRA amino acid sequence |
| `cdr3_aa_beta` | yes | TRB CDR3 |
| `cdr3_aa_alpha` | yes | TRA CDR3 |
| `v_call_beta`, `j_call_beta`, `v_call_alpha`, `j_call_alpha` | recommended | Gene calls |
| `species` | recommended | |

Any extra columns are preserved and copied through to the final exported
pickles.

---

## Output Format

After a successful run, `<output>/06_export/` contains:

```
06_export/
├── train/
│   ├── train.heavy.fasta       (chain1 = heavy / beta)
│   ├── train.light.fasta       (chain2 = light / alpha)
│   ├── train.pair.fasta        (concatenated chain1 + GGGGSGGGGS linker + chain2)
│   ├── train.metadata.tsv      (tab-separated, all annotations)
│   └── train.pkl               (pandas pickle, ready to load)
├── val/                        (same structure)
└── test/                       (same structure)
```

Loading the training data:

```python
import pandas as pd
train = pd.read_pickle("output/06_export/train/train.pkl")
print(train.shape)             # (n_train_pairs, n_columns)
print(train.columns.tolist())  # all preserved columns
```

The leakage audit results are in `<output>/07_audit/`:

```
07_audit/
├── val_vs_train_heavy.summary.tsv
├── val_vs_train_light.summary.tsv
├── test_vs_train_heavy.summary.tsv
├── test_vs_train_light.summary.tsv
└── *.top_hits.tsv              (per-query top hit)
```

The summary TSV format:

```
n_queries_with_hit    35127
median_pident         0.934
p90_pident            0.975
p95_pident            0.983
p99_pident            1.0
frac_ge_100           0.0    # fraction of val/test with >=100% identity in train
frac_ge_99            0.0
frac_ge_95            0.0    # MUST be 0 for the 95% pipeline to be valid
frac_ge_90            0.0
frac_ge_80            0.0
```

For a 95% pipeline, `frac_ge_95` must be `0.0`. For a 90% pipeline,
`frac_ge_90` must be `0.0`.

---

## Hardware Requirements

| Resource | Recommended | Minimum |
|----------|-------------|---------|
| CPU | 32+ cores | 8 cores |
| RAM | 64 GB | 16 GB |
| Disk | 100 GB scratch | 30 GB |
| GPU (optional) | 1x >=40 GB (e.g. A100/H100) for `--gpu` mode | -- |

The pipeline runs on any Linux node (laptop, workstation, HPC). GPU mode is
an optional speed-up for the search steps; the CPU path is always sufficient.

### Approximate runtime (CPU only)

| Dataset size          | 32 cores | 16 cores | 8 cores |
|-----------------------|----------|----------|---------|
| 100K paired sequences | ~10 min  | ~20 min  | ~40 min |
| 500K paired sequences | ~1.5 h   | ~3 h     | ~6 h    |
| 2M paired sequences   | ~12 h    | ~24 h    | ~48 h   |
| 3M paired sequences   | ~24 h    | ~48 h    | ~96 h   |

Most of the time is in Step 3 (all-vs-all NN search), which scales roughly
quadratically with dataset size. For very large datasets (>3M), run
overnight, in a SLURM job, or with `--gpu` (see main `README.md`).

---

## Dependencies

### MMseqs2

The pipeline auto-detects `mmseqs` from `$PATH`, with a fallback probe for
common HPC system installs (`/programs/mmseqs-*/bin/mmseqs`,
`~/miniconda3/envs/mmseqs2/bin/mmseqs`). Override via `--mmseqs PATH` if
auto-detection picks the wrong build.

Install via conda (works on any cluster):

```bash
conda create -n mmseqs2 -c conda-forge -c bioconda mmseqs2 -y
conda activate mmseqs2
```

Or build from source: https://github.com/soedinglab/MMseqs2

GPU builds. Some MMseqs2 distributions ship a separate GPU binary
(`mmseqs-gpu-*`) alongside the CPU build. The GPU build requires calling
`mmseqs makepaddedseqdb` first and passing `--gpu 1`; skipping the padded-DB
step causes MMseqs2 to abort with:

```
Database is not a valid GPU database
Please call: makepaddedseqdb ...
Error: Ungapped prefilter died
```

When `run_pipeline.sh` is called without `--gpu`, it auto-skips obvious GPU
binaries and uses a CPU build instead. With `--gpu`, the wrapper handles
`makepaddedseqdb` automatically.

### Python

Python 3.10+ with `pandas` and `numpy`. The wrapper auto-detects `python3`
on `$PATH`. Override with `--python PATH`. A minimal env:

```bash
conda create -n seq_pipeline python=3.11 pandas numpy -y
```

---

## Examples

See [examples/](examples/):

| File | What it does |
|------|--------------|
| [`run_bcr_95.sh`](examples/run_bcr_95.sh)         | BCR at 95% (CPU) |
| [`run_bcr_90.sh`](examples/run_bcr_90.sh)         | BCR at 90% (CPU) |
| [`run_tcr_95.sh`](examples/run_tcr_95.sh)         | TCR at 95% (CPU) |
| [`run_tcr_90.sh`](examples/run_tcr_90.sh)         | TCR at 90% (CPU) |
| [`run_bcr_95_gpu.sh`](examples/run_bcr_95_gpu.sh) | BCR at 95% with GPU acceleration |
| [`slurm_submit.sh`](examples/slurm_submit.sh)     | SLURM submission template |

---

## Reference Results

Calibration numbers from one CPU run on a paired human BCR pickle at the
95% threshold:

| Stage                          | Pairs           |
|--------------------------------|-----------------|
| Raw input (paired human BCR)   | 2,090,939       |
| After exact-pair dedup (SHA-1) | 1,811,977       |
| Train (96%)                    | 1,690,549       |
| Val (2%)                       | 35,215          |
| Test (2%)                      | 35,170          |

Leakage at 95% AA identity: 0.00% on all four audits (val/test x heavy/light).
Numbers will differ for your own input; treat the table as a sanity check,
not a target.

---

## Troubleshooting

`mmseqs: command not found`
Activate the conda env (`conda activate mmseqs2`) or pass
`--mmseqs /path/to/mmseqs`.

`Missing required columns: [...]`
The pkl uses non-standard column names. Either rename them in the pkl or use
`--chain1` / `--chain2` to point at the right suffixes.

Pipeline crashes mid-step
Re-run with `--resume`. Each step is idempotent and skips work if its
output already exists.

Job runs out of wall time on a SLURM cluster
Increase `--time` and re-submit with `--resume`. Step 3 (NN search) is the
slowest; for >=2M sequences allow >=24 hours on 32 cores.

Test on a small subset first
Save a small pkl: `df.head(10000).to_pickle("test.pkl")` and run the
pipeline against it. Finishes in a few minutes.

---

## Pipeline Internals

`run_pipeline.sh` chains 10 small scripts in `scripts/`. Each script is
independent and can be invoked manually for debugging:

| Script | Purpose | Input -> Output |
|--------|---------|------------------|
| `01_extract.py` | Pkl -> FASTA + metadata | pkl -> `01_raw/` |
| `02_exact_dedup.py` | Remove byte-identical pairs | `01_raw/metadata.tsv` -> `02_exact/` |
| `03_nn_search.sh` | All-vs-all MMseqs2 search | `02_exact/*.fasta` -> `03_nn/*.m8` |
| `04_pick_top_nonself_hit.py` | Top non-self hit per query | `03_nn/*.m8` -> `03_nn/*.top.tsv` |
| `05_cluster.sh` | linclust at threshold | `02_exact/*.fasta` -> `04_clusters/*/clu.tsv` |
| `06_summarize_clusters.py` | Cluster stats + purity | cluster TSV -> summary TSV |
| `07_build_groups_and_split.py` | Tuple grouping -> split assignments | clusters -> `05_split/*.assignments.tsv` |
| `08_export_splits.py` | Materialize train/val/test | assignments + pkl -> `06_export/{train,val,test}/` |
| `09_audit_leakage.sh` | Cross-search audit | exported FASTAs -> `07_audit/*.m8` |
| `10_summarize_leakage.py` | Identity histogram | `07_audit/*.m8` -> summary TSV |

By default the shell scripts call `mmseqs` without `--gpu`, so they run on
any CPU. Pass `--gpu` to `run_pipeline.sh` (and a GPU-build path via
`--mmseqs`) to enable GPU acceleration on the search steps; see
`README.md`.

---

## Citation

If you use this pipeline, please cite MMseqs2:

> Steinegger M & Söding J. *MMseqs2 enables sensitive protein sequence
> searching for the analysis of massive data sets.*
> Nat Biotechnol 35, 1026-1028 (2017).
>
> Steinegger M & Söding J. *Clustering huge protein sequence sets in linear
> time.* Nat Commun 9, 2542 (2018).

MMseqs2 source: https://github.com/soedinglab/MMseqs2
