# TCR Sequence Similarity Reduction: Usage Guide

A walkthrough for running the pipeline on a paired TCR (β + α) dataset. The
TCR code path is identical to BCR; only the column-name suffixes differ.
The pipeline is CPU-only by default and built on MMseqs2 v18+
(https://github.com/soedinglab/MMseqs2). Optional GPU acceleration is
available for the search steps; see [`README.md`](README.md).

---

## 1. Required columns in your input pickle (TCR mode)

| Column | Required? | Description |
|--------|-----------|-------------|
| `sequence_id_beta`             | yes        | TRB unique id |
| `sequence_id_alpha`            | yes        | TRA unique id |
| `sequence_alignment_aa_beta`   | yes        | TRB full V-region aa sequence (clustered on this) |
| `sequence_alignment_aa_alpha`  | yes        | TRA full V-region aa sequence |
| `cdr3_aa_beta`                 | yes        | TRB CDR3 |
| `cdr3_aa_alpha`                | yes        | TRA CDR3 |
| `productive_beta` / `productive_alpha`        | recommended | T/F |
| `v_call_beta`, `j_call_beta`, `v_call_alpha`, `j_call_alpha` | recommended | gene calls (`*01` allele suffix is fine) |
| `species`                      | recommended | e.g. human / mouse |

Any extra columns are preserved as-is and copied through to the final
`train` / `val` / `test` pickles.

If your column names differ (e.g. `chainA` / `chainB`), override the default
suffixes with `--chain1 chainA --chain2 chainB` instead of editing the
source file.

---

## 2. One command to run the full pipeline

Two ready-to-go example scripts are included:

- [`examples/run_tcr_95.sh`](examples/run_tcr_95.sh): TCR at 95% identity
- [`examples/run_tcr_90.sh`](examples/run_tcr_90.sh): TCR at 90% identity

Edit the `--input` and `--output` paths to point at your data, then:

```bash
# Make sure mmseqs is reachable (e.g. activate a conda env)
# or pass it explicitly with --mmseqs /path/to/mmseqs
bash examples/run_tcr_95.sh
```

Or call the main entry point directly:

```bash
bash run_pipeline.sh \
    --input    /path/to/your_tcr.pkl \
    --output   /path/to/output/tcr_095 \
    --receptor tcr \
    --threshold 0.95 \
    --threads  32 \
    --resume
```

`--resume` is idempotent: if any step crashes, re-running skips the steps
that already completed.

Running both thresholds is common: 95% is the default starting point, and
90% is more aggressive (clonally-related sequences from the same donor get
squeezed harder), useful for ablations.

TCR file-name note: internally the pipeline calls chain1 `heavy` and chain2
`light` just to reuse file names; the content is your β / α. So
`train.heavy.fasta` = TRB and `train.light.fasta` = TRA. This is purely a
filename convention; the metadata column names stay as `*_beta` / `*_alpha`.

---

## 3. What you get after the run

```
<output>/06_export/
  ├── train/    train.{heavy=beta,light=alpha,pair}.fasta + train.metadata.tsv + train.pkl
  ├── val/      same structure
  └── test/     same structure
<output>/07_audit/
  ├── val_vs_train_heavy.summary.tsv      # must have frac_ge_95 (or frac_ge_90) = 0.0
  ├── val_vs_train_light.summary.tsv
  ├── test_vs_train_heavy.summary.tsv
  └── test_vs_train_light.summary.tsv
```

Loading in Python:

```python
import pandas as pd
train = pd.read_pickle("<output>/06_export/train/train.pkl")
val   = pd.read_pickle("<output>/06_export/val/val.pkl")
test  = pd.read_pickle("<output>/06_export/test/test.pkl")
print(train.shape, val.shape, test.shape)
```

`train.pair.fasta` already concatenates β + a `GGGGSGGGGS` linker + α, so it
can be fed straight into ESM or other paired-chain protein language models.

---

## 4. Acceptance check

After the run, inspect the four leakage-audit summary files. Each must have
`frac_ge_95 = 0.0` (if you ran at the 0.90 threshold, check `frac_ge_90 = 0.0`
instead):

```bash
cat <output>/07_audit/val_vs_train_heavy.summary.tsv
cat <output>/07_audit/val_vs_train_light.summary.tsv
cat <output>/07_audit/test_vs_train_heavy.summary.tsv
cat <output>/07_audit/test_vs_train_light.summary.tsv
```

If any value is non-zero, do not use this split. A non-zero leakage usually
indicates a bad MMseqs2 build (at least one v18 GPU build has been observed
to silently mis-handle `--min-seq-id`); rerun the search steps on CPU and
re-audit.

---

## 5. Resource / runtime estimates (CPU)

Step 3 (all-vs-all NN search) is the bottleneck and scales roughly O(N^2).

| Dataset size | 32 cores | 16 cores |
|--------------|----------|----------|
| 100K pairs   | ~10 min  | ~20 min  |
| 500K pairs   | ~1.5 h   | ~3 h     |
| 2M pairs     | ~12 h    | ~24 h    |
| 3M pairs     | ~24 h    | ~48 h    |

GPU acceleration (`--gpu`; see main README) typically gives ~10x speedup
on the two search steps when the build behaves correctly.

For SLURM, see [`examples/slurm_submit.sh`](examples/slurm_submit.sh).

---

## 6. Dependencies

```bash
# Option A: install via conda (works anywhere)
conda create -n mmseqs2 -c conda-forge -c bioconda mmseqs2 python pandas numpy -y
conda activate mmseqs2

# Option B: many HPC clusters provide a system MMseqs2 build at
# /programs/mmseqs-*/bin/mmseqs or /opt/mmseqs/bin/mmseqs; pass it via
#   --mmseqs /path/to/mmseqs
```

For GPU mode, additionally a GPU-enabled build of MMseqs2 of the same
version. See the main [`README.md`](README.md) GPU section.

---

## 7. Troubleshooting

- `mmseqs: command not found`: activate the MMseqs2 conda env, or pass
  `--mmseqs /path/to/mmseqs`.
- `Missing required columns: [...]`: column names are not `*_beta` /
  `*_alpha`; use `--chain1` / `--chain2` to point at the right suffixes.
- Pipeline crashes mid-step: re-run with `--resume`; every step is
  idempotent.
- Test on a small subset first:
  ```python
  import pandas as pd
  pd.read_pickle("your.pkl").head(10000).to_pickle("test_small.pkl")
  ```
  Finishes in a few minutes and verifies that the input format is correct.

---

For the full input-pickle column requirements (BCR + TCR), output schema,
and per-step internals, see [`PIPELINE.md`](PIPELINE.md).
