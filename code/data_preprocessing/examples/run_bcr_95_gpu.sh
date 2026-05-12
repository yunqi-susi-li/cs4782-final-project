#!/usr/bin/env bash
# BCR at 95% identity with GPU acceleration. Requires a GPU-enabled
# MMseqs2 build of the same version as the CPU build. Pass it via MMSEQS_GPU
# below or with --mmseqs PATH. Only steps 3 and 7 run on the GPU; step 4
# (linclust) is CPU.
#
# Validate identity enforcement on a small sample first: after the run,
# 07_audit/*.summary.tsv must report frac_ge_<threshold> = 0.0. Some GPU
# builds have shipped with bugs that silently bypass --min-seq-id; rerun on
# CPU if the audit is non-zero.
set -euo pipefail

PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${INPUT:-/path/to/paired_bcr.pkl}"
OUTPUT="${OUTPUT:-/path/to/output/bcr_095_gpu}"
MMSEQS_GPU="${MMSEQS_GPU:-/path/to/mmseqs-gpu/bin/mmseqs}"

bash "$PIPELINE/run_pipeline.sh" \
    --input     "$INPUT" \
    --output    "$OUTPUT" \
    --receptor  bcr \
    --threshold 0.95 \
    --threads   32 \
    --gpu \
    --mmseqs    "$MMSEQS_GPU" \
    --resume
