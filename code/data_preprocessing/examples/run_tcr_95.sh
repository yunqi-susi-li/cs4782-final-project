#!/usr/bin/env bash
# TCR (paired beta + alpha) at 95% identity, CPU. Input pkl must have
# sequence_alignment_aa_beta / sequence_alignment_aa_alpha columns.
set -euo pipefail

PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${INPUT:-/path/to/paired_tcr.pkl}"
OUTPUT="${OUTPUT:-/path/to/output/tcr_095}"

bash "$PIPELINE/run_pipeline.sh" \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --receptor tcr \
    --threshold 0.95 \
    --threads  32 \
    --resume
