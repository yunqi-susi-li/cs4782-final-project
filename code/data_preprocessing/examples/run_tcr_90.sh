#!/usr/bin/env bash
# TCR (paired beta + alpha) at 90% identity, CPU.
set -euo pipefail

PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${INPUT:-/path/to/paired_tcr.pkl}"
OUTPUT="${OUTPUT:-/path/to/output/tcr_090}"

bash "$PIPELINE/run_pipeline.sh" \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --receptor tcr \
    --threshold 0.90 \
    --threads  32 \
    --resume
