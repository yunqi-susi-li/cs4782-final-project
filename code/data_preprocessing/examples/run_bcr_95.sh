#!/usr/bin/env bash
# BCR at 95% identity, CPU. Edit INPUT and OUTPUT to point at your data.
set -euo pipefail

PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${INPUT:-/path/to/paired_bcr.pkl}"
OUTPUT="${OUTPUT:-/path/to/output/bcr_095}"

bash "$PIPELINE/run_pipeline.sh" \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --receptor bcr \
    --threshold 0.95 \
    --threads  32 \
    --resume
