#!/usr/bin/env bash
# BCR at 90% identity (more aggressive), CPU. Edit INPUT and OUTPUT.
set -euo pipefail

PIPELINE="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="${INPUT:-/path/to/paired_bcr.pkl}"
OUTPUT="${OUTPUT:-/path/to/output/bcr_090}"

bash "$PIPELINE/run_pipeline.sh" \
    --input    "$INPUT" \
    --output   "$OUTPUT" \
    --receptor bcr \
    --threshold 0.90 \
    --threads  32 \
    --resume
