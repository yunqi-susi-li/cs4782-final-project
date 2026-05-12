#!/usr/bin/env bash
# Step 3: all-vs-all nearest-neighbor search at the chosen identity threshold.
# Usage:
#   03_nn_search.sh <fasta> <out_prefix> <threshold> <threads> <mmseqs> [use_gpu]
set -euo pipefail
FASTA="$1"
PREFIX="$2"
THRESHOLD="$3"
THREADS="$4"
MMSEQS="${5:-mmseqs}"
USE_GPU="${6:-0}"

DB="${PREFIX}.db"
TMP="${PREFIX}.tmp"
RES="${PREFIX}.res"
M8="${PREFIX}.m8"
mkdir -p "$TMP"

"$MMSEQS" createdb "$FASTA" "$DB"

SEARCH_DB="$DB"
GPU_ARGS=()
if [[ "$USE_GPU" -eq 1 ]]; then
  PAD_DB="${DB}_pad"
  "$MMSEQS" makepaddedseqdb "$DB" "$PAD_DB"
  SEARCH_DB="$PAD_DB"
  GPU_ARGS=(--gpu 1)
fi

"$MMSEQS" search "$DB" "$SEARCH_DB" "$RES" "$TMP" \
  --alignment-mode 3 --cov-mode 0 -c "$THRESHOLD" \
  --max-seqs 50 --threads "$THREADS" "${GPU_ARGS[@]}"
"$MMSEQS" convertalis "$DB" "$SEARCH_DB" "$RES" "$M8"
echo "[03_nn_search] $PREFIX done."
