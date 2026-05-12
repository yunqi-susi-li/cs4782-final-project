#!/usr/bin/env bash
# Step 7: cross-search a query split (val/test) against train to detect leakage.
# Usage:
#   09_audit_leakage.sh <query_fasta> <target_fasta> <out_prefix> \
#                       <threshold> <threads> <mmseqs> [use_gpu]
set -euo pipefail
QFASTA="$1"
TFASTA="$2"
PREFIX="$3"
THRESHOLD="$4"
THREADS="$5"
MMSEQS="${6:-mmseqs}"
USE_GPU="${7:-0}"

QDB="${PREFIX}.qdb"
TDB="${PREFIX}.tdb"
TMP="${PREFIX}.tmp"
RES="${PREFIX}.res"
M8="${PREFIX}.m8"
mkdir -p "$TMP"

"$MMSEQS" createdb "$QFASTA" "$QDB"
"$MMSEQS" createdb "$TFASTA" "$TDB"

SEARCH_TDB="$TDB"
GPU_ARGS=()
if [[ "$USE_GPU" -eq 1 ]]; then
  TDB_PAD="${TDB}_pad"
  "$MMSEQS" makepaddedseqdb "$TDB" "$TDB_PAD"
  SEARCH_TDB="$TDB_PAD"
  GPU_ARGS=(--gpu 1)
fi

"$MMSEQS" search "$QDB" "$SEARCH_TDB" "$RES" "$TMP" \
  --alignment-mode 3 --cov-mode 0 -c "$THRESHOLD" \
  --max-seqs 50 --threads "$THREADS" "${GPU_ARGS[@]}"
"$MMSEQS" convertalis "$QDB" "$SEARCH_TDB" "$RES" "$M8"
echo "[09_audit_leakage] $PREFIX done."
