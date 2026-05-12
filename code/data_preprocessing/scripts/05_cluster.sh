#!/usr/bin/env bash
# Step 4: linclust at the chosen identity threshold. linclust is CPU-only.
# Usage: 05_cluster.sh <fasta> <out_dir> <threshold> <threads> <mmseqs>
set -euo pipefail
FASTA="$1"
OUTDIR="$2"
THRESHOLD="$3"
THREADS="$4"
MMSEQS="${5:-mmseqs}"

mkdir -p "$OUTDIR"
DB="$OUTDIR/seqDB"
CLUDB="$OUTDIR/clu"
TMP="$OUTDIR/tmp"
mkdir -p "$TMP"

"$MMSEQS" createdb "$FASTA" "$DB"
"$MMSEQS" linclust "$DB" "$CLUDB" "$TMP" \
  --min-seq-id "$THRESHOLD" --alignment-mode 3 \
  --cov-mode 0 -c "$THRESHOLD" \
  --threads "$THREADS"
"$MMSEQS" createtsv "$DB" "$DB" "$CLUDB" "$OUTDIR/clu.tsv"
"$MMSEQS" createsubdb "$CLUDB" "$DB" "$OUTDIR/rep_db"
"$MMSEQS" convert2fasta "$OUTDIR/rep_db" "$OUTDIR/rep.fasta"
echo "[05_cluster] $OUTDIR done."
