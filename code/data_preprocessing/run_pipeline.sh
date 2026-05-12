#!/usr/bin/env bash
# Paired-chain sequence similarity reduction pipeline.
# Built on MMseqs2 v18+. CPU by default; --gpu accelerates the two search
# steps. See README.md for the full reference.
set -euo pipefail

INPUT=""
OUTPUT=""
RECEPTOR="bcr"
CHAIN1=""
CHAIN2=""
THRESHOLD="0.95"
TRAIN_FRAC="0.96"
VAL_FRAC="0.02"
TEST_FRAC="0.02"
THREADS="$(nproc 2>/dev/null || echo 8)"
RESUME=0
SKIP_AUDIT=0
USE_GPU=0
MMSEQS="${MMSEQS:-}"
PYTHON="${PYTHON:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$SCRIPT_DIR/scripts"

usage() {
  cat <<'EOF'
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
  --threshold FLOAT    Identity threshold in (0,1]. Default 0.95.
  --train-frac FLOAT   Train fraction (default 0.96)
  --val-frac FLOAT     Val fraction   (default 0.02)
  --test-frac FLOAT    Test fraction  (default 0.02)
  --threads INT        CPU threads (default: all available)
  --gpu                Enable --gpu 1 for steps 3 and 7.
                       Requires a GPU-enabled mmseqs build (pass --mmseqs).
                       Step 4 (linclust) always runs on CPU.
  --resume             Skip steps whose output already exists
  --skip-audit         Skip Step 7 (leakage audit)
  --mmseqs PATH        Override path to mmseqs
  --python PATH        Override path to python
  --help               Show this message
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)       INPUT="$2"; shift 2 ;;
    --output)      OUTPUT="$2"; shift 2 ;;
    --receptor)    RECEPTOR="$2"; shift 2 ;;
    --chain1)      CHAIN1="$2"; shift 2 ;;
    --chain2)      CHAIN2="$2"; shift 2 ;;
    --threshold)   THRESHOLD="$2"; shift 2 ;;
    --train-frac)  TRAIN_FRAC="$2"; shift 2 ;;
    --val-frac)    VAL_FRAC="$2"; shift 2 ;;
    --test-frac)   TEST_FRAC="$2"; shift 2 ;;
    --threads)     THREADS="$2"; shift 2 ;;
    --resume)      RESUME=1; shift ;;
    --skip-audit)  SKIP_AUDIT=1; shift ;;
    --gpu)         USE_GPU=1; shift ;;
    --mmseqs)      MMSEQS="$2"; shift 2 ;;
    --python)      PYTHON="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "ERROR: unknown option: $1" >&2; usage ;;
  esac
done

[[ -z "$INPUT"  ]] && { echo "ERROR: --input is required" >&2; exit 1; }
[[ -z "$OUTPUT" ]] && { echo "ERROR: --output is required" >&2; exit 1; }
[[ -f "$INPUT"  ]] || { echo "ERROR: input not found: $INPUT" >&2; exit 1; }

if [[ -n "$CHAIN1" && -n "$CHAIN2" ]]; then
  :
elif [[ "$RECEPTOR" == "bcr" ]]; then
  CHAIN1="heavy"; CHAIN2="light"
elif [[ "$RECEPTOR" == "tcr" ]]; then
  CHAIN1="beta";  CHAIN2="alpha"
else
  echo "ERROR: --receptor must be 'bcr' or 'tcr', or provide --chain1/--chain2" >&2
  exit 1
fi

if ! awk -v t="$THRESHOLD" 'BEGIN{ if (t+0 > 0 && t+0 <= 1) exit 0; exit 1 }'; then
  echo "ERROR: --threshold must be in (0, 1], got: $THRESHOLD" >&2
  exit 1
fi

if [[ -z "$MMSEQS" ]]; then
  for cand in \
      /programs/mmseqs-avx2-18/bin/mmseqs \
      /programs/mmseqs-sse4-18/bin/mmseqs \
      /programs/mmseqs-avx2-15/bin/mmseqs \
      /programs/mmseqs-avx2-14/bin/mmseqs \
      /opt/mmseqs/bin/mmseqs \
      "$HOME/miniconda3/envs/mmseqs2/bin/mmseqs"; do
    if [[ -x "$cand" ]]; then MMSEQS="$cand"; break; fi
  done
  if [[ -z "$MMSEQS" ]] && command -v mmseqs >/dev/null 2>&1; then
    MMSEQS="$(command -v mmseqs)"
  fi
  if [[ -z "$MMSEQS" ]]; then
    echo "ERROR: mmseqs not found. Pass --mmseqs PATH, or install via conda:" >&2
    echo "       conda install -c bioconda mmseqs2" >&2
    exit 1
  fi
fi

# GPU builds need a padded DB before search. In CPU mode, swap to a CPU build
# if the auto-detected binary is GPU-only to avoid 'not a valid GPU database'.
if [[ "$USE_GPU" -ne 1 ]]; then
  case "$MMSEQS" in
    *mmseqs-gpu*)
      for cand in /programs/mmseqs-avx2-18/bin/mmseqs /programs/mmseqs-sse4-18/bin/mmseqs /opt/mmseqs/bin/mmseqs; do
        if [[ -x "$cand" ]]; then MMSEQS="$cand"; break; fi
      done
      ;;
  esac
fi

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$HOME/miniconda3/envs/scanpy/bin/python" ]]; then
    PYTHON="$HOME/miniconda3/envs/scanpy/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    echo "ERROR: python3 not found. Pass --python PATH." >&2
    exit 1
  fi
fi

TAG="${THRESHOLD/./}"
mkdir -p "$OUTPUT/logs"

cat <<EOF | tee "$OUTPUT/logs/run.banner.txt"
==============================================================================
  Sequence Similarity Reduction Pipeline
==============================================================================
  Date:        $(date)
  Host:        $(hostname)
  Input:       $INPUT
  Output:      $OUTPUT
  Receptor:    $RECEPTOR  (chain1=$CHAIN1, chain2=$CHAIN2)
  Threshold:   $THRESHOLD
  Split:       train=$TRAIN_FRAC val=$VAL_FRAC test=$TEST_FRAC
  Threads:     $THREADS
  GPU:         $([[ "$USE_GPU" -eq 1 ]] && echo "enabled for steps 3 and 7" || echo "disabled")
  Resume:      $RESUME    Skip audit: $SKIP_AUDIT
  mmseqs:      $MMSEQS  ($("$MMSEQS" version))
  python:      $PYTHON  ($("$PYTHON" --version 2>&1))
==============================================================================
EOF

RAW="$OUTPUT/01_raw"
EXACT="$OUTPUT/02_exact"
NN="$OUTPUT/03_nn"
CLU="$OUTPUT/04_clusters"
SPLIT="$OUTPUT/05_split"
EXPORT="$OUTPUT/06_export"
AUDIT="$OUTPUT/07_audit"

step_done() {
  [[ "$RESUME" -eq 1 && -f "$1" ]]
}

# Step 1: Extract
if step_done "$RAW/summary.tsv"; then
  echo "[$(date)] === Step 1: Extract — SKIPPED (resume) ==="
else
  echo "[$(date)] === Step 1: Extract ==="
  "$PYTHON" "$SCRIPTS/01_extract.py" \
    --input "$INPUT" --output "$RAW" \
    --chain1 "$CHAIN1" --chain2 "$CHAIN2" \
    2>&1 | tee "$OUTPUT/logs/01_extract.log"
fi

# Step 2: Exact dedup
if step_done "$EXACT/exact_dedup.summary.tsv"; then
  echo "[$(date)] === Step 2: Exact dedup — SKIPPED (resume) ==="
else
  echo "[$(date)] === Step 2: Exact dedup ==="
  "$PYTHON" "$SCRIPTS/02_exact_dedup.py" \
    "$RAW/metadata.tsv" "$EXACT" \
    2>&1 | tee "$OUTPUT/logs/02_exact.log"
fi

# Step 3: NN self-search
echo "[$(date)] === Step 3: NN self-search ($([[ "$USE_GPU" -eq 1 ]] && echo GPU || echo CPU), $THREADS threads) ==="
mkdir -p "$NN"
for CHAIN in heavy light; do
  TOP="$NN/$CHAIN.top.tsv"
  if step_done "$TOP"; then
    echo "[$(date)] NN $CHAIN — SKIPPED (resume)"
    continue
  fi
  echo "[$(date)] NN self-search: $CHAIN ..."
  bash "$SCRIPTS/03_nn_search.sh" \
    "$EXACT/${CHAIN}_full.exact.fasta" "$NN/$CHAIN" \
    "$THRESHOLD" "$THREADS" "$MMSEQS" "$USE_GPU" \
    2>&1 | tee "$OUTPUT/logs/03_nn_${CHAIN}.log"
  "$PYTHON" "$SCRIPTS/04_pick_top_nonself_hit.py" "$NN/$CHAIN.m8" "$TOP" \
    2>&1 | tee -a "$OUTPUT/logs/03_nn_${CHAIN}.log"
done

# Step 4: Clustering (linclust)
echo "[$(date)] === Step 4: Clustering (linclust at $THRESHOLD) ==="
for CHAIN in heavy light; do
  OUTDIR="$CLU/${CHAIN}_${TAG}"
  if step_done "$OUTDIR/clu.tsv"; then
    echo "[$(date)] Cluster $CHAIN — SKIPPED (resume)"
    continue
  fi
  echo "[$(date)] Clustering: $CHAIN ..."
  bash "$SCRIPTS/05_cluster.sh" \
    "$EXACT/${CHAIN}_full.exact.fasta" "$OUTDIR" \
    "$THRESHOLD" "$THREADS" "$MMSEQS" \
    2>&1 | tee "$OUTPUT/logs/04_cluster_${CHAIN}.log"
  "$PYTHON" "$SCRIPTS/06_summarize_clusters.py" \
    "$EXACT/metadata.exact.tsv" "$OUTDIR/clu.tsv" "$OUTDIR/summary" \
    2>&1 | tee -a "$OUTPUT/logs/04_cluster_${CHAIN}.log"
done

# Step 5: Build groups + split
SPLIT_PREFIX="$SPLIT/scheme_c_${TAG}"
if step_done "${SPLIT_PREFIX}.assignments.tsv"; then
  echo "[$(date)] === Step 5: Split — SKIPPED (resume) ==="
else
  echo "[$(date)] === Step 5: Split (tuple mode) ==="
  mkdir -p "$SPLIT"
  "$PYTHON" "$SCRIPTS/07_build_groups_and_split.py" \
    "$EXACT/metadata.exact.tsv" \
    "$CLU/heavy_${TAG}/clu.tsv" \
    "$CLU/light_${TAG}/clu.tsv" \
    "$SPLIT_PREFIX" \
    --group-mode tuple \
    --train-frac "$TRAIN_FRAC" --val-frac "$VAL_FRAC" --test-frac "$TEST_FRAC" \
    2>&1 | tee "$OUTPUT/logs/05_split.log"
fi

# Step 6: Export train / val / test
if step_done "$EXPORT/train/train.pkl"; then
  echo "[$(date)] === Step 6: Export — SKIPPED (resume) ==="
else
  echo "[$(date)] === Step 6: Export ==="
  mkdir -p "$EXPORT"
  "$PYTHON" "$SCRIPTS/08_export_splits.py" \
    "${SPLIT_PREFIX}.assignments.tsv" "$EXPORT" \
    --collapse-level tuple --input-pickle "$INPUT" \
    2>&1 | tee "$OUTPUT/logs/06_export.log"
fi

# Step 7: Leakage audit
if [[ "$SKIP_AUDIT" -eq 1 ]]; then
  echo "[$(date)] === Step 7: Leakage audit — SKIPPED (--skip-audit) ==="
else
  echo "[$(date)] === Step 7: Leakage audit ($([[ "$USE_GPU" -eq 1 ]] && echo GPU || echo CPU)) ==="
  mkdir -p "$AUDIT"
  for CHAIN in heavy light; do
    for SPLIT_NAME in val test; do
      PREFIX="$AUDIT/${SPLIT_NAME}_vs_train_${CHAIN}"
      SUMMARY="${PREFIX}.summary.tsv"
      if step_done "$SUMMARY"; then
        echo "[$(date)] Audit ${SPLIT_NAME}_${CHAIN} — SKIPPED (resume)"
        continue
      fi
      echo "[$(date)] Audit: ${SPLIT_NAME} vs train, ${CHAIN} ..."
      bash "$SCRIPTS/09_audit_leakage.sh" \
        "$EXPORT/$SPLIT_NAME/$SPLIT_NAME.$CHAIN.fasta" \
        "$EXPORT/train/train.$CHAIN.fasta" \
        "$PREFIX" \
        "$THRESHOLD" "$THREADS" "$MMSEQS" "$USE_GPU" \
        2>&1 | tee "$OUTPUT/logs/07_audit_${SPLIT_NAME}_${CHAIN}.log"
      "$PYTHON" "$SCRIPTS/10_summarize_leakage.py" "${PREFIX}.m8" "$PREFIX" \
        2>&1 | tee -a "$OUTPUT/logs/07_audit_${SPLIT_NAME}_${CHAIN}.log"
    done
  done
fi

echo ""
echo "=============================================================================="
echo "[$(date)] PIPELINE COMPLETE"
echo "Output: $OUTPUT"
echo ""
echo "--- Cluster summary ---"
for CHAIN in heavy light; do
  SUMM="$CLU/${CHAIN}_${TAG}/summary.summary.tsv"
  [[ -f "$SUMM" ]] && { echo "[$CHAIN]"; head -5 "$SUMM"; echo; }
done
echo "--- Split summary ---"
[[ -f "${SPLIT_PREFIX}.split_summary.tsv" ]] && cat "${SPLIT_PREFIX}.split_summary.tsv"
echo ""
if [[ "$SKIP_AUDIT" -ne 1 ]]; then
  echo "--- Leakage audit (frac_ge_${TAG} should be 0.0) ---"
  for f in "$AUDIT"/*.summary.tsv; do
    [[ -f "$f" ]] || continue
    echo "[$(basename "$f")]"
    cat "$f"
    echo
  done
fi
echo "=============================================================================="
echo "Final data: $EXPORT/{train,val,test}/"
echo "=============================================================================="
