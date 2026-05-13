#!/usr/bin/env bash
# Sets up HMMER + Pfam Ig variable-domain HMMs for eval_hmmer.py.
#
# Run ONCE on whichever machine will run hmmscan (Mac or Linux).
# Idempotent: re-running is safe.
#
# Usage:
#   bash scripts/setup_hmmer.sh           # default: $HOME/ab_ld4lg/hmmer_db
#   bash scripts/setup_hmmer.sh /custom/path

set -euo pipefail

DB_DIR="${1:-$HOME/ab_ld4lg/hmmer_db}"
mkdir -p "$DB_DIR"
cd "$DB_DIR"
echo "[setup_hmmer] DB_DIR=$DB_DIR"

# ---- 1. HMMER ----
if ! command -v hmmscan >/dev/null 2>&1; then
    echo "[setup_hmmer] hmmscan not found. Attempting install..."
    OS="$(uname -s)"
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install hmmer
        else
            echo "[setup_hmmer] Install Homebrew (https://brew.sh) then re-run." >&2
            exit 1
        fi
    elif [[ "$OS" == "Linux" ]]; then
        # Linux: use conda/pip-style local install (no sudo).
        if command -v conda >/dev/null 2>&1; then
            conda install -y -c bioconda hmmer
        elif command -v module >/dev/null 2>&1 && module avail hmmer 2>&1 | grep -qi hmmer; then
            echo "[setup_hmmer] Try: module load hmmer" >&2
            exit 1
        else
            # Build from source into ~/local
            HMMER_VERSION=3.4
            cd /tmp
            curl -L -o hmmer-${HMMER_VERSION}.tar.gz \
                http://eddylab.org/software/hmmer/hmmer-${HMMER_VERSION}.tar.gz
            tar xf hmmer-${HMMER_VERSION}.tar.gz
            cd hmmer-${HMMER_VERSION}
            ./configure --prefix=$HOME/local
            make -j 4
            make install
            echo 'export PATH=$HOME/local/bin:$PATH' >> ~/.bashrc
            export PATH=$HOME/local/bin:$PATH
            cd "$DB_DIR"
        fi
    fi
fi
hmmscan -h | head -2

# ---- 2. Pfam Ig V-domain HMMs ----
# Three Pfam families that cover antibody variable domains:
#   PF07686  V-set            (Ig V-domain, both VH and VL)
#   PF00047  ig               (immunoglobulin domain)
#   PF13927  Ig_3
# Pulled individually from InterPro/Pfam mirror.

cd "$DB_DIR"
PFAM_BASE="https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam"

for acc in PF07686 PF00047 PF13927; do
    if [[ ! -f "${acc}.hmm" ]]; then
        echo "[setup_hmmer] downloading ${acc}.hmm ..."
        # InterPro hands back a JSON-wrapped HMM; the raw HMM endpoint:
        curl -L -o "${acc}.hmm" "https://www.ebi.ac.uk/interpro/api/entry/pfam/${acc}/?annotation=hmm" || true
        # If curl returned a gzipped HMM, decompress:
        if file "${acc}.hmm" | grep -qi gzip; then
            mv "${acc}.hmm" "${acc}.hmm.gz"
            gunzip "${acc}.hmm.gz"
        fi
    fi
done

# Combine into one DB
cat PF07686.hmm PF00047.hmm PF13927.hmm > Ig.hmm

# ---- 3. Press the DB so hmmscan can use it ----
hmmpress -f Ig.hmm

ls -lh Ig.hmm*
echo "[setup_hmmer] DONE. Use --hmm-db ${DB_DIR}/Ig.hmm"
