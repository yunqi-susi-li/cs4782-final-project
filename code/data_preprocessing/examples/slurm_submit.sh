#!/usr/bin/env bash
#SBATCH --job-name=seq_sim_reduction
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# SLURM submission template, CPU. Adjust --partition / --cpus-per-task /
# --mem / --time for your cluster, then submit with:
#
#   sbatch examples/slurm_submit.sh \
#          --input  /path/to/data.pkl \
#          --output /path/to/output_dir \
#          --receptor bcr --threshold 0.95
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate mmseqs2 2>/dev/null || true

PIPELINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

bash "$PIPELINE_DIR/run_pipeline.sh" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --resume \
    "$@"
