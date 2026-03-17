#!/bin/bash
#SBATCH --job-name=FMTL_Experiment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo

# --- project root (EDIT THIS) ---
PROJECT_DIR="/tc1home/FYP/eyong002/FMTL-Experiments"
cd "$PROJECT_DIR"

# Slurm stdout/err directory (because #SBATCH --output/--error uses it)
mkdir -p logs

# JSON/JSONL results directory (project root/results)
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Fail fast if not writable
touch "$RESULTS_DIR/.write_test" && rm -f "$RESULTS_DIR/.write_test"

# --- environment ---
module load cuda/11.8
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate fmtl

# Make conda-provided CUDA/cuDNN visible first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "PWD:      $(pwd)"
echo "Node:     $(node -v)"
echo "NPM:      $(npm -v)"
echo "Results:  $RESULTS_DIR"
echo

echo "== NVIDIA-SMI =="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo

# Defaults (can override with env vars when sbatch)
RUN_ID="${RUN_ID:-abl_$(date -u +"%Y-%m-%dT%H-%M-%SZ")}"

echo "== Run settings =="
echo "RUN_ID=$RUN_ID"
echo "Writing to: $RESULTS_DIR/"
echo

echo "== Build =="
npx tsc -p tsconfig.node.json

echo "== Running experiment =="
set -x

which npx
npx --version
npx --yes tsx -v

ls -la src/experiments/quantity_skew_high.ts

# npx --yes tsx src/experiments/quantity_skew_high.ts \
node dist/experiments/quantity_skew_high.js \
  --logDir "$RESULTS_DIR" \
  --runId "$RUN_ID"

set +x

echo
echo "Done. Logs written to: ${RESULTS_DIR}/${RUN_ID}.jsonl"

# sbatch shell/label_skew_med.sh