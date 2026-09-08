#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p l40s-public
#SBATCH --qos=24h_16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=reed_report

# Metrics, then the per-image iteration grids into wandb.
#   TAG           results subdirectory
#   LIMIT         restrict to the first N samples (must match the generate run)
#   WANDB_MODE    set to "offline" if the compute node cannot reach wandb;
#                 then `wandb sync` the run directory from the login node.

set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
mkdir -p logs
source venv/bin/activate

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=1
# wandb's console capture defaults to "auto", which only hooks a tty. Under slurm
# stdout is a file, so the run's Logs tab comes up empty; "wrap" captures anyway.
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}
export REED_SD15_ID=${REED_SD15_ID:-runwayml/stable-diffusion-v1-5}
export WANDB_MODE=${WANDB_MODE:-online}

TAG=${TAG:-run}
PROJECT=${PROJECT:-reed-vae-repro}
ARGS=(--tag "$TAG" --wandb_project "$PROJECT")
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "$LIMIT"); fi
# MODELS scopes the evaluation to one model so a per-model job can fire as soon as that
# model's two codec arms finish, writing metrics_<model>.json / table1_<model>.{md,csv}.
if [[ -n "${MODELS:-}" ]]; then ARGS+=(--models ${MODELS}); fi

python -u -m reed_repro.run_repro evaluate "${ARGS[@]}"
python -u -m reed_repro.run_repro report   "${ARGS[@]}"
