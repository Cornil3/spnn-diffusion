#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p l40s-public
#SBATCH --qos=4h_0g
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=reed_report_cpu

# Grid reporting needs no GPU -- it is PIL composition plus wandb uploads -- so it runs
# on the 4h_0g (0-GPU, 10-concurrent) QOS instead of queueing behind the generate arms.
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
ARGS=(--tag "$TAG" --wandb_project "${PROJECT:-reed-vae-repro}")
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "$LIMIT"); fi

python -u -m reed_repro.run_repro report "${ARGS[@]}"
