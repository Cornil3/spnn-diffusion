#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p l40s-public
#SBATCH --qos=24h_16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=reed_verify_codec

set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
mkdir -p logs reed_repro/results
source venv/bin/activate

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=1
# wandb's console capture defaults to "auto", which only hooks a tty. Under slurm
# stdout is a file, so the run's Logs tab comes up empty; "wrap" captures anyway.
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}

# athena still holds the (hub-removed) runwayml copy of SD 1.5; reuse it rather than
# pulling the byte-identical community mirror. Unset to use the mirror.
export REED_SD15_ID=${REED_SD15_ID:-runwayml/stable-diffusion-v1-5}

CKPT=${CKPT:-imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt}
WEIGHTS=${WEIGHTS:-ema}

python -u -m reed_repro.verify_codec --checkpoint "$CKPT" --weights "$WEIGHTS"
