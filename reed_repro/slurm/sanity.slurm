#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p l40s-public
#SBATCH --qos=24h_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=reed_sanity

set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
mkdir -p logs
source venv/bin/activate
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=1
export REED_SD15_ID=${REED_SD15_ID:-runwayml/stable-diffusion-v1-5}
export REED_USE_IMAGENHUB=${REED_USE_IMAGENHUB:-0}
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}

CODEC=${CODEC:-vae}
CK=${CKPT:-imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt}
ARGS=(--keys ${KEYS:-102171_t1 100081_t1} --iters ${ITERS:-3} --codec "$CODEC")
if [[ "$CODEC" == "spnn" ]]; then ARGS+=(--spnn_checkpoint "$CK" --latent_scale prescaled); fi

python -u -m reed_repro.sanity_check "${ARGS[@]}"
