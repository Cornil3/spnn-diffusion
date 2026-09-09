#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=churches_i2i

# One arm per job: ARM = compvis | 2blk | 2blk_rt
set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
mkdir -p logs
source venv/bin/activate
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}

ARM=${ARM:?set ARM}
python -u -m reed_repro.churches_img2img \
  --arms "$ARM" \
  --n_images "${N_IMAGES:-100}" \
  --strength "${STRENGTH:-0.5}" \
  --tag "${TAG:-churches_img2img}"
