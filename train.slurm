#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p a100-public
#SBATCH --qos=72h_8g
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_vae

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion

source venv/bin/activate

# Multi-GPU training via accelerate (batch_size is per-GPU)
accelerate launch --num_processes=4 main.py --train --test \
    --dataset celebahq \
    --hidden 192 \
    --scale_bound 2.0 \
    --lambda_lpips 0.5 \
    --lambda_cycle 0.3 \
    --lambda_roundtrip 0.3 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_epochs 500 \
    --output_dir checkpoints_celebahq
