#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
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
    --hidden 256 \
    --scale_bound 2.0 \
    --lambda_decoder_distill 1.0 \
    --lambda_decoder_gt 0.0 \
    --lambda_lpips 0.5 \
    --lambda_cycle 0.3 \
    --lambda_roundtrip 0.3 \
    --lambda_align 0.05 \
    --batch_size 8 \
    --max_grad_norm 1.0 \
    --lr 1e-4 \
    --num_epochs 500 \
    --img_size 512 \
    --penrose_batch_size 32 \
    --resume checkpoints_celebahq_distill_512/spnn_vae_best.pt \
    --resume_epoch 99 \
    --output_dir checkpoints_celebahq_distill_512
