#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_churches_gt

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion

source venv/bin/activate

# Train SPNN on LSUN Churches with GT loss + CompVis KL-VAE (clean start)
accelerate launch --num_processes=6 main.py --train --test \
    --dataset lsun_churches \
    --hidden 256 \
    --scale_bound 2.0 \
    --lambda_decoder_distill 0.0 \
    --lambda_decoder_gt 1.0 \
    --lambda_lpips 0.5 \
    --lambda_cycle 0.3 \
    --lambda_roundtrip 0.3 \
    --lambda_align 0.1 \
    --batch_size 16 \
    --max_grad_norm 1.0 \
    --lr 1e-4 \
    --num_epochs 500 \
    --img_size 256 \
    --penrose_batch_size 128 \
    --save_every 50 \
    --num_save_images 30 \
    --output_dir checkpoints_churches_gt \
    --sample_dir samples_churches_gt \
    --compvis_ckpt DDNM-main/models/lsun_churches.ckpt \
    --compvis_model_config DDNM-main/models/lsun_churches-ldm-kl-8.yaml
