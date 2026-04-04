#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=celebahq_spnn

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion

source venv/bin/activate

# Step 3: Distill SPNN from trained VAE on CelebA-HQ (128x128)
accelerate launch --num_processes=8 celebahq_experiment/train_spnn.py \
    --num_epochs 100 \
    --batch_size 64 \
