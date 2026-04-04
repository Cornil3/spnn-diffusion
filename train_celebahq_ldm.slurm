#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=celebahq_ldm

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion

source venv/bin/activate

# Step 2: Train unconditional LDM on CelebA-HQ (128x128)
# Uses frozen VAE from Step 1, trains UNet in latent space
accelerate launch --num_processes=7 celebahq_experiment/train_ldm.py
