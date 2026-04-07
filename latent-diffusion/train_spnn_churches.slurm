#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p h200-shared
#SBATCH --qos=72h_8g
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_churches_gan

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion/latent-diffusion

source ../venv/bin/activate

python train_spnn.py --gpus 4
