#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p a100-public
#SBATCH --qos=72h_8g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_vae

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion

source venv/bin/activate

python -u main.py --train --test \
    --dataset celebahq \
    --output_dir checkpoints_celebahq
