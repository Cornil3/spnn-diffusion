#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_churches_gan

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion/latent-diffusion

source ../venv/bin/activate

python main.py --base configs/spnn_churches.yaml -t --gpus 0, 1
