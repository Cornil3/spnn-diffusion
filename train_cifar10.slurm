#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p a100-public
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=spnn_cifar10

set -euo pipefail

mkdir -p logs
cd /home/yamitehrlich/work/spnn-diffusion

source venv/bin/activate

accelerate launch --num_processes=1 cifar10_experiment/finetune_decoder_cifar10.py \
  --checkpoint cifar10_experiment/checkpoints/dec1_lp0.1_cy1.0_rt1.0_al0.1_adv0.0_h512/spnn_cifar10_best.pt \
  --hidden 512 \
  --scale_bound 0.2 \
  --num_epochs 100 \
  --alpha 1.0 \
  --lambda_distill 0.0 \
  --lambda_recon 1.0 \
  --lambda_lpips 0.0 \
  --lambda_roundtrip 0.0 \
  --output_dir cifar10_experiment/checkpoints_finetune_decoder_v1


