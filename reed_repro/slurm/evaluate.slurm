#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p l40s-public
#SBATCH --qos=24h_16g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=reed_eval

set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
mkdir -p logs
source venv/bin/activate
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=1

python -u -m reed_repro.run_repro evaluate --tag "${TAG:-run}"
