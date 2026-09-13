#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p h200-shared
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=churches_eval4
set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
source venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u -m reed_repro.churches_eval --tag churches_img2img   --arms compvis 2blk 2blk_rt 2blk_dist
echo; echo "########## PAIRED TEST: 2blk vs 2blk_dist ##########"; echo
python -u -m reed_repro.churches_rt_test --tag churches_img2img --a 2blk --b 2blk_dist
