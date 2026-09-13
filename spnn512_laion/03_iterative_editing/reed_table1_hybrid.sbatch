#!/bin/bash
#SBATCH -J reed_hybrid
#SBATCH -p a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --array=0-9%8
#SBATCH -o /rg/shocher_prj/ron.libman/reed_table1/out/logs/arm_%A_%a.out
# REED-VAE Table 1, hybrid: one (editor, codec) arm per array task, 179 ImagenHub samples,
# 25 alternating edits each, scored at 5/15/25. SPNN arms are tasks 0-4 so they finish first.
ARMS=(IP2P:spnn MagicBrush:spnn DiffEdit:spnn PbE:spnn SDInpaint:spnn \
      IP2P:vanilla MagicBrush:vanilla DiffEdit:vanilla PbE:vanilla SDInpaint:vanilla)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
export ARM_EDITOR=${A%%:*} ARM_CODEC=${A##*:}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/rg/shocher_prj/ron.libman/pylibs_eval:/rg/shocher_prj/ron.libman/pylibs_qr${PYTHONPATH:+:$PYTHONPATH}
cd /home/ron.libman/churches-reboot || exit 1
echo "arm=$ARM_EDITOR/$ARM_CODEC node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir /rg/shocher_prj/ron.libman/reed_table1/out/logs \
    --output "executed_${ARM_EDITOR}_${ARM_CODEC}.ipynb" reed_table1_hybrid.ipynb
echo "exit=$? end=$(date)"
