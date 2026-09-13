#!/bin/bash
#SBATCH -J reed_vefull
#SBATCH -p a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-4
#SBATCH -o /rg/shocher_prj/ron.libman/reed_table1/out/logs/vefull_%A_%a.out
#SBATCH -e /rg/shocher_prj/ron.libman/reed_table1/out/logs/vefull_%A_%a.err
# Full 179-sample REED Table 1 with the hybrid codec: SD-1.5 VAE encoder + SPNN decoder.
EDS=(PbE SDInpaint DiffEdit IP2P MagicBrush)
export ARM_EDITOR=${EDS[$SLURM_ARRAY_TASK_ID]} ARM_CODEC=hybrid
export REED_OUT=/rg/shocher_prj/ron.libman/reed_table1/out_vaeenc_full
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/rg/shocher_prj/ron.libman/pylibs_eval:/rg/shocher_prj/ron.libman/pylibs_qr${PYTHONPATH:+:$PYTHONPATH}
cd /home/ron.libman/churches-reboot || exit 1
echo "arm=$ARM_EDITOR/hybrid  node=$(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)  start=$(date)"
python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir /rg/shocher_prj/ron.libman/reed_table1/out/logs \
    --output "executed_vefull_${ARM_EDITOR}.ipynb" reed_table1_vaeenc.ipynb
echo "exit=$? end=$(date)"
