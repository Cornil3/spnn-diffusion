#!/bin/bash
#SBATCH -J reed_final
#SBATCH -p a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o /rg/shocher_prj/ron.libman/reed_table1/out/logs/final_%j.out
export PYTHONPATH=/rg/shocher_prj/ron.libman/pylibs_eval:/rg/shocher_prj/ron.libman/pylibs_qr
export HF_HUB_OFFLINE=1
cd /home/ron.libman/churches-reboot && python3 reed_score_final.py
