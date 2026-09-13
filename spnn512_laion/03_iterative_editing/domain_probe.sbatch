#!/bin/bash
#SBATCH -J dom_probe
#SBATCH -p a100-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o /rg/shocher_prj/ron.libman/reed_table1/out/logs/domprobe_%j.out
export HF_HUB_OFFLINE=1
export PYTHONPATH=/rg/shocher_prj/ron.libman/pylibs_eval:/rg/shocher_prj/ron.libman/pylibs_qr
cd /home/ron.libman/churches-reboot && python3 /rg/shocher_prj/ron.libman/codec_domain_probe.py
