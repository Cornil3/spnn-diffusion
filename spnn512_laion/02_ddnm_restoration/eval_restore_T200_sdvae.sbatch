#!/bin/bash
#SBATCH -J eval_T200_sdvae
#SBATCH -p l40s-shared
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /rg/shocher_prj/ron.libman/laion1m/runs/spnn512/eval_restore/slurm_%j.out
# The same benchmark, both inpainting tasks at T=200, SD-VAE only (fp32; same images, seeds and settings as the
# T=100 run). Results land next to the T=100 ones: res/sdvae_fp32/{mask,scatter66}_T200_*. Resumable.
# SPNN at T=200 later (same seeds, so the comparison is paired):
#   python3 eval_restore_fid.py snapshot --out $OUT --force          # only to switch to a newer best.pt
#   torchrun --standalone --nproc_per_node=4 eval_restore_fid.py run --out $OUT --tasks mask,scatter66 --T 200 --codecs spnn
cd /home/ron.libman/churches-reboot || exit 1
OUT=/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/eval_restore
torchrun --standalone --nproc_per_node=4 eval_restore_fid.py run --out $OUT --tasks mask,scatter66 --T 200 \
    --codecs sdvae --vae-dtype fp32 || exit 1
python3 eval_restore_fid.py report --out $OUT
