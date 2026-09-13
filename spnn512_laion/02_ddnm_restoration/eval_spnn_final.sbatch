#!/bin/bash
#SBATCH -J eval_spnn_final
#SBATCH -p l40s-shared
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /rg/shocher_prj/ron.libman/laion1m/runs/spnn512/eval_restore/slurm_%j.out
# The finished phase-2 codec (final best.pt, EMA) on the benchmark: inpainting (DDNM mask + 66% scattered squares)
# at T=100 and T=200, and colorization at T=20. SPNN only - every SD-VAE result is already cached and reused,
# with the same images, seeds and settings, so the comparison stays paired.
cd /home/ron.libman/churches-reboot || exit 1
OUT=/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/eval_restore
python3 eval_restore_fid.py snapshot --out $OUT --force || exit 1     # freeze the final best.pt -> its own spnn_s<step> tag
torchrun --standalone --nproc_per_node=4 eval_restore_fid.py run --out $OUT --tasks mask,scatter66,color \
    --T 100 --T-color 20 --codecs spnn || exit 1
torchrun --standalone --nproc_per_node=4 eval_restore_fid.py run --out $OUT --tasks mask,scatter66 \
    --T 200 --codecs spnn || exit 1
python3 eval_restore_fid.py report --out $OUT
