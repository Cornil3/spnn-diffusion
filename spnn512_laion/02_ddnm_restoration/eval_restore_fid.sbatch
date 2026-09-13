#!/bin/bash
#SBATCH -J eval_restore
#SBATCH -p l40s-shared
#SBATCH --qos=24h_4g
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /rg/shocher_prj/ron.libman/laion1m/runs/spnn512/eval_restore/slurm_%j.out
# Restoration benchmark (eval_restore_fid.py): latent DDNM inpainting + colorization, SD-VAE vs SPNN, FID/KID/PSNR/SSIM/LPIPS.
# Resumable: resubmit the same command to continue.
# Override: TASKS=mask,scatter66 T=100 T_COLOR=20 VAE_DTYPE=bf16 sbatch eval_restore_fid.sbatch
cd /home/ron.libman/churches-reboot || exit 1
OUT=/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/eval_restore
TASKS=${TASKS:-mask,scatter66,color}
T=${T:-100}                 # inpainting DDNM steps
T_COLOR=${T_COLOR:-20}      # colorization DDNM steps
VAE_DTYPE=${VAE_DTYPE:-fp32}
python3 eval_restore_fid.py prepare  --out $OUT || exit 1        # keeps an existing manifest
python3 eval_restore_fid.py snapshot --out $OUT || exit 1        # freezes best.pt (EMA) once; later resubmits reuse it
torchrun --standalone --nproc_per_node=4 eval_restore_fid.py run --out $OUT --tasks $TASKS --T $T --T-color $T_COLOR \
    --vae-dtype $VAE_DTYPE || exit 1
python3 eval_restore_fid.py report --out $OUT
