#!/bin/bash
#SBATCH --job-name=ddnm_sweep
#SBATCH --partition=compute-gpu
#SBATCH --account=cycle1_iit_shocher_prj
#SBATCH --qos=owner_95
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model/logs/%x_%j.out
#
# Sweep lambda-schedule + damping (and a couple of step/time-travel variants) for
# latent DDNM inpainting. One GPU job runs all configs sequentially; each writes to
# exp/image_samples/sweep_<name>/{VAE,SPNN}/ and prints its average PSNR.

IMAGE=docker://nvcr.io/nvidia/pytorch:25.04-py3

srun --container-image="$IMAGE" \
     --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj \
     --container-workdir=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model/DDNM-main-scartch \
     bash -c '
       set -e
       export HF_HOME=/shared/cycle1_iit_shocher_prj/ron/hf_cache
       export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
       pip install --quiet diffusers transformers scipy pyyaml

       declare -a CONFIGS=(
         "base|"
         "damp0|--damping_floor 0.0"
         "damp05|--damping_floor 0.5"
         "lam_exp|--lambda_mode exp --lambda_rate 5 --lambda_floor 0.1"
         "lam_noise|--lambda_mode noise --lambda_floor 0.1"
         "T20|--T_sampling 20"
         "tt|--travel_length 10 --travel_repeat 2"
       )

       for entry in "${CONFIGS[@]}"; do
         name="${entry%%|*}"; xargs="${entry#*|}"
         echo ""
         echo "=================== SWEEP RUN: ${name} | args: ${xargs} ==================="
         python main.py --config sd15_inpaint_latent.yml --deg inpainting \
           --ni --sigma_y 0 -i "sweep_${name}" ${xargs} || echo "RUN ${name} FAILED"
       done

       echo ""
       echo "=================== SWEEP SUMMARY ==================="
       grep -rH "" exp/image_samples/sweep_*/PSNR.txt 2>/dev/null || true
     '
echo "=== sweep job finished ==="
