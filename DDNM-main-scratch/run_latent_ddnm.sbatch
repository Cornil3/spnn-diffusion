#!/bin/bash
#SBATCH --job-name=ddnm_latent
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
# Clean-room latent-space DDNM (SD1.5 UNet + {KL-VAE, SPNN-512}), 512x512 inpainting.
# Usage:
#   sbatch run_latent_ddnm.sbatch <run_name> [extra args for main.py]
# Examples:
#   sbatch run_latent_ddnm.sbatch box50
#   sbatch run_latent_ddnm.sbatch half       --mask_kind half
#   sbatch run_latent_ddnm.sbatch vae_only    --codec_type vae
# Outputs land in DDNM-main-scartch/exp/image_samples/<run_name>/{VAE,SPNN}/.

PROJ=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model
DDNM=$PROJ/DDNM-main-scartch
IMAGE=docker://nvcr.io/nvidia/pytorch:25.04-py3
export HF_HOME=/shared/cycle1_iit_shocher_prj/ron/hf_cache

RUN_NAME="${1:-ddnm_latent}"
shift || true
EXTRA_ARGS="$*"

echo "=== latent DDNM: run_name=$RUN_NAME  extra_args=$EXTRA_ARGS ==="

srun --container-image="$IMAGE" \
     --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj \
     --container-workdir="$DDNM" \
     bash -c "
       set -e
       export HF_HOME=$HF_HOME
       export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # load SD1.5 from cache only
       export PYTHONUNBUFFERED=1
       pip install --quiet diffusers transformers scipy pyyaml
       python main.py \
         --config sd15_inpaint_latent.yml \
         --deg inpainting \
         --ni --sigma_y 0 \
         -i $RUN_NAME \
         $EXTRA_ARGS
     "
echo "=== done: $RUN_NAME (exit $?) ==="
