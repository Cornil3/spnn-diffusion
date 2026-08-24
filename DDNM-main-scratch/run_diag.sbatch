#!/bin/bash
#SBATCH --job-name=ddnm_diag
#SBATCH --partition=compute-gpu
#SBATCH --account=cycle1_iit_shocher_prj
#SBATCH --qos=owner_95
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model/logs/%x_%j.out
#
# Codec round-trip diagnostic (no diffusion).  Usage: sbatch run_diag.sbatch

PROJ=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model
DDNM=$PROJ/DDNM-main-scartch
IMAGE=docker://nvcr.io/nvidia/pytorch:25.04-py3
export HF_HOME=/shared/cycle1_iit_shocher_prj/ron/hf_cache

srun --container-image="$IMAGE" \
     --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj \
     --container-workdir="$DDNM" \
     bash -c "
       set -e
       export HF_HOME=$HF_HOME
       export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
       export PYTHONUNBUFFERED=1
       pip install --quiet diffusers transformers scipy pyyaml
       python diag_codec.py --config sd15_inpaint_latent.yml
     "
