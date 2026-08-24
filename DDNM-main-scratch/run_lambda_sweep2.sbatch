#!/bin/bash
#SBATCH --job-name=ddnm_sweep2
#SBATCH --partition=compute-gpu
#SBATCH --account=cycle1_iit_shocher_prj
#SBATCH --qos=owner_95
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model/logs/%x_%j.out
#
# Lambda-schedule sweep for latent DDNM inpainting:
#   lambda modes: const(base), exp@rate{0.5,1,2,5}, noise, linear   x   T in {100,200}
#   time-travel OFF (length=repeat=1), damping OFF (damping_floor=1.0, the diffusion.py value).
# Each config -> exp/image_samples/sweep2_T<T>_<mode>/{VAE,SPNN}/ + PSNR.txt.

IMAGE=docker://nvcr.io/nvidia/pytorch:25.04-py3

srun --container-image="$IMAGE" \
     --container-mounts=/shared/cycle1_iit_shocher_prj:/shared/cycle1_iit_shocher_prj \
     --container-workdir=/shared/cycle1_iit_shocher_prj/ron/spnn/simple-latent-diffusion-model/DDNM-main-scartch \
     bash -c '
       set -e
       export HF_HOME=/shared/cycle1_iit_shocher_prj/ron/hf_cache
       export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
       pip install --quiet diffusers transformers scipy pyyaml

       declare -a LAM=(
         "base|"
         "exp0.5|--lambda_mode exp --lambda_rate 0.5"
         "exp1|--lambda_mode exp --lambda_rate 1"
         "exp2|--lambda_mode exp --lambda_rate 2"
         "exp5|--lambda_mode exp --lambda_rate 5"
         "noise|--lambda_mode noise"
         "linear|--lambda_mode linear"
       )

       for T in 100 200; do
         for entry in "${LAM[@]}"; do
           lname="${entry%%|*}"; largs="${entry#*|}"
           name="T${T}_${lname}"
           echo ""
           echo "=================== SWEEP RUN: ${name} | T=${T} ${largs} ==================="
           python main.py --config sd15_inpaint_latent.yml --deg inpainting \
             --ni --sigma_y 0 -i "sweep2_${name}" --T_sampling ${T} ${largs} \
             || echo "RUN ${name} FAILED"
         done
       done

       echo ""
       echo "=================== SWEEP2 SUMMARY (config : codec PSNR) ==================="
       for f in exp/image_samples/sweep2_*/PSNR.txt; do
         [ -f "$f" ] && echo "$(basename "$(dirname "$f")"): $(tr "\n" " " < "$f")"
       done
     '
echo "=== sweep2 job finished ==="
