#!/bin/bash
# Fan out one job per (model, codec) arm.
#
#   bash reed_repro/slurm/submit.sh pilot10 10       # pilot: 10 of the 179 samples
#   bash reed_repro/slurm/submit.sh full179          # full run
#
# Add --dryrun as a third argument to preview without submitting.

set -euo pipefail
TAG=${1:-run}
LIMIT=${2:-}
DRY=${3:-}

cd /home/yamitehrlich/work/spnn-diffusion
MODELS=(ip2p magicbrush diffedit pbe sd_inpaint)
CODECS=(vae spnn)

for m in "${MODELS[@]}"; do
  for c in "${CODECS[@]}"; do
    CMD=(sbatch --job-name="reed_${m}_${c}_${TAG}"
         --export=ALL,MODEL="$m",CODEC="$c",TAG="$TAG",LIMIT="$LIMIT"
         reed_repro/slurm/generate.slurm)
    if [[ "$DRY" == "--dryrun" ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
  done
done
