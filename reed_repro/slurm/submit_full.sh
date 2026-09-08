#!/bin/bash
# Submit both codec arms for each model, plus a per-model evaluation job that fires
# automatically once BOTH of that model's arms finish (--dependency=afterok).
#
#   bash reed_repro/slurm/submit_full.sh                      # the 4 non-IP2P models
#   bash reed_repro/slurm/submit_full.sh "pbe sd_inpaint"     # a chosen subset
#
# Each eval writes metrics_<model>.json and table1_<model>.{md,csv} so the per-model
# jobs never clobber one another; a final all-models pass produces the combined Table 1.
set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion

TAG=${TAG:-full179}
CK=${CKPT:-imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt}
MODELS=${1:-"magicbrush diffedit pbe sd_inpaint"}
SEED_MODE=${SEED_MODE:-varying}
FIXED_SEED=${FIXED_SEED:-42}

sub() { sbatch --parsable "$@" 2>/dev/null | grep -oE '^[0-9]+' | head -1; }

for m in $MODELS; do
  A=$(sub --job-name="reed_${m}_vae_${TAG}" \
        --export=ALL,MODEL="$m",CODEC=vae,TAG="$TAG",CKPT="$CK",WEIGHTS=ema,LATENT_SCALE=prescaled,SEED_MODE="$SEED_MODE",FIXED_SEED="$FIXED_SEED" \
        reed_repro/slurm/generate.slurm)
  B=$(sub --job-name="reed_${m}_spnn_${TAG}" \
        --export=ALL,MODEL="$m",CODEC=spnn,TAG="$TAG",CKPT="$CK",WEIGHTS=ema,LATENT_SCALE=prescaled,SEED_MODE="$SEED_MODE",FIXED_SEED="$FIXED_SEED" \
        reed_repro/slurm/generate.slurm)
  E=$(sub --qos=24h_1g --dependency=afterok:"$A":"$B" --job-name="reed_eval_${m}_${TAG}" \
        --export=ALL,TAG="$TAG",MODELS="$m",PROJECT=reed-vae-repro \
        reed_repro/slurm/report.slurm)
  echo "$m: vae=$A spnn=$B eval=$E (eval waits on both arms)"
done
