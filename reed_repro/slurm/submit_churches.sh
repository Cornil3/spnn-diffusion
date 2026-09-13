#!/bin/bash
# One generation job per arm (they run concurrently), then a single evaluation job
# gated on all three with afterok so it can never score a partial sweep.
set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
TAG=${TAG:-churches_img2img}
N_IMAGES=${N_IMAGES:-100}
STRENGTH=${STRENGTH:-0.5}

ids=()
for arm in compvis 2blk 2blk_rt; do
  jid=$(ARM=$arm TAG=$TAG N_IMAGES=$N_IMAGES STRENGTH=$STRENGTH \
        sbatch --parsable --export=ALL --job-name="ci2i_$arm" \
        reed_repro/slurm/churches_img2img.slurm)
  echo "  $arm -> $jid"
  ids+=("$jid")
done
dep=$(IFS=:; echo "${ids[*]}")
eid=$(TAG=$TAG sbatch --parsable --export=ALL --dependency=afterok:"$dep" \
      reed_repro/slurm/churches_eval.slurm)
echo "  eval -> $eid (afterok:$dep)"
