#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p l40s-public
#SBATCH --qos=4d_1g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=reed_gen

# One (model, codec) arm per job. Parameterized by env vars:
#   MODEL   ip2p | magicbrush | diffedit | pbe | sd_inpaint
#   CODEC   vae | spnn
#   TAG     results subdirectory name
#   LIMIT   number of samples (unset = all 179)
#   CKPT / WEIGHTS / LATENT_SCALE  (spnn arm only)
#
# Generation is resumable: re-submitting skips samples whose snapshots exist.

set -euo pipefail
cd /home/yamitehrlich/work/spnn-diffusion
mkdir -p logs
source venv/bin/activate

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=1
# wandb's console capture defaults to "auto", which only hooks a tty. Under slurm
# stdout is a file, so the run's Logs tab comes up empty; "wrap" captures anyway.
export WANDB_CONSOLE=${WANDB_CONSOLE:-wrap}

# athena still holds the (hub-removed) runwayml copy of SD 1.5; reuse it rather than
# pulling the byte-identical community mirror. Unset to use the mirror.
export REED_SD15_ID=${REED_SD15_ID:-runwayml/stable-diffusion-v1-5}
# Set to 0 to force the vendored wrappers instead of importing imagen_hub.
export REED_USE_IMAGENHUB=${REED_USE_IMAGENHUB:-1}

MODEL=${MODEL:?set MODEL}
CODEC=${CODEC:?set CODEC}
TAG=${TAG:-run}
CKPT=${CKPT:-imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt}
WEIGHTS=${WEIGHTS:-ema}
LATENT_SCALE=${LATENT_SCALE:-prescaled}

# SNAPSHOTS_ONLY=1 keeps only iterations 1/5/15/25 instead of all 25.
# Metrics are identical; only the wandb grids get sparser. ~4.5x less disk.
SNAPSHOTS_ONLY=${SNAPSHOTS_ONLY:-0}
SEED_MODE=${SEED_MODE:-varying}
FIXED_SEED=${FIXED_SEED:-42}

ARGS=(--tag "$TAG" --models "$MODEL" --codecs "$CODEC"
      --spnn_checkpoint "$CKPT" --spnn_weights "$WEIGHTS" --latent_scale "$LATENT_SCALE"
      --seed_mode "$SEED_MODE" --fixed_seed "$FIXED_SEED")
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ "$SNAPSHOTS_ONLY" == "1" ]]; then ARGS+=(--snapshots_only); fi

python -u -m reed_repro.run_repro generate "${ARGS[@]}"
