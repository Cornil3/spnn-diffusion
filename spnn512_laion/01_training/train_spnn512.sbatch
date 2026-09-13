#!/bin/bash
#SBATCH --job-name=spnn512
#SBATCH --partition=b200-shared,a100-public,l40s-public
#SBATCH --qos=24h_16g
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/slurm_%j.out
#SBATCH --error=/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/slurm_%j.err

# Phase 2 (from the end of epoch 4): + LPIPS(VGG) perceptual loss on every reconstruction branch - the
# loss the SD1.5 VAE was trained with; pixel MSE+L1 alone gave blurry decodes. One-off 1000-step LR
# re-warmup into the same 2e-4 peak; best.pt now tracks val cycle LPIPS. --resume auto continues the
# same WSD schedule (10 epochs = 54,260 steps, decay over the last 20%).
# Keeps itself running: a clean exit (23.4 h wall) resubmits this script until OUT/DONE exists.
# To stop the chain: touch /rg/shocher_prj/ron.libman/laion1m/runs/spnn512/NO_RESUBMIT
cd /home/ron.libman/churches-reboot
OUT=/rg/shocher_prj/ron.libman/laion1m/runs/spnn512
mkdir -p $OUT
export WANDB_DIR=$OUT
NG=$(nvidia-smi -L | wc -l)
echo "node=$(hostname) gpus=$NG start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The job starts on whichever partition frees 8 GPUs first (B200, A100 or L40S). The effective batch stays 128 so the
# LR (scaled from it: 2e-4) and the step-based schedule are identical everywhere; L40S (48 GB) uses micro-batch 2.
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *L40S*) BS=2 ;; *) BS=4 ;; esac
ACC=$(( 128 / (BS * NG) ))
[ $((BS * ACC * NG)) -eq 128 ] || { echo "cannot keep eff_bs 128 with $NG GPUs of $GPU"; exit 1; }
echo "gpu=$GPU bs=$BS accum=$ACC eff_bs=$((BS * ACC * NG))"
torchrun --standalone --nproc_per_node=$NG train_spnn512.py \
  --out $OUT --resume auto \
  --bs $BS --accum $ACC --workers 8 \
  --epochs 10 --max-hours 23.4 \
  --warmup 500 --val-every 500 --log-every 25 --save-every-min 20 \
  --val-shards 4 --test-shards 4 \
  --w-perc 1.0 --perc-type lpips --perc-branches clean,traj,self \
  --best-metric lpips --rewarmup-steps 1000 \
  --wandb --wandb-project spnn512 --wandb-run spnn512-laion-coco "$@"
RC=$?
echo "exit=$RC end=$(date)"
# a crash (RC != 0) does NOT resubmit, so a bug cannot loop; the 23.4 h wall exits cleanly with 0
if [ $RC -eq 0 ] && [ ! -f $OUT/DONE ] && [ ! -f $OUT/NO_RESUBMIT ]; then
  sbatch /home/ron.libman/churches-reboot/train_spnn512.sbatch && echo "resubmitted $(date)"
fi
