#!/bin/bash
# Keep the sweep alive unattended.
#
# Slurm reporting a job as RUNNING is not evidence it is doing anything: when the disk
# filled, six arms sat "RUNNING" at 0% GPU for 7-9 hours while every sample failed. The
# only reliable liveness signal is new output files, so that is what this checks.
#
# Every CHECK_MIN minutes, per model under $TAG:
#   * skip if its table1_<model>.csv already exists (done)
#   * if an arm is absent from the queue and incomplete -> resubmit the model
#   * if an arm is RUNNING but has produced nothing for STALL_MIN -> resubmit the model
# Generation resumes from disk, so a resubmit costs only the in-flight sample. A
# per-model cooldown stops it fighting itself.
#
#   nohup bash reed_repro/slurm/watchdog.sh > watchdog.log 2>&1 &
set -u
cd /home/yamitehrlich/work/spnn-diffusion
R=/rg/shocher_prj/yamitehrlich/spnn-diffusion/reed_repro/results
TAG=${TAG:-full179_seed42}
MODELS=${MODELS:-"ip2p magicbrush diffedit pbe sd_inpaint"}
CHECK_MIN=${CHECK_MIN:-10}
STALL_MIN=${STALL_MIN:-40}
COOLDOWN_MIN=${COOLDOWN_MIN:-60}
PARTITION=${PARTITION:-h200-shared}
export REED_USE_IMAGENHUB=0 SEED_MODE=fixed FIXED_SEED=42 TAG PARTITION

declare -A LASTCOUNT LASTMOVE LASTFIX
now() { date +%s; }
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

log "watchdog start: tag=$TAG models='$MODELS' check=${CHECK_MIN}m stall=${STALL_MIN}m"
while true; do
  free_g=$(df -BG --output=avail /rg/shocher_prj 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "${free_g:-}" ] && [ "$free_g" -lt 15 ] && log "WARNING: only ${free_g}G free on the shared quota"

  alldone=1
  for m in $MODELS; do
    [ -f "$R/$TAG/table1_$m.csv" ] && continue      # finished and evaluated
    alldone=0
    need_fix=""
    for c in vae spnn; do
      # Completion is measured by finished samples (iter_25), but LIVENESS must use
      # total PNGs across all iterations: a slow arm can take longer than STALL_MIN to
      # finish one sample, and counting only iter_25 would call it stalled while it is
      # working normally.
      done_n=$(ls "$R/$TAG/$m/$c/iter_25"/*.png 2>/dev/null | wc -l)
      [ "$done_n" -ge 179 ] && continue             # this arm is complete
      n=$(find "$R/$TAG/$m/$c" -name '*.png' 2>/dev/null | wc -l)
      st=$(squeue -u yamitehrlich -h -n "reed_${m}_${c}_${TAG}" -o "%t" 2>/dev/null | head -1)
      key="$m/$c"
      if [ -z "$st" ]; then
        need_fix="absent:$key"; break
      elif [ "$st" = "R" ]; then
        prev=${LASTCOUNT[$key]:--1}
        if [ "$n" -ne "$prev" ]; then
          LASTCOUNT[$key]=$n; LASTMOVE[$key]=$(now)
        else
          since=$(( ( $(now) - ${LASTMOVE[$key]:-$(now)} ) / 60 ))
          [ "$since" -ge "$STALL_MIN" ] && { need_fix="stalled(${since}m):$key"; break; }
        fi
      fi
    done

    if [ -n "$need_fix" ]; then
      last=${LASTFIX[$m]:-0}
      if [ $(( ( $(now) - last ) / 60 )) -lt "$COOLDOWN_MIN" ]; then
        log "  $m needs fix ($need_fix) but is in cooldown; leaving it"
      else
        log "RESTART $m ($need_fix)"
        squeue -u yamitehrlich -h -o "%i %j" | grep -E "reed_(${m}_(vae|spnn)|eval_${m})_${TAG}\$" \
          | awk '{print $1}' | xargs -r scancel
        sleep 5
        bash reed_repro/slurm/submit_full.sh "$m" 2>&1 | sed 's/^/    /'
        LASTFIX[$m]=$(now)
        for c in vae spnn; do unset 'LASTCOUNT[$m/$c]' 'LASTMOVE[$m/$c]'; done
      fi
    fi
  done

  [ "$alldone" -eq 1 ] && { log "all models have tables — watchdog exiting"; break; }
  sleep $(( CHECK_MIN * 60 ))
done
