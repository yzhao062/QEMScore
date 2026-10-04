#!/usr/bin/env bash
# One full ghx4-interactive node (288 cores, at most 2 h) runs the shot sweep and the
# strong-learner Part A side by side, 140 workers each, because the batch queue would
# start them a day later. The stages are the same scripts the batch chain runs.
set -uo pipefail
SW=/projects/bhph/yzhao13/qemscore-sweep
ST=/projects/bhph/yzhao13/qemscore-strong
FITB_JOB="${FITB_JOB:?}"
export WORKERS=140
note() { printf '%s\tnote\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" >> "$1/run.log"; }
note "$SW" "Job ${SLURM_JOB_ID} on ghx4-interactive ($(hostname)) runs prepare, fitAC, predict, fitall (all levels), analyze (levels in parallel), and hypotheses in order with WORKERS=$WORKERS; the batch queue estimated a start a day later."
note "$ST" "Job ${SLURM_JOB_ID} on ghx4-interactive ($(hostname)) runs fitA with WORKERS=$WORKERS, then analyze once Part B (batch job $FITB_JOB) has completed."

sweep() {
  local st pids=() L rc=0
  for st in prepare fitAC predict fitall; do
    bash "$SW/sweep_stage.sh" "$st" || { note "$SW" "stage $st failed; later stages not run"; return 1; }
  done
  for L in 256 1024 2048 8192 32768 131072 exact; do
    ONLY_LEVELS=$L bash "$SW/sweep_stage.sh" analyze > "$SW/slurm/analyze-$L-${SLURM_JOB_ID}.out" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  [ "$rc" = 0 ] || { note "$SW" "a per-level analysis failed; hypotheses not run"; return 1; }
  bash "$SW/sweep_stage.sh" hypotheses
}

strong() {
  bash "$ST/strong_stage.sh" probe
  bash "$ST/strong_stage.sh" fitA || { note "$ST" "fitA failed; analyze not run"; return 1; }
  while squeue -h -j "$FITB_JOB" 2>/dev/null | grep -q .; do sleep 30; done
  local state
  state=$(sacct -n -X -j "$FITB_JOB" -o State | head -1 | tr -d ' ')
  [ "$state" = COMPLETED ] || { note "$ST" "Part B job $FITB_JOB ended $state; analyze not run"; return 1; }
  bash "$ST/strong_stage.sh" analyze
}

sweep > "$SW/slurm/interactive-sweep-${SLURM_JOB_ID}.out" 2>&1 &
p1=$!
strong > "$ST/slurm/interactive-strong-${SLURM_JOB_ID}.out" 2>&1 &
p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
echo "sweep=$r1 strong=$r2"
