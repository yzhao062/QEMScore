#!/usr/bin/env bash
# One full DeltaAI node (ghx4-interactive: 288 cores, about 454 GB, at most 2 h) per call.
#   follow: B, C, D, and G fit side by side, then their analyses.
#   fresh:  the fresh-panel stages in order (fresh_check through fresh_score).
# A call that reaches the time limit can be repeated: every run command skips fits already written.
set -uo pipefail
MODE="$1"
ROOT="${ROOT:-/projects/bhph/yzhao13/qemscore-r8}"
ST="${ST:-$ROOT/round8_stage.sh}"
[ -f "$ST" ] || ST="$(cd "$(dirname "$0")" && pwd)/round8_stage.sh"
note() { printf '%s\tnode\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$ROOT/run.log"; }
waitall() { local pid rc=0; for pid in "$@"; do wait "$pid" || rc=1; done; return $rc; }
note "job ${SLURM_JOB_ID:-none} on $(hostname): mode $MODE"
case "$MODE" in
follow)
  pids=()
  WORKERS=120 bash "$ST" B_fit > "$ROOT/slurm/B_fit-${SLURM_JOB_ID:-x}.out" 2>&1 & pids+=($!)
  WORKERS=50 bash "$ST" C_fit > "$ROOT/slurm/C_fit-${SLURM_JOB_ID:-x}.out" 2>&1 & pids+=($!)
  WORKERS=30 bash "$ST" D_fit > "$ROOT/slurm/D_fit-${SLURM_JOB_ID:-x}.out" 2>&1 & pids+=($!)
  WORKERS=80 bash "$ST" G_fit > "$ROOT/slurm/G_fit-${SLURM_JOB_ID:-x}.out" 2>&1 & pids+=($!)
  waitall "${pids[@]}" || { note "a fit stage failed; analyses not run"; exit 1; }
  pids=()
  for st in B_analyze C_analyze D_analyze G_analyze; do
    bash "$ST" "$st" > "$ROOT/slurm/$st-${SLURM_JOB_ID:-x}.out" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { note "an analysis stage failed"; exit 1; }
  ;;
fresh)
  for st in fresh_check fresh_fitAC fresh_predict fresh_fitall fresh_analyze fresh_score; do
    if [ "$st" = fresh_predict ] && [ -s "$ROOT/fresh/predictions-fresh.json" ]; then continue; fi
    WORKERS=280 R4_WORKERS=24 bash "$ST" "$st" > "$ROOT/slurm/$st-${SLURM_JOB_ID:-x}.out" 2>&1 \
      || { note "stage $st failed; later stages not run"; exit 1; }
  done
  ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
note "mode $MODE done"
