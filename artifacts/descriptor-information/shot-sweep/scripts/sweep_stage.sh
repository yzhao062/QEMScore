#!/usr/bin/env bash
# Shot-count sweep of the descriptor-information experiment on NCSA DeltaAI.
# Frozen rule: docs/frozen-rules/2026-10-03-shot-sweep.md. One stage per Slurm job:
#   gen2048   generate the 2,048-shot level and run check 1 (faithfulness)
#   genrest   generate the other six levels and run check 2 (exact level)
#   prepare   prepare --sweep every level and run check 3 (level caches)
#   gate      R0 refits of the earlier gate on this platform (reported, no pass criterion)
#   fitAC     fitting step 1: arms A and C at the 2,048-shot level
#   predict   fitting step 2: the predictions file, hashed and timed in the run log
#   fitall    fitting step 3: arms A, C, F, P at every level (skips fits already present)
#   analyze   analyses A and B, reconcile, F - R, and H1/H2
set -euo pipefail
STAGE="$1"
ROOT=/projects/bhph/yzhao13/qemscore-sweep
CODE="$ROOT/code"
IN="$ROOT/inputs"
LV="$ROOT/levels"
RUNS="$ROOT/runs"
REP="$ROOT/reports"
LOG="$ROOT/run.log"
PY=/projects/bhph/yzhao13/miniforge3/envs/py312/bin/python
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-72}}"
SEEDS="101 211 307"
ALL_LEVELS="256 1024 2048 8192 32768 131072 exact"
# ONLY_LEVELS restricts fitall and analyze to some levels, so one job per level can run in parallel.
LEVELS="${ONLY_LEVELS:-$ALL_LEVELS}"
RULE=docs/frozen-rules/2026-10-03-shot-sweep.md
ARCHIVE="$IN/campaign-archive-v1"
GATE_FILE=artifacts/descriptor-information/gate/gate_r0.json
STRENGTH_FITS="$IN/descriptor-information-strength-v1/fits"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$CODE"
mkdir -p "$LV" "$RUNS" "$REP"
cd "$CODE"

log() { printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STAGE" "$*" >> "$LOG"; }
datasets() { for s in $SEEDS; do printf '%s ' "$LV/shots-$1/regen-shipped-s$s-n640"; done; }
run() { log "cmd: $*"; "$@"; }

hypotheses_step() {
  local A_ARGS="" B_ARGS="" L
  for L in $ALL_LEVELS; do
    A_ARGS="$A_ARGS --analysis-a $L=$RUNS/shots-$L/analysis/analysis-a.json"
    B_ARGS="$B_ARGS --analysis-b $L=$RUNS/shots-$L/analysis/analysis-b.json"
  done
  run $PY tools/shot_sweep_hypotheses.py --predictions "$RUNS/predictions.json" $A_ARGS $B_ARGS \
      --out "$RUNS/hypotheses.json"
}

log "start job=${SLURM_JOB_ID:-none} levels=[$LEVELS] host=$(hostname) commit=$(git -C "$CODE" rev-parse HEAD) rule_sha256=$(sha256sum "$RULE" | cut -d' ' -f1)"

case "$STAGE" in
gen2048)
  log "versions: $($PY -c 'import sys,numpy,scipy,sklearn,qiskit,qiskit_aer,platform;print(sys.version.split()[0],numpy.__version__,scipy.__version__,sklearn.__version__,qiskit.__version__,qiskit_aer.__version__,platform.machine())')"
  for s in $SEEDS; do
    run $PY tools/shot_sweep.py generate --source "$IN/data/regen-shipped-s$s-n640" --shots 2048 \
        --out "$LV/shots-2048/regen-shipped-s$s-n640" --workers "$WORKERS"
    run $PY tools/shot_sweep.py check-faithful --source "$IN/data/regen-shipped-s$s-n640" \
        --regen "$LV/shots-2048/regen-shipped-s$s-n640" --report "$REP/check1-faithful-s$s.json"
  done
  log "check 1 passed on DeltaAI for all three datasets"
  ;;
genrest)
  for L in 256 1024 8192 32768 131072; do
    for s in $SEEDS; do
      run $PY tools/shot_sweep.py generate --source "$IN/data/regen-shipped-s$s-n640" --shots "$L" \
          --out "$LV/shots-$L/regen-shipped-s$s-n640" --workers "$WORKERS"
    done
  done
  for s in $SEEDS; do
    run $PY tools/shot_sweep.py generate --source "$IN/data/regen-shipped-s$s-n640" --exact \
        --out "$LV/shots-exact/regen-shipped-s$s-n640" --workers "$WORKERS"
    run $PY tools/shot_sweep.py check-exact --exact "$LV/shots-exact/regen-shipped-s$s-n640" \
        --finite "$LV/shots-131072/regen-shipped-s$s-n640" --report "$REP/check2-exact-s$s.json"
  done
  ( cd "$LV" && find . -name items.jsonl -print0 | sort -z | xargs -0 sha256sum ) > "$REP/level-items-sha256.txt"
  log "check 2 passed; level items.jsonl SHA-256 in reports/level-items-sha256.txt"
  ;;
prepare)
  # The levels were generated on macOS (rule check-1 fallback): verify every copied file first.
  ( cd "$LV" && sha256sum -c --quiet "$REP/mac-gen/level-tree-sha256.txt" ) \
    || { log "level tree differs from reports/mac-gen/level-tree-sha256.txt; nothing prepared"; exit 4; }
  log "level tree verified: $(wc -l < "$REP/mac-gen/level-tree-sha256.txt") files match the macOS SHA-256 list"
  for L in $LEVELS; do
    run $PY tools/descriptor_ladder.py prepare --sweep --datasets $(datasets "$L") --archive "$ARCHIVE" \
        --out "$RUNS/shots-$L" --workers "$WORKERS"
  done
  for L in $LEVELS; do
    [ "$L" = 2048 ] && continue
    run $PY tools/shot_sweep.py check-caches --reference "$RUNS/shots-2048" --level "$RUNS/shots-$L" \
        --report "$REP/check3-caches-$L.json"
  done
  log "check 3 passed for all levels"
  ;;
gate)
  AMENDMENT="$($PY -c 'import json; print(json.load(open("artifacts/descriptor-information/gate/gate_r0.json"))["amendment"])')"
  GATE_SEEDS="${GATE_SEEDS:-$SEEDS}"; log "gate seeds: $GATE_SEEDS"
  run $PY tools/descriptor_ladder.py gate --datasets $(for s in $GATE_SEEDS; do printf '%s ' "$IN/data/regen-shipped-s$s-n640"; done) \
      --archive "$ARCHIVE" --out "$RUNS/gate-deltaai" --tolerance 1.4e-11 --amendment "$AMENDMENT" \
      --workers "$WORKERS" || log "gate command exited nonzero (reported; no pass criterion)"
  ;;
fitAC)
  log "blas_fpe_probe: $($PY -c 'import json,sys; sys.path.insert(0,"tools"); import descriptor_common as c; print(json.dumps(c.blas_fpe_probe()))')"
  log "numpy config: $($PY -c 'import numpy, json; c = numpy.show_config(mode="dicts"); print(json.dumps(c.get("Build Dependencies", {}).get("blas", {})))')"
  run $PY tools/descriptor_ladder.py run --sweep --strength-indicator --rungs R0 N1 N2 R5 --arms A C \
      --seeds 1-20 --datasets $(datasets 2048) --archive "$ARCHIVE" --gate-file "$GATE_FILE" \
      --out "$RUNS/shots-2048" --workers "$WORKERS"
  ;;
predict)
  LEVEL_ARGS=""
  for L in $LEVELS; do LEVEL_ARGS="$LEVEL_ARGS --level $L=$RUNS/shots-$L"; done
  run $PY tools/shot_sweep_predict.py $LEVEL_ARGS --c-fits "$RUNS/shots-2048/fits" \
      --reference-c-fits "$STRENGTH_FITS" \
      --reference-analysis artifacts/descriptor-information/strength-indicator/analysis-a.json \
      --measurement-floor artifacts/descriptor-information/posthoc-measurement-floor.json \
      --out "$RUNS/predictions.json" --run-log "$LOG"
  log "predictions sha256=$(sha256sum "$RUNS/predictions.json" | cut -d' ' -f1)"
  ;;
fitall)
  test -s "$RUNS/predictions.json" || { log "no predictions file; refusing to fit F or P"; exit 3; }
  for L in $LEVELS; do
    run $PY tools/descriptor_ladder.py run --sweep --strength-indicator --rungs R0 N1 N2 R5 --arms A C F P \
        --seeds 1-20 --datasets $(datasets "$L") --archive "$ARCHIVE" --gate-file "$GATE_FILE" \
        --out "$RUNS/shots-$L" --workers "$WORKERS"
  done
  ;;
analyze)
  for L in $LEVELS; do
    D="$RUNS/shots-$L/analysis"; mkdir -p "$D"
    if [ "$L" = 2048 ]; then ORIG="$STRENGTH_FITS"; else ORIG="$RUNS/shots-2048/fits"; fi
    run $PY tools/descriptor_ladder_analysis.py --fits "$RUNS/shots-$L/fits" --cache "$RUNS/shots-$L/cache" \
        --original-fits "$ORIG" --follow-up-rule "$RULE" --out "$D/analysis-a.json" --quiet
    run $PY tools/descriptor_ladder_analysis_b.py --fits "$RUNS/shots-$L/fits" --cache "$RUNS/shots-$L/cache" \
        --follow-up-rule "$RULE" --out "$D/analysis-b.json"
    run $PY tools/descriptor_ladder_reconcile.py "$D/analysis-a.json" "$D/analysis-b.json" --out "$D/reconcile.json"
    run $PY tools/descriptor_ladder_posthoc.py --part A --fits "$RUNS/shots-$L/fits" --cache "$RUNS/shots-$L/cache" \
        --out "$D/posthoc-arm-minus-raw.json" --quiet
  done
  [ -n "${ONLY_LEVELS:-}" ] || hypotheses_step
  ;;
hypotheses)
  hypotheses_step
  ;;
*)
  echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
log "done"
