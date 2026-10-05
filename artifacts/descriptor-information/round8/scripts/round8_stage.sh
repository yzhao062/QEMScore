#!/usr/bin/env bash
# Round-8 follow-ups of the descriptor-information experiment on NCSA DeltaAI.
# Frozen rules: docs/frozen-rules/2026-10-04-fresh-confirmation.md (fresh panel)
# and docs/frozen-rules/2026-10-04-crossed-follow-ups.md (B, C, D, G). Stages:
#   setup          clone the code at the rule commit; record versions
#   fresh_check    verify the macOS fresh trees against reports/tree-sha256.txt
#   fresh_fitAC    original candidates, A and C at 2,048 shots (R0, N1, N2, R5)
#   fresh_predict  the predictions file (validation rows only), hashed in the run log
#   fresh_fitall   everything else of the fresh panel (both candidate sets, all levels)
#   fresh_analyze  analyses, derived baselines, learner adequacy, stacking at R0 exact
#   fresh_score    P1 to P8, once
#   B_fit B_analyze  strong candidates at the six released levels other than 2,048
#   C_fit C_analyze  the well-trained MLP at 2,048 shots and the exact level
#   D_fit D_analyze  learner seeds 21 to 40 of C and F, and the A/A calibration
#   G_fit G_analyze  training sizes 40, 80, 160, 320
set -euo pipefail
STAGE="$1"
# ROOT, CODE, SW, STRONG, PY, FSEEDS, and SHA256 are overridden only for a local rehearsal
# (the fresh stages on throwaway seeds) or a test of the stage commands.
ROOT="${ROOT:-/projects/bhph/yzhao13/qemscore-r8}"
CODE="${CODE:-$ROOT/code}"
SW="${SW:-/projects/bhph/yzhao13/qemscore-sweep}"
STRONG="${STRONG:-/projects/bhph/yzhao13/qemscore-strong}"
IN="$SW/inputs"
FR="$ROOT/fresh"
FU="$ROOT/follow"
LOG="$ROOT/run.log"
COMMIT="${COMMIT:?set COMMIT to the rule commit}"
PY="${PY:-/projects/bhph/yzhao13/miniforge3/envs/py312/bin/python}"
SHA256="${SHA256:-sha256sum}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-72}}"
RULE_FRESH=docs/frozen-rules/2026-10-04-fresh-confirmation.md
RULE_FU=docs/frozen-rules/2026-10-04-crossed-follow-ups.md
GATE_FILE=artifacts/descriptor-information/gate/gate_r0.json
ARCHIVE="$IN/campaign-archive-v1"
FSEEDS="${FSEEDS:-401 503 607}"
LEVELS="256 1024 2048 8192 32768 131072 exact"
SWEEP_LEVELS="256 1024 8192 32768 131072 exact"
SWA=artifacts/descriptor-information/shot-sweep/levels
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$CODE"
mkdir -p "$ROOT"
log() { printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STAGE" "$*" >> "$LOG"; }
run() { log "cmd: $*"; "$@"; }
fds() { for s in $FSEEDS; do printf '%s ' "$FR/levels/shots-$1/regen-shipped-s$s-n640"; done; }
sds() { for s in 101 211 307; do printf '%s ' "$SW/levels/shots-$1/regen-shipped-s$s-n640"; done; }
# A tree: a prepared tree's cache, descriptors, encoder cache, and dataset index, without fits.
tree() {
  local src="$1" dst="$2" part
  mkdir -p "$dst"
  for part in cache descriptors encoder-cache; do
    [ -d "$src/$part" ] && [ ! -d "$dst/$part" ] && cp -R "$src/$part" "$dst/"
  done
  for part in datasets.json dataset_checks.json; do
    [ -f "$src/$part" ] && [ ! -f "$dst/$part" ] && cp "$src/$part" "$dst/"
  done
  return 0
}
probe() {
  log "versions: $($PY -c 'import sys,numpy,scipy,sklearn,platform;print(sys.version.split()[0],numpy.__version__,scipy.__version__,sklearn.__version__,platform.machine())')"
  log "numpy blas: $($PY -c 'import numpy, json; c = numpy.show_config(mode="dicts"); print(json.dumps(c.get("Build Dependencies", {}).get("blas", {})))')"
  log "threadpool_info: $($PY -c 'import json, numpy, scipy.linalg, sklearn.ensemble, threadpoolctl; print(json.dumps(threadpoolctl.threadpool_info()))')"
  log "blas_fpe_probe: $($PY -c 'import json,sys; sys.path.insert(0,"tools"); import descriptor_common as c; print(json.dumps(c.blas_fpe_probe()))')"
}
analyses() {  # analyses A and B, reconcile, and F - R for one fits tree
  local T="$1" rule="$2" seeds="$3" orig="${4:-}"
  local D="$T/analysis"; mkdir -p "$D"
  run $PY tools/descriptor_ladder_analysis.py --fits "$T/fits" --cache "$T/cache" --dataset-seeds $seeds \
      ${orig:+--original-fits "$orig"} --follow-up-rule "$rule" --out "$D/analysis-a.json" --quiet
  run $PY tools/descriptor_ladder_analysis_b.py --fits "$T/fits" --cache "$T/cache" --dataset-seeds $seeds \
      --follow-up-rule "$rule" --out "$D/analysis-b.json"
  run $PY tools/descriptor_ladder_reconcile.py "$D/analysis-a.json" "$D/analysis-b.json" --out "$D/reconcile.json"
  run $PY tools/descriptor_ladder_posthoc.py --part A --fits "$T/fits" --cache "$T/cache" --dataset-seeds $seeds \
      --out "$D/posthoc-arm-minus-raw.json" --quiet
}

[ "$STAGE" = setup ] || cd "$CODE"
log "start job=${SLURM_JOB_ID:-none} host=$(hostname) commit=${COMMIT}"

case "$STAGE" in
setup)
  [ -d "$CODE/.git" ] || git clone -q https://github.com/yzhao062/QEMScore.git "$CODE"
  git -C "$CODE" fetch -q origin && git -C "$CODE" checkout -q "$COMMIT"
  cd "$CODE"; probe
  ;;
fresh_check)
  ( cd "$FR" && $SHA256 -c --quiet reports/tree-sha256.txt ) \
    || { log "fresh tree differs from reports/tree-sha256.txt; nothing fitted"; exit 4; }
  for L in $LEVELS; do
    for set in orig strong; do tree "$FR/prepared/shots-$L" "$FR/$set/shots-$L"; done
  done
  log "fresh trees verified and copied"
  ;;
fresh_fitAC)
  probe
  run $PY tools/descriptor_ladder.py run --strength-indicator --rungs R0 N1 N2 R5 --arms A C --seeds 1-20 \
      --datasets $(fds 2048) --gate-file "$GATE_FILE" --cache-record "$FR/records/cache-record-2048.json" \
      --out "$FR/orig/shots-2048" --workers "$WORKERS"
  ;;
fresh_predict)
  LV=(); for L in $LEVELS; do LV+=(--level "$L=$FR/orig/shots-$L"); done
  run $PY tools/fresh_confirmation.py predict --dataset-seeds $FSEEDS "${LV[@]}" \
      --c-fits "$FR/orig/shots-2048/fits" --out "$FR/predictions-fresh.json" --run-log "$LOG"
  ;;
fresh_fitall)
  test -s "$FR/predictions-fresh.json" || { log "no predictions file; refusing to fit F or P"; exit 3; }
  probe
  for set in orig strong; do
    OPT=""; [ "$set" = strong ] && OPT="--strong-learners"
    for L in $LEVELS; do
      if [ "$L" = 2048 ]; then RUNGS="R0 N1 N2 N3 N4 R3-TFI R3-Heis R5"; else RUNGS="R0 N1 N2 R5"; fi
      SWEEP=""; [ "$L" = 2048 ] || SWEEP="--sweep"
      run $PY tools/descriptor_ladder.py run $SWEEP --strength-indicator $OPT --rungs $RUNGS --arms A C F P \
          --seeds 1-20 --datasets $(fds "$L") --gate-file "$GATE_FILE" \
          --cache-record "$FR/records/cache-record-$L.json" --out "$FR/$set/shots-$L" --workers "$WORKERS"
      # R4 separately: a strong R4 fit can need 10 to 20 GB, so it runs with fewer workers.
      if [ "$L" = 2048 ]; then
        run $PY tools/descriptor_ladder.py run --strength-indicator $OPT --rungs R4 --arms A C F P \
            --seeds 1-20 --datasets $(fds "$L") --gate-file "$GATE_FILE" \
            --cache-record "$FR/records/cache-record-$L.json" --out "$FR/$set/shots-$L" \
            --workers "${R4_WORKERS:-$WORKERS}"
      fi
    done
  done
  ;;
fresh_analyze)
  for L in $LEVELS; do
    if [ "$L" = 2048 ]; then ORIG=""; else ORIG="$FR/orig/shots-2048/fits"; fi
    analyses "$FR/orig/shots-$L" "$RULE_FRESH" "$FSEEDS" "$ORIG"
    S="$FR/strong/shots-$L"
    run $PY tools/strong_learner_derive.py --fits "$S/fits" --out "$S/derived/fits"
    analyses "$S" "$RULE_FRESH" "$FSEEDS" "$S/derived/fits"
    mkdir -p "$S/derived/analysis"
    run $PY tools/descriptor_ladder_analysis.py --fits "$S/derived/fits" --cache "$S/cache" --dataset-seeds $FSEEDS \
        --original-fits "$FR/orig/shots-$L/fits" --follow-up-rule "$RULE_FRESH" \
        --out "$S/derived/analysis/analysis-a.json" --quiet
  done
  # Learner adequacy and stacking on all 14 trees (both fit sets at every level). The original
  # exact-level set keeps the name fresh-exact, which P8 reads.
  SETS=()
  for L in $LEVELS; do
    for set in orig strong; do
      T="$FR/$set/shots-$L"
      run $PY tools/learner_adequacy.py --fits "$T/fits" --cache "$T/cache" --dataset-seeds $FSEEDS \
          --analysis-a "$T/analysis/analysis-a.json" --out "$T/analysis/learner-adequacy.json" --quiet
      NAME="fresh-$set-$L"; [ "$set-$L" = orig-exact ] && NAME=fresh-exact
      SETS+=(--set "$NAME" "$T/fits" "$FR/levels/shots-$L" "$T/cache" "$T/analysis/analysis-a.json")
    done
  done
  run $PY tools/stacking_increment_all.py "${SETS[@]}" --unpinned --follow-up-rule "$RULE_FRESH" \
      --dataset-seeds $FSEEDS --out "$FR/stacking-fresh.json"
  ;;
fresh_score)
  OR=(); for L in $LEVELS; do OR+=(--original "$L=$FR/orig/shots-$L/analysis/analysis-a.json"); done
  run $PY tools/fresh_confirmation.py score --dataset-seeds $FSEEDS --predictions "$FR/predictions-fresh.json" \
      "${OR[@]}" --strong "2048=$FR/strong/shots-2048/analysis/analysis-a.json" \
      --stacking "$FR/stacking-fresh.json" --stacking-set fresh-exact --out "$FR/fresh-confirmation.json"
  ;;
B_fit)
  probe
  for L in $SWEEP_LEVELS; do
    tree "$SW/runs/shots-$L" "$FU/B/shots-$L"
    run $PY tools/descriptor_ladder.py run --sweep --strength-indicator --strong-learners --rungs R0 N1 N2 R5 \
        --arms A C F P --seeds 1-20 --datasets $(sds "$L") --archive "$ARCHIVE" --gate-file "$GATE_FILE" \
        --cache-record "$SWA/$L/analysis-a.json" --out "$FU/B/shots-$L" --workers "$WORKERS"
  done
  ;;
B_analyze)
  for L in $SWEEP_LEVELS; do
    T="$FU/B/shots-$L"
    run $PY tools/strong_learner_derive.py --fits "$T/fits" --out "$T/derived/fits"
    analyses "$T" "$RULE_FU" "101 211 307" "$SW/runs/shots-$L/fits"
    mkdir -p "$T/derived/analysis"
    run $PY tools/descriptor_ladder_analysis.py --fits "$T/derived/fits" --cache "$T/cache" \
        --original-fits "$SW/runs/shots-$L/fits" --follow-up-rule "$RULE_FU" \
        --out "$T/derived/analysis/analysis-a.json" --quiet
  done
  ;;
C_fit)
  probe
  for L in 2048 exact; do
    tree "$SW/runs/shots-$L" "$FU/C/shots-$L"
    run $PY tools/descriptor_ladder.py run --sweep --strength-indicator --neural-es --rungs R0 N1 N2 R5 \
        --arms A C F P --seeds 1-20 --datasets $(sds "$L") --archive "$ARCHIVE" --gate-file "$GATE_FILE" \
        --cache-record "$SWA/$L/analysis-a.json" --out "$FU/C/shots-$L" --workers "$WORKERS"
  done
  ;;
C_analyze)
  for L in 2048 exact; do
    T="$FU/C/shots-$L"
    analyses "$T" "$RULE_FU" "101 211 307" "$SW/runs/shots-$L/fits"
    run $PY tools/learner_adequacy.py --fits "$T/fits" --cache "$T/cache" \
        --analysis-a "$T/analysis/analysis-a.json" --out "$T/analysis/learner-adequacy.json" --quiet
  done
  ;;
D_fit)
  probe
  tree "$SW/runs/shots-2048" "$FU/D/shots-2048"
  run $PY tools/descriptor_ladder.py run --sweep --strength-indicator --rungs R0 N1 N2 R5 --arms C F \
      --seeds 21-40 --datasets $(sds 2048) --archive "$ARCHIVE" --gate-file "$GATE_FILE" \
      --cache-record "$SWA/2048/analysis-a.json" --out "$FU/D/shots-2048" --workers "$WORKERS"
  ;;
D_analyze)
  T="$FU/D/shots-2048"
  run $PY tools/aa_calibration.py --fits "$SW/runs/shots-2048/fits" --fits "$T/fits" --cache "$T/cache" \
      --rungs R0 N1 N2 R5 --set1-seeds 1-20 --set2-seeds 21-40 --out "$FU/D/aa-calibration.json" --quiet
  ;;
G_fit)
  probe
  for N in 40 80 160 320; do
    tree "$IN/partA-prepared" "$FU/G/n$N"
    run $PY tools/descriptor_ladder.py run --strength-indicator --train-size "$N" --rungs R0 N1 N2 R4 R5 \
        --arms A C F P --seeds 1-20 --datasets $(for s in 101 211 307; do printf '%s ' "$IN/data/regen-shipped-s$s-n640"; done) \
        --archive "$ARCHIVE" --gate-file "$GATE_FILE" \
        --cache-record artifacts/descriptor-information/strength-indicator/analysis-a.json \
        --out "$FU/G/n$N" --workers "$WORKERS"
  done
  ;;
G_analyze)
  # The N = 640 comparator: the shot sweep's 2,048-shot fits at R0, N1, N2, and R5 and the
  # strong-learner rerun's derived baseline at R4, linked into one directory for --original-fits.
  REF="$FU/G/reference-640/fits"; mkdir -p "$REF"
  for f in "$SW/runs/shots-2048/fits"/shipped-s*-n640__{R0,N1,N2,R5}__*; do ln -sf "$f" "$REF/"; done
  for f in "$STRONG/runs/derived-baseline/fits"/shipped-s*-n640__R4__*; do ln -sf "$f" "$REF/"; done
  for s in 101 211 307; do for r in R0 N1 N2 R4 R5; do for k in $(seq -w 1 20); do for a in C F; do
    for e in json npz; do
      f="$REF/shipped-s$s-n640__${r}__k${k}__$a.$e"
      [ -f "$f" ] || { log "N = 640 reference fit missing: $f; nothing analyzed"; exit 5; }
    done
  done; done; done; done
  log "N = 640 reference: $(ls "$REF" | wc -l | tr -d ' ') files"
  for N in 40 80 160 320; do analyses "$FU/G/n$N" "$RULE_FU" "101 211 307" "$REF"; done
  ;;
*)
  echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
log "done"
