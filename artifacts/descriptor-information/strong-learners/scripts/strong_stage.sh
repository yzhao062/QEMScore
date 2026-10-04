#!/usr/bin/env bash
# Strong-learner rerun of the descriptor-information experiment on NCSA DeltaAI.
# Frozen rule: docs/frozen-rules/2026-10-03-strong-learners.md (QEMScore 832ec36). Stages:
#   setup    clone the code at the rule commit, copy the prepared trees, record versions
#   fitA     Part A: 9 rungs x 3 datasets x (A + 20 x C, F, P) = 1,647 fits
#   fitB     Part B: 3 rungs x 3 datasets x (A + 20 x C, F, P) + 3 GBT = 552 fits
#   analyze  derived baseline, analyses A and B with reconcile (rerun and derived baseline),
#            platform comparison, learner adequacy, and the strong-learner report
set -euo pipefail
STAGE="$1"
ROOT=/projects/bhph/yzhao13/qemscore-strong
CODE="$ROOT/code"
IN=/projects/bhph/yzhao13/qemscore-sweep/inputs
RUNS="$ROOT/runs"
OUT="$ROOT/analysis"
LOG="$ROOT/run.log"
COMMIT=832ec36
PY=/projects/bhph/yzhao13/miniforge3/envs/py312/bin/python
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-72}}"
RULE=docs/frozen-rules/2026-10-03-strong-learners.md
GATE_FILE=artifacts/descriptor-information/gate/gate_r0.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$CODE"
mkdir -p "$RUNS" "$OUT"
log() { printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STAGE" "$*" >> "$LOG"; }
run() { log "cmd: $*"; "$@"; }

case "$STAGE" in
setup)
  [ -d "$CODE/.git" ] || git clone -q https://github.com/yzhao062/QEMScore.git "$CODE"
  git -C "$CODE" checkout -q "$COMMIT"
  mkdir -p "$RUNS/partA-strong" "$RUNS/partB-strong"
  cp -R "$IN/partA-prepared/cache" "$IN/partA-prepared/descriptors" "$IN/partA-prepared/encoder-cache" "$RUNS/partA-strong/"
  cp "$IN/partA-prepared/datasets.json" "$IN/partA-prepared/dataset_checks.json" "$RUNS/partA-strong/"
  ;;
probe)
  cd "$CODE"
  log "threadpool_info (re-recorded for fitB, whose first record imported no BLAS user): $($PY -c 'import json, numpy, scipy.linalg, sklearn.ensemble, threadpoolctl; print(json.dumps(threadpoolctl.threadpool_info()))')"
  ;;
fitA|fitB)
  cd "$CODE"
  log "start job=${SLURM_JOB_ID:-none} host=$(hostname) commit=$(git -C "$CODE" rev-parse HEAD) rule_sha256=$(sha256sum "$RULE" | cut -d' ' -f1)"
  log "versions: $($PY -c 'import sys,numpy,scipy,sklearn,platform;print(sys.version.split()[0],numpy.__version__,scipy.__version__,sklearn.__version__,platform.machine())')"
  log "numpy blas: $($PY -c 'import numpy, json; c = numpy.show_config(mode="dicts"); print(json.dumps(c.get("Build Dependencies", {}).get("blas", {})))')"
  log "threadpool_info: $($PY -c 'import json, numpy, scipy.linalg, sklearn.ensemble, threadpoolctl; print(json.dumps(threadpoolctl.threadpool_info()))')"
  log "blas_fpe_probe: $($PY -c 'import json,sys; sys.path.insert(0,"tools"); import descriptor_common as c; print(json.dumps(c.blas_fpe_probe()))')"
  if [ "$STAGE" = fitA ]; then
    run $PY tools/descriptor_ladder.py run --strength-indicator --strong-learners \
        --rungs R0 N1 N2 N3 N4 R3-TFI R3-Heis R4 R5 --arms A C F P --seeds 1-20 \
        --datasets "$IN/data/regen-shipped-s101-n640" "$IN/data/regen-shipped-s211-n640" "$IN/data/regen-shipped-s307-n640" \
        --archive "$IN/campaign-archive-v1" --gate-file "$GATE_FILE" \
        --out "$RUNS/partA-strong" --workers "$WORKERS"
  else
    run $PY tools/qaoa_intermediate.py run --strength-indicator --strong-learners \
        --data-root "$IN/qaoa-data-deltaai" --gate-file "$GATE_FILE" \
        --out "$RUNS/partB-strong" --workers "$WORKERS"
  fi
  ;;
analyze)
  cd "$CODE"
  log "start job=${SLURM_JOB_ID:-none} host=$(hostname) commit=$(git -C "$CODE" rev-parse HEAD)"
  A="$RUNS/partA-strong"; B="$RUNS/partB-strong"; DER="$RUNS/derived-baseline"
  run $PY tools/strong_learner_derive.py --fits "$A/fits" --fits "$B/fits" --out "$DER/fits"
  # Primary comparison: the rerun against its derived same-platform baseline.
  run $PY tools/descriptor_ladder_analysis.py --fits "$A/fits" --fits "$B/fits" --cache "$A/cache" --cache "$B/cache" \
      --original-fits "$DER/fits" --follow-up-rule "$RULE" --out "$OUT/analysis-a.json" --quiet
  run $PY tools/descriptor_ladder_analysis_b.py --fits "$A/fits" "$B/fits" --cache "$A/cache" "$B/cache" \
      --follow-up-rule "$RULE" --out "$OUT/analysis-b.json"
  run $PY tools/descriptor_ladder_reconcile.py "$OUT/analysis-a.json" "$OUT/analysis-b.json" --out "$OUT/reconcile.json"
  # The derived baseline's own statements, and the platform comparison with the macOS strength fits.
  run $PY tools/descriptor_ladder_analysis.py --fits "$DER/fits" --cache "$A/cache" --cache "$B/cache" \
      --original-fits "$IN/descriptor-information-strength-v1/fits" --follow-up-rule "$RULE" \
      --out "$OUT/derived-analysis-a.json" --quiet
  run $PY tools/descriptor_ladder_analysis_b.py --fits "$DER/fits" --cache "$A/cache" "$B/cache" \
      --follow-up-rule "$RULE" --out "$OUT/derived-analysis-b.json"
  run $PY tools/descriptor_ladder_reconcile.py "$OUT/derived-analysis-a.json" "$OUT/derived-analysis-b.json" \
      --out "$OUT/derived-reconcile.json"
  # Stated in advance outside the agreement criterion.
  run $PY tools/learner_adequacy.py --fits "$A/fits" --fits "$B/fits" --cache "$A/cache" --cache "$B/cache" \
      --analysis-a "$OUT/analysis-a.json" --out "$OUT/learner-adequacy.json" --quiet
  run $PY tools/strong_learner_report.py --fits "$A/fits" --fits "$B/fits" --cache "$A/cache" --cache "$B/cache" \
      --out "$OUT/strong-learner-report.json"
  ;;
*)
  echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
log "done"
