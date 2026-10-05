#!/usr/bin/env bash
# Fresh-circuit confirmation panel, macOS part (frozen rule
# docs/frozen-rules/2026-10-04-fresh-confirmation.md): generate seeds 401, 503,
# and 607, check determinism, derive the six other shot levels with the sweep
# rule's three checks, prepare every tree, and record each tree's cache SHA-256
# for the DeltaAI copies. Usage: bash mac_fresh.sh CODE OUT
set -euo pipefail
CODE="$1"; OUT="$2"
PY=/Users/yzhao062/miniforge3/envs/py312/bin/python
# SEEDS and COUNTS are overridden only for a rehearsal on throwaway seeds.
SEEDS="${SEEDS:-401 503 607}"
COUNTS="${COUNTS:-}"
VERIFY_SEED="${SEEDS%% *}"
LV="$OUT/levels"; PR="$OUT/prepared"; REP="$OUT/reports"; LOG="$OUT/run.log"
RULE=docs/frozen-rules/2026-10-04-fresh-confirmation.md
export PYTHONPATH="$CODE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LV" "$PR" "$REP"
cd "$CODE"
log() { printf '%s\tmac\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }
run() { log "cmd: $*"; "$@"; }
ds() { for s in $SEEDS; do printf '%s ' "$LV/shots-$1/regen-shipped-s$s-n640"; done; }

log "start commit=$(git -C "$CODE" rev-parse HEAD) rule_sha256=$(shasum -a 256 "$RULE" | cut -d' ' -f1)"
log "versions: $($PY -c 'import sys,numpy,scipy,sklearn,qiskit,qiskit_aer,platform;print(sys.version.split()[0],numpy.__version__,scipy.__version__,sklearn.__version__,qiskit.__version__,qiskit_aer.__version__,platform.machine())')"
# The seeds are independent, so each runs in its own process; any failure stops the script.
waitall() { local pid rc=0; for pid in "$@"; do wait "$pid" || rc=1; done; return $rc; }
# 1. Generation and the determinism check.
pids=()
for s in $SEEDS; do
  run $PY tools/fresh_panel.py generate --seeds "$s" --out "$LV/shots-2048" --size 640 ${COUNTS:+--counts $COUNTS} &
  pids+=($!)
done
waitall "${pids[@]}"
run $PY tools/fresh_panel.py verify --seed "$VERIFY_SEED" --out "$LV/shots-2048" --target "$OUT/verify-$VERIFY_SEED" ${COUNTS:+--counts $COUNTS}
# 2. Derived levels with the sweep rule's checks (faithfulness, exact level).
derive() {
  local s="$1" L
  SRC="$LV/shots-2048/regen-shipped-s$s-n640"
  run $PY tools/shot_sweep.py generate --source "$SRC" --shots 2048 --out "$OUT/faithful/regen-shipped-s$s-n640" --workers 4
  run $PY tools/shot_sweep.py check-faithful --source "$SRC" --regen "$OUT/faithful/regen-shipped-s$s-n640" \
      --report "$REP/check1-faithful-s$s.json"
  for L in 256 1024 8192 32768 131072; do
    run $PY tools/shot_sweep.py generate --source "$SRC" --shots "$L" --out "$LV/shots-$L/regen-shipped-s$s-n640" --workers 4
  done
  run $PY tools/shot_sweep.py generate --source "$SRC" --exact --out "$LV/shots-exact/regen-shipped-s$s-n640" --workers 4
  run $PY tools/shot_sweep.py check-exact --exact "$LV/shots-exact/regen-shipped-s$s-n640" \
      --finite "$LV/shots-131072/regen-shipped-s$s-n640" --report "$REP/check2-exact-s$s.json"
}
pids=()
for s in $SEEDS; do derive "$s" & pids+=($!); done
waitall "${pids[@]}"
# 3. Preparation: the 2,048-shot tree in full, the other levels in sweep mode, then the cache check.
run $PY tools/descriptor_ladder.py prepare --fresh --datasets $(ds 2048) --out "$PR/shots-2048" --workers 4
for L in 256 1024 8192 32768 131072 exact; do
  run $PY tools/descriptor_ladder.py prepare --sweep --fresh --fresh-source "$LV/shots-2048" \
      --datasets $(ds "$L") --out "$PR/shots-$L" --workers 4
  run $PY tools/shot_sweep.py check-caches --dataset-seeds $SEEDS --reference "$PR/shots-2048" \
      --level "$PR/shots-$L" --reference-only-encoder-cache --report "$REP/check3-caches-$L.json"
done
# 4. Cache records for the copies, and the SHA-256 of every file DeltaAI receives.
for L in 256 1024 2048 8192 32768 131072 exact; do
  run $PY tools/fresh_panel.py record-caches --tree "$PR/shots-$L" --out "$OUT/records/cache-record-$L.json"
done
( cd "$OUT" && find levels prepared records -type f ! -name '*.tmp' -print0 | sort -z | xargs -0 shasum -a 256 ) > "$REP/tree-sha256.txt"
log "done: $(wc -l < "$REP/tree-sha256.txt") files listed in reports/tree-sha256.txt"
