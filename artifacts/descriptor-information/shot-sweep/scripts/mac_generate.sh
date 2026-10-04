#!/usr/bin/env bash
# Shot sweep, rule 2026-10-03-shot-sweep.md, check 1 fallback: generate every level on the
# machine that produced the earlier datasets (this Mac), after check 1 passes here.
set -euo pipefail
CODE=/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/r6/mac-gen/code
SRC=/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/data
OUT=/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/r6/mac-gen
LV="$OUT/levels"; REP="$OUT/reports"; LOG="$OUT/mac-run.log"
PY=/Users/yzhao062/miniforge3/envs/py312/bin/python
W=16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$CODE"
mkdir -p "$LV" "$REP"
log() { printf '%s\tmac\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }
run() { log "cmd: $*"; "$@"; }
cd "$CODE"
log "start commit=$(git -C "$CODE" rev-parse HEAD) host=$(hostname) machine=$(uname -m)"
log "versions: $($PY -c 'import sys,numpy,scipy,sklearn,qiskit,qiskit_aer,stim,platform;print(sys.version.split()[0],numpy.__version__,scipy.__version__,sklearn.__version__,qiskit.__version__,qiskit_aer.__version__,stim.__version__,platform.machine())')"
for s in 101 211 307; do
  run $PY tools/shot_sweep.py generate --source "$SRC/regen-shipped-s$s-n640" --shots 2048 \
      --out "$LV/shots-2048/regen-shipped-s$s-n640" --workers $W
  run $PY tools/shot_sweep.py check-faithful --source "$SRC/regen-shipped-s$s-n640" \
      --regen "$LV/shots-2048/regen-shipped-s$s-n640" --report "$REP/check1-faithful-s$s.json"
done
log "check 1 passed on this machine for all three datasets"
for L in 256 1024 8192 32768 131072; do
  for s in 101 211 307; do
    run $PY tools/shot_sweep.py generate --source "$SRC/regen-shipped-s$s-n640" --shots "$L" \
        --out "$LV/shots-$L/regen-shipped-s$s-n640" --workers $W
  done
done
for s in 101 211 307; do
  run $PY tools/shot_sweep.py generate --source "$SRC/regen-shipped-s$s-n640" --exact \
      --out "$LV/shots-exact/regen-shipped-s$s-n640" --workers $W
  run $PY tools/shot_sweep.py check-exact --exact "$LV/shots-exact/regen-shipped-s$s-n640" \
      --finite "$LV/shots-131072/regen-shipped-s$s-n640" --report "$REP/check2-exact-s$s.json"
done
( cd "$LV" && find . -name items.jsonl -print0 | sort -z | xargs -0 shasum -a 256 ) > "$REP/level-items-sha256.txt"
log "check 2 passed; all levels generated; items.jsonl SHA-256 in reports/level-items-sha256.txt"
echo MAC-GEN-DONE
