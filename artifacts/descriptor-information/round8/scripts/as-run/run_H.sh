#!/usr/bin/env bash
# Part H of docs/frozen-rules/2026-10-04-crossed-follow-ups.md on the released shot-sweep inputs (as run, QEMScore b04f49b).
set -euo pipefail
cd /Users/yzhao062/qemscore-r8/code
A=${ASSETS:-/Users/yzhao062/qemscore-r8/assets}/shot-sweep-v1
R=artifacts/descriptor-information/shot-sweep/levels
ARGS=()
for L in 256 1024 2048 8192 32768 131072 exact; do
  ARGS+=(--analysis-a $L=$R/$L/analysis-a.json --data $L=$A/levels/shots-$L --cache $L=$A/runs/shots-$L/cache)
done
export PYTHONPATH=. OMP_NUM_THREADS=1
P=/Users/yzhao062/miniforge3/envs/py312/bin/python
$P tools/equal_spend.py "${ARGS[@]}" --predictions artifacts/descriptor-information/shot-sweep/predictions.json \
  --rungs R0 N1 N2 R5 --out /Users/yzhao062/qemscore-r8/follow-local/H/dry-run.json --dry-run
$P tools/equal_spend.py "${ARGS[@]}" --predictions artifacts/descriptor-information/shot-sweep/predictions.json \
  --rungs R0 N1 N2 R5 --out /Users/yzhao062/qemscore-r8/follow-local/H/equal-spend.json
