#!/usr/bin/env bash
# Part E of docs/frozen-rules/2026-10-04-crossed-follow-ups.md on the released fits (QEMScore b04f49b).
set -euo pipefail
cd /Users/yzhao062/qemscore-r8/code
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/Users/yzhao062/miniforge3/envs/py312/bin/python tools/measurement_free_stack.py $(cat /Users/yzhao062/qemscore-r8/follow-local/E/args.txt) \
  --g-fits /private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/r7/assets/strong-learners-v1/partA-fits \
  --stacking-all artifacts/descriptor-information/posthoc-stacking-all.json --rungs R0 N1 N2 \
  --out /Users/yzhao062/qemscore-r8/follow-local/E/measurement-free-stack.json
echo "E exit $?"
