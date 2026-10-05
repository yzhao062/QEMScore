#!/usr/bin/env bash
# ML-QEM own-data fits under docs/frozen-rules/2026-10-04-mlqem-own-data.md (QEMScore b04f49b).
set -euo pipefail
cd /Users/yzhao062/qemscore-r8/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
/Users/yzhao062/miniforge3/envs/mlqem/bin/python -W ignore -m reanalysis.mlqem.run \
  --settings no_readout readout coherent --models ols rf mlp --arms F C P R Rcal \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --data-root /private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/r8/upstream/ml-qem \
  --output-dir /Users/yzhao062/qemscore-r8/mlqem-own-data/fits --workers 8 \
  --frozen-rule docs/frozen-rules/2026-10-04-mlqem-own-data.md
echo "run.py exit $?"
