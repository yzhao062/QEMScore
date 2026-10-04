#!/usr/bin/env bash
# Completes Part A after fitA in job 3308354 stopped at fit 483 (BrokenProcessPool: the node ran out
# of memory with 140 workers; one R4 fit peaks at 7.1 GB, one R0 fit at 0.44 GB; R4 runs with 16 workers). The same command
# runs in two invocations with bounded workers; run skips the fits already written. Then analyze.
set -euo pipefail
ROOT=/projects/bhph/yzhao13/qemscore-strong; IN=/projects/bhph/yzhao13/qemscore-sweep/inputs
PY=/projects/bhph/yzhao13/miniforge3/envs/py312/bin/python; LOG=$ROOT/run.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$ROOT/code
log() { printf "%s\tfitA-resume\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> $LOG; }
cd $ROOT/code
for spec in "120:R0 N1 N2 N3 N4 R3-TFI R3-Heis R5" "16:R4"; do
  w=${spec%%:*}; rungs=${spec#*:}
  log "cmd (workers $w): descriptor_ladder.py run --strength-indicator --strong-learners --rungs $rungs --arms A C F P --seeds 1-20"
  $PY tools/descriptor_ladder.py run --strength-indicator --strong-learners --rungs $rungs --arms A C F P --seeds 1-20 \
      --datasets $IN/data/regen-shipped-s101-n640 $IN/data/regen-shipped-s211-n640 $IN/data/regen-shipped-s307-n640 \
      --archive $IN/campaign-archive-v1 --gate-file artifacts/descriptor-information/gate/gate_r0.json \
      --out $ROOT/runs/partA-strong --workers $w
done
n=$(ls $ROOT/runs/partA-strong/fits/*.json | wc -l); log "Part A fits present: $n of 1647"
[ "$n" = 1647 ] || { log "Part A incomplete; analyze not run"; exit 3; }
log "done"
bash $ROOT/strong_stage.sh analyze
