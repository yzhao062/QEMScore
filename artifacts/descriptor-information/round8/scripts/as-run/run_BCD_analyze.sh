#!/usr/bin/env bash
# B, C, D analyses of the crossed follow-ups on macOS, from copies of the DeltaAI fits (QEMScore b04f49b).
export ROOT=/Users/yzhao062/qemscore-r8/follow-dai CODE=/Users/yzhao062/qemscore-r8/code SW=/private/tmp/claude-501/-Users-yzhao062-PycharmProjects-internal-writing/45633ba5-81f7-4ac2-97b2-23227db37d1c/scratchpad/r7/assets/shot-sweep-v1
export PY=/Users/yzhao062/miniforge3/envs/py312/bin/python SHA256="shasum -a 256" COMMIT=b04f49b737f310b3cd0673d8e9d63fb2b8ce8fe9
ST=/Users/yzhao062/qemscore-r8/code/artifacts/descriptor-information/round8/scripts/round8_stage.sh
pids=()
for st in B_analyze C_analyze D_analyze; do
  bash $ST $st > $ROOT/$st.out 2>&1 & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
echo "BCD analyses exit $rc" >> $ROOT/BCD.done
