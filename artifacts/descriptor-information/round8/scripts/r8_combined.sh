#!/usr/bin/env bash
# One ghx4-interactive node: G (8 workers), then the fresh panel in the rule order (160 workers, R4 12).
# Worker counts only bound memory; every fit is single-threaded and seeded. Each call can be repeated:
# fits already written are skipped, the predictions file is kept, and the score is written once.
set -u
R=/projects/bhph/yzhao13/qemscore-r8; ST=$R/round8_stage.sh; J=${SLURM_JOB_ID:-x}
note() { printf "%s\tnode\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> $R/run.log; }
note "job $J on $(hostname): G with 8 workers, then the fresh panel with 160 workers and 12 for R4"
if WORKERS=8 bash $ST G_fit > $R/slurm/G_fit-$J.out 2>&1 && bash $ST G_analyze > $R/slurm/G_analyze-$J.out 2>&1; then
  note "G fits and analyses done"
else
  note "G failed; the fresh panel still runs"
fi
for st in fresh_check fresh_fitAC fresh_predict fresh_fitall fresh_analyze fresh_score; do
  if [ "$st" = fresh_predict ] && [ -s "$R/fresh/predictions-fresh.json" ]; then continue; fi
  if [ "$st" = fresh_score ] && [ -s "$R/fresh/fresh-confirmation.json" ]; then continue; fi
  WORKERS=160 R4_WORKERS=12 bash $ST $st > $R/slurm/$st-$J.out 2>&1 || { note "stage $st failed; later stages not run"; exit 1; }
done
note "fresh panel done"
