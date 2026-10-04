#!/usr/bin/env bash
# Submit the shot-sweep fitting stages with one node per level for fitall and analyze.
# Order: prepare -> fitAC -> predict -> fitall x 7 levels (parallel) -> analyze x 7 levels
# (parallel, after every fitall, since each analysis reads the 2,048-shot fits) -> hypotheses.
# Usage: submit_fanout.sh [prepare|fitAC|predict|fitall]   (first stage; default prepare)
set -euo pipefail
ROOT=/projects/bhph/yzhao13/qemscore-sweep
mkdir -p "$ROOT/slurm"
LEVELS=(256 1024 2048 8192 32768 131072 exact)
declare -A TIME=([prepare]=02:00:00 [fitAC]=01:00:00 [predict]=00:30:00 [fitall]=04:00:00
                 [analyze]=03:00:00 [hypotheses]=00:30:00)
FIRST="${1:-prepare}"
submit() {  # stage dependency [level]
  local stage="$1" dep="$2" level="${3:-}"
  sbatch --parsable --account=bhph-dtai-gh --partition=ghx4 --nodes=1 --ntasks=1 \
    --gpus-per-node=1 --cpus-per-task=72 --mem=110G --time="${TIME[$stage]}" \
    --job-name="qs-$stage${level:+-$level}" --output="$ROOT/slurm/%x-%j.out" \
    --export="ALL${level:+,ONLY_LEVELS=$level}" ${dep:+--dependency=afterok:$dep} \
    --wrap="bash $ROOT/sweep_stage.sh $stage"
}
dep=""; on=0
for st in prepare fitAC predict; do
  [ "$st" = "$FIRST" ] && on=1
  [ "$on" = 1 ] || continue
  dep="$(submit "$st" "$dep")"; echo "$st $dep"
done
fit_ids=()
for L in "${LEVELS[@]}"; do
  id="$(submit fitall "$dep" "$L")"; echo "fitall-$L $id"; fit_ids+=("$id")
done
fit_dep="$(IFS=:; echo "${fit_ids[*]}")"
an_ids=()
for L in "${LEVELS[@]}"; do
  id="$(submit analyze "$fit_dep" "$L")"; echo "analyze-$L $id"; an_ids+=("$id")
done
an_dep="$(IFS=:; echo "${an_ids[*]}")"
echo "hypotheses $(submit hypotheses "$an_dep")"
