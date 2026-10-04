#!/usr/bin/env bash
# Strong-learner rerun on DeltaAI: setup on the login node, then Part A and Part B on two
# nodes in parallel, then one analysis job after both.
set -euo pipefail
ROOT=/projects/bhph/yzhao13/qemscore-strong
mkdir -p "$ROOT/slurm"
bash "$ROOT/strong_stage.sh" setup
submit() {  # stage time [dependency]
  sbatch --parsable --account=bhph-dtai-gh --partition=ghx4 --nodes=1 --ntasks=1 \
    --gpus-per-node=1 --cpus-per-task=72 --mem=110G --time="$2" \
    --job-name="sl-$1" --output="$ROOT/slurm/%x-%j.out" ${3:+--dependency=afterok:$3} \
    --wrap="bash $ROOT/strong_stage.sh $1"
}
a=$(submit fitA 03:00:00); echo "fitA $a"
b=$(submit fitB 01:00:00); echo "fitB $b"
echo "analyze $(submit analyze 03:00:00 "$a:$b")"
