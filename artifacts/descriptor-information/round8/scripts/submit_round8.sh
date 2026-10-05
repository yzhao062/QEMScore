#!/usr/bin/env bash
# Submit one round8_node.sh call to a full ghx4-interactive node. Usage: submit_round8.sh follow|fresh
set -euo pipefail
ROOT=/projects/bhph/yzhao13/qemscore-r8
mkdir -p "$ROOT/slurm"
sbatch --parsable --account=bhph-dtai-gh --partition=ghx4-interactive --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --cpus-per-task=288 --mem=0 --time=02:00:00 \
  --job-name="r8-$1" --output="$ROOT/slurm/%x-%j.out" \
  --export="ALL,COMMIT=${COMMIT:?set COMMIT}" --wrap="bash $ROOT/round8_node.sh $1"
