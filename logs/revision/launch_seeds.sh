#!/usr/bin/env bash
# Full rerun of the 10-seed (42-51) sweep at N=94 behind Table 1, the full
# model comparison (Table S1) and the grid-search comparison (Table S2):
#   - backprop: all five architectures x 10 seeds
#   - hopf-grid: Coupled Hopf grid search x 10 seeds
#
# The seed is the unit of analysis: each seed is an independent refit with its
# own subject-level partition. The connectome prior and omega are estimated
# from each partition's training subjects (first halves only).
#
# Writes results/runs/ts_young_<model>_n94_seed<seed>.json,
#        results/runs/ts_young_hopf_grid_seed<seed>.json
# and the matching checkpoints/ts_young_<model>_n94_seed<seed>.pt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root

DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
COMMON="$DATA_ARGS --no-wandb --skip-figures --max-subjects 94"
PY=".venv/bin/python examples/train_models.py"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
JOBIDS="logs/revision/seed_jobids.txt"
: > "$JOBIDS"

submit () {  # $1=log stem  $2=time limit  $3...=train_models.py arguments
  local stem="$1"; local tlim="$2"; shift 2
  local jid
  jid=$(sbatch -M tinygpu --gres=gpu:1 --time="$tlim" -o "logs/revision/${stem}_%j.log" \
    --wrap="$PY $* $COMMON" | awk '{print $4}')
  echo "$jid  $stem" | tee -a "$JOBIDS"
}

for s in "${SEEDS[@]}"; do
  for model in hopf nsde hybrid_hopf gnn_hopf hybrid_neural; do
    submit "${model}_n94_seed${s}" 03:00:00 \
      backprop --model "$model" --seed "$s" --run-suffix "n94_seed${s}"
  done
  submit "hopf_grid_seed${s}" 03:00:00 hopf-grid --seed "$s" --run-suffix "seed${s}"
done

echo "Submitted $(wc -l < "$JOBIDS") jobs."
