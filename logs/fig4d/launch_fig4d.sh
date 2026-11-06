#!/usr/bin/env bash
# Data-efficiency sweep behind Figure 4d: 3 architectures x 2 training-set sizes
# x 10 seeds (42-51) = 60 jobs.
#
# The subject partition is fixed (--split-seed 42), so every fit is scored on
# the same 15 test subjects. Each fit trains on K subjects drawn with its --seed
# from the 65-subject training pool: K=10 or K=65 (the whole pool). The
# connectome prior and omega are estimated from those K subjects (first halves
# only).
#
# Writes results/runs/ts_young_<model>_fixtest_tr<K>_seed<seed>.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root

DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
COMMON="$DATA_ARGS --no-wandb --skip-figures --max-subjects 94 --split-seed 42"
PY=".venv/bin/python examples/train_models.py backprop"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
SIZES=(10 65)
JOBIDS="logs/fig4d/fig4d_jobids.txt"
: > "$JOBIDS"

for s in "${SEEDS[@]}"; do
  for k in "${SIZES[@]}"; do
    for model in hopf nsde hybrid_hopf; do
      suffix="fixtest_tr${k}_seed${s}"
      jid=$(sbatch -M tinygpu --gres=gpu:1 --time=03:00:00 -o "logs/fig4d/${model}_${suffix}_%j.log" \
        --wrap="$PY --model $model $COMMON --seed $s --train-subset $k --run-suffix $suffix" \
        | awk '{print $4}')
      echo "$jid  $model  $suffix" | tee -a "$JOBIDS"
    done
  done
done

echo "Submitted $(wc -l < "$JOBIDS") jobs."
