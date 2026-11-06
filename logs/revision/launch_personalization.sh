#!/usr/bin/env bash
# Personalization (N = 1) fits behind Table S3 and Figure 3e: one scratch fit per
# (architecture, target subject), seed 42.
#
# The group connectome prior of every fit is the prior of the split-seed-42
# cohort fit (mean FC over the first halves of its 65 training subjects). The
# targets are therefore the first ten subjects outside that training pool, so no
# data of a personalized subject enters the prior; train_personalized.py refuses
# any target inside the pool.
#
# Writes results/personalization/ts_young_<model>_sub<idx>_seed42.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root

DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
COMMON="$DATA_ARGS --no-wandb --skip-figures --max-subjects 94 --seed 42"
PY=".venv/bin/python examples/train_personalized.py"
TARGETS=(1 2 14 20 21 23 29 32 37 41)
JOBIDS="logs/revision/personalization_jobids.txt"
: > "$JOBIDS"

for model in hopf nsde hybrid_hopf; do
  for subj in "${TARGETS[@]}"; do
    stem="pers_${model}_sub${subj}"
    jid=$(sbatch -M tinygpu --gres=gpu:1 --time=02:00:00 -o "logs/revision/${stem}_%j.log" \
      --wrap="$PY --model $model --subject $subj $COMMON" | awk '{print $4}')
    echo "$jid  $stem" | tee -a "$JOBIDS"
  done
done

echo "Submitted $(wc -l < "$JOBIDS") jobs."
