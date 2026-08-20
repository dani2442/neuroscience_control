#!/usr/bin/env bash
# Posterior-predictive / personalization evaluation over fitted checkpoints.
#
# Evaluation-time only: loads each checkpoint, draws --n-realizations independent
# trajectories per subject, and writes per-realization + per-subject records to
# results/generative/. No retraining.
#
# Usage: launch_generative.sh [model ...]   (default: the three main architectures)
set -euo pipefail
cd /home/hpc/dsaa/dsaa110h/projects/neuroscience_control

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then MODELS=(hopf nsde hybrid_hopf); fi

DATA="data/ts_young/ts_young_TR0.72.mat"
PY=".venv/bin/python examples/evaluate_generative.py"
JOBIDS="logs/revision/generative_jobids.txt"
mkdir -p results/generative
: >> "$JOBIDS"

n=0
for m in "${MODELS[@]}"; do
  for ckpt in checkpoints/ts_young_${m}_n94_seed*.pt; do
    [ -e "$ckpt" ] || continue
    stem=$(basename "$ckpt" .pt)
    out="results/generative/${stem}.json"
    [ -e "$out" ] && { echo "skip (exists): $stem"; continue; }
    jid=$(sbatch -M tinygpu --gres=gpu:1 --time=01:30:00 \
      -o "logs/revision/gen_${stem}_%j.log" \
      --wrap="$PY --checkpoint $ckpt --data-path $DATA --n-realizations 20 --out $out" \
      | awk '{print $4}')
    echo "$jid  $stem" | tee -a "$JOBIDS"
    n=$((n+1))
  done
done
echo "Submitted $n generative-eval jobs."
