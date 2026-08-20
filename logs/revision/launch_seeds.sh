#!/usr/bin/env bash
# Complete the 10-seed (42-51) set at N=94 for every architecture in Table 1 / S1.
#
# Rationale (reviewer point 3): the main table previously reported the spread
# across held-out test *batches* of a single fit, which excludes fitting
# variability entirely. The revised protocol treats the *seed* (independent
# refit) as the unit of analysis, so every architecture needs the same 10 seeds.
#
# Already present: hopf (9/10, missing 45), nsde (10/10), hybrid_hopf (10/10).
# Missing entirely: gnn_hopf, hybrid_neural.
#
# Writes results/runs/ts_young_<model>_n94_seed<seed>.json
set -euo pipefail
cd /home/hpc/dsaa/dsaa110h/projects/neuroscience_control

DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
COMMON="$DATA_ARGS --no-wandb --skip-figures --max-subjects 94"
PY=".venv/bin/python examples/train_models.py backprop"
JOBIDS="logs/revision/seed_jobids.txt"
: > "$JOBIDS"

submit () {  # $1=model $2=seed $3=time
  local model="$1"; local s="$2"; local tlim="$3"
  local suffix="n94_seed${s}"
  local out="logs/revision/${model}_${suffix}_%j.log"
  local jid
  jid=$(sbatch -M tinygpu --gres=gpu:1 --time="$tlim" -o "$out" \
    --wrap="$PY --model $model $COMMON --seed $s --run-suffix $suffix" \
    | awk '{print $4}')
  echo "$jid  $model  $suffix" | tee -a "$JOBIDS"
}

# Coupled Hopf: only seed 45 is missing from the existing sweep.
submit hopf 45 02:00:00

# GNN-Hopf and Hopf+Neural: no multi-seed runs existed at all.
for s in 42 43 44 45 46 47 48 49 50 51; do
  submit gnn_hopf      "$s" 03:00:00
  submit hybrid_neural "$s" 03:00:00
done

echo "Submitted $(wc -l < "$JOBIDS") jobs."
