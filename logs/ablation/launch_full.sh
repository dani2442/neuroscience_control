#!/usr/bin/env bash
# Launch the full ablation batch on ts_young: 8 configs x 3 seeds = 24 GPU jobs.
# Writes results to results/runs/ts_young_<model>_<ablkey>_seed<seed>.json
set -euo pipefail
cd /home/hpc/dsaa/dsaa110h/projects/neuroscience_control

DATA="data/ts_young/ts_young_TR0.72.mat"
COMMON="--dataset-type ts_young --data-path $DATA --no-wandb --skip-figures"
SEEDS=(42 43 44)
PY=".venv/bin/python examples/train_models.py backprop"
JOBIDS="logs/ablation/full_jobids.txt"
: > "$JOBIDS"

submit () {  # $1=ablkey  $2=model  $3=time  $4...=extra flags
  local ablkey="$1"; local model="$2"; local tlim="$3"; shift 3
  local extra="$*"
  for s in "${SEEDS[@]}"; do
    local suffix="${ablkey}_seed${s}"
    local out="logs/ablation/${model}_${suffix}_%j.log"
    local jid
    jid=$(sbatch -M tinygpu --gres=gpu:1 --time="$tlim" -o "$out" \
      --wrap="$PY --model $model $COMMON $extra --seed $s --run-suffix $suffix" \
      | awk '{print $4}')
    echo "$jid  $model  $suffix" | tee -a "$JOBIDS"
  done
}

#        ablkey         model         time      extra flags
submit   baseline       hybrid_hopf   03:00:00
submit   pmatch         nsde          01:30:00  --hidden-dim 4 --n-layers 1
submit   scshuffle      hybrid_hopf   03:00:00  --sc-mode shuffled
submit   scrandom       hybrid_hopf   03:00:00  --sc-mode random
submit   fixedcoupling  hybrid_hopf   03:00:00  --no-learnable-coupling
submit   nolocal        hybrid_hopf   03:00:00  --disable-local
submit   nofdm          hybrid_hopf   03:00:00  --zero-loss fdm
submit   nodyn          hybrid_hopf   03:00:00  --zero-loss fcd phfcd phase_fc_correlation metastability

echo "Submitted $(wc -l < "$JOBIDS") jobs."
