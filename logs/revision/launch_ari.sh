#!/usr/bin/env bash
# FC adjusted Rand index per (model, seed) over the N=94 checkpoints of
# launch_seeds.sh, behind the modularity claim. One GPU job over the whole glob;
# submit it once launch_seeds.sh has finished.
#
# Writes results/ari_per_seed.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root

PY=".venv/bin/python examples/compute_ari.py"
JOBIDS="logs/revision/ari_jobids.txt"
: > "$JOBIDS"

jid=$(sbatch -M tinygpu --gres=gpu:1 --time=02:00:00 -o "logs/revision/ari_%j.log" \
  --wrap="$PY --checkpoints 'checkpoints/ts_young_*_n94_seed*.pt' --out results/ari_per_seed.json" \
  | awk '{print $4}')
echo "$jid  ari" | tee -a "$JOBIDS"

echo "Submitted $(wc -l < "$JOBIDS") job."
