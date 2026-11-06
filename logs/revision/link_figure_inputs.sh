#!/usr/bin/env bash
# Inputs of the figure-panel notebooks (Figure 4a-c, supplementary figure
# fig:supp_modularity_complexity) and of plot_grid_search_fc_similarity.py
# (Figure 2a): the seed-42 fit of every
# architecture from launch_seeds.sh. No GPU; run once those jobs have finished.
#
# The notebooks load every checkpoint in their directory and keep one per
# architecture, and metrics_over_epochs keeps the newest history per model, so
# they read these dedicated directories instead of the sweep's. Creates
# relative symlinks:
#   checkpoints/figures/ts_young_<model>.pt     -> ts_young_<model>_n94_seed42.pt
#   checkpoints/figures/ts_young_hopf_grid.pt   -> ts_young_hopf_grid_seed42.pt
#   results/metrics_figures/<model>_backprop_ts_young
#                                               -> metrics/<model>_backprop_ts_young_n94_seed42
#   results/grid_search/grid_search_results.json -> seed42/grid_search_results.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root

SEED="${SEED:-42}"
mkdir -p checkpoints/figures results/metrics_figures results/grid_search

for model in hopf nsde hybrid_hopf gnn_hopf hybrid_neural; do
  ln -sfn "../ts_young_${model}_n94_seed${SEED}.pt" "checkpoints/figures/ts_young_${model}.pt"
  ln -sfn "../metrics/${model}_backprop_ts_young_n94_seed${SEED}" \
    "results/metrics_figures/${model}_backprop_ts_young"
done
ln -sfn "../ts_young_hopf_grid_seed${SEED}.pt" checkpoints/figures/ts_young_hopf_grid.pt
ln -sfn "seed${SEED}/grid_search_results.json" results/grid_search/grid_search_results.json

# A dangling link means the run it points to has not finished (or failed).
missing=$(find checkpoints/figures results/metrics_figures results/grid_search -maxdepth 1 -xtype l)
if [ -n "$missing" ]; then
  echo "Missing targets (run not finished?):" >&2
  echo "$missing" | sed 's/^/  /' >&2
  exit 1
fi
echo "Linked the seed-${SEED} fits into checkpoints/figures and results/metrics_figures."
