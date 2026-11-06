#!/usr/bin/env bash
# Runtime cost of every architecture behind the supplementary timing table
# (tab:timing: training s/epoch, grid-search s/trial, inference s, peak GPU
# memory). One GPU job.
#
# The paper reports an RTX 2080 Ti, so the job requests that GPU type; timings
# on another GPU are not comparable.
#
# Writes results/timing_benchmark.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # repository root

DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
PY=".venv/bin/python examples/benchmark_timing.py"
JOBIDS="logs/revision/timing_jobids.txt"
: > "$JOBIDS"

jid=$(sbatch -M tinygpu --gres=gpu:rtx2080ti:1 --time=02:00:00 --job-name=bench_timing \
  -o "logs/revision/timing_%j.log" --wrap="$PY $DATA_ARGS" | awk '{print $4}')
echo "$jid  timing" | tee -a "$JOBIDS"

echo "Submitted $(wc -l < "$JOBIDS") job."
