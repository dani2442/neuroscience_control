# Reproducing the paper

End-to-end commands to regenerate every figure and table in `paper_new/main.tex` from a fresh checkout.

All paths are relative to the repo root. Every Python invocation assumes you've activated the project venv (`uv sync` then `source .venv/bin/activate`) or use `.venv/bin/python` explicitly as shown.

---

## 0. Prerequisites

```bash
# Clone + sync
git clone https://github.com/dani2442/neuroscience_control.git
cd neuroscience_control
git submodule update --init --recursive
uv sync
```

Data must live at:

| Dataset | Expected path | Required for |
|---|---|---|
| HCP `ts_young` (`.mat`) | `data/ts_young/ts_young_TR0.72.mat` | All figures |

Outputs land in:

| Folder | Contents |
|---|---|
| `checkpoints/` | `ts_young_{model}.pt` from each training run |
| `results/` | Canonical per-model JSON (`ts_young_{model}.json`) and aggregated `paper_metrics_*.json` |
| `results/runs/` | Per-seed and per-(size, seed) JSONs from `--run-suffix` runs |
| `results/metrics/` | Per-model training metric histories (epoch curves) |
| `paper_new/images*/` | Figure panels and intermediate exports |

---

## 1. Quick reproduction (canonical paper)

One command trains every model variant and produces the canonical checkpoints + per-model JSONs:

```bash
.venv/bin/python examples/train_models.py paper \
    --dataset-type ts_young \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --no-wandb
```

The `paper` run sequentially trains: `hopf-grid` → `hopf` → `nsde` → `hybrid_hopf` → `gnn_hopf` → `hybrid_neural` and writes:
- `checkpoints/ts_young_{model}.pt` for each
- `results/ts_young_{model}.json` (test_inter + test_intra metrics)

If you want them in parallel on a cluster instead of sequential, see [§5 Slurm equivalents](#5-slurm-equivalents) below.

Then run the post-training pipeline:

```bash
# Patches LaTeX tables in paper_new/, generates comparison figures
.venv/bin/python examples/postprocess.py pipeline \
    --dataset-type ts_young \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --no-wandb
```

---

## 2. Per-seed runs for Fig. 2 (grid vs gradient with error bars)

[`fig2_grid_vs_gradient.ipynb`](examples/fig2_grid_vs_gradient.ipynb) plots mean ± std across **three seeds (42, 43, 44)**. Required output: `results/runs/ts_young_{hopf|hopf_grid}_seed{S}.json` for `S ∈ {42, 43, 44}`.

```bash
DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
for s in 42 43 44; do
  .venv/bin/python examples/train_models.py hopf-grid \
      $DATA_ARGS --no-wandb --seed $s --run-suffix seed$s
  .venv/bin/python examples/train_models.py backprop --model hopf \
      $DATA_ARGS --no-wandb --seed $s --run-suffix seed$s
done
```

Produces 6 JSONs: 3 seeds × 2 methods.

---

## 3. Dataset-size sweep for Fig. 3E (robustness)

[`fig3_size_robustness.ipynb`](examples/fig3_size_robustness.ipynb) plots three models trained on `N ∈ {10, 25, 50, 94}` subjects across **three seeds**. Required output: `results/runs/ts_young_{model}_n{N}_seed{S}.json`.

```bash
DATA_ARGS="--dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"
for s in 42 43 44; do
  for n in 10 25 50 94; do
    for m in hopf nsde hybrid_hopf; do
      .venv/bin/python examples/train_models.py backprop --model $m \
          $DATA_ARGS --no-wandb \
          --max-subjects $n --seed $s \
          --run-suffix n${n}_seed${s}
    done
  done
done
```

Produces 36 JSONs: 3 seeds × 4 sizes × 3 models.

---

## 4. Generate figures from notebooks

After §1–§3 finish, execute notebooks in any order. Each writes panels into `paper_new/images/`, `paper_new/images_png/`, `paper_new/images_svg/`.

```bash
for nb in examples/fig1_*.ipynb examples/fig2_*.ipynb examples/fig3_*.ipynb; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 "$nb"
done
```

| Notebook | Reads | Writes |
|---|---|---|
| [`fig1_data_pipeline.ipynb`](examples/fig1_data_pipeline.ipynb) | `data/ts_young/` | `paper_new/images*/representation/` (26 SVG/PNG each) |
| [`fig1_brain_renderings.ipynb`](examples/fig1_brain_renderings.ipynb) | `checkpoints/ts_young_{hopf,nsde,hybrid_hopf}.pt`, clones [`dagush/WholeBrain`](https://github.com/dagush/WholeBrain) into `.cache/` | `paper_new/images*/wholebrain/wholebrain_timestep_lateral_magma.{pdf,png}` |
| [`fig2_grid_vs_gradient.ipynb`](examples/fig2_grid_vs_gradient.ipynb) | `results/runs/ts_young_{hopf,hopf_grid}_seed{42,43,44}.json` (see §2) | `paper_new/images*/comparison/ts_young_old_vs_new_hopf_seeds.{svg,png}` |
| [`fig3_fc_fcd_phfcd.ipynb`](examples/fig3_fc_fcd_phfcd.ipynb) | `checkpoints/ts_young_{model}.pt` | `paper_new/images*/wholebrain_matrices/` + first-run cache in `cache/` |
| [`fig3_training_curves.ipynb`](examples/fig3_training_curves.ipynb) | `results/metrics/{model}/` (written during training) | `paper_new/images*/ts_young/metrics_over_epochs*.{pdf,png,svg}` |
| [`fig3_clustering_complexity.ipynb`](examples/fig3_clustering_complexity.ipynb) | `checkpoints/ts_young_{model}.pt` | `paper_new/images*/wholebrain/` complexity panels + `cache/` |
| [`fig3_personalization.ipynb`](examples/fig3_personalization.ipynb) | `results/ts_young_{model}.json` from §1 | `paper_new/images*/comparison/intra_vs_inter*.{pdf,png,svg}` |
| [`fig3_size_robustness.ipynb`](examples/fig3_size_robustness.ipynb) | `results/runs/ts_young_{model}_n{N}_seed{S}.json` (see §3) | `paper_new/images*/comparison/ts_young_dataset_size_sweep_*.{svg,png}` |
| [`fig3_brain_panels.ipynb`](examples/fig3_brain_panels.ipynb) | `checkpoints/ts_young_{model}.pt`, [`dagush/WholeBrain`](https://github.com/dagush/WholeBrain) | `paper_new/images*/wholebrain/wholebrain_{fc,fcd,phfcd}_panel.png`, `_fc_residuals.png` |

The four figures actually referenced by `paper_new/main.tex` are in `paper_new/images/diagrams/` and are exported from drawio sources in the same folder. Drawios embed the notebook outputs as base64; re-export after re-importing updated panels.

---

## 5. Slurm equivalents

Wrap each `.venv/bin/python ...` line with the Slurm headers your cluster expects. Example for `tinygpu`:

```bash
sbatch -M tinygpu --gres=gpu:1 --time=02:00:00 \
    --wrap=".venv/bin/python examples/train_models.py paper \
        --dataset-type ts_young \
        --data-path data/ts_young/ts_young_TR0.72.mat \
        --no-wandb"
```

The same headers work for `srun` for synchronous runs. Loops in §2–§3 can be parallelised by wrapping the inner command line:

```bash
for s in 42 43 44; do
  for n in 10 25 50 94; do
    for m in hopf nsde hybrid_hopf; do
      sbatch -M tinygpu --gres=gpu:1 --time=02:00:00 \
          --wrap=".venv/bin/python examples/train_models.py backprop --model $m \
              --dataset-type ts_young \
              --data-path data/ts_young/ts_young_TR0.72.mat \
              --no-wandb --max-subjects $n --seed $s \
              --run-suffix n${n}_seed${s}"
    done
  done
done
```

For per-job parallelism in §1, replace the single `paper` command with six individual `backprop`/`hopf-grid` commands wrapped in `sbatch`.

---

## 6. Verifying your draft commands

The commands you provided are correct. Differences and recommendations:

| Your command | Status | Note |
|---|---|---|
| Loop with `for s in 42..51` over hopf-grid + hopf backprop | ✅ Works, but wasteful | The notebook reads only `SEEDS = [42, 43, 44]`. Using 10 seeds writes extra JSONs that the notebook ignores. Trim to `42 43 44` unless you want to extend the notebook's SEEDS list. |
| Dataset-size sweep loop | ✅ Correct | Matches the notebook's `SIZES = [10, 25, 50, 94]`, `SEEDS = [42, 43, 44]`, models `{hopf, nsde, hybrid_hopf}`. Suffix `n${n}_seed${s}` matches the expected filename. |
| Six individual `hopf-grid` + `backprop --model ...` commands for ts_young | ⚠️ Redundant | Equivalent to `train_models.py paper --dataset-type ts_young ...` which runs all six in one process. Use the paper command unless you want each model in its own Slurm job for parallelism. |
| `postprocess.py pipeline` for ts_young | ✅ Correct | |

---

## 7. Determinism notes

- `train_models.py` calls `seed_all(cfg.seed)` (default 42) — single-process runs are deterministic on identical hardware/CUDA versions. `--seed N` overrides.
- All training scripts use `matplotlib.use("Agg")`; figure rendering is headless and deterministic.
- Notebooks now self-seed where needed (`fig3_brain_panels`, `fig3_fc_fcd_phfcd`, `fig3_personalization` have an explicit seed cell; the other notebooks are either already seeded or contain no stochastic ops).
- Notebooks auto-create parent directories for `savefig` (monkey-patched at the top), so a fresh checkout where `paper_new/images*/representation/` etc. don't yet exist still works.

## 8. Common gotchas

- `paper_new` is an Overleaf submodule. After regenerating figures, `cd paper_new && git add … && git commit && git push` to sync upstream.
- Re-running `fig3_fc_fcd_phfcd.ipynb` or `fig3_clustering_complexity.ipynb` after a checkpoint change requires deleting their `cache/wholebrain_*` entries; otherwise they reuse stale forward-pass results.
- `fig1_brain_renderings.ipynb` and `fig3_brain_panels.ipynb` shell out to clone `dagush/WholeBrain` into `.cache/WholeBrain` on first run. Requires git + network access.
