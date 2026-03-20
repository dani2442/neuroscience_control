# First Training Run

This tutorial covers the current training entry point, the dataset knobs it uses, and the files it writes.

## 1. Pick a dataset backend

The default config points at:

```text
data/ts_young/ts_young_TR0.72.mat
```

You can keep that default, or override the loader from the CLI with `--dataset-type` and the matching dataset arguments.

Common choices:

- `--dataset-type ts_young --data-path ...`
- `--dataset-type lsd --lsd-data-dir ...`
- `--dataset-type nilearn --nilearn-dataset development_fmri`
- `--dataset-type bids --bids-root ...`

## 2. Run a short backprop experiment

```bash
python examples/train_models.py backprop \
  --model hopf \
  --n-epochs 2 \
  --no-wandb \
  --device cpu
```

Supported `--model` values in the CLI are:

- `hopf`
- `hybrid_hopf`
- `nsde`

Under the hood this command:

1. Loads data with `src.dataset.load_dataset()`.
2. Builds random-window loaders with `src.dataset.create_data_loaders()`.
3. Constructs the selected model with `src.models.build_model()`.
4. Trains it with `src.training.Trainer`.
5. Evaluates it with `src.utils.evaluate_model_loader_metrics()`.

## 3. Run Hopf grid search

```bash
python examples/train_models.py hopf-grid \
  --no-wandb \
  --device cpu
```

This searches over `g_values`, `a_values`, and `kappa_values` from `src/training/config.py`.

## 4. Run the paper suite

```bash
python examples/train_models.py paper \
  --output-json results/paper_metrics.json \
  --no-wandb
```

This runs:

1. Hopf grid search.
2. Backprop training for Hopf, Hybrid Hopf, and Neural SDE.
3. Aggregated metric export to a JSON file in `results/`.

## 5. Outputs

- Checkpoints are written to `checkpoints/`.
- Figures are written under `paper/images/<dataset_type>/`.
- Per-run metrics stores are written under `results/metrics/<experiment_name>/`.
- The `paper` command writes an aggregated JSON report to `results/`.

## 6. Post-training steps

Use `examples/postprocess.py` after training:

```bash
python examples/postprocess.py compare \
  --data-path data/ts_young/ts_young_TR0.72.mat \
  --hopf-checkpoint checkpoints/hopf_backprop_ts_young_best_<run>.pt \
  --nsde-checkpoint checkpoints/nsde_backprop_ts_young_best_<run>.pt
```

Other subcommands:

- `update-tables`: patch LaTeX tables from a metrics JSON.
- `compare-conditions`: LSD control-condition comparison.
- `pipeline`: run all post-training steps in sequence.
