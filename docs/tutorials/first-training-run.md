# First Training Run

This tutorial runs one short backprop experiment and saves outputs locally.

## 1. Prepare data

By default, training config points to:

```text
data/ts_young/ts_young_TR0.72.mat
```

If your dataset path differs, edit `TrainingConfig.data_path` in `src/training/config.py`.

## 2. Run a quick experiment

```bash
python examples/train_models.py backprop \
  --model hopf \
  --n-epochs 2 \
  --no-wandb \
  --device cpu
```

## 3. Outputs

- Checkpoints: `checkpoints/`
- Figures: `paper/images/`
- Optional metrics JSON if you run the `paper` subcommand

## 4. Reproduce paper pipeline

```bash
python examples/train_models.py paper \
  --output-json results/paper_metrics.json \
  --no-wandb
```

This executes:

1. Hopf grid search
2. Backprop training for Hopf, Hybrid Hopf, and Neural SDE
3. Metrics comparison report
