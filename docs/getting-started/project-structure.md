# Project Structure

This repository is organized around the `src/` package plus a small set of example entry points.

## Top Level

```text
neuroscience_control/
├── src/
├── examples/
├── tests/
├── docs/
├── paper/
├── data/
├── checkpoints/
├── results/
├── mkdocs.yml
└── pyproject.toml
```

## `src/`

The current import namespace is `src`, not `neuroscience_control`.

### `src/dataset/`

- `data_loader.py`: `NeuroscienceDataset`, alternate constructors for LSD, nilearn, OpenNeuro, DataLad, and BIDS, plus FC/bandpass/Hilbert helpers.
- `preprocessing.py`: random window sampling, train/validation/test loader creation, and intrinsic-frequency estimation with `compute_omega_from_timeseries`.

### `src/models/`

- `base_model.py`: common model interface, FC computation, checkpoint save/load scaffolding.
- `hopf_model.py`: `CoupledHopfModel`.
- `hybrid_hopf_model.py`: `HybridHopfModel` with learnable complex coupling.
- `gnn_hopf_model.py`: `GNNHopfModel`.
- `neural_sde.py`: `NeuralSDE`.
- `factory.py`: `build_model()` used by the training scripts.
- `checkpointing.py`: `load_model_from_checkpoint()`.

### `src/metrics/`

- `fc_metrics.py`: `FCCorrelation`, `FCMSE`, `compute_static_fc`.
- `timeseries_metrics.py`: spectrum, temporal correlation, autocorrelation, `L2Timeseries`, `AmplitudeLoss`, `OmegaLoss`, reference-stat helpers.
- `dynamics_metrics.py`: `FCD`, `PhFCD`, `Metastability`, `PhaseFC`, phase/FCD helper functions.
- `metrics_store.py`: JSON-backed metrics accumulation during training.

### `src/training/`

- `config.py`: `TrainingConfig` plus `HopfConfig`, `HybridHopfConfig`, `GNNHopfConfig`, and `NeuralSDEConfig`.
- `trainer.py`: backprop trainer with early stopping, W&B logging, checkpointing, and test loops.
- `grid_search.py`: Hopf grid search with optional composite scoring.
- `losses.py`: `CompositeLoss` and registry wiring for the metric/loss modules.
- `train_utils.py`: backward-compatible `load_dataset` re-export.

### `src/utils/`

- `evaluation.py`: `evaluate_model_loader_metrics`, figure generation helpers, checkpoint helpers, `EVAL_METRIC_KEYS`.
- `runtime.py`: device resolution, seeding, W&B wrappers.
- `visualization.py`: FC comparison plots, training curves, multi-grid simulation plots, and report figures.

## `examples/`

- `train_models.py`: main CLI for `backprop`, `hopf-grid`, and `paper`.
- `postprocess.py`: post-training CLI for `update-tables`, `compare`, `compare-conditions`, and `pipeline`.
- `cli_args.py`: shared dataset and atlas arguments used by the CLIs.
- `visualization.py`: 2D animated trajectory example.
- `visualization_3d.py`: 3D surface animation example.
- `example.ipynb`, `pipeline_visualization.ipynb`: notebooks.

## `tests/`

The current tracked tests focus on model behavior and grid-search composition:

- `test_hopf_model.py`
- `test_hybrid_hopf_model.py`
- `test_gnn_hopf_model.py`
- `test_grid_search_composite.py`

## Outputs and Supporting Assets

- `data/`: local `.mat` data, LSD inputs, and downloaded dataset caches.
- `checkpoints/`: saved model checkpoints.
- `results/`: metrics JSON and evaluation outputs.
- `paper/`: figures and LaTeX tables used for manuscript artefacts.
- `docs/`: MkDocs source for the documentation site.
