# Project Structure

This repository is organized around the `src/` package plus the example entry points
and analysis scripts that reproduce the manuscript.

## Top Level

```text
neuroscience-modeling/
├── src/                # Python package (import namespace `src`)
├── examples/           # CLI entry points, paper analysis scripts, figure notebooks
├── tests/              # Unit tests (pytest)
├── docs/               # MkDocs source for the documentation site
├── logs/               # SLURM launch scripts used for the paper experiments
├── paper_submission/   # Manuscript sources (git submodule, Overleaf)
├── data/               # Local datasets (ignored by git)
├── checkpoints/        # Saved model checkpoints (ignored by git)
├── results/            # Metrics JSON and evaluation outputs (ignored by git)
├── figures/            # Default output directory for generated figures (ignored by git)
├── mkdocs.yml
└── pyproject.toml
```

## `src/`

The import namespace is `src`.

### `src/dataset/`

- `data_loader.py`: `NeuroscienceDataset`, alternate constructors for LSD, nilearn, OpenNeuro, DataLad, and BIDS, plus FC/bandpass/Hilbert helpers.
- `preprocessing.py`: random window sampling, train/validation/test loader creation, and intrinsic-frequency estimation with `compute_omega_from_timeseries`.

### `src/models/`

- `base_model.py`: common model interface, FC computation, checkpoint save/load scaffolding.
- `hopf_model.py`: `CoupledHopfModel`.
- `hybrid_hopf_model.py`: `HybridHopfModel` with learnable complex coupling.
- `hybrid_neural_model.py`: `HybridHopfNeuralModel` (Hopf drift plus a neural drift correction).
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

- `config.py`: `TrainingConfig` plus `HopfConfig`, `HybridHopfConfig`, `HybridNeuralConfig`, `GNNHopfConfig`, and `NeuralSDEConfig`.
- `trainer.py`: backprop trainer with early stopping, W&B logging, checkpointing, and test loops.
- `grid_search.py`: Hopf grid search with optional composite scoring.
- `losses.py`: `CompositeLoss` and registry wiring for the metric/loss modules.
- `evaluation.py`: loader-level evaluation helpers.

### `src/utils/`

- `evaluation.py`: `evaluate_model_loader_metrics`, figure generation helpers, checkpoint helpers, `EVAL_METRIC_KEYS`.
- `runtime.py`: device resolution, seeding, W&B wrappers.
- `visualization.py`: FC comparison plots, training curves, multi-grid simulation plots, and report figures.

## `examples/`

Entry points:

- `train_models.py`: main CLI for `backprop`, `hopf-grid`, and `paper`.
- `train_personalized.py`: single-subject (N = 1) personalization fits.
- `postprocess.py`: post-training CLI for `update-tables`, `compare`, `plot-metrics`, and `pipeline`.
- `cli_args.py`: shared dataset and atlas arguments used by the CLIs.

Paper analysis scripts (see the README section *Reproducing the paper analyses*):

- `aggregate_seeds.py`, `aggregate_grid_vs_gradient.py`, `aggregate_ablations.py`,
  `aggregate_generative.py`, `aggregate_personalization.py`: build the LaTeX tables.
- `evaluate_generative.py`, `coupling_diagnostics.py`, `compute_ari.py`,
  `benchmark_timing.py`, `build_run_manifest.py`: evaluation-time analyses.
- `plot_fig4_panels.py`, `plot_grid_search_fc_similarity.py`: figure panels.
- `visualization.py`, `visualization_3d.py`: trajectory animation examples.

Notebooks (`*.ipynb`) hold the figure-generation code for the manuscript panels; outputs
are stripped, so run them to regenerate the figures.

## `tests/`

- `test_dataset_backends.py`
- `test_evaluation_consistency.py`
- `test_fc_metrics.py`
- `test_grid_search_composite.py`
- `test_hopf_model.py`
- `test_hybrid_hopf_model.py`
- `test_gnn_hopf_model.py`

## Outputs and Supporting Assets

- `data/`: local `.mat` data and downloaded dataset caches.
- `checkpoints/`: saved model checkpoints.
- `results/`: metrics JSON and evaluation outputs (`results/runs/` holds the per-seed records).
- `figures/`: default location for figures written by the training and comparison code.
- `paper_submission/`: manuscript sources; the aggregation scripts write LaTeX tables into `paper_submission/sections/`.
