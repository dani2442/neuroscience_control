# Neuroscience Control

[![CI](https://github.com/dani2442/neuroscience_control/actions/workflows/ci.yml/badge.svg)](https://github.com/dani2442/neuroscience_control/actions/workflows/ci.yml)
[![Docs](https://github.com/dani2442/neuroscience_control/actions/workflows/docs.yml/badge.svg)](https://github.com/dani2442/neuroscience_control/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/neuroscience-control.svg)](https://pypi.org/project/neuroscience-control/)
[![Python versions](https://img.shields.io/pypi/pyversions/neuroscience-control.svg)](https://pypi.org/project/neuroscience-control/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/dani2442/neuroscience_control/branch/main/graph/badge.svg)](https://app.codecov.io/gh/dani2442/neuroscience_control)

<p align="center">
  <b>Brain dynamics simulation and control using Coupled Hopf and Neural SDE models</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#models">Models</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#training">Training</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

A PyTorch framework for **whole-brain modeling** of resting-state fMRI BOLD signals. It implements three complementary approaches:

| Model | Description |
|-------|-------------|
| **Coupled Hopf** | Physics-based coupled oscillators at the supercritical Hopf bifurcation, informed by structural connectivity |
| **Hybrid Hopf** | Hopf oscillators with a learnable complex-valued graph-coupling network replacing fixed linear diffusive coupling |
| **Neural SDE** | Data-driven neural networks parameterizing stochastic differential equations |

All three models operate in **complex-valued** space — state, drift, diffusion, and Brownian motion are complex tensors. The observed BOLD signal is the real part of the complex state.

### Key Features

- **Biologically-grounded** modeling with structural connectivity integration
- **Native complex-valued SDEs** via [`torchsde`](https://github.com/dani2442/torchsde) with complex Brownian motion support
- **Comprehensive evaluation**: FC, FCD, phFCD, phase-coherence FC, metastability, and timeseries metrics
- **Flexible data loading**: local `.mat` files, LSD pharmacological data, nilearn, OpenNeuro, DataLad, and BIDS derivatives
- **Weights & Biases** integration for experiment tracking
- **GPU-accelerated** training and simulation

---

## Installation

Requires **Python ≥ 3.13** and **PyTorch ≥ 2.2**.

### From PyPI

```bash
pip install neuroscience-control
```

Or with `uv`:

```bash
uv add neuroscience-control
```

### From source

```bash
git clone https://github.com/dani2442/neuroscience_control.git
cd neuroscience_control
uv sync
```

### Optional dataset integrations

`nilearn` is included by default. For OpenNeuro/DataLad dataset download support:

```bash
pip install "neuroscience-control[datasets]"
# or
uv sync --group datasets
```

### Verify

```bash
python -c "import neuroscience_control as nc; print(nc.__version__)"
```

> **Note:** The project depends on a [complex-valued fork of `torchsde`](https://github.com/dani2442/torchsde). When installing from source with `uv`, this is resolved automatically via the `[tool.uv.sources]` override in `pyproject.toml`.

---

## Quick Start

```python
import torch
from neuroscience_control.models import CoupledHopfModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CoupledHopfModel(
    n_rois=68,
    initial_a=-0.02,   # Bifurcation parameter (near criticality)
    initial_g=0.5,      # Global coupling strength
    initial_kappa=0.1,  # Scaling factor for local dynamics
    noise_sigma=0.5,    # Noise amplitude (default)
    device=device,
)

# Simulate 200 timepoints from 10 initial conditions
initial_state = torch.randn(10, 68, dtype=torch.complex64, device=device)
with torch.no_grad():
    timeseries = model.forward(initial_state=initial_state, n_steps=200)  # (10, 68, 200) complex
    fc_matrix = model.compute_fc(timeseries)                               # (10, 68, 68)
```

---

## Models

### Coupled Hopf Model

Each brain region is a nonlinear oscillator at the supercritical Hopf bifurcation:

$$dz_i = \Bigl[\bigl(\kappa a + i\omega_i - \kappa \lvert z_i \rvert^2\bigr) z_i + G \sum_{j=1}^{N} C_{ij} (z_j - z_i)\Bigr] dt + \sigma\, dW_i$$

where $W_i = W_{1,i} + i\, W_{2,i}$ is complex Brownian motion.

| Symbol | Description |
|--------|-------------|
| $z_i \in \mathbb{C}$ | Complex state of region $i$ |
| $a$ | Bifurcation parameter ($a < 0$: damped; $a > 0$: oscillatory) |
| $\kappa$ | Scaling factor for local dynamics amplitude |
| $\omega_i$ | Intrinsic frequency of region $i$ (rad/s) |
| $G$ | Global coupling strength |
| $C_{ij}$ | Row-normalised structural connectivity from DTI |
| $\sigma$ | Noise amplitude |

The BOLD signal is $s_i(t) = \Re(z_i(t))$. The brain is hypothesised to operate near criticality ($a \approx 0$).

```python
from neuroscience_control.models import CoupledHopfModel

model = CoupledHopfModel(
    n_rois=68,
    structural_connectivity=sc_matrix,  # (68, 68) tensor
    initial_a=-0.02,
    initial_g=0.5,
    initial_kappa=0.1,
    omega=intrinsic_frequencies,        # (68,) tensor in rad/s
    noise_sigma=0.5,
    learnable_a=True,
    learnable_g=True,
    learnable_kappa=False,
)
```

### Hybrid Hopf Model

Combines Hopf local oscillator dynamics with a learnable graph-coupling network $\psi_\theta$:

$$dz_i = \Bigl[\bigl(\kappa a + i\omega_i - \kappa \lvert z_i \rvert^2\bigr) z_i + G \sum_j \psi_\theta(z_j - z_i,\; C_{ij})\Bigr] dt + \sigma\, dW_i$$

The network $\psi_\theta : \mathbb{C}^2 \to \mathbb{C}$ is a complex-valued harmonic network with phase-preserving activations ($f(z) = \frac{z}{|z|} \cdot \tanh(|z|)$), replacing the fixed linear coupling of the classical model. Magnitude is soft-clamped at 10.0 to prevent runaway growth.

```python
from neuroscience_control.models import HybridHopfModel

model = HybridHopfModel(
    n_rois=68,
    structural_connectivity=sc_matrix,
    initial_a=-0.02,
    initial_g=0.5,
    initial_kappa=0.1,
    omega=intrinsic_frequencies,
    coupling_hidden_dim=16,   # Complex-valued coupling network width
    coupling_n_layers=1,      # Number of coupling network layers
)
```

### Neural SDE Model

Uses neural networks to parameterise drift and diffusion of a complex-valued SDE:

$$dz_t = f_\theta(z_t)\, dt + g_\phi(z_t)\, dW_t$$

where $f_\theta$ and $g_\phi$ are real-valued MLPs operating on `torch.view_as_real` / `torch.view_as_complex` conversions. The drift network uses `tanh` activations; the diffusion network uses `Softplus`.

```python
from neuroscience_control.models import NeuralSDE

model = NeuralSDE(
    n_rois=68,
    hidden_dim=128,                       # MLP width (training default via NeuralSDEConfig)
    n_layers=2,
    structural_connectivity=sc_matrix,    # Optional SC-based coupling
    coupling_strength=0.1,
)
```

### Common Forward Interface

All three models share the same `forward()` signature:

```python
timeseries = model.forward(
    initial_state,           # (batch, n_rois) complex tensor
    n_steps=100,             # Number of output timepoints
    dt=0.72,                 # TR (time between observed samples)
    dt_min=0.05,             # Internal Euler–Maruyama sub-step (seconds)
    method="euler",          # SDE solver method
    use_adjoint=False,       # Use torchsde adjoint solver for memory-efficient backprop
    control=None,            # Optional (batch, n_control_dims) control input
)
# Returns: (batch, n_rois, n_steps) complex tensor
```

---

## Metrics

Each metric and differentiable loss is a `nn.Module` subclass living in `src/metrics/`.
All classes accept `(ts_pred, ts_target)` complex analytic signals `(batch, n_rois, T)`.

- `forward(ts_pred, ts_target)` → differentiable scalar loss tensor (for training)
- `evaluate(ts_pred, ts_target)` → `dict[str, float]` (for logging)

### Metric / Loss Classes

| Class | Module | `forward()` loss | `evaluate()` keys |
|-------|--------|-----------------|-------------------|
| `FCCorrelation` | `src/metrics/fc_metrics.py` | `1 − FC correlation` | `fc_correlation` |
| `FCMSE` | `src/metrics/fc_metrics.py` | FC MSE | `fc_mse` |
| `PowerSpectrumDistance` | `src/metrics/timeseries_metrics.py` | power spectrum MSE | `power_spectrum_distance` |
| `TemporalCorrelation` | `src/metrics/timeseries_metrics.py` | `1 − temporal correlation` | `temporal_correlation` |
| `AutocorrelationDistance` | `src/metrics/timeseries_metrics.py` | autocorrelation MSE | `autocorr_distance` |
| `FCD(tr, fcd_win_sec, fcd_step_sec)` | `src/metrics/dynamics_metrics.py` | FCD MSE (surrogate) | `fcd_mse`, `fcd_ks` |
| `PhFCD` | `src/metrics/dynamics_metrics.py` | phFCD MSE (surrogate) | `phfcd_mse`, `phfcd_ks` |
| `Metastability` | `src/metrics/dynamics_metrics.py` | metastability L1 diff | `metastability_diff` |
| `PhaseFC` | `src/metrics/dynamics_metrics.py` | `1 − phase-coherence FC corr` | `phase_fc_correlation` |
| `L2Timeseries` | `src/training/losses.py` | L² timeseries error | — |
| `AmplitudeLoss(ref_amplitude, tr)` | `src/training/losses.py` | amplitude L² | — |
| `OmegaLoss(ref_omega, tr)` | `src/training/losses.py` | frequency L² | — |

> **Note:** `FCD` and `PhFCD` use differentiable MSE surrogates for training; the non-differentiable KS distances are only computed in `evaluate()`.

### Evaluation Metrics (evaluate() keys)

| Key | Range | Direction | Description |
|-----|-------|-----------|-------------|
| `fc_correlation` | $[-1, 1]$ | Higher ↑ | Pearson correlation of upper-triangular FC |
| `fc_mse` | $[0, \infty)$ | Lower ↓ | MSE of upper-triangular FC |
| `fcd_ks` | $[0, 1]$ | Lower ↓ | KS distance between FCD distributions (sliding-window) |
| `fcd_mse` | $[0, \infty)$ | Lower ↓ | MSE between FCD matrices |
| `phfcd_ks` | $[0, 1]$ | Lower ↓ | KS distance between phase-FCD distributions |
| `phfcd_mse` | $[0, \infty)$ | Lower ↓ | MSE between phFCD matrices |
| `phase_fc_correlation` | $[-1, 1]$ | Higher ↑ | Pearson correlation of phase-coherence FC |
| `metastability_diff` | $[0, \infty)$ | Lower ↓ | Absolute difference in metastability (Kuramoto) |
| `temporal_correlation` | $[-1, 1]$ | Higher ↑ | Mean per-ROI Pearson correlation over time |
| `power_spectrum_distance` | $[0, \infty)$ | Lower ↓ | MSE between normalised power spectra |
| `autocorr_distance` | $[0, \infty)$ | Lower ↓ | MSE between autocorrelation functions |

### Mathematical Details

**Functional Connectivity (FC)** — Static Pearson correlation between regional time series from the real part $s_n(t) = \Re(z_n(t))$:

$$\text{FC}\_{nm} = \frac{\text{Cov}(s_n,\, s_m)}{\text{SD}(s_n) \cdot \text{SD}(s_m)}$$

**Functional Connectivity Dynamics (FCD)** — Sliding-window FC vectors (windowed with configurable `fcd_win_sec` and `fcd_step_sec`), z-scored, then correlated pairwise across windows. Compared via two-sample Kolmogorov–Smirnov distance.

**Phase FCD (phFCD)** — The paper's main model-fitting metric. Uses instantaneous phase coherence:

$$P_{nm}(t) = \cos\bigl(\phi_n(t) - \phi_m(t)\bigr)$$

instead of windowed Pearson FC. The phFCD matrix is built from cosine similarity of the upper-triangular phase-coherence vectors across time.

**Metastability** — Temporal variability of global synchronisation via the Kuramoto order parameter:

$$R(t) = \left\lvert \frac{1}{N} \sum_{n=1}^{N} e^{i\phi_n(t)} \right\rvert, \qquad \text{Metastability} = \text{std}_t\bigl(R(t)\bigr)$$

### Composite Loss

Losses are configured via `loss_weights` in `TrainingConfig` — a plain dict mapping term names to scalar weights. Zero-weight terms are never computed. The `CompositeLoss` nn.Module assembles the active terms automatically.

```python
from src.training.losses import CompositeLoss

# Explicit weight dict — no preset lookup needed
loss_fn = CompositeLoss(
    weights={"fc_correlation": 1.0, "phfcd": 1.0, "metastability": 1.0},
    tr=0.72, fcd_win_sec=30.0, fcd_step_sec=2.0,
)
total, components = loss_fn(ts_pred, ts_target)
# components: {"loss_fc_correlation": ..., "loss_phfcd": ..., "loss_metastability": ...}
```

Available term names: `fc_correlation`, `fc_mse`, `l2`, `amplitude`, `omega`, `power_spectrum`, `temporal_correlation`, `autocorrelation`, `fcd`, `phfcd`, `phase_fc_correlation`, `metastability`.

Configure via `TrainingConfig`:

```python
cfg = HopfConfig(loss_weights={"fc_correlation": 1.0, "phfcd": 1.0, "metastability": 1.0})
```

---

## Training

### Unified Entry Point

All training goes through `examples/train_models.py` with three subcommands:

```bash
# Hopf grid search over (G, a, κ)
python examples/train_models.py hopf-grid

# Backprop training for any model
python examples/train_models.py backprop --model nsde
python examples/train_models.py backprop --model hopf
python examples/train_models.py backprop --model hybrid_hopf

# Full paper reproduction: grid search + all backprop models + comparison report
python examples/train_models.py paper --output-json results/paper_metrics.json
```

**Common flags:** `--no-wandb`, `--device {auto,cuda,cpu}`, `--skip-figures`, `--n-epochs N`

Training hyperparameters (learning rate, loss function, window size, etc.) are configured via the `TrainingConfig` dataclass in `src/training/config.py`.

### Dataset Backends

The training scripts support multiple data sources:

| Backend | Description |
|---------|-------------|
| `ts_young` / `mat` | Local `.mat` file with `FC_all`, `FC_mean`, `timeseries_all` |
| `lsd` | Local LSD directory (`time_series_data.mat`, `condition_names.mat`) |
| `nilearn` | Download + extract ROI timeseries from nilearn datasets |
| `openneuro` | Download with `openneuro-py`, then load BIDS derivatives |
| `datalad` | Install/get with DataLad, then load BIDS derivatives |
| `bids` | Load a local BIDS derivatives directory directly |

```bash
# nilearn fetcher dataset
python examples/train_models.py backprop --model hopf \
  --dataset-type nilearn \
  --nilearn-dataset development_fmri \
  --nilearn-n-subjects 20 \
  --atlas-n-rois 100 \
  --no-wandb

# OpenNeuro via openneuro-py
python examples/train_models.py backprop --model nsde \
  --dataset-type openneuro \
  --openneuro-dataset ds000030 \
  --openneuro-target-dir data/openneuro \
  --bids-derivatives-dir derivatives/fmriprep \
  --bids-space MNI152NLin2009cAsym \
  --bids-desc preproc \
  --no-wandb

# DataLad + BIDS derivatives
python examples/train_models.py backprop --model hybrid_hopf \
  --dataset-type datalad \
  --datalad-source https://github.com/OpenNeuroDatasets/ds000030.git \
  --datalad-dataset-dir data/datalad/ds000030 \
  --datalad-get-paths 'derivatives/fmriprep/sub-*/**/*_desc-preproc_bold.nii.gz' \
  --bids-space MNI152NLin2009cAsym \
  --bids-desc preproc \
  --no-wandb
```

> **Note:** `openneuro`, `datalad`, and `bids` backends expect preprocessed BOLD files (typically fMRIPrep outputs). ROI extraction uses the Schaefer atlas via nilearn (`--atlas-n-rois`, `--atlas-yeo-networks`, `--atlas-resolution-mm`). If BOLD runs have different lengths, the loader trims all runs to the shortest length.

### Fine-Tuning a Neural SDE

```bash
python examples/train_nsde_finetune.py \
    --checkpoint checkpoints/best_nsde_backprop.pt \
    --fine-tune-epochs 20 \
    --fine-tune-lr 1e-4
```

---

## Data Format

`NeuroscienceDataset` loads a `.mat` file (via `scipy.io.loadmat`) with these keys:

```python
{
    'timeseries_all': np.ndarray,  # (n_rois, n_timepoints, n_subjects)
    'FC_all':         np.ndarray,  # (n_rois, n_rois, n_subjects)
    'FC_mean':        np.ndarray,  # (n_rois, n_rois)
}
```

### Preprocessing Pipeline

1. **Z-score** each ROI time series
2. **FFT bandpass** denoising (optional; default 0.008–0.08 Hz)
3. **Hilbert transform** to complex analytic signal $z(t)$

All downstream metrics and losses operate on this complex signal — FCD and timeseries metrics use `.real`, phase-based metrics extract phases via `torch.angle(z)`.

The SDE solver uses an internal Euler–Maruyama sub-step of `dt_min = 0.05` s (configurable), with the output sampled at the TR interval (default 0.72 s).

---

## Import Path

Use the public namespace in new code:

```python
from neuroscience_control.models import CoupledHopfModel, NeuralSDE
from neuroscience_control.metrics import FCCorrelation, FCMSE, FCD, PhFCD
from neuroscience_control.training import Trainer, TrainingConfig
```

Legacy imports via `src.*` are still supported:

```python
from src.models import NeuralSDE
```

---

## Project Structure

```text
neuroscience_control/
├── src/
│   ├── dataset/                   # Data loading and preprocessing
│   │   ├── data_loader.py         # NeuroscienceDataset, Hilbert transform
│   │   └── preprocessing.py       # Windowing, omega estimation
│   ├── models/                    # Brain dynamics models
│   │   ├── base_model.py          # Abstract base class (forward, compute_fc, save/load)
│   │   ├── hopf_model.py          # Coupled Hopf oscillator SDE
│   │   ├── hybrid_hopf_model.py   # Hybrid mechanistic–neural Hopf SDE
│   │   ├── neural_sde.py          # Neural SDE (drift + diffusion MLPs)
│   │   ├── factory.py             # build_model() dispatcher
│   │   └── checkpointing.py       # Checkpoint loading with version migration
│   ├── metrics/                   # nn.Module metric/loss classes
│   │   ├── fc_metrics.py          # FCCorrelation, FCMSE
│   │   ├── dynamics_metrics.py    # FCD, PhFCD, Metastability, PhaseFC
│   │   ├── timeseries_metrics.py  # PowerSpectrumDistance, TemporalCorrelation, AutocorrelationDistance
│   │   └── metrics_store.py       # MetricsStore JSON-backed accumulator
│   ├── training/                  # Training utilities
│   │   ├── trainer.py             # Backprop trainer (train/val/test loops)
│   │   ├── grid_search.py         # Hopf grid search over (G, a, κ)
│   │   ├── fine_tuning.py         # Fine-tuning pipeline
│   │   ├── losses.py              # CompositeLoss, L2Timeseries, AmplitudeLoss, OmegaLoss
│   │   ├── backprop.py            # run_backprop_training() orchestration
│   │   └── config.py              # TrainingConfig / HopfConfig / NeuralSDEConfig
│   └── utils/                     # Visualization, evaluation, runtime
│       ├── visualization.py       # FC comparison plots, training curves, multigrid
│       ├── evaluation.py          # EVAL_METRIC_KEYS, checkpoint evaluation helpers
│       └── runtime.py             # Device resolution, seeding, W&B wrappers
├── examples/                      # Entry-point scripts
│   ├── train_models.py            # Unified training (grid / backprop / paper)
│   ├── train_nsde_finetune.py     # Neural SDE fine-tuning
│   ├── compare_models.py          # Side-by-side model comparison
│   ├── test.py                    # Checkpoint evaluation
│   └── visualization.py           # Standalone plotting utilities
├── tests/                         # Unit tests (pytest)
├── paper/                         # LaTeX source for accompanying paper
├── docs/                          # MkDocs Material documentation site
├── data/                          # Data directory (.mat files)
├── checkpoints/                   # Saved model weights
├── mkdocs.yml
└── pyproject.toml
```

---

## Examples

```bash
# Evaluate a saved checkpoint
python examples/test.py --checkpoint checkpoints/best_nsde_backprop.pt

# Compare two trained models
python examples/compare_models.py \
    --hopf-checkpoint checkpoints/best_hopf_backprop.pt \
    --nsde-checkpoint checkpoints/best_nsde_backprop.pt \
    --n-simulations 10
```

---

## Documentation Website

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).  
Live site: **https://dani2442.github.io/neuroscience_control/**

```bash
# Install docs dependencies
uv sync --group docs

# Local preview
uv run mkdocs serve

# Build static site
uv run mkdocs build --strict
```

---

## Development and Test Coverage

```bash
uv sync --group dev
uv run pytest --cov=src --cov=neuroscience_control --cov-report=term-missing
```

---

## Release and Publishing

Automated workflows:

- **CI** (`.github/workflows/ci.yml`): tests + coverage
- **Docs** (`.github/workflows/docs.yml`): docs build + GitHub Pages deploy
- **Publish** (`.github/workflows/publish.yml`): manual dispatch → TestPyPI; GitHub release → PyPI

```bash
uv run python -m build
uv run twine check dist/*
```

---

## Related Work

- [Deco et al., 2017](https://doi.org/10.1038/s41598-017-03073-5) — Whole-brain coupled Hopf model
- [torchsde](https://github.com/dani2442/torchsde) — SDE solvers for PyTorch (complex-valued fork)
- [The Virtual Brain](https://www.thevirtualbrain.org/) — Open-source brain simulation platform

---

## Citation

If you use this project in academic work, please cite this repository and the accompanying paper:

```bibtex
@software{neuroscience_control,
  author = {López Montero, Daniel and Kobeleva, Xenia and Liverani, Lorenzo},
  title  = {Neuroscience Control: Brain Dynamics Simulation and Control
            with Coupled Hopf and Neural SDE Models},
  url    = {https://github.com/dani2442/neuroscience_control},
  year   = {2026},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Paper Commands (`.venv`)

Run from the repository root. See each script's docstring for full usage details.

```bash
# 1) Paper suite — train all models on a dataset and generate metrics JSON
#    LSD dataset:
.venv/bin/python examples/train_models.py paper \
  --dataset-type lsd --lsd-data-dir data/lsd

#    ts_young dataset:
.venv/bin/python examples/train_models.py paper \
  --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat

# 2) Update paper LaTeX tables from metrics JSON (see examples/update_paper_tables.py)
.venv/bin/python examples/update_paper_tables.py \
  --metrics results/lsd_paper_metrics_<timestamp>.json

.venv/bin/python examples/update_paper_tables.py \
  --metrics results/ts_young_paper_metrics_<timestamp>.json

# 3) Compare LSD control conditions (see examples/compare_control_conditions.py)
.venv/bin/python examples/compare_control_conditions.py \
  --checkpoints checkpoints/hopf_grid_lsd_best_*.pt \
                checkpoints/hopf_backprop_lsd_best_*.pt \
                checkpoints/nsde_backprop_lsd_best_*.pt \
                checkpoints/hybrid_hopf_backprop_lsd_best_*.pt \
                checkpoints/gnn_hopf_backprop_lsd_best_*.pt \
  --lsd-data-dir data/lsd
```
