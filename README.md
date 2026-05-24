# Data-Driven Modeling of Whole-Brain Dynamics

<!-- [![CI](https://github.com/dani2442/neuroscience_control/actions/workflows/ci.yml/badge.svg)](https://github.com/dani2442/neuroscience_control/actions/workflows/ci.yml)
[![Docs](https://github.com/dani2442/neuroscience_control/actions/workflows/docs.yml/badge.svg)](https://github.com/dani2442/neuroscience_control/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/neuroscience-control.svg)](https://pypi.org/project/neuroscience-control/)
[![Python versions](https://img.shields.io/pypi/pyversions/neuroscience-control.svg)](https://pypi.org/project/neuroscience-control/) -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/dani2442/neuroscience_control/branch/main/graph/badge.svg)](https://app.codecov.io/gh/dani2442/neuroscience_control) -->

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

A PyTorch framework for **whole-brain modeling** of resting-state fMRI BOLD signals. It provides three main training workflows and an additional experimental model class:

| Model | Description |
|-------|-------------|
| **Coupled Hopf** | Physics-based coupled oscillators at the supercritical Hopf bifurcation, informed by structural connectivity |
| **Hybrid Hopf** | Hopf oscillators with a learnable complex-valued graph-coupling network replacing fixed linear diffusive coupling |
| **Neural SDE** | Data-driven neural networks parameterizing stochastic differential equations |
| **GNN Hopf** | A node-wise neural coupling variant of the Hopf model, exposed as a model class and covered by tests |

All model families operate in **complex-valued** space — state, drift, diffusion, and Brownian motion are complex tensors. The observed BOLD signal is the real part of the complex state.

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

The distribution is named `neuroscience-control`, but the importable package in this repository is currently `src`.

```bash
python -c "import src; print(src.__version__)"
```

> **Note:** The project depends on a [complex-valued fork of `torchsde`](https://github.com/dani2442/torchsde). When installing from source with `uv`, this is resolved automatically via the `[tool.uv.sources]` override in `pyproject.toml`.

---

## Quick Start

```python
import torch
from src.models import CoupledHopfModel

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
from src.models import CoupledHopfModel

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
from src.models import HybridHopfModel

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
from src.models import NeuralSDE

model = NeuralSDE(
    n_rois=68,
    hidden_dim=128,                       # MLP width (training default via NeuralSDEConfig)
    n_layers=2,
    structural_connectivity=sc_matrix,    # Optional SC-based coupling
    coupling_strength=0.1,
)
```

### GNN Hopf Model

The repository also exports `GNNHopfModel`, a node-wise neural coupling variant of the Hopf model. It is covered by tests and checkpoint loading, but it is not currently exposed through the main `examples/train_models.py backprop --model ...` CLI.

```python
from src.models import GNNHopfModel

model = GNNHopfModel(
    n_rois=68,
    structural_connectivity=sc_matrix,
    node_hidden_dim=16,
    node_n_layers=1,
)
```

### Common Forward Interface

`CoupledHopfModel`, `HybridHopfModel`, `GNNHopfModel`, and `NeuralSDE` share the same `forward()` signature:

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
from src.training import CompositeLoss

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

Training is orchestrated by [`examples/train_models.py`](examples/train_models.py). It exposes three subcommands:

```bash
# Hopf grid search over (G, a, κ)
python examples/train_models.py hopf-grid

# Backprop training for the CLI-supported models
python examples/train_models.py backprop --model nsde
python examples/train_models.py backprop --model hopf
python examples/train_models.py backprop --model hybrid_hopf

# Full paper suite: grid search + all backprop models + metrics JSON
python examples/train_models.py paper --output-json results/paper_metrics.json
```

**Common flags:** `--no-wandb`, `--device {auto,cuda,cpu}`, `--skip-figures`, `--n-epochs N`

Training hyperparameters (learning rate, loss weights, dataset backend, solver settings, atlas settings, and split ratios) are configured via the `TrainingConfig` dataclasses in `src/training/config.py`.

The `paper` subcommand saves an aggregated metrics JSON. Post-training report generation, table patching, and figure comparison live in [`examples/postprocess.py`](examples/postprocess.py).

### Dataset Backends

The training scripts support multiple data sources:

| Backend | Description |
|---------|-------------|
| `ts_young` / `mat` | Local `.mat` file with `FC_all`, `FC_mean`, `timeseries_all` |
| `lsd` | Local LSD directory (`time_series_data.mat`, `condition_names.mat`) |
| `abide` | ABIDE PCP func_preproc files from local cache or nilearn download |
| `adhd200` | ADHD-200 rs-fMRI from nilearn, or locally mirrored PCP/NITRC/S3 files |
| `nilearn` | Download + extract ROI timeseries from nilearn datasets |
| `openneuro` | Download with `openneuro-py`, then load BIDS derivatives |
| `datalad` | Install/get with DataLad, then load BIDS derivatives |
| `bids` | Load a local BIDS derivatives directory directly |

```bash
# Default local ts_young dataset
python examples/train_models.py backprop --model hopf --no-wandb

# ABIDE PCP (uses existing data/abide/ABIDE_pcp files first; otherwise nilearn downloads)
python examples/train_models.py backprop --model hopf \
  --dataset-type abide \
  --abide-data-dir data/abide \
  --abide-n-subjects 50 \
  --atlas-n-rois 100 \
  --tr 2.0 \
  --no-wandb

# ADHD-200 nilearn preprocessed sample
python examples/train_models.py backprop --model nsde \
  --dataset-type adhd200 \
  --adhd200-data-dir data/adhd200 \
  --adhd200-n-subjects 40 \
  --atlas-n-rois 100 \
  --tr 2.0 \
  --no-wandb

# ADHD-200 full local PCP/NITRC/S3 mirror
python examples/train_models.py backprop --model nsde \
  --dataset-type adhd200 \
  --adhd200-data-dir data/adhd200_full \
  --adhd200-local-pattern '**/sfnwmrda*.nii.gz' \
  --tr 2.0 \
  --no-wandb

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

> **Note:** `openneuro`, `datalad`, and `bids` backends expect preprocessed BOLD files, typically fMRIPrep derivatives. ROI extraction uses the Schaefer atlas via nilearn (`--atlas-n-rois`, `--atlas-yeo-networks`, `--atlas-resolution-mm`). If BOLD runs have different lengths, the loader trims all runs to the shortest length.
> ABIDE and ADHD-200 contain multi-site acquisitions; use `--tr` to match the subset you train on when the single default TR is not appropriate.

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

The current import namespace in this repository is `src`:

```python
from src.models import NeuralSDE
from src.metrics import FCCorrelation, FCMSE, FCD, PhFCD
from src.training import Trainer, TrainingConfig
```

---

## Project Structure

```text
neuroscience_control/
├── src/
│   ├── dataset/                   # Data loading, preprocessing, alternate dataset backends
│   ├── metrics/                   # FC, dynamics, phase, spectrum, and auxiliary losses
│   ├── models/                    # Coupled Hopf, Hybrid Hopf, GNN Hopf, Neural SDE, checkpoint loading
│   ├── training/                  # Config dataclasses, grid search, trainer, composite loss
│   └── utils/                     # Evaluation helpers, plotting, runtime/W&B utilities
├── examples/
│   ├── train_models.py            # Training entry point: backprop, hopf-grid, paper
│   ├── postprocess.py             # Table updates, model comparison, LSD condition comparisons
│   ├── cli_args.py                # Shared dataset-related CLI flags
│   ├── visualization.py           # 2D latent-trajectory animation example
│   ├── visualization_3d.py        # Surface-mapped 3D animation example
│   ├── example.ipynb              # Notebook demo
│   └── pipeline_visualization.ipynb
├── tests/                         # Model and grid-search unit tests
├── docs/                          # MkDocs Material documentation site
├── paper/                         # Figures, LaTeX tables, manuscript sources
├── data/                          # Local datasets (.mat, LSD, downloaded BIDS derivatives)
├── checkpoints/                   # Saved model checkpoints
├── results/                       # Metrics JSON, comparison outputs, metrics-store runs
├── mkdocs.yml
└── pyproject.toml
```

---

## Examples

```bash
# Train a short Hopf run on the default ts_young dataset
python examples/train_models.py backprop --model hopf --n-epochs 2 --no-wandb

# Compare trained Hopf and Neural SDE checkpoints
python examples/postprocess.py compare \
  --data-path data/ts_young/ts_young_TR0.72.mat \
  --hopf-checkpoint checkpoints/hopf_backprop_ts_young_best_<run>.pt \
  --nsde-checkpoint checkpoints/nsde_backprop_ts_young_best_<run>.pt

# Patch paper tables from the latest metrics JSON
python examples/postprocess.py update-tables --metrics results/ts_young_paper_metrics_<timestamp>.json
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
uv run pytest --cov=src --cov-report=term-missing
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

## Citation

If you use this project in academic work, please cite this repository and the accompanying paper:

```bibtex
@software{neuroscience_control,
  author = {López-Montero, Daniel and Liverani, Lorenzo and Zuazua, Enrique and Kobeleva, Xenia},
  title  = {Data-Driven Modeling of Whole-Brain Dynamics},
  url    = {https://github.com/dani2442/neuroscience_control},
  year   = {2026},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

