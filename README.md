# Neuroscience Modeling

[![CI](https://github.com/DCN-FAU-AvH/neuroscience-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/DCN-FAU-AvH/neuroscience-modeling/actions/workflows/ci.yml)
[![Docs](https://github.com/DCN-FAU-AvH/neuroscience-modeling/actions/workflows/docs.yml/badge.svg)](https://github.com/DCN-FAU-AvH/neuroscience-modeling/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <b>Hybrid generative whole-brain models: Coupled Hopf, Neural SDE, and hybrid architectures fitted by gradient descent</b>
</p>

Code accompanying the manuscript *Hybrid generative models of whole-brain dynamics balance data fidelity and interpretability* (D. López-Montero, L. Liverani, E. Zuazua, X. Kobeleva). The manuscript sources live in [`paper_submission/`](paper_submission/), and every table and figure panel can be regenerated with the scripts listed under [Reproducing the paper analyses](#reproducing-the-paper-analyses).

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

A PyTorch framework for **whole-brain modeling** of resting-state fMRI BOLD signals. It implements the five whole-brain model (WBM) architectures compared in the manuscript: three are discussed in the main text and two further hybrid variants are reported in the Supplementary Material.

| Model | CLI name | Description |
|-------|----------|-------------|
| **Coupled Hopf** | `hopf` | Physics-based coupled oscillators at the supercritical Hopf bifurcation, coupled through a connectome prior; fitted by grid search or by gradient descent |
| **Neural SDE** | `nsde` | Data-driven neural networks parameterizing the drift and diffusion of a stochastic differential equation |
| **Hybrid Hopf** | `hybrid_hopf` | Hopf oscillators with a learnable complex-valued coupling network replacing the fixed linear diffusive coupling |
| **GNN Hopf** (supplement) | `gnn_hopf` | Hopf oscillators with node-wise neural coupling |
| **Hopf + Neural** (supplement) | `hybrid_neural` | Hopf drift plus a neural drift correction, with a learnable noise amplitude |

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

### From source

```bash
git clone https://github.com/DCN-FAU-AvH/neuroscience-modeling.git
cd neuroscience-modeling
uv sync               # creates .venv with the pinned dependencies
uv sync --group dev   # also installs pytest and ruff
```

### Optional dataset integrations

`nilearn` is included by default. For OpenNeuro/DataLad dataset download support:

```bash
uv sync --group datasets
```

### Verify

The importable package is `src`.

```bash
uv run python -c "import src; print(src.__version__)"
uv run pytest
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

### Supplementary hybrid variants: GNN Hopf and Hopf + Neural

Two further hybrid architectures are reported in the Supplementary Material. Both are trained with the same CLI (`--model gnn_hopf`, `--model hybrid_neural`) and the same losses.

- `GNNHopfModel` replaces the diffusive coupling with a node-wise neural coupling.
- `HybridHopfNeuralModel` keeps the Hopf drift and adds a neural drift correction (near-zero initialization, same network as the Neural SDE) with a learnable noise amplitude.

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

`CoupledHopfModel`, `HybridHopfModel`, `HybridHopfNeuralModel`, `GNNHopfModel`, and `NeuralSDE` share the same `forward()` signature:

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

# Backprop training (one architecture per call)
python examples/train_models.py backprop --model hopf
python examples/train_models.py backprop --model nsde
python examples/train_models.py backprop --model hybrid_hopf
python examples/train_models.py backprop --model gnn_hopf
python examples/train_models.py backprop --model hybrid_neural

# Full paper suite: grid search + all backprop models + metrics JSON
python examples/train_models.py paper --output-json results/paper_metrics.json
```

**Common flags:** `--no-wandb`, `--device {auto,cuda,cpu}`, `--skip-figures`, `--n-epochs N`

**Ablation flags** (`backprop`): `--sc-mode`, `--no-learnable-coupling`, `--disable-local`, `--zero-loss`, `--hidden-dim/--n-layers` — see [Ablations](#4-ablations).

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
neuroscience-modeling/
├── src/
│   ├── dataset/                   # Data loading, preprocessing, alternate dataset backends
│   ├── metrics/                   # FC, dynamics, phase, spectrum, and auxiliary losses
│   ├── models/                    # Coupled Hopf, Hybrid Hopf, Hopf+Neural, GNN Hopf, Neural SDE
│   ├── training/                  # Config dataclasses, grid search, trainer, composite loss
│   └── utils/                     # Evaluation helpers, plotting, runtime/W&B utilities
├── examples/
│   ├── train_models.py            # Training entry point: backprop, hopf-grid, paper
│   ├── train_personalized.py      # Single-subject (N = 1) personalization fits
│   ├── postprocess.py             # Table updates, model comparison, metric curves
│   ├── cli_args.py                # Shared dataset-related CLI flags
│   ├── aggregate_seeds.py         # Seed-level model tables + paired Wilcoxon/Holm + composite
│   ├── aggregate_grid_vs_gradient.py  # Grid search vs gradient descent table
│   ├── aggregate_ablations.py     # Ablation table from per-seed ablation runs
│   ├── evaluate_generative.py     # Repeated-realization, coverage and personalization evaluation
│   ├── aggregate_generative.py    # Variance decomposition, coverage, personalization tables
│   ├── aggregate_personalization.py  # Within- vs between-subject contrasts of the N = 1 fits
│   ├── coupling_diagnostics.py    # Learned-coupling vs connectome-initialization diagnostics
│   ├── compute_ari.py             # FC adjusted Rand index per (model, seed)
│   ├── benchmark_timing.py        # Training / grid-search / inference runtime benchmark
│   ├── build_run_manifest.py      # One row per run behind every table and figure
│   ├── plot_fig4_panels.py        # Figure 4 panels (d) and (e)
│   ├── plot_grid_search_fc_similarity.py
│   ├── visualization.py, visualization_3d.py   # Trajectory animation examples
│   └── *.ipynb                    # Figure-panel notebooks (outputs stripped)
├── tests/                         # Unit tests (pytest)
├── docs/                          # MkDocs Material documentation site
├── logs/                          # SLURM launch scripts used for the paper experiments
├── paper_submission/              # Manuscript sources (git submodule synced with Overleaf)
├── data/                          # Local datasets (ignored by git)
├── checkpoints/                   # Saved model checkpoints (ignored by git)
├── results/                       # Metrics JSON, per-seed run records (ignored by git)
├── figures/                       # Default output directory for generated figures (ignored by git)
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

## Reproducing the paper analyses

The manuscript in `paper_submission/` is generated from the artifacts below. All
statistics use an explicit replication unit — see *Statistical reporting* in the
paper's Methods, and the note at the end of this section.

The aggregation scripts read every matching file in `results/runs/`,
`results/personalization/` and `results/generative/`, and the evaluation scripts
glob `checkpoints/`. Start a rerun from empty directories (move earlier
outputs into an archive folder), or old and new runs are pooled.

Every GPU step has a SLURM launcher under `logs/` (written for the FAU
`tinygpu` cluster; adjust `-M`/partition for yours). Training jobs are
independent and can be submitted together; the evaluation launchers read the
checkpoints of `launch_seeds.sh`, so submit them once those jobs have finished:

```bash
./logs/revision/launch_seeds.sh            # Table 1, S1, S2   (60 jobs)
./logs/ablation/launch_full.sh             # ablations         (24 jobs)
./logs/fig4d/launch_fig4d.sh               # Figure 4d         (60 jobs)
./logs/revision/launch_personalization.sh  # personalization   (30 jobs)
./logs/revision/launch_timing.sh           # timing table      (1 job)
# after launch_seeds.sh has finished:
./logs/revision/launch_generative.sh hopf nsde hybrid_hopf gnn_hopf hybrid_neural
./logs/revision/launch_ari.sh
./logs/revision/link_figure_inputs.sh      # figure inputs, no GPU (step 7)
```

### 1. Seed sweep (the unit of analysis for every architecture comparison)

Each architecture is refitted from scratch under 10 random seeds (42–51) at the
full sample size. Model-comparison intervals are standard deviations **across
these independent fits**, so they include fitting variability; an earlier version
reported the spread across held-out test batches of a *single* fit, which does
not.

```bash
# Submits every (model, seed) job, including the hopf-grid runs of step 5;
# writes results/runs/ts_young_<model>_n94_seed<seed>.json and ts_young_hopf_grid_seed<seed>.json
./logs/revision/launch_seeds.sh

# Main-text table (3 architectures) and full supplementary table (all 5)
python examples/aggregate_seeds.py --ds ts_young --main \
  --out paper_submission/sections/ts_young_model_table.tex \
  --summary-out results/seed_summary_main.json
python examples/aggregate_seeds.py --ds ts_young \
  --out paper_submission/sections/ts_young_full_model_table.tex
```

`aggregate_seeds.py` reports mean ± SD across seeds, a two-sided paired Wilcoxon
signed-rank test against the Hybrid Hopf reference (paired on seed, Holm-corrected
across architectures within each metric), and a **pre-specified composite error**
— the mean of {FC corr, phFC corr, FCD KS, phFCD KS} with upward metrics flipped
to `1 − v`. The composite is fixed in advance so that "best balanced performance"
is a computed quantity rather than a reading across columns.

### 2. Generative evaluation (repeated stochastic realizations)

These are stochastic generative models, so one rollout is a single draw. This
step redraws 20 independent trajectories per subject from each fitted checkpoint
and records per-realization values, enabling a variance decomposition into
*seed*, *subject*, and *simulation* components, plus a posterior-predictive
coverage check. Evaluation-time only — no retraining.

```bash
# Per-checkpoint evaluation → results/generative/<stem>.json  (GPU)
./logs/revision/launch_generative.sh hopf nsde hybrid_hopf gnn_hopf hybrid_neural

# Variance, coverage and personalization tabulars
python examples/aggregate_generative.py --ds ts_young \
  --out-dir paper_submission/sections
```

The same script computes **personalization** with the subject as the replication
unit: for each subject it simulates from that subject's own initial condition and
correlates the simulated FC against that subject's held-out second half
(`within`) versus every other subject (`between`), giving a paired per-subject
`delta` that supports a signed-rank test and a bootstrap CI, together with top-1
subject identification accuracy against a `1/n` chance level.

> **Read `delta` against the initial-condition reference, not against zero.**
> Every simulation is seeded with the subject's own initial state, which is a
> subject-specific input, so `delta > 0` alone does not show that a model is
> personalized. The Coupled Hopf makes this concrete: it shares all parameters
> across the cohort and uses a group-average connectome, yet reaches
> `delta = +0.184` and 43% top-1 identification (chance 1.1%) — more than either
> learned architecture. `aggregate_generative.py` therefore reports
> `delta - delta_IC` against the Coupled Hopf, and both learned models score
> *negative* increments. The conclusion the paper draws is subject-specific
> reconstruction, not personalization.

### 3. Coupling diagnostics (effective vs structural connectivity)

The coupling `C = LRᵀ` is *initialized* from the group-averaged connectome and
then learned without symmetry, positivity, or sign constraints. This script
quantifies how far it departs, whether it stays connectome-like, and whether it
is reproducible across seeds — the evidence behind describing it as an
*effective learned coupling* rather than a learned connectome.

```bash
python examples/coupling_diagnostics.py \
  --checkpoints "checkpoints/ts_young_hybrid_hopf_n94_seed*.pt" \
  --out results/coupling_diagnostics.json \
  --tex-out paper_submission/sections/ts_young_coupling_table.tex
```

### 4. Ablations

The controls-and-ablations table in the supplement isolates **why** the Hybrid
Hopf model performs well by changing one ingredient at a time. All runs use the
`backprop` subcommand with the flags below; each run also takes `--seed S` and
`--run-suffix TAG` and writes `results/runs/<dataset>_<model>_<TAG>.json`.

| Flag | Effect |
| --- | --- |
| `--hidden-dim N` `--n-layers L` | Override the Neural SDE drift/diffusion size (used for the **parameter-matched** baseline). |
| `--sc-mode {real,shuffled,random}` | Connectome used to seed the coupling: unchanged, edge-weight **shuffled** (topology destroyed), or a **random** symmetric graph. Seeded by `--seed`. |
| `--no-learnable-coupling` | Freeze the coupling `C` at its connectome initialization (**fixed** vs. learnable connectivity; Hybrid Hopf only). |
| `--disable-local` | Zero the mechanistic **Hopf local term**, keeping the learned coupling architecture (Hybrid Hopf only). |
| `--zero-loss KEY [KEY ...]` | Set the listed loss weights to 0. Keys: `fc_correlation fc_mse l2 amplitude omega power_spectrum temporal_correlation autocorrelation fcd phfcd phase_fc_correlation metastability fdm`. |

The eight configurations (one seed shown; the table uses seeds 42, 43, 44):

```bash
DATA=data/ts_young/ts_young_TR0.72.mat
COMMON="--dataset-type ts_young --data-path $DATA --no-wandb --skip-figures --seed 42"
PY="python examples/train_models.py backprop"

$PY --model hybrid_hopf $COMMON --run-suffix baseline_seed42                            # reference
$PY --model nsde        $COMMON --hidden-dim 4 --n-layers 1 --run-suffix pmatch_seed42  # parameter-matched Neural SDE
$PY --model hybrid_hopf $COMMON --sc-mode shuffled --run-suffix scshuffle_seed42
$PY --model hybrid_hopf $COMMON --sc-mode random   --run-suffix scrandom_seed42
$PY --model hybrid_hopf $COMMON --no-learnable-coupling --run-suffix fixedcoupling_seed42
$PY --model hybrid_hopf $COMMON --disable-local --run-suffix nolocal_seed42
$PY --model hybrid_hopf $COMMON --zero-loss fdm --run-suffix nofdm_seed42
$PY --model hybrid_hopf $COMMON \
  --zero-loss fcd phfcd phase_fc_correlation metastability --run-suffix nodyn_seed42
```

[`logs/ablation/launch_full.sh`](logs/ablation/launch_full.sh) submits the full
8 × 3 batch as SLURM jobs (adjust `-M`/partition for your cluster). Once the
`results/runs/*.json` files exist, build the table:

```bash
bash logs/ablation/launch_full.sh
python examples/aggregate_ablations.py --ds ts_young \
  --out paper_submission/sections/ts_young_ablation_table.tex
```

### 5. Grid search vs. gradient descent (Supplementary Table S2)

Compares the Coupled Hopf fitted by exhaustive grid search (`train_models.py hopf-grid`,
one run per seed written to `results/runs/ts_young_hopf_grid_seed<s>.json` by
`logs/revision/launch_seeds.sh`) against the
gradient-fitted Coupled Hopf runs of Table 1, paired on the seed:

```bash
python examples/aggregate_grid_vs_gradient.py --seeds 42-51 \
  --out paper_submission/sections/grid_vs_gradient_table.tex
```

### 6. Personalization (N = 1 fits)

`train_personalized.py` fits one model to a single subject and evaluates it on the
held-out second half of that subject's scan (`test_intra`) and on every other subject
(`test_inter`). `aggregate_personalization.py` turns the per-subject records into the
within- vs. between-subject contrasts reported in the supplement.

The group connectome prior of every fit is that of the split-seed-42 cohort fit,
so the ten targets are subjects outside its training pool (the script refuses any
other target and lists the eligible ones).

```bash
./logs/revision/launch_personalization.sh   # 3 architectures x 10 targets, seed 42
python examples/aggregate_personalization.py --ds ts_young
```

### 7. Figure panels, ARI, timing, and run bookkeeping

```bash
# Figure 4d runs: 3 architectures x {10, 65} training subjects x seeds 42-51,
# all scored on the same 15 test subjects (--split-seed 42)
./logs/fig4d/launch_fig4d.sh

# Figure 4 panels (d) data efficiency and (e) personalization
python examples/plot_fig4_panels.py --panel e
python examples/plot_fig4_panels.py --panel d --seeds 42-51

# FC adjusted Rand index per (model, seed), behind the modularity claim (GPU,
# after step 1); writes results/ari_per_seed.json
./logs/revision/launch_ari.sh

# Runtime cost of every model (training s/epoch, grid-search s/trial, inference s)
# on the RTX 2080 Ti the paper reports; writes results/timing_benchmark.json
./logs/revision/launch_timing.sh

# One row per (table/figure, model, seed) with inclusion status and exclusion reason
python examples/build_run_manifest.py --out results/run_manifest.csv
```

The remaining figure panels (training curves, brain maps, FC/FCD/phFCD
matrices, clustering, stacked trajectories, pipeline illustrations) are produced
by the notebooks under `examples/`, and the grid-search landscape of Figure 2a by
`plot_grid_search_fc_similarity.py`. They show the seed-42 fit of each
architecture. The notebooks load every checkpoint in their directory, so they
read `checkpoints/figures/` and `results/metrics_figures/`, which
`link_figure_inputs.sh` fills with links to the seed-42 runs of step 1:

```bash
./logs/revision/link_figure_inputs.sh      # after launch_seeds.sh has finished
python examples/plot_grid_search_fc_similarity.py
```

Their outputs are stripped from the repository; run a notebook to regenerate its
panels.

### A note on the connectivity matrix

The `structural_connectivity` argument threaded through the models is the
**group-averaged functional connectome** of the training subjects, computed over
their training segments (first halves) by `src.dataset.training_view`; the
intrinsic frequencies use the same segments. The HCP release in `data/` contains BOLD time series only and
no diffusion-weighted data, so despite the argument name there is no tractogram
anywhere in this repository. The models are unaffected — all architectures
receive the same matrix — but the connectome ablations should be read as testing
a data-derived network prior rather than anatomical structure, and the static-FC
part of that comparison is partly circular. This is stated in the paper's
Methods and Limitations.

---

## Documentation Website

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).  
Live site: **https://dcn-fau-avh.github.io/neuroscience-modeling/**

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

## Continuous Integration

- **CI** (`.github/workflows/ci.yml`): runs the test suite with coverage on every push and pull request.
- **Docs** (`.github/workflows/docs.yml`): builds the MkDocs site and deploys it to GitHub Pages.

---

## Related Work

- [Deco et al., 2017](https://doi.org/10.1038/s41598-017-03073-5) — Whole-brain coupled Hopf model
- [torchsde](https://github.com/dani2442/torchsde) — SDE solvers for PyTorch (complex-valued fork)
- [The Virtual Brain](https://www.thevirtualbrain.org/) — Open-source brain simulation platform

---

## Citation

If you use this code in academic work, please cite the manuscript and the repository:

```bibtex
@article{lopezmontero2026hybrid,
  author = {L{\'o}pez-Montero, Daniel and Liverani, Lorenzo and Zuazua, Enrique and Kobeleva, Xenia},
  title  = {Hybrid generative models of whole-brain dynamics balance data fidelity and interpretability},
  year   = {2026},
  note   = {Manuscript submitted for publication},
}

@software{neuroscience_modeling,
  author = {L{\'o}pez-Montero, Daniel and Liverani, Lorenzo and Zuazua, Enrique and Kobeleva, Xenia},
  title  = {neuroscience-modeling: hybrid generative whole-brain models fitted by gradient descent},
  url    = {https://github.com/DCN-FAU-AvH/neuroscience-modeling},
  year   = {2026},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
