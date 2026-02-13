# Neuroscience Control

<p align="center">
  <b>Brain dynamics simulation and control using Coupled Hopf and Neural SDE models</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#models">Models</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This project provides a PyTorch-based framework for **whole-brain modeling** of resting-state fMRI BOLD signals. It implements two complementary approaches:

1. **Coupled Hopf Model** — A physics-based approach using coupled nonlinear oscillators at the supercritical Hopf bifurcation, informed by structural connectivity from diffusion MRI.

2. **Neural SDE Model** — A data-driven approach using neural networks to parameterize stochastic differential equations (SDEs) for flexible, learnable brain dynamics.

Both models operate entirely in **complex-valued** space: state, drift, diffusion, and Brownian motion are all complex tensors.  The observed BOLD signal is taken as the real part of the complex state.

### Key Features

- 🧠 **Biologically-grounded modeling** with structural connectivity integration
- 🔢 **Native complex-valued SDEs** — state, drift, diffusion, and Brownian motion are complex tensors (`torch.complex64` / `complex128`)
- 📈 **Multiple evaluation metrics**: Functional Connectivity (FC), FC Dynamics (FCD), and Metastability
- 🔬 **Stochastic simulation** via [`torchsde`](https://github.com/dani2442/torchsde) with complex Brownian motion support
- 📊 **Weights & Biases** integration for experiment tracking
- ⚡ **GPU-accelerated** training and simulation

---

## Installation

### Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.10

### Install with uv (recommended)

```bash
git clone https://github.com/your-username/neuroscience-control.git
cd neuroscience-control

# Install with uv
uv sync
```

### Install with pip

```bash
pip install -e .
```

### Dependencies

The project requires the following main dependencies (automatically installed):

| Package     | Description                           |
|-------------|---------------------------------------|
| `torch`     | Deep learning framework               |
| `torchsde`  | SDE solvers for PyTorch (complex-valued fork) |
| `numpy`     | Numerical computing                   |
| `scipy`     | Scientific computing utilities        |
| `matplotlib`| Visualization                         |
| `wandb`     | Experiment tracking                   |
| `tqdm`      | Progress bars                         |

---

## Quick Start

### Basic Simulation

```python
import torch
from src.models import CoupledHopfModel, NeuralSDE

# Create a Coupled Hopf model for 68 ROIs (e.g., Desikan-Killiany atlas)
model = CoupledHopfModel(
    n_rois=68,
    initial_a=-0.02,  # Bifurcation parameter (near criticality)
    initial_g=0.5,    # Global coupling strength
    noise_sigma=0.01,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Simulate 200 timepoints
with torch.no_grad():
    initial_state = dataset.timeseries[:10, :, 0]  # complex initial conditions
    timeseries = model.forward(initial_state=initial_state, n_steps=200)  # (10, 68, 200) complex
    fc_matrix = model.compute_fc(timeseries)                 # (10, 68, 68)

print(f"Simulated timeseries shape: {timeseries.shape}")
print(f"FC matrix shape: {fc_matrix.shape}")
```

### Generate BOLD Signal

```python
# Generate realistic BOLD signal with TR=0.72s
bold_signal = model.generate_bold(
    n_timepoints=400,      # Number of BOLD volumes
    tr=0.72,               # Repetition time in seconds
    dt=0.001,              # Integration time step
    batch_size=5,
    initial_transient=1000 # Discard initial transient
)
# Shape: (5, 68, 400)
```

---

## Models

### Coupled Hopf Model

The **Coupled Hopf Model** represents each brain region as a nonlinear oscillator governed by a supercritical Hopf bifurcation. The dynamics are described by the complex-valued stochastic differential equation:

$$
dz_i = \left[ \left( a + i\omega_i - |z_i|^2 \right) z_i + G \sum_{j=1}^{N} C_{ij} z_j \right] dt + \sigma \, dW_i
$$

where $W_i = W_{1,i} + i\,W_{2,i}$ is a **complex Brownian motion** constructed from two independent standard real Brownian motions.

| Symbol | Description |
|--------|-------------|
| $z_i \in \mathbb{C}$ | Complex oscillator state for region $i$ |
| $a \in \mathbb{R}$ | **Bifurcation parameter** ($a < 0$: damped, $a > 0$: oscillatory) |
| $\omega_i$ | **Intrinsic frequency** of region $i$ (estimated from data) |
| $G \geq 0$ | **Global coupling strength** |
| $C_{ij}$ | **Structural connectivity** matrix from DTI tractography |
| $\sigma$ | **Noise amplitude** |
| $W_i$ | Complex Wiener process ($W_{1,i} + i W_{2,i}$) |

The state, drift, and diffusion are all complex-valued.  The simulated BOLD signal is the real part: $s_i(t) = \Re(z_i(t))$.

**Key insight**: The brain is hypothesized to operate near the critical point ($a \approx 0$), balancing flexibility and stability.

```python
from src.models import CoupledHopfModel

model = CoupledHopfModel(
    n_rois=68,
    structural_connectivity=sc_matrix,  # From DTI tractography
    initial_a=-0.02,                     # Near criticality
    initial_g=0.5,                       # Global coupling
    omega=intrinsic_frequencies,         # From power spectrum analysis
    noise_sigma=0.01,
    learnable_a=True,                    # Make bifurcation learnable
    learnable_g=True                     # Make coupling learnable
)
```

### Neural SDE Model

The **Neural SDE Model** uses neural networks to parameterize the drift and diffusion terms of a **complex-valued** SDE:

$$
dZ_t = f_\theta(Z_t) \, dt + g_\phi(Z_t) \, dW_t
$$

where $Z_t \in \mathbb{C}^N$, $f_\theta, g_\phi$ are learnable neural networks that accept and return complex tensors, and $W_t$ is complex Brownian motion.  Internally each network converts the complex state to a real representation via `torch.view_as_real`, processes it through a standard real-valued MLP, and converts back to complex via `torch.view_as_complex`.  This provides maximum flexibility for learning complex brain dynamics directly from data.

```python
from src.models import NeuralSDE

model = NeuralSDE(
    n_rois=68,
    hidden_dim=64,                       # Hidden layer dimension
    n_layers=2,                          # Number of drift network layers
    structural_connectivity=sc_matrix,  # Optional SC coupling
    coupling_strength=0.1
)
```

---

## Metrics

The framework evaluates model fit using three complementary metrics aligned with the neuroscience literature:

### 1. Functional Connectivity (FC)

Static correlation between regional time series:

$$
\text{FC}_{ij} = \frac{\text{Cov}_{ij}}{\text{SD}_i \cdot \text{SD}_j}
$$

**Training terms**:
- `loss_fc_correlation = 1 - corr(v(FC_pred), v(FC_target))`
- `loss_fc_mse = MSE(v(FC_pred), v(FC_target))`

**Evaluation metrics**:
- `fc_correlation`
- `fc_mse`

### 2. Functional Connectivity Dynamics (FCD)

Captures how FC evolves over time using sliding windows:

1. Compute windowed FC at each time window
2. Build FCD matrix: correlation between windowed FC patterns
3. Extract distribution of FCD values

**Training term**:
- `loss_fcd`: differentiable surrogate using MSE between FCD matrices (`fcd_mse_loss`).

**Evaluation metric**:
- `fcd_ks`: Kolmogorov-Smirnov distance between empirical and simulated FCD value distributions.

`fcd_ks` can be `NaN` when FCD windowing is not feasible for the current `--tr`, `--fcd-win-sec`, `--fcd-step-sec`, and time-series length (for example, short trajectories).

### 3. Metastability

Temporal variability of global synchronization using the Kuramoto order parameter:

$$
R(t) = \left| \frac{1}{N} \sum_{i=1}^{N} e^{i\phi_i(t)} \right|
$$

$$
\text{Metastability} = \text{std}_t(R(t))
$$

**Training term**:
- `loss_metastability = |Meta(pred) - Meta(target)|`

**Evaluation metric**:
- `metastability_diff`

### Total Objective

$$
\mathcal{L}(G, \sigma, a) = w_{\text{FC}} \cdot \mathcal{L}_{\text{FC}} + w_{\text{FCD}} \cdot \mathcal{L}_{\text{FCD}} + w_{\text{Meta}} \cdot \mathcal{L}_{\text{Meta}}
$$

For backpropagation, `--loss-fn fc_fcd_meta` uses:
- `loss_fc_correlation` (FC term)
- `loss_fcd` (MSE surrogate, not KS)
- `loss_metastability`

For Hopf grid search (`examples/train_hopf.py`), model selection uses the composite score `w_FC·fc_correlation − w_FCD·fcd_mse − w_Meta·metastability_diff` (weights configurable via `--weight-fc-correlation`, `--weight-fcd-mse`, `--weight-metastability-diff`; defaults 1.0, 0.5, 0.5). FCD/Metastability are also reported as evaluation metrics.

### Metric Usage by Script

| Script | Training / selection objective | Reported metrics |
|--------|--------------------------------|------------------|
| [`examples/train_hopf.py`](examples/train_hopf.py) | Grid search by composite `w_FC·fc_correlation - w_FCD·fcd_mse - w_Meta·metastability_diff` | `fc_correlation`, `fc_mse`, `fcd_ks`, `metastability_diff` |
| [`examples/train_backprop.py`](examples/train_backprop.py) | `--loss-fn` composite (`loss_*` terms) | Epoch/test: FC + timeseries + dynamics metrics; final: `fc_correlation`, `fc_mse`, `fcd_ks`, `metastability_diff` |
| [`examples/train_nsde_finetune.py`](examples/train_nsde_finetune.py) | Fine-tuning via `Trainer` composite loss | Test/summary include FC + timeseries + dynamics metrics; final: `fc_correlation`, `fc_mse`, `fcd_ks`, `metastability_diff` |
| [`examples/test.py`](examples/test.py) | No training (checkpoint evaluation) | Loader-based metrics + per-run real-vs-sim: FC + timeseries + dynamics metrics |

---

## Training

### Hopf Model: Grid Search

The Coupled Hopf model supports grid search over $(G, a)$:

```bash
python examples/train_hopf.py \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --experiment-name hopf_experiment \
    --wandb-project neuroscience-control
```

Backpropagation (Hopf or NSDE) is handled by:

```bash
python examples/train_backprop.py --model hopf --n-epochs 50 --loss-fn fc_fcd_meta
```

### Neural SDE: Backpropagation

The Neural SDE model is trained end-to-end via backpropagation using `train_backprop.py`:

```bash
python examples/train_backprop.py --model nsde \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --n-epochs 50 \
    --lr 1e-3 \
    --hidden-dim 32 \
    --loss-fn fc_fcd_meta \
    --experiment-name nsde_experiment
```

### Unified Backpropagation (NSDE + Hopf)

Use one script for both models:

```bash
python examples/train_backprop.py --model nsde --n-epochs 50 --loss-fn fc_fcd_meta
python examples/train_backprop.py --model hopf --n-epochs 50 --initial-a -0.02 --initial-g 0.5
```

### Neural SDE: Fine-tuning

Fine-tune a pretrained NSDE checkpoint:

```bash
python examples/train_nsde_finetune.py \
    --checkpoint checkpoints/nsde_best_<run_name>.pt \
    --fine-tune-epochs 20 \
    --fine-tune-lr 1e-4
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tr` | Repetition time (seconds) | 0.72 |
| `--f-lo` | Bandpass low cutoff (Hz) | 0.04 |
| `--f-hi` | Bandpass high cutoff (Hz) | 0.07 |
| `--fcd-win-sec` | FCD window length (seconds) | 60.0 |
| `--fcd-step-sec` | FCD window step (seconds) | 2.0 |
| `--no-fcd-ks` | Disable `fcd_ks` metric computation | False |
| `--no-metastability-diff` | Disable `metastability_diff` metric computation | False |
| `--loss-fn` | Loss preset (`mse`, `correlation`, `combined`, `fc_fcd_meta`, `full`, `custom`) or individual term | `combined` |
| `--loss-weight-fc-correlation` | `loss_fc_correlation` weight override (backprop) | preset default |
| `--loss-weight-fcd` | `loss_fcd` weight override (backprop) | preset default |
| `--loss-weight-metastability` | `loss_metastability` weight override (backprop) | preset default |

---

## Examples

### Training Scripts

| Script | Description |
|--------|-------------|
| [`examples/train_hopf.py`](examples/train_hopf.py) | Train Coupled Hopf via grid search |
| [`examples/train_backprop.py`](examples/train_backprop.py) | Unified backprop training entrypoint for NSDE or Hopf |
| [`examples/train_nsde_finetune.py`](examples/train_nsde_finetune.py) | Fine-tune a pretrained Neural SDE checkpoint |
| [`examples/compare_models.py`](examples/compare_models.py) | Compare trained models |
| [`examples/test.py`](examples/test.py) | Evaluate a checkpoint and visualize trajectories |

### Model Comparison

```bash
python examples/compare_models.py \
    --hopf-checkpoint checkpoints/hopf_best.pt \
    --nsde-checkpoint checkpoints/nsde_best.pt \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --n-simulations 10
```

This generates:
- FC comparison figures
- Simulated timeseries plots
- Model comparison bar charts
- Comprehensive comparison report

---

## Project Structure

```
neuroscience_control/
├── src/
│   ├── dataset/           # Data loading and preprocessing
│   │   ├── data_loader.py # NeuroscienceDataset class
│   │   └── preprocessing.py # Windowed dataset for training
│   ├── models/            # Brain dynamics models
│   │   ├── hopf_model.py  # Coupled Hopf oscillator model
│   │   ├── neural_sde.py  # Neural SDE model
│   │   └── base_model.py  # Base class for models
│   ├── metrics/           # Evaluation metrics
│   │   ├── fc_metrics.py  # FC correlation, MSE
│   │   ├── dynamics_metrics.py # FCD, metastability
│   │   └── timeseries_metrics.py # Power spectrum, autocorrelation
│   ├── training/          # Training utilities
│   │   ├── trainer.py     # Backpropagation trainer
│   │   ├── grid_search.py # Grid search for Hopf
│   │   ├── fine_tuning.py # Fine-tuning utilities
│   │   └── config.py      # Configuration dataclasses
│   └── utils/             # Visualization and utilities
├── examples/              # Training and comparison scripts
├── paper/                 # LaTeX source for accompanying paper
└── notebooks/             # Jupyter notebooks for exploration
```

---

## Preprocessing Pipeline

The code path uses three related preprocessing stages:

1. **Dataset preprocessing (`NeuroscienceDataset`)**
- Z-score each ROI time series.
- Optional FFT brick-wall bandpass denoising via `--fourier-denoise` (`--denoise-f-lo`, `--denoise-f-hi`; defaults 0.01–0.1 Hz when enabled).
- Convert to complex analytic signal via Hilbert transform.

2. **Dynamics-metric preprocessing (`compute_dynamics_fit_metrics`)**
- Per sample: linear detrend -> FFT bandpass (`--f-lo`, `--f-hi`, defaults 0.04–0.07 Hz) -> z-score.
- Used before FCD and metastability calculations.

3. **Intrinsic frequency estimation for Hopf (`compute_omega_from_timeseries`)**
- FFT-based estimation in `[f_lo, f_hi]` (default 0.04–0.07 Hz), using peak-power (default) or weighted mode.
- Returns angular frequencies (rad/s).

```python
from src.dataset import NeuroscienceDataset

dataset = NeuroscienceDataset(
    filepath="data/ts_young/ts_young_TR0.72.mat",
    normalize=True,
    device="cuda"
)

print(f"Subjects: {dataset.n_subjects}")
print(f"ROIs: {dataset.n_rois}")
print(f"Timepoints: {dataset.n_timepoints}")
print(f"Mean FC shape: {dataset.fc_mean.shape}")
```

---

## Data Format

The loader expects a `.mat` file with these keys:

```python
{
    'timeseries_all': np.ndarray,  # Shape: (n_rois, n_timepoints, n_subjects)
    'FC_all': np.ndarray,          # Shape: (n_rois, n_rois, n_subjects)
    'FC_mean': np.ndarray,         # Shape: (n_rois, n_rois)
}
```

`FC_mean` can be precomputed in file or recomputed from `FC_all` when subject subsampling is requested.

---

## Theoretical Background

This project is based on the theoretical framework described in the accompanying paper:

> **Continuous-Time Coupled Hopf Model for fMRI: Definitions and Fitting Objectives**
>
> This note defines a continuous-time preprocessing and fitting objective for a coupled Hopf whole-brain model of resting-state fMRI BOLD signals. We formalize detrending, zero-phase bandpass filtering, normalization, intrinsic frequency estimation, and summary statistics based on functional connectivity (FC) and functional connectivity dynamics (FCD).

Key theoretical contributions:
- Rigorous continuous-time formulation of preprocessing
- Formal definitions of FC, FCD, and metastability losses
- Data-driven controllability tests for linear systems

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neuroscience_control,
  title = {Neuroscience Control: Brain Dynamics Simulation and Control},
  author = {Authors},
  year = {2025},
  url = {https://github.com/your-username/neuroscience-control}
}
```

---

## Related Work

- [Deco et al., 2017](https://doi.org/10.1038/s41598-017-03073-5) — Whole-brain coupled Hopf model
- [torchsde](https://github.com/dani2442/torchsde) — SDE solvers for PyTorch (complex-valued fork)
- [The Virtual Brain](https://www.thevirtualbrain.org/) — Open-source brain simulation platform

---

## License

This project is licensed under the MIT License.
