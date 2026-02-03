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

### Key Features

- 🧠 **Biologically-grounded modeling** with structural connectivity integration
- 📈 **Multiple evaluation metrics**: Functional Connectivity (FC), FC Dynamics (FCD), and Metastability
- 🔬 **Stochastic simulation** via [`torchsde`](https://github.com/google-research/torchsde) for proper SDE integration
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
| `torchsde`  | SDE solvers for PyTorch               |
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
    timeseries = model.forward(n_steps=200, batch_size=10)  # (10, 68, 200)
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

The **Coupled Hopf Model** represents each brain region as a nonlinear oscillator governed by a supercritical Hopf bifurcation. The dynamics are described by the stochastic differential equation:

$$
dz_i = \left[ \left( a + i\omega_i - |z_i|^2 \right) z_i + G \sum_{j=1}^{N} C_{ij}(z_j - z_i) \right] dt + \sigma \, dW_i
$$

Where:
- $z_i \in \mathbb{C}$ — Complex oscillator state for region $i$
- $a \in \mathbb{R}$ — **Bifurcation parameter** controlling local dynamics ($a < 0$: damped, $a > 0$: oscillatory)
- $\omega_i$ — **Intrinsic frequency** of region $i$ (estimated from data)
- $G \geq 0$ — **Global coupling strength**
- $C_{ij}$ — **Structural connectivity** matrix from DTI tractography
- $\sigma$ — **Noise amplitude**
- $W_i$ — Complex Wiener process

The simulated BOLD signal is the real part: $s_i(t) = \Re(z_i(t)) = x_i(t)$

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

The **Neural SDE Model** uses neural networks to parameterize the drift and diffusion terms of an SDE:

$$
dX_t = f_\theta(X_t) \, dt + g_\phi(X_t) \, dW_t
$$

Where $f_\theta$ and $g_\phi$ are learnable neural networks. This provides maximum flexibility for learning complex brain dynamics directly from data.

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

**Loss**: $\mathcal{L}_{\text{FC}} = 1 - \text{corr}(v(\tilde{x}), v(\tilde{s}))$ where $v(\cdot)$ extracts upper-triangular elements.

### 2. Functional Connectivity Dynamics (FCD)

Captures how FC evolves over time using sliding windows:

1. Compute windowed FC at each time window
2. Build FCD matrix: correlation between windowed FC patterns
3. Extract distribution of FCD values

**Loss**: Kolmogorov-Smirnov distance between empirical and simulated FCD distributions.

### 3. Metastability

Temporal variability of global synchronization using the Kuramoto order parameter:

$$
R(t) = \left| \frac{1}{N} \sum_{i=1}^{N} e^{i\phi_i(t)} \right|
$$

$$
\text{Metastability} = \text{std}_t(R(t))
$$

**Loss**: $\mathcal{L}_{\text{Meta}} = |\text{Meta}(\tilde{x}) - \text{Meta}(\tilde{s})|$

### Total Objective

$$
\mathcal{L}(G, \sigma, a) = w_{\text{FC}} \cdot \mathcal{L}_{\text{FC}} + w_{\text{FCD}} \cdot \mathcal{L}_{\text{FCD}} + w_{\text{Meta}} \cdot \mathcal{L}_{\text{Meta}}
$$

---

## Training

### Hopf Model: Grid Search

The Coupled Hopf model is typically trained via grid search over the parameter space $(G, a)$:

```bash
python examples/train_hopf.py \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --experiment-name hopf_experiment \
    --wandb-project neuroscience-control
```

### Neural SDE: Backpropagation

The Neural SDE model is trained end-to-end via backpropagation:

```bash
python examples/train_nsde.py \
    --data-path data/ts_young/ts_young_TR0.72.mat \
    --n-epochs 50 \
    --lr 1e-3 \
    --hidden-dim 32 \
    --experiment-name nsde_experiment
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tr` | Repetition time (seconds) | 0.72 |
| `--f-lo` | Bandpass low cutoff (Hz) | 0.04 |
| `--f-hi` | Bandpass high cutoff (Hz) | 0.07 |
| `--fcd-win-sec` | FCD window length (seconds) | 60.0 |
| `--fcd-step-sec` | FCD window step (seconds) | 2.0 |
| `--no-fcd` | Disable FCD metrics | False |
| `--no-metastability` | Disable metastability metrics | False |

---

## Examples

### Training Scripts

| Script | Description |
|--------|-------------|
| [`examples/train_hopf.py`](examples/train_hopf.py) | Train Coupled Hopf via grid search |
| [`examples/train_nsde.py`](examples/train_nsde.py) | Train Neural SDE via backpropagation |
| [`examples/compare_models.py`](examples/compare_models.py) | Compare trained models |

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

The framework implements a continuous-time preprocessing pipeline following best practices in fMRI analysis:

1. **Bandpass Filtering** (0.01–0.1 Hz) — Isolate neural activity band using zero-phase filtering
2. **Z-scoring** — Standardize each ROI to zero mean and unit variance
3. **Intrinsic Frequency Estimation** — Welch's method to estimate dominant oscillation frequency

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

The framework expects fMRI timeseries data in `.mat` format with the following structure:

```python
{
    'ts': np.ndarray,  # Shape: (n_subjects, n_rois, n_timepoints)
    'SC': np.ndarray   # Optional: (n_rois, n_rois) structural connectivity
}
```

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
- [torchsde](https://github.com/google-research/torchsde) — SDE solvers for PyTorch
- [The Virtual Brain](https://www.thevirtualbrain.org/) — Open-source brain simulation platform

---

## License

This project is licensed under the MIT License.
