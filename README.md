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
- 📈 **Multiple evaluation metrics**: Functional Connectivity (FC), FC Dynamics (FCD), Phase FCD (phFCD), and Metastability
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

---

## Models

### Coupled Hopf Model

The **Coupled Hopf Model** represents each brain region as a nonlinear oscillator governed by a supercritical Hopf bifurcation. The dynamics are described by the complex-valued stochastic differential equation:

$$
dz_i = \left[ \left( a + i\omega_i - |z_i|^2 \right) z_i + G \sum_{j=1}^{N} C_{ij} (z_j - z_i) \right] dt + \sigma \, dW_i
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

The framework evaluates model fit using three complementary metrics aligned with the neuroscience literature.

All time-domain metrics use the discrete time-average $\frac{1}{T}\sum_{t=1}^T f(t)$ as the uniform-$\Delta t$ approximation to $\frac{1}{T}\int_0^T f(t)\,dt$, then averaged over the batch and ROIs.  Since both data and model outputs are **complex analytic signals** (bandpass-filtered and Hilbert-transformed at dataset-load time), no additional signal preprocessing is applied inside the metrics.

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

Captures how FC evolves over time using sliding windows on the **real part** of the complex analytic signal (no additional preprocessing):

1. Compute windowed FC at each time window
2. Build FCD matrix: correlation between windowed FC patterns
3. Extract distribution of FCD values

**Training term**:
- `loss_fcd`: differentiable surrogate using MSE between FCD matrices (`fcd_mse_loss`).

**Evaluation metric**:
- `fcd_ks`: Kolmogorov-Smirnov distance between empirical and simulated FCD value distributions.

`fcd_ks` can be `NaN` when FCD windowing is not feasible for the current `--tr`, `--fcd-win-sec`, `--fcd-step-sec`, and time-series length (for example, short trajectories).

### 3. Phase Functional Connectivity Dynamics (phFCD)

The paper's **main model-fitting metric**. Instead of sliding-window Pearson FC, phFCD uses instantaneous phase coherence to capture time-varying functional connectivity:

1. Extract instantaneous phase from the complex analytic signal: $\phi_n(t) = \arg(z_n(t))$
2. Compute phase coherence: $P_{nm}(t) = \cos(\phi_n(t) - \phi_m(t))$
3. Vectorise upper-triangular entries of $P(t)$ at each time point
4. Build phFCD similarity matrix via cosine similarity between time points:

$$
\mathrm{phFCD}_{ij} = \frac{\mathbf{p}(t_i)^\top \mathbf{p}(t_j)}{\|\mathbf{p}(t_i)\|_2 \, \|\mathbf{p}(t_j)\|_2}
$$

5. Extract upper-triangular phFCD values as the tv-FC summary distribution
6. Compare empirical vs simulated distributions with Kolmogorov-Smirnov distance

**Training term**:
- `loss_phfcd`: differentiable surrogate using MSE between phFCD matrices (`phfcd_mse_loss`).

**Evaluation metric**:
- `phfcd_ks`: Kolmogorov-Smirnov distance between empirical and simulated phFCD distributions. This is the paper's decisive goodness-of-fit metric; **lower = better**.

Unlike `fcd_ks`, `phfcd_ks` has **no windowing dependency** — it works at full temporal resolution.

### 4. Grand-Average Phase-Coherence FC

The time-averaged instantaneous phase-coherence matrix (Eq. 11 in Deco et al. 2019), used in the effective connectivity (EC) optimisation delta-rule:

$$
\mathrm{FC}^{\phi}_{ij} = \left\langle \cos\!\bigl(\phi_i(t) - \phi_j(t)\bigr) \right\rangle_t
$$

where $\phi_i(t) = \arg(z_i(t))$.  Each entry is the mean phase coherence between two ROIs over time, producing a symmetric matrix with diagonal identically 1 and entries in $[-1, 1]$.

**Evaluation metric**:
- `phase_fc_corr`: Pearson correlation between upper-triangular entries of predicted and target phase-coherence FC (analogous to `fc_correlation` for Pearson FC).

### 5. Metastability

Temporal variability of global synchronization using the Kuramoto order parameter.
Phases are extracted directly from the complex analytic signal via $\phi_i(t) = \arg(z_i(t))$ — no Hilbert transform or bandpass is needed at metric time:

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

Alternatively, `--loss-fn fc_phfcd_meta` uses the **phase-based** phFCD instead of windowed FCD:
- `loss_fc_correlation` (FC term)
- `loss_phfcd` (MSE between phFCD matrices)
- `loss_metastability`

For Hopf grid search (`examples/train_hopf.py`), model selection uses the composite score `w_FC·fc_correlation − w_FCD·fcd_mse − w_Meta·metastability_diff` (weights configurable via `--weight-fc-correlation`, `--weight-fcd-mse`, `--weight-metastability-diff`; defaults 1.0, 0.5, 0.5). FCD/Metastability are also reported as evaluation metrics.

### Metric Usage by Script

| Script | Training / selection objective | Reported metrics |
|--------|--------------------------------|------------------|
| [`examples/train_hopf.py`](examples/train_hopf.py) | Grid search by composite `w_FC·fc_correlation - w_FCD·fcd_mse - w_Meta·metastability_diff` | `fc_correlation`, `fc_mse`, `fcd_ks`, `phfcd_ks`, `phase_fc_corr`, `metastability_diff` |
| [`examples/train_backprop.py`](examples/train_backprop.py) | `--loss-fn` composite (`loss_*` terms) | Epoch/test: FC + timeseries + dynamics metrics; final: `fc_correlation`, `fc_mse`, `fcd_ks`, `phfcd_ks`, `phase_fc_corr`, `metastability_diff` |
| [`examples/train_nsde_finetune.py`](examples/train_nsde_finetune.py) | Fine-tuning via `Trainer` composite loss | Test/summary include FC + timeseries + dynamics metrics; final: `fc_correlation`, `fc_mse`, `fcd_ks`, `phfcd_ks`, `phase_fc_corr`, `metastability_diff` |
| [`examples/test.py`](examples/test.py) | No training (checkpoint evaluation) | Loader-based metrics + per-run real-vs-sim: FC + timeseries + dynamics metrics |

### Complete Metrics Reference

The table below lists every metric and loss term in the framework, grouped by category.

#### Evaluation Metrics

These are used for reporting model quality. They are **not** directly optimized during training.

| Name | Module | Input | Formula | Range | Direction |
|------|--------|-------|---------|-------|-----------|
| `fc_correlation` | `src.metrics.fc_metrics` | FC matrices `(B, N, N)` | Pearson correlation between upper-triangle vectors of predicted and target FC | $[-1, 1]$ | Higher = better |
| `fc_mse` | `src.metrics.fc_metrics` | FC matrices `(B, N, N)` | MSE between upper-triangle entries of predicted and target FC | $[0, \infty)$ | Lower = better |
| `power_spectrum_distance` | `src.metrics.timeseries_metrics` | Time series `(B, N, T)` | MSE between normalised power spectra (via FFT), averaged over batch and ROIs | $[0, \infty)$ | Lower = better |
| `temporal_correlation` | `src.metrics.timeseries_metrics` | Time series `(B, N, T)` | Mean per-ROI Pearson correlation over time between predicted and target, averaged over batch | $[-1, 1]$ | Higher = better |
| `autocorr_distance` | `src.metrics.timeseries_metrics` | Time series `(B, N, T)` | MSE between autocorrelation functions (up to `max_lag` lags), averaged over batch, ROIs, and lags | $[0, \infty)$ | Lower = better |
| `fcd_ks` | `src.metrics.dynamics_metrics` | Time series `(B, N, T)` | Two-sample Kolmogorov-Smirnov distance between FCD value distributions of predicted and target | $[0, 1]$ | Lower = better |
| `phfcd_ks` | `src.metrics.dynamics_metrics` | Time series `(B, N, T)` complex | KS distance between phase-based FCD (phFCD) distributions of predicted and target — the paper's main fitting metric | $[0, 1]$ | Lower = better |
| `phase_fc_corr` | `src.metrics.dynamics_metrics` | Time series `(B, N, T)` complex | Pearson correlation between upper-triangular entries of predicted and target grand-average phase-coherence FC | $[-1, 1]$ | Higher = better |
| `metastability_diff` | `src.metrics.dynamics_metrics` | Time series `(B, N, T)` | $\lvert\text{Meta}(\text{pred}) - \text{Meta}(\text{target})\rvert$ where Meta = $\text{std}_t(R(t))$ (Kuramoto order parameter) | $[0, \infty)$ | Lower = better |
| `symmetric_kl_divergence` | `src.metrics.dynamics_metrics` | Probability vectors `(k,)` | $\frac{1}{2}\text{KL}(P\|Q) + \frac{1}{2}\text{KL}(Q\|P)$ — symmetrized KL divergence for PMS probability mismatch | $[0, \infty)$ | Lower = better |
| `tpm_entropy_distance` | `src.metrics.dynamics_metrics` | TPMs `(k, k)` | $\lvert S(\mathbf{T}_{\text{pred}}) - S(\mathbf{T}_{\text{target}})\rvert$ — absolute Markov entropy-rate difference between TPMs | $[0, \infty)$ | Lower = better |

#### Training Loss Terms

These are differentiable loss functions used during gradient-based optimization. All are oriented so that **lower = better** and can be combined via `CompositeLoss`.

| Name | Registry key | Module | Input | Formula | Notes |
|------|-------------|--------|-------|---------|-------|
| `loss_fc_correlation` | `fc_correlation` | `src.training.losses` | FC matrices `(B, N, N)` | $1 - \text{fc\_correlation}$ | Converts similarity into a minimizable loss |
| `loss_fc_mse` | `fc_mse` | `src.training.losses` | FC matrices `(B, N, N)` | Same as `fc_mse` (thin wrapper) | Identical to the evaluation metric |
| `loss_l2_timeseries` | `l2` | `src.training.losses` | Time series `(B, N, T)` | $\frac{1}{BNT}\sum\lvert z^{\text{pred}} - z^{\text{target}}\rvert^2$ (squared modulus for complex) | Supports both real and complex tensors |
| `loss_amplitude` | `amplitude` | `src.training.losses` | Time series `(B, N, T)` | $\frac{1}{N}\sum_n(\overline{\lvert z_n^{\text{pred}}\rvert} - A_n^{\text{ref}})^2$ | L² error of mean envelope amplitude per ROI; can use dataset-level `ref_amplitude` |
| `loss_omega` | `omega` | `src.training.losses` | Time series `(B, N, T)` | $\frac{1}{N}\sum_n(\bar{\omega}_n^{\text{pred}} - \omega_n^{\text{ref}})^2$ | L² error of mean instantaneous frequency per ROI; can use dataset-level `ref_omega` |
| `loss_fcd` | `fcd` | `src.training.losses` | Time series `(B, N, T)` | MSE between FCD matrices (differentiable surrogate) | Proxy for the non-differentiable KS distance (`fcd_ks`) |
| `loss_phfcd` | `phfcd` | `src.training.losses` | Time series `(B, N, T)` complex | MSE between phFCD matrices (differentiable surrogate) | Proxy for the non-differentiable KS distance (`phfcd_ks`); no windowing needed |
| `loss_metastability` | `metastability` | `src.training.losses` | Time series `(B, N, T)` | $\lvert\text{Meta}(\text{pred}) - \text{Meta}(\text{target})\rvert$ | L1 difference of Kuramoto metastability |

#### Loss Presets

| Preset name | Included loss terms (default weights) |
|-------------|--------------------------------------|
| `mse` | `fc_mse: 1.0` |
| `correlation` | `fc_correlation: 1.0` |
| `combined` | `fc_mse: 1.0`, `fc_correlation: 0.5` |
| `fc_fcd_meta` | `fc_correlation: 1.0`, `fcd: 1.0`, `metastability: 1.0` |
| `fc_phfcd_meta` | `fc_correlation: 1.0`, `phfcd: 1.0`, `metastability: 1.0` |
| `full` | `fc_correlation: 1.0`, `l2: 1.0`, `amplitude: 1.0`, `omega: 1.0`, `fcd: 1.0`, `phfcd: 1.0`, `metastability: 1.0` |

---

## Training

### Unified Training Entrypoint

`examples/train_models.py` now handles Hopf grid-search, backprop training, and paper-style comparison reporting:

```bash
python examples/train_models.py hopf-grid
python examples/train_models.py backprop --model nsde --n-epochs 50
python examples/train_models.py backprop --model hopf --n-epochs 50
python examples/train_models.py backprop --model hybrid_hopf --n-epochs 50
python examples/train_models.py paper --output-json results/paper_metrics.json
```

Compatibility wrappers remain available:

```bash
python examples/train_hopf.py              # same as: train_models.py hopf-grid
python examples/train_backprop.py --model nsde   # same as: train_models.py backprop --model nsde
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
| `--f-lo` | Bandpass low cutoff (Hz) — used only for Hopf intrinsic frequency estimation | 0.04 |
| `--f-hi` | Bandpass high cutoff (Hz) — used only for Hopf intrinsic frequency estimation | 0.07 |
| `--fcd-win-sec` | FCD window length (seconds) | 60.0 |
| `--fcd-step-sec` | FCD window step (seconds) | 2.0 |
| `--loss-fn` | Loss preset (`mse`, `correlation`, `combined`, `fc_fcd_meta`, `full`, `custom`) or individual term | `combined` |
| `--loss-weight-fc-correlation` | `loss_fc_correlation` weight override (backprop) | preset default |
| `--loss-weight-fcd` | `loss_fcd` weight override (backprop) | preset default |
| `--loss-weight-phfcd` | `loss_phfcd` weight override (backprop) | preset default |
| `--loss-weight-metastability` | `loss_metastability` weight override (backprop) | preset default |

---

## Examples

### Training Scripts

| Script | Description |
|--------|-------------|
| [`examples/train_models.py`](examples/train_models.py) | Unified entrypoint: Hopf grid-search, backprop models, and paper report |
| [`examples/train_hopf.py`](examples/train_hopf.py) | Compatibility wrapper for `train_models.py hopf-grid` |
| [`examples/train_backprop.py`](examples/train_backprop.py) | Compatibility wrapper for `train_models.py backprop` |
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
│   │   ├── _utils.py      # Shared helpers (to_real, ensure_batch, zscore, …)
│   │   ├── fc_metrics.py  # FC correlation, MSE, compute_static_fc
│   │   ├── dynamics_metrics.py # FCD, phFCD, phase-coherence FC, metastability, sym-KL, Markov entropy
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

The code path uses two related preprocessing stages:

1. **Dataset preprocessing (`NeuroscienceDataset`)**
- Z-score each ROI time series.
- Optional FFT brick-wall bandpass denoising via `--fourier-denoise` (`--denoise-f-lo`, `--denoise-f-hi`; defaults 0.01–0.1 Hz when enabled).
- Convert to complex analytic signal via Hilbert transform.

All downstream metrics and losses operate directly on this complex analytic signal — FCD uses the real part (`.real`), metastability extracts phases via `torch.angle(z)`, and timeseries metrics (power spectrum, temporal correlation, autocorrelation) use the real part.

2. **Intrinsic frequency estimation for Hopf (`compute_omega_from_timeseries`)**
- FFT-based estimation in `[f_lo, f_hi]` (default 0.04–0.07 Hz), using peak-power (default) or weighted mode.
- Returns angular frequencies (rad/s).
- `--f-lo` and `--f-hi` are only used for this omega estimation, not for metric preprocessing.

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
