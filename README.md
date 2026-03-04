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


## Overview

A PyTorch framework for **whole-brain modeling** of resting-state fMRI BOLD signals. It implements three complementary approaches:

| Model | Description |
|-------|-------------|
| **Coupled Hopf** | Physics-based coupled oscillators at the supercritical Hopf bifurcation, informed by structural connectivity |
| **Hybrid Hopf** | Hopf oscillators with a learnable graph-coupling network replacing fixed linear diffusive coupling |
| **Neural SDE** | Data-driven neural networks parameterizing stochastic differential equations |

All three models operate in **complex-valued** space — state, drift, diffusion, and Brownian motion are complex tensors. The observed BOLD signal is the real part of the complex state.

### Key Features

- **Biologically-grounded** modeling with structural connectivity integration
- **Native complex-valued SDEs** via [`torchsde`](https://github.com/dani2442/torchsde) with complex Brownian motion support
- **Comprehensive evaluation**: FC, FCD, phFCD, phase-coherence FC, metastability, and timeseries metrics
- **Weights & Biases** integration for experiment tracking
- **GPU-accelerated** training and simulation

---

## Installation

Requires **Python ≥ 3.13** and **PyTorch ≥ 2.10**.

```bash
git clone https://github.com/dani2442/neuroscience_control.git
cd neuroscience_control
uv sync
```

---

## Quick Start

```python
import torch
from src.models import CoupledHopfModel

model = CoupledHopfModel(
    n_rois=68,
    initial_a=-0.02,   # Bifurcation parameter (near criticality)
    initial_g=0.5,      # Global coupling strength
    noise_sigma=0.01,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Simulate 200 timepoints from 10 initial conditions
with torch.no_grad():
    timeseries = model.forward(initial_state=initial_state, n_steps=200)  # (10, 68, 200) complex
    fc_matrix = model.compute_fc(timeseries)                               # (10, 68, 68)
```

---

## Models

### Coupled Hopf Model

Each brain region is a nonlinear oscillator at the supercritical Hopf bifurcation:

$$dz_i = \Bigl[\bigl(a + i\omega_i - \lvert z_i \rvert^2\bigr) z_i + G \sum_{j=1}^{N} C_{ij} (z_j - z_i)\Bigr] dt + \sigma\, dW_i$$

where $W_i = W_{1,i} + i\, W_{2,i}$ is complex Brownian motion.

| Symbol | Description |
|--------|-------------|
| $z_i \in \mathbb{C}$ | Complex state of region $i$ |
| $a$ | Bifurcation parameter ($a < 0$: damped; $a > 0$: oscillatory) |
| $\omega_i$ | Intrinsic frequency of region $i$ |
| $G$ | Global coupling strength |
| $C_{ij}$ | Structural connectivity from DTI |
| $\sigma$ | Noise amplitude |

The BOLD signal is $s_i(t) = \Re(z_i(t))$. The brain is hypothesized to operate near criticality ($a \approx 0$).

```python
from src.models import CoupledHopfModel

model = CoupledHopfModel(
    n_rois=68,
    structural_connectivity=sc_matrix,
    initial_a=-0.02,
    initial_g=0.5,
    omega=intrinsic_frequencies,
    noise_sigma=0.01,
    learnable_a=True,
    learnable_g=True,
)
```

### Hybrid Hopf Model

Combines Hopf local oscillator dynamics with a learnable graph-coupling network $\psi_\theta$:

$$dz_i = \Bigl[\bigl(a + i\omega_i - \lvert z_i \rvert^2\bigr) z_i + G \sum_j \psi_\theta(z_j - z_i,\; C_{ij})\Bigr] dt + \sigma\, dW_i$$

The network $\psi_\theta : \mathbb{C}^2 \to \mathbb{C}$ is a complex-valued harmonic network with phase-preserving activations, replacing the fixed linear coupling of the classical Hopf model.

```python
from src.models import HybridHopfModel

model = HybridHopfModel(
    n_rois=68,
    structural_connectivity=sc_matrix,
    initial_a=-0.02,
    initial_g=0.5,
    omega=intrinsic_frequencies,
)
```

### Neural SDE Model

Uses neural networks to parameterize drift and diffusion of a complex-valued SDE:

$$dz_t = f_\theta(z_t)\, dt + g_\phi(z_t)\, dW_t$$

where $f_\theta$ and $g_\phi$ are real-valued MLPs operating on `torch.view_as_real` / `torch.view_as_complex` conversions.

```python
from src.models import NeuralSDE

model = NeuralSDE(
    n_rois=68,
    hidden_dim=64,
    n_layers=2,
    structural_connectivity=sc_matrix,
    coupling_strength=0.1,
)
```

---

## Metrics

All evaluation metrics are defined in `EVAL_METRIC_KEYS` and reported by every training and evaluation script. Metrics operate on complex analytic signals produced by the preprocessing pipeline (z-score → optional bandpass → Hilbert transform).

### Evaluation Metrics

| Metric | Range | Direction | Description |
|--------|-------|-----------|-------------|
| `fc_correlation` | $[-1, 1]$ | Higher | Pearson correlation of upper-triangular FC |
| `fc_mse` | $[0, \infty)$ | Lower | MSE of upper-triangular FC |
| `fcd_ks` | $[0, 1]$ | Lower | KS distance between FCD distributions (sliding-window) |
| `phfcd_ks` | $[0, 1]$ | Lower | KS distance between phase-FCD distributions |
| `phase_fc_correlation` | $[-1, 1]$ | Higher | Pearson correlation of phase-coherence FC |
| `metastability_diff` | $[0, \infty)$ | Lower | Absolute difference in metastability |
| `temporal_correlation` | $[-1, 1]$ | Higher | Mean per-ROI Pearson correlation over time |
| `power_spectrum_distance` | $[0, \infty)$ | Lower | MSE between normalised power spectra |
| `autocorr_distance` | $[0, \infty)$ | Lower | MSE between autocorrelation functions |

### Mathematical Details

**Functional Connectivity (FC)** — Static Pearson correlation between regional time series from the real part $s_n(t) = \Re(z_n(t))$:

$$\text{FC}\_{nm} = \frac{\text{Cov}(s_n,\, s_m)}{\text{SD}(s_n) \cdot \text{SD}(s_m)}$$

**Functional Connectivity Dynamics (FCD)** — Sliding-window FC vectors, z-scored, then correlated pairwise across windows. Compared via two-sample KS distance.

**Phase FCD (phFCD)** — The paper's main model-fitting metric. Uses instantaneous phase coherence $P_{nm}(t) = \cos\bigl(\phi_n(t) - \phi_m(t)\bigr)$ instead of windowed Pearson FC. The phFCD matrix is built from cosine similarity of the upper-triangular phase-coherence vectors across time.

**Metastability** — Temporal variability of global synchronisation via the Kuramoto order parameter:

$$R(t) = \left\lvert \frac{1}{N} \sum_{n=1}^{N} e^{i\phi_n(t)} \right\rvert, \qquad \text{Metastability} = \text{std}_t\bigl(R(t)\bigr)$$

### Loss Presets

Losses are composable via `CompositeLoss`. Preset names can be passed in the training config:

| Preset | Terms (default weights) |
|--------|------------------------|
| `mse` | `fc_mse: 1.0` |
| `correlation` | `fc_correlation: 1.0` |
| `combined` | `fc_mse: 1.0`, `fc_correlation: 0.5` |
| `fc_fcd_meta` | `fc_correlation: 1.0`, `fcd: 1.0`, `metastability: 1.0` |
| `fc_phfcd_meta` | `fc_correlation: 1.0`, `phfcd: 1.0`, `metastability: 1.0` |
| `full` | `fc_correlation`, `l2`, `amplitude`, `omega`, `fcd`, `phfcd`, `metastability` (all 1.0) |

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

### Fine-Tuning a Neural SDE

```bash
python examples/train_nsde_finetune.py \
    --checkpoint checkpoints/nsde_best.pt \
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
3. **Hilbert transform** to complex analytic signal

All downstream metrics and losses operate on this complex signal — FCD and timeseries metrics use `.real`, phase-based metrics extract phases via `torch.angle(z)`.

---

## Project Structure

```
neuroscience_control/
├── src/
│   ├── dataset/               # Data loading and preprocessing
│   │   ├── data_loader.py     # NeuroscienceDataset, Hilbert transform
│   │   └── preprocessing.py   # Windowing, omega estimation
│   ├── models/                # Brain dynamics models
│   │   ├── base_model.py      # Abstract base class
│   │   ├── hopf_model.py      # Coupled Hopf oscillator
│   │   ├── hybrid_hopf_model.py # Hybrid mechanistic–neural Hopf
│   │   ├── neural_sde.py      # Neural SDE
│   │   ├── factory.py         # build_model() dispatcher
│   │   └── checkpointing.py   # Checkpoint loading
│   ├── metrics/               # Evaluation metrics
│   │   ├── fc_metrics.py      # FC correlation, MSE
│   │   ├── dynamics_metrics.py    # FCD, phFCD, metastability
│   │   ├── timeseries_metrics.py  # Power spectrum, autocorrelation
│   │   └── metrics_store.py   # MetricsStore accumulator
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Backprop trainer
│   │   ├── grid_search.py     # Hopf grid search
│   │   ├── fine_tuning.py     # Fine-tuning
│   │   ├── losses.py          # CompositeLoss and loss functions
│   │   ├── backprop.py        # Backprop training loop
│   │   └── config.py          # TrainingConfig dataclass
│   └── utils/                 # Visualization, evaluation, runtime
├── examples/                  # Entry-point scripts
│   ├── train_models.py        # Unified training (grid / backprop / paper)
│   ├── train_nsde_finetune.py # Neural SDE fine-tuning
│   ├── compare_models.py      # Side-by-side model comparison
│   ├── test.py                # Checkpoint evaluation
│   └── visualization.py       # Plotting utilities
├── paper/                     # LaTeX source for accompanying paper
├── data/                      # Data directory
└── checkpoints/               # Saved model weights
```

---

## Examples

```bash
# Evaluate a saved checkpoint
python examples/test.py --checkpoint checkpoints/best_nsde_backprop.pt

# Compare two trained models
python examples/compare_models.py \
    --hopf-checkpoint checkpoints/hopf_best.pt \
    --nsde-checkpoint checkpoints/nsde_best.pt \
    --n-simulations 10
```

---

## Related Work

- [Deco et al., 2017](https://doi.org/10.1038/s41598-017-03073-5) — Whole-brain coupled Hopf model
- [torchsde](https://github.com/dani2442/torchsde) — SDE solvers for PyTorch (complex-valued fork)
- [The Virtual Brain](https://www.thevirtualbrain.org/) — Open-source brain simulation platform

---

## License

MIT License
