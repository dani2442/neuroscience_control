# Neuroscience Control

`neuroscience-control` is a PyTorch framework for **whole-brain modeling** of resting-state fMRI BOLD signals. It implements three complementary model families — all operating in **complex-valued** space — with a shared training and evaluation pipeline.

| Model | Description |
|-------|-------------|
| **Coupled Hopf** | Physics-based coupled oscillators at the supercritical Hopf bifurcation, informed by structural connectivity |
| **Hybrid Hopf** | Hopf oscillators with a learnable complex-valued graph-coupling network replacing fixed linear diffusive coupling |
| **Neural SDE** | Data-driven neural networks parameterising stochastic differential equations |

The observed BOLD signal is the real part of the complex state, $s_i(t) = \Re(z_i(t))$.

## Key Features

- **Biologically-grounded** modeling with structural connectivity from DTI
- **Native complex-valued SDEs** via a [complex-valued fork of `torchsde`](https://github.com/dani2442/torchsde)
- **Comprehensive evaluation**: FC, FCD, phFCD, phase-coherence FC, metastability, power spectrum, autocorrelation
- **Composable losses** via `CompositeLoss` with presets (`mse`, `combined`, `fc_phfcd_meta`, `full`, …)
- **Flexible data loading**: local `.mat` files, LSD pharmacological data, nilearn, OpenNeuro, DataLad, BIDS
- **Weights & Biases** integration for experiment tracking
- **GPU-accelerated** training and simulation
- **CI / Docs / PyPI** workflows ready out of the box

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
    noise_sigma=0.5,
    device=device,
)

initial_state = torch.randn(10, 68, dtype=torch.complex64, device=device)
with torch.no_grad():
    timeseries = model.forward(initial_state=initial_state, n_steps=200)
    fc_matrix  = model.compute_fc(timeseries)
```

## Documentation

| Page | Contents |
|------|----------|
| [Installation](getting-started/installation.md) | Requirements, PyPI/source install, optional extras, torchsde note |
| [First Training Run](tutorials/first-training-run.md) | End-to-end training walkthrough, CLI flags, dataset backends |
| [Metrics & Evaluation](tutorials/metrics-evaluation.md) | All 9 evaluation metrics, mathematical definitions, loss presets |
| [API Overview](api/index.md) | Module-by-module reference: models, metrics, training, dataset, utils |
| [Publishing](publishing.md) | TestPyPI / PyPI release checklist |
