# Neuroscience Control

`neuroscience-control` is a PyTorch framework for **whole-brain modeling** of resting-state fMRI BOLD signals. The repository centers on three main training workflows:

| Workflow / Model | Description |
|------------------|-------------|
| **Coupled Hopf** | Physics-based coupled oscillators at the supercritical Hopf bifurcation, informed by structural connectivity |
| **Hybrid Hopf** | Hopf oscillators with learnable complex coupling networks |
| **Neural SDE** | Data-driven neural networks parameterizing stochastic differential equations |

The codebase also exports an experimental `GNNHopfModel` class for node-wise neural coupling experiments.

The observed BOLD signal is the real part of the complex state, $s_i(t) = \Re(z_i(t))$.

## Key Facts

- The install name is `neuroscience-control`, but the current Python import namespace is `src`.
- All model families operate on **complex-valued** analytic signals.
- Training uses `examples/train_models.py`; post-training comparison and paper artefacts use `examples/postprocess.py`.
- Dataset backends include local `.mat`, LSD, nilearn, OpenNeuro, DataLad, and local BIDS derivatives.
- Evaluation covers FC, FCD, phFCD, phase-coherence FC, metastability, temporal correlation, power spectrum, and autocorrelation.

## Quick Start

```python
import torch
from src.models import CoupledHopfModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CoupledHopfModel(
    n_rois=68,
    initial_a=-0.02,
    initial_g=0.5,
    initial_kappa=0.1,
    noise_sigma=0.5,
    device=device,
)

initial_state = torch.randn(10, 68, dtype=torch.complex64, device=device)
with torch.no_grad():
    timeseries = model.forward(initial_state=initial_state, n_steps=200)
    fc_matrix = model.compute_fc(timeseries)
```

## Documentation

| Page | Contents |
|------|----------|
| [Installation](getting-started/installation.md) | Requirements, install commands, import namespace, dependency notes |
| [Project Structure](getting-started/project-structure.md) | Repository map for `src/`, `examples/`, `tests/`, `docs/`, and output folders |
| [First Training Run](tutorials/first-training-run.md) | Backprop, grid-search, and paper-suite entry points |
| [Metrics & Evaluation](tutorials/metrics-evaluation.md) | Metric modules, composite loss, and loader-level evaluation helpers |
| [API Overview](api/index.md) | Actual exported modules and common imports |
| [Publishing](publishing.md) | Preflight checks and release workflow |
