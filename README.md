# Neuroscience Control

[![CI](https://github.com/dani2442/neuroscience_control/actions/workflows/ci.yml/badge.svg)](https://github.com/dani2442/neuroscience_control/actions/workflows/ci.yml)
[![Docs](https://github.com/dani2442/neuroscience_control/actions/workflows/docs.yml/badge.svg)](https://github.com/dani2442/neuroscience_control/actions/workflows/docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/neuroscience-control.svg)](https://pypi.org/project/neuroscience-control/)
[![Python versions](https://img.shields.io/pypi/pyversions/neuroscience-control.svg)](https://pypi.org/project/neuroscience-control/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/dani2442/neuroscience_control/branch/main/graph/badge.svg)](https://app.codecov.io/gh/dani2442/neuroscience_control)

Whole-brain resting-state fMRI simulation and control using:

- Coupled Hopf oscillators
- Hybrid Hopf (learnable coupling)
- Neural SDE models

Built with PyTorch, complex-valued dynamics, and domain metrics for FC/FCD/phFCD/metastability.

## Features

- Unified training entry point (`examples/train_models.py`)
- Grid search and backprop training pipelines
- Evaluation metrics for FC, dynamics, and timeseries fidelity
- Experiment logging integration (Weights & Biases)
- Package + CI + docs + release workflows ready for TestPyPI/PyPI

## Installation

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

### Verify

```bash
python -c "import neuroscience_control as nc; print(nc.__version__)"
```

## Quick Start

```python
import torch
from neuroscience_control.models import CoupledHopfModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CoupledHopfModel(n_rois=68, initial_a=-0.02, initial_g=0.5, device=device)

initial_state = torch.randn(10, 68, dtype=torch.complex64, device=device)
with torch.no_grad():
    ts = model.forward(initial_state=initial_state, n_steps=200)
    fc = model.compute_fc(ts)
```

## Training Commands

```bash
# Hopf grid search
python examples/train_models.py hopf-grid

# Backprop training
python examples/train_models.py backprop --model hopf
python examples/train_models.py backprop --model hybrid_hopf
python examples/train_models.py backprop --model nsde

# Full paper-style pipeline
python examples/train_models.py paper --output-json results/paper_metrics.json
```

## Import Path

Use the public namespace in new code:

```python
from neuroscience_control.models import NeuralSDE
```

Legacy imports are still supported:

```python
from src.models import NeuralSDE
```

## Documentation Website

Docs are built with MkDocs Material.

- Local preview:

```bash
uv sync --group docs
uv run mkdocs serve
```

- Build static site:

```bash
uv run mkdocs build --strict
```

Main pages:

- [Getting Started](docs/getting-started/installation.md)
- [First Training Run Tutorial](docs/tutorials/first-training-run.md)
- [Publishing Guide](docs/publishing.md)

## Development and Test Coverage

```bash
uv sync --group dev
uv run pytest --cov=src --cov=neuroscience_control --cov-report=term-missing
```

## Release and Publishing

Automated workflows included:

- `CI` (`.github/workflows/ci.yml`): tests + coverage
- `Docs` (`.github/workflows/docs.yml`): docs build + GitHub Pages deploy
- `Publish` (`.github/workflows/publish.yml`):
  - manual dispatch -> TestPyPI
  - GitHub release -> PyPI

Before publishing:

```bash
uv run python -m build
uv run twine check dist/*
```

## Project Layout

```text
neuroscience_control/
├── src/                         # Core implementation (legacy namespace)
├── neuroscience_control/        # Public package namespace
├── examples/                    # Training and evaluation scripts
├── tests/                       # Unit tests
├── docs/                        # Documentation site
├── mkdocs.yml
└── pyproject.toml
```

## Citation

If you use this project in academic work, cite this repository and the associated paper in `paper/`.
