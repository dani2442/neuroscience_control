# Installation

## Requirements

- **Python ≥ 3.13**
- **PyTorch ≥ 2.2**
- Linux / macOS / Windows
- Optional CUDA-enabled GPU for training speed

## Install from source

The project is installed from the repository with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DCN-FAU-AvH/neuroscience-modeling.git
cd neuroscience-modeling
uv sync
```

`uv sync` creates `.venv/` with the pinned dependencies. Add `--group dev` for the
test and lint tools and `--group docs` for the documentation site.

## Optional dataset integrations

`nilearn` is included by default. For OpenNeuro and DataLad dataset download support,
install the `datasets` group:

```bash
uv sync --group datasets
```

## Verify installation

```bash
uv run python -c "import src; print(src.__version__)"
uv run pytest
```

## Import namespace

The importable package is `src`:

```python
from src.models import CoupledHopfModel, NeuralSDE
from src.metrics import FCCorrelation, PhFCD
from src.training import Trainer, HopfConfig
```

## Dependency notes

!!! note "Complex-valued `torchsde` fork"
    This project depends on a [fork of `torchsde`](https://github.com/dani2442/torchsde)
    that adds complex Brownian motion support. When installing with `uv`, this
    is resolved automatically via the `[tool.uv.sources]` override in `pyproject.toml`.
    If you install with plain `pip`, it will use the PyPI `torchsde` release (which
    lacks complex-valued support).
