# Installation

## Requirements

- **Python ≥ 3.13**
- **PyTorch ≥ 2.2**
- Linux / macOS / Windows
- Optional CUDA-enabled GPU for training speed

## Install from PyPI

```bash
pip install neuroscience-control
```

Or with `uv`:

```bash
uv add neuroscience-control
```

## Install from source

```bash
git clone https://github.com/dani2442/neuroscience_control.git
cd neuroscience_control
uv sync
```

## Optional dataset integrations

`nilearn` is included by default. For OpenNeuro and DataLad dataset download support, install the `datasets` extra:

```bash
pip install "neuroscience-control[datasets]"
# or
uv sync --group datasets
```

## Verify installation

```bash
python -c "import neuroscience_control as nc; print(nc.__version__)"
```

## Dependency notes

!!! note "Complex-valued `torchsde` fork"
    This project depends on a [fork of `torchsde`](https://github.com/dani2442/torchsde)
    that adds complex Brownian motion support. When installing from source with `uv`, this
    is resolved automatically via the `[tool.uv.sources]` override in `pyproject.toml`.
    If you install with plain `pip`, it will use the PyPI `torchsde` release (which may
    lack complex-valued support).
