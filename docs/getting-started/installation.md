# Installation

## Requirements

- Python 3.13+
- Linux/macOS/Windows
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

## Verify installation

```bash
python -c "import neuroscience_control as nc; print(nc.__version__)"
```

## Dependency notes

`uv` is configured to use your `torchsde` fork through `tool.uv.sources`.
If you install with plain `pip`, it will use the PyPI `torchsde` release.
