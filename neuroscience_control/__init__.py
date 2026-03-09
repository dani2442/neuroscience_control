"""Public package namespace for ``neuroscience-control``.

This package currently re-exports modules from the legacy ``src`` namespace
for backwards compatibility.
"""

from importlib import import_module
import sys

from src import dataset, metrics, models, training, utils

for _name in ("dataset", "metrics", "models", "training", "utils"):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"src.{_name}")

__version__ = import_module("src").__version__
__all__ = ["dataset", "metrics", "models", "training", "utils"]
