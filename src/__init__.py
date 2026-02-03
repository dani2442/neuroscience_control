"""
Neuroscience Control - Brain dynamics simulation and optimization.

This package provides tools for:
- Loading and processing neuroscience timeseries data
- Simulating brain dynamics with Coupled Hopf and Neural SDE models
- Training models via grid search and backpropagation
- Evaluating functional connectivity metrics
- Visualizing results and comparing models
"""

from . import dataset
from . import models
from . import metrics
from . import training
from . import utils

__version__ = "0.1.0"
__all__ = ["dataset", "models", "metrics", "training", "utils"]
