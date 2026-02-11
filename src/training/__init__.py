"""Training module for neuroscience models."""

from .trainer import Trainer
from .grid_search import GridSearch, grid_search_hopf
from .fine_tuning import FineTuner
from .config import TrainingConfig, HopfConfig, NeuralSDEConfig
from .losses import CompositeLoss, build_loss, LOSS_REGISTRY

__all__ = [
    "Trainer",
    "GridSearch",
    "grid_search_hopf",
    "FineTuner",
    "TrainingConfig",
    "HopfConfig",
    "NeuralSDEConfig",
    "CompositeLoss",
    "build_loss",
    "LOSS_REGISTRY",
]
