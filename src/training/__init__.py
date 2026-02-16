"""Training module for neuroscience models."""

from .trainer import Trainer
from .grid_search import GridSearch, grid_search_hopf
from .fine_tuning import FineTuner
from .backprop import create_windowed_loaders, run_backprop_training
from .config import TrainingConfig, HopfConfig, HybridHopfConfig, NeuralSDEConfig
from .losses import CompositeLoss, build_loss, LOSS_REGISTRY

__all__ = [
    "Trainer",
    "GridSearch",
    "grid_search_hopf",
    "FineTuner",
    "create_windowed_loaders",
    "run_backprop_training",
    "TrainingConfig",
    "HopfConfig",
    "HybridHopfConfig",
    "NeuralSDEConfig",
    "CompositeLoss",
    "build_loss",
    "LOSS_REGISTRY",
]
