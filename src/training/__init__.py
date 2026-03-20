"""Training module for neuroscience models."""

from .trainer import Trainer
from .grid_search import GridSearch, grid_search_hopf
from .backprop import create_windowed_loaders, load_dataset, run_backprop_training
from .config import TrainingConfig, HopfConfig, HybridHopfConfig, GNNHopfConfig, NeuralSDEConfig
from .losses import CompositeLoss

__all__ = [
    "Trainer",
    "GridSearch",
    "grid_search_hopf",
    "create_windowed_loaders",
    "load_dataset",
    "run_backprop_training",
    "TrainingConfig",
    "HopfConfig",
    "HybridHopfConfig",
    "GNNHopfConfig",
    "NeuralSDEConfig",
    "CompositeLoss",
]
