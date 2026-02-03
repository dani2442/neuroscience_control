"""Dataset module for loading and processing neuroscience data."""

from .data_loader import NeuroscienceDataset, load_mat_data
from .preprocessing import (
    WindowedDataset, 
    create_data_loaders,
    compute_omega_from_timeseries,
    compute_omega_uniform
)

__all__ = [
    "NeuroscienceDataset",
    "load_mat_data",
    "WindowedDataset",
    "create_data_loaders",
    "compute_omega_from_timeseries",
    "compute_omega_uniform",
]
