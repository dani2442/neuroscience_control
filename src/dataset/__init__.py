"""Dataset module for loading and processing neuroscience data."""

from .data_loader import (
    NeuroscienceDataset,
    load_dataset,
    load_mat_data,
    load_lsd_data,
    compute_fc_from_timeseries,
    fft_bandpass_3d,
    hilbert_transform,
    LSD_CONDITION_MAP,
)
from .preprocessing import (
    RandomWindowDataset,
    create_data_loaders,
    compute_omega_from_timeseries,
)

__all__ = [
    "NeuroscienceDataset",
    "load_dataset",
    "load_mat_data",
    "load_lsd_data",
    "compute_fc_from_timeseries",
    "fft_bandpass_3d",
    "hilbert_transform",
    "LSD_CONDITION_MAP",
    "RandomWindowDataset",
    "create_data_loaders",
    "compute_omega_from_timeseries",
]
