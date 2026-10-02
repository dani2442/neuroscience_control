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
    LSD_N_CONTROL_DIMS,
)
from .preprocessing import (
    RandomWindowDataset,
    compute_split_indices,
    create_data_loaders,
    compute_omega_from_timeseries,
)
from .priors import SplitPriors, estimate_split_priors, training_segment_stop

__all__ = [
    "NeuroscienceDataset",
    "load_dataset",
    "load_mat_data",
    "load_lsd_data",
    "compute_fc_from_timeseries",
    "fft_bandpass_3d",
    "hilbert_transform",
    "LSD_CONDITION_MAP",
    "LSD_N_CONTROL_DIMS",
    "RandomWindowDataset",
    "compute_split_indices",
    "create_data_loaders",
    "compute_omega_from_timeseries",
    "SplitPriors",
    "estimate_split_priors",
    "training_segment_stop",
]
