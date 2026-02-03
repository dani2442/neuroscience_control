"""Data loading utilities for neuroscience timeseries data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


def load_mat_data(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load .mat file containing neuroscience data.
    
    Args:
        filepath: Path to the .mat file
        
    Returns:
        Dictionary containing:
        - FC_all: Functional connectivity (ROIs x ROIs x subjects)
        - FC_mean: Mean functional connectivity (ROIs x ROIs)
        - timeseries_all: Timeseries data (ROIs x timepoints x subjects)
    """
    data = loadmat(filepath)
    
    # Extract relevant fields
    result = {
        'FC_all': data['FC_all'],
        'FC_mean': data['FC_mean'],
        'timeseries_all': data['timeseries_all'],
    }
    
    return result


class NeuroscienceDataset(Dataset):
    """
    PyTorch Dataset for neuroscience timeseries data.
    
    Attributes:
        timeseries: Tensor of shape (n_subjects, n_rois, n_timepoints)
        fc_matrices: Tensor of shape (n_subjects, n_rois, n_rois)
        fc_mean: Tensor of shape (n_rois, n_rois)
        ts: Tensor of shape (n_timepoints,) - time array
        dt: Time step (TR) in seconds
    """
    
    def __init__(
        self,
        filepath: str,
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72
    ):
        """
        Initialize the dataset.
        
        Args:
            filepath: Path to the .mat file
            normalize: Whether to z-score normalize timeseries
            device: Device to store tensors on
            max_subjects: Optional limit on number of subjects (first N)
            dt: Time step (TR) in seconds (default 0.72)
        """
        self.device = device
        self.normalize = normalize
        self.dt = dt
        
        # Load data
        data = load_mat_data(filepath)
        
        # Optionally limit number of subjects
        if max_subjects is not None:
            n_subjects_total = data['timeseries_all'].shape[2]
            if max_subjects <= 0 or max_subjects > n_subjects_total:
                raise ValueError(f"max_subjects must be in [1, {n_subjects_total}], got {max_subjects}")
            data['timeseries_all'] = data['timeseries_all'][:, :, :max_subjects]
            data['FC_all'] = data['FC_all'][:, :, :max_subjects]
            data['FC_mean'] = data['FC_all'].mean(axis=2)

        # Convert to tensors and rearrange dimensions
        # Original: (ROIs, timepoints, subjects) -> (subjects, ROIs, timepoints)
        self.timeseries = torch.tensor(
            data['timeseries_all'].transpose(2, 0, 1),
            dtype=torch.float32,
            device=device
        )
        
        # FC: (ROIs, ROIs, subjects) -> (subjects, ROIs, ROIs)
        self.fc_matrices = torch.tensor(
            data['FC_all'].transpose(2, 0, 1),
            dtype=torch.float32,
            device=device
        )
        
        self.fc_mean = torch.tensor(
            data['FC_mean'],
            dtype=torch.float32,
            device=device
        )
        
        # Normalize timeseries if requested
        if normalize:
            self._normalize_timeseries()
        
        self.n_subjects = self.timeseries.shape[0]
        self.n_rois = self.timeseries.shape[1]
        self.n_timepoints = self.timeseries.shape[2]
        
        # Create time array
        self.ts = torch.linspace(
            0, 
            (self.n_timepoints - 1) * self.dt, 
            self.n_timepoints, 
            device=device
        )
    
    def _normalize_timeseries(self):
        """Z-score normalize timeseries for each subject and ROI."""
        mean = self.timeseries.mean(dim=2, keepdim=True)
        std = self.timeseries.std(dim=2, keepdim=True) + 1e-8
        self.timeseries = (self.timeseries - mean) / std
    
    def __len__(self) -> int:
        return self.n_subjects
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single subject's data.
        
        Args:
            idx: Subject index
            
        Returns:
            Tuple of (timeseries, fc_matrix)
        """
        return self.timeseries[idx], self.fc_matrices[idx]
    
    def get_all_timeseries(self) -> torch.Tensor:
        """Return all timeseries data."""
        return self.timeseries
    
    def get_fc_mean(self) -> torch.Tensor:
        """Return mean functional connectivity matrix."""
        return self.fc_mean
    
    def compute_fc(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Compute functional connectivity from timeseries.
        
        Args:
            timeseries: Tensor of shape (batch, n_rois, n_timepoints)
            
        Returns:
            FC matrices of shape (batch, n_rois, n_rois)
        """
        # Center the timeseries
        ts_centered = timeseries - timeseries.mean(dim=2, keepdim=True)
        
        # Compute correlation
        std = ts_centered.std(dim=2, keepdim=True) + 1e-8
        ts_normalized = ts_centered / std
        
        # Batch matrix multiplication for correlation
        n_timepoints = timeseries.shape[2]
        fc = torch.bmm(ts_normalized, ts_normalized.transpose(1, 2)) / n_timepoints
        
        return fc
