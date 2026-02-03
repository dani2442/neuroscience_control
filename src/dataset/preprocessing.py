"""Preprocessing utilities for windowed training."""

import torch
import torch.fft
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional, Union
import numpy as np


def compute_omega_from_timeseries(
    timeseries: torch.Tensor,
    dt: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    method: str = "peak"
) -> torch.Tensor:
    """
    Compute intrinsic frequencies (omega) for each ROI from timeseries data.
    
    Uses FFT to find the dominant frequency in the specified band for each ROI.
    
    Args:
        timeseries: Tensor of shape (n_subjects, n_rois, n_timepoints) or (n_rois, n_timepoints)
        dt: Time step (TR) in seconds
        f_lo: Low frequency cutoff in Hz
        f_hi: High frequency cutoff in Hz
        method: 'peak' for peak frequency, 'weighted' for power-weighted mean
        
    Returns:
        omega: Tensor of shape (n_rois,) with angular frequencies (rad/s)
    """
    # Handle 2D input
    if timeseries.dim() == 2:
        timeseries = timeseries.unsqueeze(0)
    
    n_subjects, n_rois, n_timepoints = timeseries.shape
    device = timeseries.device
    
    # Compute FFT for all subjects and ROIs
    fft_result = torch.fft.rfft(timeseries, dim=2)
    power_spectrum = torch.abs(fft_result) ** 2
    
    # Frequency axis
    freqs = torch.fft.rfftfreq(n_timepoints, d=dt).to(device)
    
    # Create mask for frequency band of interest
    freq_mask = (freqs >= f_lo) & (freqs <= f_hi)
    
    if not freq_mask.any():
        # Fallback: use all positive frequencies if band is empty
        freq_mask = freqs > 0
    
    # Extract power in band
    power_in_band = power_spectrum[:, :, freq_mask]  # (n_subjects, n_rois, n_freqs_in_band)
    freqs_in_band = freqs[freq_mask]
    
    if method == "peak":
        # Find peak frequency for each subject and ROI
        peak_indices = power_in_band.argmax(dim=2)  # (n_subjects, n_rois)
        peak_freqs = freqs_in_band[peak_indices]  # (n_subjects, n_rois)
        
        # Average across subjects
        omega_hz = peak_freqs.mean(dim=0)  # (n_rois,)
    
    elif method == "weighted":
        # Power-weighted mean frequency
        power_sum = power_in_band.sum(dim=2, keepdim=True) + 1e-10
        weights = power_in_band / power_sum
        weighted_freqs = (weights * freqs_in_band.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        
        # Average across subjects
        omega_hz = weighted_freqs.mean(dim=0)  # (n_rois,)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'peak' or 'weighted'.")
    
    # Convert to angular frequency (rad/s)
    omega = 2 * np.pi * omega_hz
    
    return omega


def compute_omega_uniform(
    n_rois: int,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute uniformly spaced omega values across a frequency band.
    
    This is a simple alternative to data-driven omega estimation.
    
    Args:
        n_rois: Number of brain regions
        f_lo: Low frequency in Hz
        f_hi: High frequency in Hz
        device: Device for tensor
        
    Returns:
        omega: Tensor of shape (n_rois,) with angular frequencies (rad/s)
    """
    omega_hz = torch.linspace(f_lo, f_hi, n_rois, device=device)
    omega = 2 * np.pi * omega_hz
    return omega


class WindowedDataset(Dataset):
    """
    Dataset that provides windowed segments of timeseries data.
    
    This is used for mini-batch training with a sliding window approach.
    """
    
    def __init__(
        self,
        timeseries: torch.Tensor,
        fc_matrices: torch.Tensor,
        window_size: int,
        stride: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize windowed dataset.
        
        Args:
            timeseries: Tensor of shape (n_subjects, n_rois, n_timepoints)
            fc_matrices: Tensor of shape (n_subjects, n_rois, n_rois)
            window_size: Size of the sliding window
            stride: Step size between windows
            device: Device to store tensors on
        """
        self.device = device
        self.window_size = window_size
        self.stride = stride
        
        self.timeseries = timeseries.to(device)
        self.fc_matrices = fc_matrices.to(device)
        
        n_subjects, n_rois, n_timepoints = timeseries.shape
        
        # Compute number of windows per subject
        self.n_windows_per_subject = (n_timepoints - window_size) // stride + 1
        
        # Pre-compute all windows
        self.windows = []
        self.fc_targets = []
        self.subject_ids = []
        
        for subj_idx in range(n_subjects):
            for win_idx in range(self.n_windows_per_subject):
                start = win_idx * stride
                end = start + window_size
                
                window = self.timeseries[subj_idx, :, start:end]
                self.windows.append(window)
                self.fc_targets.append(self.fc_matrices[subj_idx])
                self.subject_ids.append(subj_idx)
        
        self.windows = torch.stack(self.windows)
        self.fc_targets = torch.stack(self.fc_targets)
        self.subject_ids = torch.tensor(self.subject_ids, device=device)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a windowed sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (window, fc_target, subject_id)
        """
        return self.windows[idx], self.fc_targets[idx], self.subject_ids[idx]


def create_data_loaders(
    dataset: "NeuroscienceDataset",
    window_size: int,
    stride: int = 1,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str = "cpu"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset: NeuroscienceDataset instance
        window_size: Window size for mini-batch training
        stride: Stride for windowing
        batch_size: Batch size for data loaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        device: Device to use
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create windowed dataset
    windowed = WindowedDataset(
        dataset.timeseries,
        dataset.fc_matrices,
        window_size=window_size,
        stride=stride,
        device=device
    )
    
    # Split into train/val/test
    n_samples = len(windowed)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        windowed,
        [n_train, n_val, n_test],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def split_subjects(
    dataset: "NeuroscienceDataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split subjects into train/val/test sets.
    
    Args:
        dataset: NeuroscienceDataset instance
        train_ratio: Fraction of subjects for training
        val_ratio: Fraction of subjects for validation
        seed: Random seed
        
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    np.random.seed(seed)
    
    n_subjects = dataset.n_subjects
    indices = np.random.permutation(n_subjects)
    
    n_train = int(train_ratio * n_subjects)
    n_val = int(val_ratio * n_subjects)
    
    train_subjects = indices[:n_train].tolist()
    val_subjects = indices[n_train:n_train + n_val].tolist()
    test_subjects = indices[n_train + n_val:].tolist()
    
    return train_subjects, val_subjects, test_subjects
