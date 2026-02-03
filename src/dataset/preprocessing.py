"""Preprocessing utilities for windowed training."""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional
import numpy as np


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
