"""Preprocessing utilities for windowed training."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


def compute_omega_from_timeseries(
    timeseries: torch.Tensor,
    dt: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    method: str = "peak",
) -> torch.Tensor:
    """Compute intrinsic angular frequencies per ROI from timeseries via FFT.

    Args:
        timeseries: (n_subjects, n_rois, n_timepoints) or (n_rois, n_timepoints).
                    Accepts complex tensors (uses real part).
        dt: TR in seconds.
        f_lo, f_hi: Frequency band of interest (Hz).
        method: 'peak' or 'weighted'.

    Returns:
        omega: (n_rois,) angular frequencies (rad/s).
    """
    ts = timeseries.real if torch.is_complex(timeseries) else timeseries
    if ts.dim() == 2:
        ts = ts.unsqueeze(0)

    n_subjects, n_rois, n_timepoints = ts.shape
    power = torch.abs(torch.fft.rfft(ts, dim=2)) ** 2
    freqs = torch.fft.rfftfreq(n_timepoints, d=dt).to(ts.device)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        mask = freqs > 0

    power_band = power[:, :, mask]
    freqs_band = freqs[mask]

    if method == "peak":
        omega_hz = freqs_band[power_band.argmax(dim=2)].mean(dim=0)
    elif method == "weighted":
        w = power_band / (power_band.sum(dim=2, keepdim=True) + 1e-10)
        omega_hz = (w * freqs_band).sum(dim=2).mean(dim=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return 2 * np.pi * omega_hz


def compute_omega_uniform(
    n_rois: int, f_lo: float = 0.04, f_hi: float = 0.07, device: str = "cpu",
) -> torch.Tensor:
    """Uniformly spaced angular frequencies across a band."""
    return 2 * np.pi * torch.linspace(f_lo, f_hi, n_rois, device=device)


class RandomWindowDataset(Dataset):
    """Dataset that randomly samples windows from timeseries each call.

    Each ``__getitem__`` picks a random subject and a random start position,
    so every epoch sees different window positions — better for reducing
    overfitting compared to pre-computed sliding windows.
    """

    def __init__(
        self,
        timeseries: torch.Tensor,
        fc_matrices: torch.Tensor,
        window_size: int,
        n_windows: int,
        device: str = "cpu",
    ):
        self.timeseries = timeseries.to(device)    # (n_subjects, n_rois, T), complex
        self.fc_matrices = fc_matrices.to(device)  # (n_subjects, n_rois, n_rois), real
        self.window_size = window_size
        self.n_windows = n_windows
        self.n_subjects = timeseries.shape[0]
        self.max_start = timeseries.shape[2] - window_size

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        subj = torch.randint(0, self.n_subjects, (1,)).item()
        start = torch.randint(0, self.max_start + 1, (1,)).item()
        window = self.timeseries[subj, :, start:start + self.window_size]
        return window, self.fc_matrices[subj], subj


def create_data_loaders(
    dataset: "NeuroscienceDataset",
    window_size: int,
    batch_size: int = 32,
    n_windows_per_epoch: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with random windowing and subject-level split."""
    rng = np.random.RandomState(seed)
    n_subjects = dataset.n_subjects
    indices = rng.permutation(n_subjects)

    n_train = max(1, int(train_ratio * n_subjects))
    n_val = max(1, int(val_ratio * n_subjects))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if len(test_idx) == 0:
        test_idx = val_idx  # fallback

    def _loader(subj_idx, n_win, shuffle=True):
        ds = RandomWindowDataset(
            dataset.timeseries[subj_idx],
            dataset.fc_matrices[subj_idx],
            window_size, n_win, device,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    n_val_win = max(batch_size, n_windows_per_epoch // 4)
    n_test_win = max(batch_size, n_windows_per_epoch // 4)

    return (
        _loader(train_idx, n_windows_per_epoch),
        _loader(val_idx, n_val_win, shuffle=False),
        _loader(test_idx, n_test_win, shuffle=False),
    )
