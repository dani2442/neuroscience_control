"""Data loading utilities for neuroscience timeseries data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from typing import Dict, Tuple, Optional


def load_mat_data(filepath: str) -> Dict[str, np.ndarray]:
    """Load .mat file. Returns dict with FC_all, FC_mean, timeseries_all."""
    data = loadmat(filepath)
    return {
        'FC_all': data['FC_all'],
        'FC_mean': data['FC_mean'],
        'timeseries_all': data['timeseries_all'],
    }


def fft_bandpass_3d(x: torch.Tensor, dt: float, f_lo: float, f_hi: float) -> torch.Tensor:
    """FFT brick-wall bandpass on last dimension. x: (..., T) real."""
    T = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=dt).to(device=x.device, dtype=x.dtype)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    for _ in range(X.ndim - 1):
        mask = mask.unsqueeze(0)
    return torch.fft.irfft(X * mask, n=T, dim=-1)


def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """Analytic signal via Hilbert transform on last dimension. x: (..., T) real → complex."""
    T = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    h = torch.zeros(T, device=x.device, dtype=X.dtype)
    if T % 2 == 0:
        h[0] = 1.0; h[T // 2] = 1.0; h[1:T // 2] = 2.0  # noqa: E702
    else:
        h[0] = 1.0; h[1:(T + 1) // 2] = 2.0  # noqa: E702
    for _ in range(X.ndim - 1):
        h = h.unsqueeze(0)
    return torch.fft.ifft(X * h, dim=-1)


class NeuroscienceDataset(Dataset):
    """PyTorch Dataset for neuroscience timeseries data.

    Stores **complex** timeseries (analytic signal via Hilbert transform).
    FC matrices remain real (empirical targets from .mat file).

    Attributes:
        timeseries: Complex tensor (n_subjects, n_rois, n_timepoints)
        fc_matrices: Real tensor (n_subjects, n_rois, n_rois)
        fc_mean: Real tensor (n_rois, n_rois)
        ts: Time array (n_timepoints,)
        dt: TR in seconds
    """

    def __init__(
        self,
        filepath: str,
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = False,
        denoise_f_lo: float = 0.01,
        denoise_f_hi: float = 0.1,
    ):
        self.device = device
        self.dt = dt

        data = load_mat_data(filepath)

        if max_subjects is not None:
            n_total = data['timeseries_all'].shape[2]
            if max_subjects <= 0 or max_subjects > n_total:
                raise ValueError(f"max_subjects must be in [1, {n_total}], got {max_subjects}")
            data['timeseries_all'] = data['timeseries_all'][:, :, :max_subjects]
            data['FC_all'] = data['FC_all'][:, :, :max_subjects]
            data['FC_mean'] = data['FC_all'].mean(axis=2)

        # (ROIs, timepoints, subjects) → (subjects, ROIs, timepoints)
        timeseries = torch.tensor(
            data['timeseries_all'].transpose(2, 0, 1),
            dtype=torch.float32, device=device,
        )
        self.fc_matrices = torch.tensor(
            data['FC_all'].transpose(2, 0, 1),
            dtype=torch.float32, device=device,
        )
        self.fc_mean = torch.tensor(
            data['FC_mean'], dtype=torch.float32, device=device,
        )

        # Z-score normalize
        if normalize:
            mean = timeseries.mean(dim=2, keepdim=True)
            std = timeseries.std(dim=2, keepdim=True) + 1e-8
            timeseries = (timeseries - mean) / std

        # Optional Fourier denoising (bandpass)
        if fourier_denoise:
            timeseries = fft_bandpass_3d(timeseries, dt, denoise_f_lo, denoise_f_hi)

        # Hilbert transform → complex analytic signal
        self.timeseries = hilbert_transform(timeseries)

        self.n_subjects, self.n_rois, self.n_timepoints = self.timeseries.shape
        self.ts = torch.linspace(
            0, (self.n_timepoints - 1) * dt, self.n_timepoints, device=device,
        )

    def __len__(self) -> int:
        return self.n_subjects

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.timeseries[idx], self.fc_matrices[idx]

    def get_all_timeseries(self) -> torch.Tensor:
        return self.timeseries

    def get_fc_mean(self) -> torch.Tensor:
        return self.fc_mean
