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
        control: Optional[torch.Tensor] = None,
    ):
        self.timeseries = timeseries.to(device)    # (n_subjects, n_rois, T), complex
        self.fc_matrices = fc_matrices.to(device)  # (n_subjects, n_rois, n_rois), real
        self.window_size = window_size
        self.n_windows = n_windows
        self.n_subjects = timeseries.shape[0]
        self.max_start = timeseries.shape[2] - window_size
        # control: (n_subjects, n_control_dims) or None
        if control is not None:
            self.control = control.to(device)
        else:
            self.control = torch.zeros(self.n_subjects, 0, device=device)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        subj = torch.randint(0, self.n_subjects, (1,)).item()
        start = torch.randint(0, self.max_start + 1, (1,)).item()
        window = self.timeseries[subj, :, start:start + self.window_size]
        return window, self.fc_matrices[subj], subj, self.control[subj]


def compute_split_indices(
    dataset: "NeuroscienceDataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val/test split indices.

    When the dataset has ``patient_ids`` (e.g. LSD with 25 patients × 3
    conditions), the split is done at the patient level so that all conditions
    for the same physical patient stay in the same partition.  Otherwise falls
    back to a standard per-timeseries split.

    Returns numpy arrays of timeseries indices for train, val, and test.
    """
    rng = np.random.RandomState(seed)

    if getattr(dataset, "patient_ids", None) is not None:
        patient_ids = dataset.patient_ids  # (n_subjects,) long tensor
        unique_patients = torch.unique(patient_ids).cpu().numpy()
        n_patients = len(unique_patients)
        perm = rng.permutation(n_patients)

        n_train_p = max(1, int(train_ratio * n_patients))
        n_val_p = max(1, int(val_ratio * n_patients))
        train_patients = set(unique_patients[perm[:n_train_p]].tolist())
        val_patients = set(unique_patients[perm[n_train_p:n_train_p + n_val_p]].tolist())
        test_patients = set(unique_patients[perm[n_train_p + n_val_p:]].tolist())
        if not test_patients:
            test_patients = val_patients

        pid_np = patient_ids.cpu().numpy()
        train_idx = np.where(np.isin(pid_np, list(train_patients)))[0]
        val_idx = np.where(np.isin(pid_np, list(val_patients)))[0]
        test_idx = np.where(np.isin(pid_np, list(test_patients)))[0]

        print(f"  Patient-level split: {len(train_patients)} train / {len(val_patients)} val / "
              f"{len(test_patients)} test patients  →  {len(train_idx)}/{len(val_idx)}/{len(test_idx)} timeseries")
    else:
        n_subjects = dataset.n_subjects
        indices = rng.permutation(n_subjects)

        n_train = max(1, int(train_ratio * n_subjects))
        n_val = max(1, int(val_ratio * n_subjects))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        if len(test_idx) == 0:
            test_idx = val_idx  # fallback

    return train_idx, val_idx, test_idx


def create_data_loaders(
    dataset: "NeuroscienceDataset",
    window_size: int,
    batch_size: int = 32,
    n_windows_per_epoch: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create train/val/test_inter/test_intra loaders with random windowing.

    Uses :func:`compute_split_indices` for the split (patient-aware when
    ``dataset.patient_ids`` is set).

    Also creates a temporal intra-patient split: first half of training-subject
    timeseries → training, second half → intra-patient test loader.
    """
    train_idx, val_idx, test_idx = compute_split_indices(
        dataset, train_ratio, val_ratio, seed,
    )

    # Extract per-subject control tensor (may be empty dim-1 for uncontrolled)
    ctrl = getattr(dataset, "control", None)

    def _loader(subj_idx, n_win, shuffle=True):
        subj_ctrl = ctrl[subj_idx] if ctrl is not None else None
        ds = RandomWindowDataset(
            dataset.timeseries[subj_idx],
            dataset.fc_matrices[subj_idx],
            window_size, n_win, device,
            control=subj_ctrl,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def _loader_from_tensors(ts, fc, subj_ctrl, n_win, shuffle=True):
        ds = RandomWindowDataset(ts, fc, window_size, n_win, device, control=subj_ctrl)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    n_val_win = max(batch_size, n_windows_per_epoch // 4)
    n_test_win = max(batch_size, n_windows_per_epoch // 4)

    # Temporal split for intra-patient test set
    T_split = dataset.n_timepoints // 2
    train_ts  = dataset.timeseries[train_idx, :, :T_split]
    intra_ts  = dataset.timeseries[train_idx, :, T_split:]
    train_fc  = dataset.fc_matrices[train_idx]
    if ctrl is not None:
        # Control is per-subject (n_subjects, n_control_dims), not time-varying.
        # Both temporal halves share the same per-subject control.
        train_ctrl = ctrl[train_idx]
        intra_ctrl = ctrl[train_idx]
    else:
        train_ctrl = intra_ctrl = None

    return (
        _loader_from_tensors(train_ts, train_fc, train_ctrl, n_windows_per_epoch),
        _loader(val_idx, n_val_win, shuffle=False),
        _loader(test_idx, n_test_win, shuffle=False),                                      # inter-patient
        _loader_from_tensors(intra_ts, train_fc, intra_ctrl, n_test_win, shuffle=False),   # intra-patient
    )
