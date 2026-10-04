"""Preprocessing utilities for windowed training."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Sequence


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


def _resolve_excluded(
    dataset: "NeuroscienceDataset",
    exclude_subjects: Optional[Sequence[int]],
) -> set:
    """Timeseries indices that must stay out of *every* split.

    When the dataset groups timeseries by physical patient, excluding one
    timeseries excludes every timeseries of that patient: the flag exists so a
    held-out individual contributes nothing to the fit, and a second condition
    recorded from the same brain would reintroduce them.
    """
    if not exclude_subjects:
        return set()
    n = int(dataset.n_subjects)
    excluded = {int(i) for i in exclude_subjects}
    bad = sorted(i for i in excluded if not 0 <= i < n)
    if bad:
        raise ValueError(f"exclude_subjects out of range [0, {n - 1}]: {bad}")
    pids = getattr(dataset, "patient_ids", None)
    if pids is not None:
        pid_np = pids.cpu().numpy()
        drop = set(pid_np[sorted(excluded)].tolist())
        excluded = set(np.where(np.isin(pid_np, list(drop)))[0].tolist())
    return excluded


def compute_split_indices(
    dataset: "NeuroscienceDataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    exclude_subjects: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val/test split indices.

    When the dataset has ``patient_ids`` (e.g. LSD with 25 patients × 3
    conditions), the split is done at the patient level so that all conditions
    for the same physical patient stay in the same partition.  Otherwise falls
    back to a standard per-timeseries split.

    ``exclude_subjects`` drops those subjects from all three splits, so they
    influence neither the gradients (train) nor model selection (val). This is
    what makes a pretrained base model usable as a fine-tuning starting point in
    the personalization experiment: the base must never have seen the subjects
    it will later be adapted to, otherwise every adapted model already encodes
    every target and the own-vs-other contrast is suppressed by construction.
    Passing ``None`` or an empty sequence reproduces the original splits exactly.

    Returns numpy arrays of timeseries indices for train, val, and test.
    """
    rng = np.random.RandomState(seed)
    excluded = _resolve_excluded(dataset, exclude_subjects)

    if getattr(dataset, "patient_ids", None) is not None:
        patient_ids = dataset.patient_ids  # (n_subjects,) long tensor
        pid_np = patient_ids.cpu().numpy()
        kept = np.array([i for i in range(len(pid_np)) if i not in excluded], dtype=int)
        unique_patients = np.unique(pid_np[kept])
        n_patients = len(unique_patients)
        perm = rng.permutation(n_patients)

        n_train_p = max(1, int(train_ratio * n_patients))
        n_val_p = max(1, int(val_ratio * n_patients))
        train_patients = set(unique_patients[perm[:n_train_p]].tolist())
        val_patients = set(unique_patients[perm[n_train_p:n_train_p + n_val_p]].tolist())
        test_patients = set(unique_patients[perm[n_train_p + n_val_p:]].tolist())
        if not test_patients:
            test_patients = val_patients

        train_idx = np.where(np.isin(pid_np, list(train_patients)))[0]
        val_idx = np.where(np.isin(pid_np, list(val_patients)))[0]
        test_idx = np.where(np.isin(pid_np, list(test_patients)))[0]

        print(f"  Patient-level split: {len(train_patients)} train / {len(val_patients)} val / "
              f"{len(test_patients)} test patients  →  {len(train_idx)}/{len(val_idx)}/{len(test_idx)} timeseries")
    else:
        n_subjects = int(dataset.n_subjects)
        # Permute positions within the kept pool, then map back to subject
        # indices. With nothing excluded the pool is the identity, so this draws
        # exactly the permutation the un-filtered version did.
        pool = np.array([i for i in range(n_subjects) if i not in excluded], dtype=int)
        n_pool = len(pool)
        if n_pool < 2:
            raise ValueError(
                f"Only {n_pool} subject(s) left after excluding {len(excluded)} of "
                f"{n_subjects}; need at least 2 to form a split."
            )
        indices = pool[rng.permutation(n_pool)]

        n_train = max(1, int(train_ratio * n_pool))
        n_val = max(1, int(val_ratio * n_pool))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        if len(test_idx) == 0:
            test_idx = val_idx  # fallback

    if excluded:
        print(f"  Excluded {len(excluded)} subject(s) from every split: "
              f"{sorted(excluded)}")

    return train_idx, val_idx, test_idx


def select_train_subset(
    train_idx: np.ndarray,
    n_keep: Optional[int],
    seed: int,
) -> np.ndarray:
    """Keep ``n_keep`` subjects of the training pool, drawn with ``seed``.

    The subset is a prefix of one seeded permutation of the (sorted) pool, so
    subsets drawn with the same seed are nested: every smaller subset is
    contained in every larger one. ``None`` or ``n_keep >= len(pool)`` keeps the
    whole pool.
    """
    pool = np.sort(np.asarray(train_idx))
    if n_keep is None or n_keep >= len(pool):
        return pool
    if n_keep < 1:
        raise ValueError(f"train_subset must be >= 1, got {n_keep}")
    perm = np.random.RandomState(seed).permutation(len(pool))
    return np.sort(pool[perm[:n_keep]])


def resolve_split(
    dataset: "NeuroscienceDataset",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    exclude_subjects: Optional[Sequence[int]] = None,
    split_seed: Optional[int] = None,
    train_subset: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/val/test indices with the partition and the training subset decoupled.

    By default the fit ``seed`` also seeds the partition, so two seeds are two
    different partitions. ``split_seed`` fixes the partition independently of
    the fit seed, which holds validation and test subjects constant across
    fits. ``train_subset`` then keeps that many training subjects, drawn with
    the fit ``seed`` (see :func:`select_train_subset`). With both left at
    ``None`` this is exactly :func:`compute_split_indices`.
    """
    train_idx, val_idx, test_idx = compute_split_indices(
        dataset, train_ratio, val_ratio, seed if split_seed is None else split_seed,
        exclude_subjects=exclude_subjects,
    )
    if train_subset is not None:
        n_pool = len(train_idx)
        train_idx = select_train_subset(train_idx, train_subset, seed)
        print(f"  Training subset: {len(train_idx)}/{n_pool} subjects (subset seed={seed}); "
              f"val/test fixed at {len(val_idx)}/{len(test_idx)}")
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
    use_full_timeseries: bool = False,
    exclude_subjects: Optional[Sequence[int]] = None,
    split_seed: Optional[int] = None,
    train_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train/val/test_inter/test_intra loaders with random windowing.

    Uses :func:`resolve_split` for the split (patient-aware when
    ``dataset.patient_ids`` is set, and honouring ``exclude_subjects``,
    ``split_seed`` and ``train_subset``).
    """
    train_idx, val_idx, test_idx = resolve_split(
        dataset, train_ratio, val_ratio, seed, exclude_subjects=exclude_subjects,
        split_seed=split_seed, train_subset=train_subset,
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

    train_fc = dataset.fc_matrices[train_idx]
    train_ctrl = ctrl[train_idx] if ctrl is not None else None

    T_split = dataset.n_timepoints // 2
    if use_full_timeseries:
        # Use the entire timeseries for training; intra-patient test is not applicable.
        train_loader = _loader_from_tensors(
            dataset.timeseries[train_idx], train_fc, train_ctrl, n_windows_per_epoch,
        )
        intra_loader = _loader_from_tensors(
            dataset.timeseries[train_idx, :, T_split:], train_fc, train_ctrl, n_test_win, shuffle=False,
        )
    else:
        train_loader = _loader_from_tensors(
            dataset.timeseries[train_idx, :, :T_split], train_fc, train_ctrl, n_windows_per_epoch,
        )
        intra_loader = _loader_from_tensors(
            dataset.timeseries[train_idx, :, T_split:], train_fc, train_ctrl, n_test_win, shuffle=False,
        )

    return (
        train_loader,
        _loader(val_idx, n_val_win, shuffle=False),
        _loader(test_idx, n_test_win, shuffle=False),  # inter-patient
        intra_loader,                                   # intra-patient (None if use_full_timeseries)
    )
