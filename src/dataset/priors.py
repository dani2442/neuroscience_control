"""Training-only model priors: the functional connectome and intrinsic frequencies.

The functional connectome C0 is the mean Pearson FC over the training segments
of the training subjects, and omega is the per-ROI spectral peak over the same
segments.  Both are computed once from the split and then kept fixed, so no
held-out subject (val / test_inter) and no held-out segment (test_intra)
enters model construction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .data_loader import compute_fc_from_timeseries
from .preprocessing import compute_omega_from_timeseries

if TYPE_CHECKING:
    from ..training.config import TrainingConfig
    from .data_loader import NeuroscienceDataset


def training_segment_stop(n_timepoints: int, use_full_timeseries: bool) -> int:
    """End index of the training segment, matching :func:`create_data_loaders`."""
    return n_timepoints if use_full_timeseries else n_timepoints // 2


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class SplitPriors:
    """Functional connectome and omega estimated from one train split."""

    connectome: torch.Tensor  # (n_rois, n_rois), mean FC (not row-normalised)
    omega: torch.Tensor       # (n_rois,), rad/s
    train_idx: np.ndarray
    t_stop: int

    def fingerprint(self) -> dict[str, object]:
        """Provenance record saved with every results JSON."""
        return {
            "connectome_sha256": _sha256(self.connectome),
            "omega_sha256": _sha256(self.omega),
            "train_subjects": sorted(int(i) for i in self.train_idx),
            "train_segment": [0, int(self.t_stop)],
        }


def estimate_split_priors(
    dataset: "NeuroscienceDataset",
    train_idx: np.ndarray,
    cfg: "TrainingConfig",
) -> SplitPriors:
    """Estimate the functional connectome and omega from training data only."""
    train_idx = np.asarray(train_idx)
    t_stop = training_segment_stop(dataset.n_timepoints, cfg.use_full_timeseries)
    idx = torch.as_tensor(train_idx, dtype=torch.long, device=dataset.timeseries.device)
    train_ts = dataset.timeseries[idx, :, :t_stop]

    connectome = compute_fc_from_timeseries(train_ts).mean(dim=0)
    omega = compute_omega_from_timeseries(
        train_ts, dt=dataset.dt, f_lo=cfg.f_lo, f_hi=cfg.f_hi, method="peak",
    )
    return SplitPriors(connectome=connectome, omega=omega, train_idx=train_idx, t_stop=t_stop)
