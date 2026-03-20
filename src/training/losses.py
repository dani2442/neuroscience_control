"""
Loss functions and composite loss for neuroscience model training.

Training-specific loss classes (L2Timeseries, AmplitudeLoss, OmegaLoss) live
here alongside CompositeLoss.  All metric/loss classes that correspond to
evaluation metrics live in ``src/metrics/``.

All classes follow the nn.Module convention: ``forward(ts_pred, ts_target)``
returns a differentiable scalar loss tensor.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..metrics._utils import ensure_batch
from ..metrics.fc_metrics import FCCorrelation, FCMSE
from ..metrics.timeseries_metrics import (
    AutocorrelationDistance,
    PowerSpectrumDistance,
    TemporalCorrelation,
)
from ..metrics.dynamics_metrics import FCD, Metastability, PhaseFC, PhFCD


# ---------------------------------------------------------------------------
# Dataset-level reference statistics
# ---------------------------------------------------------------------------

def compute_ref_amplitude(timeseries: torch.Tensor) -> torch.Tensor:
    """Mean envelope amplitude per ROI across the full dataset.

    Args:
        timeseries: ``(n_subjects, n_rois, T)`` complex analytic signal.

    Returns:
        ``(n_rois,)`` — mean |z| over subjects and time.
    """
    ts = timeseries if timeseries.dim() == 3 else timeseries.unsqueeze(0)
    return ts.abs().mean(dim=(0, 2))


def compute_ref_omega(
    timeseries: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
) -> torch.Tensor:
    """Per-ROI intrinsic angular frequency via FFT peak detection.

    Args:
        timeseries: ``(n_subjects, n_rois, T)`` complex analytic signal.
        tr: Repetition time in seconds.
        f_lo, f_hi: Frequency band of interest (Hz).

    Returns:
        ``(n_rois,)`` — angular frequencies (rad/s).
    """
    from ..dataset.preprocessing import compute_omega_from_timeseries

    return compute_omega_from_timeseries(
        timeseries, dt=tr, f_lo=f_lo, f_hi=f_hi, method="peak",
    )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _complex_amplitude_omega(
    ts: torch.Tensor,
    tr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-region amplitude and instantaneous-frequency from complex analytic signal.

    Args:
        ts: ``(batch, n_rois, n_timepoints)``, complex.
        tr: Repetition time (seconds).

    Returns:
        amplitude: ``(batch, n_rois, n_timepoints)``  — |z|
        omega:     ``(batch, n_rois, n_timepoints - 1)``  — rad/s
    """
    ts = ensure_batch(ts)
    if not torch.is_complex(ts):
        raise ValueError(
            "_complex_amplitude_omega expects complex input; "
            "got real tensor.  Ensure the model output is complex."
        )
    amplitude = ts.abs()
    phase = torch.angle(ts)
    dphi = torch.diff(phase, dim=2)
    dphi = dphi - 2.0 * math.pi * torch.round(dphi / (2.0 * math.pi))
    omega = dphi / tr
    return amplitude, omega


# ---------------------------------------------------------------------------
# Training-specific nn.Module loss classes
# ---------------------------------------------------------------------------

class L2Timeseries(nn.Module):
    r"""L² error between predicted and target timeseries.

    Supports both real and complex tensors.  For complex inputs the loss
    is :math:`\frac{1}{B N T}\sum |x^{\mathrm{pred}} - x^{\mathrm{target}}|^2`.
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        if ts_pred.ndim == 2:
            ts_pred = ts_pred.unsqueeze(0)
        if ts_target.ndim == 2:
            ts_target = ts_target.unsqueeze(0)
        T = min(ts_pred.shape[2], ts_target.shape[2])
        B = min(ts_pred.shape[0], ts_target.shape[0])
        diff = ts_pred[:B, :, :T] - ts_target[:B, :, :T]
        if torch.is_complex(diff):
            return (diff.real ** 2 + diff.imag ** 2).mean()
        return (diff ** 2).mean()


class AmplitudeLoss(nn.Module):
    r"""L² error between mean predicted amplitude and a per-ROI reference.

    When *ref_amplitude* ``(n_rois,)`` is provided (precomputed from the
    full dataset via :func:`compute_ref_amplitude`), it is used as the
    target.  Otherwise falls back to the per-window mean of *ts_target*.

    Args:
        ref_amplitude: Optional per-ROI reference amplitude ``(n_rois,)``.
        tr: Repetition time in seconds.
    """

    def __init__(
        self,
        ref_amplitude: Optional[torch.Tensor] = None,
        tr: float = 0.72,
    ):
        super().__init__()
        self.tr = tr
        if ref_amplitude is not None:
            self.register_buffer("ref_amplitude", ref_amplitude)
        else:
            self.ref_amplitude = None

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        amp_pred, _ = _complex_amplitude_omega(ts_pred, self.tr)
        mean_pred = amp_pred.mean(dim=2)   # (B, N)
        if self.ref_amplitude is not None:
            target = self.ref_amplitude.to(mean_pred.device)
        else:
            amp_targ, _ = _complex_amplitude_omega(ts_target, self.tr)
            target = amp_targ.mean(dim=2)  # (B, N)
        return ((mean_pred - target) ** 2).mean()


class OmegaLoss(nn.Module):
    r"""L² error between mean predicted instantaneous frequency and a per-ROI reference.

    When *ref_omega* ``(n_rois,)`` is provided (precomputed from the full
    dataset via :func:`compute_ref_omega`), it is used as the target.
    Otherwise falls back to the per-window mean of *ts_target*.

    Args:
        ref_omega: Optional per-ROI reference angular frequency ``(n_rois,)``.
        tr: Repetition time in seconds.
    """

    def __init__(
        self,
        ref_omega: Optional[torch.Tensor] = None,
        tr: float = 0.72,
    ):
        super().__init__()
        self.tr = tr
        if ref_omega is not None:
            self.register_buffer("ref_omega", ref_omega)
        else:
            self.ref_omega = None

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        _, omega_pred = _complex_amplitude_omega(ts_pred, self.tr)
        mean_pred = omega_pred.mean(dim=(0, 2))  # (N,)
        if self.ref_omega is not None:
            target = self.ref_omega.to(mean_pred.device)
        else:
            _, omega_targ = _complex_amplitude_omega(ts_target, self.tr)
            target = omega_targ.mean(dim=(0, 2))  # (N,)
        return ((mean_pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Registry of metric/loss classes
# ---------------------------------------------------------------------------

# Maps loss weight key → nn.Module class.
# FCD takes extra kwargs (tr, fcd_win_sec, fcd_step_sec); others take none.
METRIC_REGISTRY: Dict[str, type] = {
    "fc_correlation":       FCCorrelation,
    "fc_mse":               FCMSE,
    "l2":                   L2Timeseries,
    "amplitude":            AmplitudeLoss,
    "omega":                OmegaLoss,
    "power_spectrum":       PowerSpectrumDistance,
    "temporal_correlation": TemporalCorrelation,
    "autocorrelation":      AutocorrelationDistance,
    "fcd":                  FCD,
    "phfcd":                PhFCD,
    "phase_fc_correlation": PhaseFC,
    "metastability":        Metastability,
}

# Keys whose constructor accepts (tr, fcd_win_sec, fcd_step_sec)
_DYN_KWARGS_KEYS = {"fcd"}

# Keys whose constructor accepts tr only
_TR_KWARGS_KEYS = {"amplitude", "omega"}


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """Weighted sum of named metric/loss terms.

    Each term is an ``nn.Module`` whose ``forward(ts_pred, ts_target)``
    returns a scalar loss tensor.  Zero-weight terms are never instantiated
    or called.

    Example::

        loss_fn = CompositeLoss(
            weights={"fc_correlation": 1.0, "fcd": 0.5},
            tr=0.72, fcd_win_sec=30.0, fcd_step_sec=2.0,
        )
        total, components = loss_fn(ts_pred, ts_target)

    Args:
        weights: Mapping of loss term name → scalar weight.  Terms with
            weight 0 are dropped.
        tr: Repetition time (seconds), forwarded to FCD, AmplitudeLoss, OmegaLoss.
        fcd_win_sec: FCD window length in seconds.
        fcd_step_sec: FCD step size in seconds.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        tr: float = 0.72,
        fcd_win_sec: float = 30.0,
        fcd_step_sec: float = 2.0,
    ) -> None:
        super().__init__()
        unknown = set(weights) - set(METRIC_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown loss terms: {unknown}. "
                f"Available: {sorted(METRIC_REGISTRY)}"
            )
        # Drop zero-weight terms
        self.weights = {k: v for k, v in weights.items() if v != 0.0}

        modules: Dict[str, nn.Module] = {}
        for name in self.weights:
            cls = METRIC_REGISTRY[name]
            if name in _DYN_KWARGS_KEYS:
                modules[name] = cls(tr=tr, fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec)
            elif name in _TR_KWARGS_KEYS:
                modules[name] = cls(tr=tr)
            else:
                modules[name] = cls()
        self.terms = nn.ModuleDict(modules)

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the weighted composite loss.

        Args:
            ts_pred: ``(batch, n_rois, T)`` complex analytic signal.
            ts_target: ``(batch, n_rois, T)`` complex analytic signal.

        Returns:
            total: weighted scalar loss for back-propagation.
            components: ``{"loss_<name>": tensor, ...}`` for logging.
        """
        device = ts_pred.device
        dtype = ts_pred.real.dtype if torch.is_complex(ts_pred) else ts_pred.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        components: Dict[str, torch.Tensor] = {}

        for name, weight in self.weights.items():
            value = self.terms[name](ts_pred, ts_target)
            components[f"loss_{name}"] = value
            total = total + weight * value

        return total, components

    @property
    def component_names(self) -> list[str]:
        """Return the ``loss_<name>`` keys that will appear in components."""
        return [f"loss_{name}" for name in self.weights]
