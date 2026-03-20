"""Timeseries metrics and training-specific signal losses.

All inputs are ``(batch, n_rois, n_timepoints)`` complex analytic signals.
The **real part** is used for spectral and temporal comparisons.

Time-domain averages (.mean over the time axis) are the discrete
approximation to :math:`\\frac{1}{T}\\int_0^T f(t)\\,dt` for uniform
sampling, then averaged over batch and ROIs.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from ._utils import align_batch_and_time, ensure_batch, to_real


# ---------------------------------------------------------------------------
# nn.Module metric/loss classes
# ---------------------------------------------------------------------------

class PowerSpectrumDistance(nn.Module):
    """MSE between normalised power spectra. Metric and loss are the same value.

    ``forward()`` returns the loss. ``evaluate()`` returns ``{"power_spectrum_distance": float}``.
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        pred, target, _, _ = align_batch_and_time(ts_pred, ts_target)
        pred, target = to_real(pred), to_real(target)

        pp = torch.abs(torch.fft.rfft(pred, dim=2)) ** 2
        pt = torch.abs(torch.fft.rfft(target, dim=2)) ** 2
        pp = pp / (pp.sum(dim=2, keepdim=True) + 1e-8)
        pt = pt / (pt.sum(dim=2, keepdim=True) + 1e-8)
        return ((pp - pt) ** 2).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"power_spectrum_distance": self(ts_pred, ts_target).item()}


class TemporalCorrelation(nn.Module):
    """1 − mean per-ROI Pearson temporal correlation.

    ``forward()`` returns the loss (lower is better).
    ``evaluate()`` returns ``{"temporal_correlation": float}`` (higher is better).
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        pred, target, _, _ = align_batch_and_time(ts_pred, ts_target)
        pred, target = to_real(pred), to_real(target)

        p = pred - pred.mean(dim=2, keepdim=True)
        t = target - target.mean(dim=2, keepdim=True)
        num = (p * t).sum(dim=2)
        den = torch.sqrt((p ** 2).sum(dim=2) * (t ** 2).sum(dim=2)) + 1e-8
        return 1.0 - (num / den).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"temporal_correlation": 1.0 - self(ts_pred, ts_target).item()}


class AutocorrelationDistance(nn.Module):
    """MSE between autocorrelation functions. Metric and loss are the same value.

    ``forward()`` returns the loss. ``evaluate()`` returns ``{"autocorr_distance": float}``.
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor, max_lag: int = 50) -> torch.Tensor:
        pred, target, _, _ = align_batch_and_time(ts_pred, ts_target)
        pred, target = to_real(pred), to_real(target)
        return (_autocorr(pred, max_lag) - _autocorr(target, max_lag)).pow(2).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"autocorr_distance": self(ts_pred, ts_target).item()}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _autocorr(ts: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Autocorrelation up to *max_lag* for ``(B, N, T)`` real tensor."""
    tc = ts - ts.mean(dim=2, keepdim=True)
    var = (tc ** 2).mean(dim=2, keepdim=True) + 1e-8
    acs = []
    for lag in range(1, min(max_lag, ts.shape[2] // 2)):
        acs.append((tc[:, :, :-lag] * tc[:, :, lag:]).mean(dim=2) / var.squeeze(2))
    return torch.stack(acs, dim=2)


def _complex_amplitude_omega(
    ts: torch.Tensor,
    tr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-region amplitude and instantaneous frequency from complex analytic signal.

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
