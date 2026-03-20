"""Timeseries metrics.

All inputs are ``(batch, n_rois, n_timepoints)`` complex analytic signals.
The **real part** is used for spectral and temporal comparisons.

Time-domain averages (.mean over the time axis) are the discrete
approximation to :math:`\\frac{1}{T}\\int_0^T f(t)\\,dt` for uniform
sampling, then averaged over batch and ROIs.
"""

import torch
import torch.nn as nn

from ._utils import align_batch_and_time, to_real


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
