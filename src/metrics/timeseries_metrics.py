"""Timeseries metrics.

All inputs are ``(batch, n_rois, n_timepoints)`` complex analytic signals.
The **real part** is used for spectral and temporal comparisons.

Time-domain averages (.mean over the time axis) are the discrete
approximation to :math:`\\frac{1}{T}\\int_0^T f(t)\\,dt` for uniform
sampling, then averaged over batch and ROIs.
"""

from typing import Dict

import torch

from ._utils import align_batch_and_time, to_real


def power_spectrum_distance(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between normalised power spectra.

    Result is averaged over batch and ROIs.
    """
    pred, target, _, _ = align_batch_and_time(ts_pred, ts_target)
    pred, target = to_real(pred), to_real(target)

    pp = torch.abs(torch.fft.rfft(pred, dim=2)) ** 2
    pt = torch.abs(torch.fft.rfft(target, dim=2)) ** 2
    pp = pp / (pp.sum(dim=2, keepdim=True) + 1e-8)
    pt = pt / (pt.sum(dim=2, keepdim=True) + 1e-8)
    return ((pp - pt) ** 2).mean()


def temporal_correlation(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-ROI Pearson correlation over time, averaged over batch."""
    pred, target, _, _ = align_batch_and_time(ts_pred, ts_target)
    pred, target = to_real(pred), to_real(target)

    p = pred - pred.mean(dim=2, keepdim=True)
    t = target - target.mean(dim=2, keepdim=True)
    num = (p * t).sum(dim=2)
    den = torch.sqrt((p ** 2).sum(dim=2) * (t ** 2).sum(dim=2)) + 1e-8
    return (num / den).mean()


def autocorrelation_distance(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    max_lag: int = 50,
) -> torch.Tensor:
    """MSE between autocorrelation functions up to *max_lag*."""
    pred, target, _, _ = align_batch_and_time(ts_pred, ts_target)
    pred, target = to_real(pred), to_real(target)
    return ((_autocorr(pred, max_lag) - _autocorr(target, max_lag)) ** 2).mean()


def compute_all_timeseries_metrics(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all timeseries metrics. Returns dict of name -> value."""
    return {
        "power_spectrum_distance": power_spectrum_distance(ts_pred, ts_target).item(),
        "temporal_correlation": temporal_correlation(ts_pred, ts_target).item(),
        "autocorr_distance": autocorrelation_distance(ts_pred, ts_target).item(),
    }


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
