"""Timeseries metrics."""

import torch
from typing import Dict


def _to_real(ts: torch.Tensor) -> torch.Tensor:
    """Extract real part if complex, else pass through."""
    return ts.real if torch.is_complex(ts) else ts


def power_spectrum_distance(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between normalised power spectra (batch, n_rois, n_timepoints)."""
    ts_pred, ts_target = _to_real(ts_pred), _to_real(ts_target)
    T = min(ts_pred.shape[2], ts_target.shape[2])
    B = min(ts_pred.shape[0], ts_target.shape[0])
    pp = torch.abs(torch.fft.rfft(ts_pred[:B, :, :T], dim=2)) ** 2
    pt = torch.abs(torch.fft.rfft(ts_target[:B, :, :T], dim=2)) ** 2
    pp = pp / (pp.sum(dim=2, keepdim=True) + 1e-8)
    pt = pt / (pt.sum(dim=2, keepdim=True) + 1e-8)
    return ((pp - pt) ** 2).mean()


def temporal_correlation(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-ROI Pearson correlation over time (batch, n_rois, T)."""
    ts_pred, ts_target = _to_real(ts_pred), _to_real(ts_target)
    T = min(ts_pred.shape[2], ts_target.shape[2])
    B = min(ts_pred.shape[0], ts_target.shape[0])
    p = ts_pred[:B, :, :T] - ts_pred[:B, :, :T].mean(dim=2, keepdim=True)
    t = ts_target[:B, :, :T] - ts_target[:B, :, :T].mean(dim=2, keepdim=True)
    num = (p * t).sum(dim=2)
    den = torch.sqrt((p ** 2).sum(dim=2) * (t ** 2).sum(dim=2)) + 1e-8
    return (num / den).mean()


def autocorrelation_distance(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    max_lag: int = 50,
) -> torch.Tensor:
    """MSE between autocorrelation functions up to *max_lag*."""
    ts_pred, ts_target = _to_real(ts_pred), _to_real(ts_target)
    T = min(ts_pred.shape[2], ts_target.shape[2])
    B = min(ts_pred.shape[0], ts_target.shape[0])

    def _autocorr(ts, max_lag):
        tc = ts - ts.mean(dim=2, keepdim=True)
        var = (tc ** 2).mean(dim=2, keepdim=True) + 1e-8
        acs = []
        for lag in range(1, min(max_lag, ts.shape[2] // 2)):
            acs.append((tc[:, :, :-lag] * tc[:, :, lag:]).mean(dim=2) / var.squeeze(2))
        return torch.stack(acs, dim=2)

    return ((_autocorr(ts_pred[:B, :, :T], max_lag) - _autocorr(ts_target[:B, :, :T], max_lag)) ** 2).mean()


def compute_all_timeseries_metrics(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all timeseries metrics. Returns dict of name → value."""
    return {
        'power_spectrum_distance': power_spectrum_distance(ts_pred, ts_target).item(),
        'temporal_correlation': temporal_correlation(ts_pred, ts_target).item(),
        'autocorr_distance': autocorrelation_distance(ts_pred, ts_target).item(),
    }
