"""Timeseries metrics."""

import torch
from typing import Dict


def power_spectrum_distance(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute distance between power spectra of predicted and target timeseries.
    
    Args:
        ts_pred: Predicted timeseries (batch, n_rois, n_timepoints)
        ts_target: Target timeseries (batch, n_rois, n_timepoints)
        
    Returns:
        Mean power spectrum distance
    """
    # Compute FFT
    fft_pred = torch.fft.rfft(ts_pred, dim=2)
    fft_target = torch.fft.rfft(ts_target, dim=2)
    
    # Compute power spectra
    power_pred = torch.abs(fft_pred) ** 2
    power_target = torch.abs(fft_target) ** 2
    
    # Normalize
    power_pred = power_pred / (power_pred.sum(dim=2, keepdim=True) + 1e-8)
    power_target = power_target / (power_target.sum(dim=2, keepdim=True) + 1e-8)
    
    # Compute distance (Jensen-Shannon-like)
    distance = ((power_pred - power_target) ** 2).mean()
    
    return distance


def temporal_correlation(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute temporal correlation between predicted and target timeseries.
    
    Args:
        ts_pred: Predicted timeseries (batch, n_rois, n_timepoints)
        ts_target: Target timeseries (batch, n_rois, n_timepoints)
        
    Returns:
        Mean temporal correlation across ROIs and batch
    """
    # Center timeseries
    pred_centered = ts_pred - ts_pred.mean(dim=2, keepdim=True)
    target_centered = ts_target - ts_target.mean(dim=2, keepdim=True)
    
    # Compute correlation
    numerator = (pred_centered * target_centered).sum(dim=2)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=2) * (target_centered ** 2).sum(dim=2)
    ) + 1e-8
    
    correlations = numerator / denominator
    
    return correlations.mean()


def autocorrelation_distance(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    max_lag: int = 50
) -> torch.Tensor:
    """
    Compute distance between autocorrelation functions.
    
    Args:
        ts_pred: Predicted timeseries
        ts_target: Target timeseries
        max_lag: Maximum lag to consider
        
    Returns:
        Mean autocorrelation distance
    """
    def compute_autocorr(ts, max_lag):
        """Compute autocorrelation for multiple lags."""
        n_timepoints = ts.shape[2]
        ts_centered = ts - ts.mean(dim=2, keepdim=True)
        var = (ts_centered ** 2).mean(dim=2, keepdim=True) + 1e-8
        
        autocorrs = []
        for lag in range(1, min(max_lag, n_timepoints // 2)):
            corr = (ts_centered[:, :, :-lag] * ts_centered[:, :, lag:]).mean(dim=2)
            autocorrs.append(corr / var.squeeze())
        
        return torch.stack(autocorrs, dim=2)
    
    ac_pred = compute_autocorr(ts_pred, max_lag)
    ac_target = compute_autocorr(ts_target, max_lag)
    
    return ((ac_pred - ac_target) ** 2).mean()


def compute_all_timeseries_metrics(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all timeseries metrics.
    
    Args:
        ts_pred: Predicted timeseries
        ts_target: Target timeseries
        
    Returns:
        Dictionary of metrics
    """
    return {
        'power_spectrum_distance': power_spectrum_distance(ts_pred, ts_target).item(),
        'temporal_correlation': temporal_correlation(ts_pred, ts_target).item(),
        'autocorr_distance': autocorrelation_distance(ts_pred, ts_target).item(),
    }
