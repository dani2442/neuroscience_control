"""Functional Connectivity metrics."""

import torch
from typing import Dict


def _to_real(x: torch.Tensor) -> torch.Tensor:
    """Extract real part if complex, else pass through."""
    return x.real if torch.is_complex(x) else x


def fc_correlation(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    use_upper_triangle: bool = True,
) -> torch.Tensor:
    """Pearson correlation between predicted and target FC (upper triangle by default)."""
    fc_pred = _to_real(fc_pred)
    fc_target = _to_real(fc_target)

    if fc_pred.dim() == 2:
        fc_pred = fc_pred.unsqueeze(0)
        fc_target = fc_target.unsqueeze(0)

    n_rois = fc_pred.shape[1]

    if use_upper_triangle:
        idx = torch.triu_indices(n_rois, n_rois, offset=1)
        pred_flat = fc_pred[:, idx[0], idx[1]]
        target_flat = fc_target[:, idx[0], idx[1]]
    else:
        pred_flat = fc_pred.reshape(fc_pred.shape[0], -1)
        target_flat = fc_target.reshape(fc_target.shape[0], -1)

    # Vectorised Pearson correlation across batch
    pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    targ_c = target_flat - target_flat.mean(dim=1, keepdim=True)
    num = (pred_c * targ_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) * (targ_c ** 2).sum(dim=1)) + 1e-8
    return (num / den).mean()


def fc_mse(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    use_upper_triangle: bool = True,
) -> torch.Tensor:
    """MSE between predicted and target FC (upper triangle by default)."""
    fc_pred = _to_real(fc_pred)
    fc_target = _to_real(fc_target)

    if fc_pred.dim() == 2:
        fc_pred = fc_pred.unsqueeze(0)
        fc_target = fc_target.unsqueeze(0)

    n_rois = fc_pred.shape[1]

    if use_upper_triangle:
        idx = torch.triu_indices(n_rois, n_rois, offset=1)
        pred_flat = fc_pred[:, idx[0], idx[1]]
        target_flat = fc_target[:, idx[0], idx[1]]
    else:
        pred_flat = fc_pred.reshape(fc_pred.shape[0], -1)
        target_flat = fc_target.reshape(fc_target.shape[0], -1)

    return ((pred_flat - target_flat) ** 2).mean()


def compute_all_fc_metrics(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all FC metrics. Returns dict of metric name → value."""
    return {
        'fc_correlation': fc_correlation(fc_pred, fc_target).item(),
        'fc_mse': fc_mse(fc_pred, fc_target).item(),
    }
