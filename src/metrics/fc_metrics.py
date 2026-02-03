"""Functional Connectivity metrics."""

import torch
from typing import Dict, Tuple


def fc_correlation(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    use_upper_triangle: bool = True
) -> torch.Tensor:
    """
    Compute Pearson correlation between predicted and target FC.
    
    Args:
        fc_pred: Predicted FC of shape (batch, n_rois, n_rois) or (n_rois, n_rois)
        fc_target: Target FC of same shape
        use_upper_triangle: If True, only use upper triangle values
        
    Returns:
        Correlation value(s)
    """
    if fc_pred.dim() == 2:
        fc_pred = fc_pred.unsqueeze(0)
        fc_target = fc_target.unsqueeze(0)
    
    batch_size = fc_pred.shape[0]
    n_rois = fc_pred.shape[1]
    
    if use_upper_triangle:
        # Get upper triangle indices
        idx = torch.triu_indices(n_rois, n_rois, offset=1)
        pred_flat = fc_pred[:, idx[0], idx[1]]
        target_flat = fc_target[:, idx[0], idx[1]]
    else:
        pred_flat = fc_pred.reshape(batch_size, -1)
        target_flat = fc_target.reshape(batch_size, -1)
    
    # Compute correlation for each batch element
    correlations = []
    for i in range(batch_size):
        pred_centered = pred_flat[i] - pred_flat[i].mean()
        target_centered = target_flat[i] - target_flat[i].mean()
        
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (target_centered ** 2).sum()
        ) + 1e-8
        
        correlations.append(numerator / denominator)
    
    return torch.stack(correlations).mean()


def fc_mse(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    use_upper_triangle: bool = True
) -> torch.Tensor:
    """
    Compute MSE between predicted and target FC.
    
    Args:
        fc_pred: Predicted FC
        fc_target: Target FC
        use_upper_triangle: If True, only use upper triangle
        
    Returns:
        MSE value
    """
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


def fc_upper_triangle_correlation(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute correlation using only upper triangle of FC matrices.
    
    This is the standard way to compare FC matrices as they are symmetric
    and the diagonal is always 1.
    
    Args:
        fc_pred: Predicted FC (batch, n_rois, n_rois)
        fc_target: Target FC (batch, n_rois, n_rois)
        
    Returns:
        Mean correlation across batch
    """
    return fc_correlation(fc_pred, fc_target, use_upper_triangle=True)


def compute_all_fc_metrics(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all FC metrics.
    
    Args:
        fc_pred: Predicted FC
        fc_target: Target FC
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        'fc_correlation': fc_correlation(fc_pred, fc_target).item(),
        'fc_mse': fc_mse(fc_pred, fc_target).item(),
        'fc_upper_corr': fc_upper_triangle_correlation(fc_pred, fc_target).item(),
    }
