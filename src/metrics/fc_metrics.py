"""Functional Connectivity metrics.

FC matrices are real-valued ``(batch, n_rois, n_rois)`` tensors.  If a
complex tensor is passed the real part is used automatically.
"""

from typing import Dict

import torch

from ._utils import ensure_batch, to_real, upper_tri_vec


def compute_static_fc(ts: torch.Tensor) -> torch.Tensor:
    r"""Compute static FC (Pearson correlation) from timeseries.

    .. math::

        FC_{nm} = \mathrm{corr}(s_n, s_m)
        = \frac{\sum_t (s_n(t)-\bar s_n)(s_m(t)-\bar s_m)}
               {\sqrt{\sum_t (s_n(t)-\bar s_n)^2}\,
                \sqrt{\sum_t (s_m(t)-\bar s_m)^2}}

    Uses the real part of the signal if the input is complex.

    Args:
        ts: ``(batch, n_rois, T)`` or ``(n_rois, T)`` complex or real tensor.

    Returns:
        ``(batch, n_rois, n_rois)`` FC matrices.
    """
    ts = to_real(ensure_batch(ts))
    # z-score each ROI over time
    ts_c = ts - ts.mean(dim=2, keepdim=True)
    ts_n = ts_c / (ts_c.std(dim=2, keepdim=True) + 1e-8)
    T = ts.shape[2]
    return torch.bmm(ts_n, ts_n.transpose(1, 2)) / max(T - 1, 1)


def _extract_upper_tri(
    fc: torch.Tensor, n_rois: int,
) -> torch.Tensor:
    """Flatten the strict upper triangle of a batch of FC matrices.

    Args:
        fc: ``(batch, n_rois, n_rois)``

    Returns:
        ``(batch, M)`` where ``M = n_rois*(n_rois-1)/2``.
    """
    idx = torch.triu_indices(n_rois, n_rois, offset=1, device=fc.device)
    return fc[:, idx[0], idx[1]]


def fc_correlation(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    use_upper_triangle: bool = True,
) -> torch.Tensor:
    """Pearson correlation between predicted and target FC.

    By default only the upper triangle (excluding diagonal) is compared.
    The result is averaged over the batch.
    """
    fc_pred = to_real(fc_pred)
    fc_target = to_real(fc_target)

    if fc_pred.dim() == 2:
        fc_pred = fc_pred.unsqueeze(0)
        fc_target = fc_target.unsqueeze(0)

    if use_upper_triangle:
        pred_flat = _extract_upper_tri(fc_pred, fc_pred.shape[1])
        targ_flat = _extract_upper_tri(fc_target, fc_target.shape[1])
    else:
        pred_flat = fc_pred.reshape(fc_pred.shape[0], -1)
        targ_flat = fc_target.reshape(fc_target.shape[0], -1)

    # Vectorised Pearson correlation across batch
    pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    targ_c = targ_flat - targ_flat.mean(dim=1, keepdim=True)
    num = (pred_c * targ_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) * (targ_c ** 2).sum(dim=1)) + 1e-8
    return (num / den).mean()


def fc_mse(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    use_upper_triangle: bool = True,
) -> torch.Tensor:
    """MSE between predicted and target FC (upper triangle by default)."""
    fc_pred = to_real(fc_pred)
    fc_target = to_real(fc_target)

    if fc_pred.dim() == 2:
        fc_pred = fc_pred.unsqueeze(0)
        fc_target = fc_target.unsqueeze(0)

    if use_upper_triangle:
        pred_flat = _extract_upper_tri(fc_pred, fc_pred.shape[1])
        targ_flat = _extract_upper_tri(fc_target, fc_target.shape[1])
    else:
        pred_flat = fc_pred.reshape(fc_pred.shape[0], -1)
        targ_flat = fc_target.reshape(fc_target.shape[0], -1)

    return ((pred_flat - targ_flat) ** 2).mean()


def compute_all_fc_metrics(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all FC metrics. Returns dict of metric name -> value."""
    return {
        "fc_correlation": fc_correlation(fc_pred, fc_target).item(),
        "fc_mse": fc_mse(fc_pred, fc_target).item(),
    }


def compute_fc_from_timeseries_and_compare(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute static FC from timeseries and return all FC metrics.

    A convenience wrapper that combines :func:`compute_static_fc` with
    :func:`compute_all_fc_metrics`.  Mean FC is computed across the batch
    before comparison.

    Args:
        ts_pred: ``(batch, n_rois, T)`` complex or real.
        ts_target: ``(batch, n_rois, T)`` complex or real.

    Returns:
        ``{"fc_correlation": float, "fc_mse": float}``
    """
    fc_pred = compute_static_fc(ts_pred).mean(dim=0, keepdim=True)
    fc_target = compute_static_fc(ts_target).mean(dim=0, keepdim=True)
    return compute_all_fc_metrics(fc_pred, fc_target)
