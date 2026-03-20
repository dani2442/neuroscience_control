"""Functional Connectivity metrics.

FC matrices are real-valued ``(batch, n_rois, n_rois)`` tensors.  If a
complex tensor is passed the real part is used automatically.
"""

import torch
import torch.nn as nn

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
    ts_c = ts - ts.mean(dim=2, keepdim=True)
    ts_n = ts_c / (ts_c.std(dim=2, keepdim=True) + 1e-8)
    T = ts.shape[2]
    return torch.bmm(ts_n, ts_n.transpose(1, 2)) / max(T - 1, 1)


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
        pred_flat = upper_tri_vec(fc_pred, k=1)   # (B, M)
        targ_flat = upper_tri_vec(fc_target, k=1)
    else:
        pred_flat = fc_pred.reshape(fc_pred.shape[0], -1)
        targ_flat = fc_target.reshape(fc_target.shape[0], -1)

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
        pred_flat = upper_tri_vec(fc_pred, k=1)   # (B, M)
        targ_flat = upper_tri_vec(fc_target, k=1)
    else:
        pred_flat = fc_pred.reshape(fc_pred.shape[0], -1)
        targ_flat = fc_target.reshape(fc_target.shape[0], -1)

    return ((pred_flat - targ_flat) ** 2).mean()


# ---------------------------------------------------------------------------
# nn.Module metric/loss classes
# ---------------------------------------------------------------------------

class FCCorrelation(nn.Module):
    """1 − Pearson correlation between FC matrices.

    ``forward()`` returns the loss (lower is better).
    ``evaluate()`` returns ``{"fc_correlation": float}`` (higher is better).
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        fc_pred = compute_static_fc(ts_pred)
        fc_target = compute_static_fc(ts_target)
        return 1.0 - fc_correlation(fc_pred, fc_target)

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"fc_correlation": 1.0 - self(ts_pred, ts_target).item()}


class FCMSE(nn.Module):
    """MSE between FC matrices. Metric and loss are the same value.

    ``forward()`` returns the loss. ``evaluate()`` returns ``{"fc_mse": float}``.
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        fc_pred = compute_static_fc(ts_pred)
        fc_target = compute_static_fc(ts_target)
        return fc_mse(fc_pred, fc_target)

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"fc_mse": self(ts_pred, ts_target).item()}
