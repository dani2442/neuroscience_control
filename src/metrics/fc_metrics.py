"""Functional Connectivity metrics.

FC matrices are real-valued ``(batch, n_rois, n_rois)`` tensors.  If a
complex tensor is passed the real part is used automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import ensure_batch, fisher_batch_average, reshape_for_groups, to_real, upper_tri_vec


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


# ---------------------------------------------------------------------------
# nn.Module metric/loss classes
# ---------------------------------------------------------------------------

class FCCorrelation(nn.Module):
    """1 − Pearson correlation between FC matrices.

    ``forward()`` returns the loss (lower is better).
    ``evaluate()`` returns ``{"fc_correlation": float}`` (higher is better).

    Batched FC tensors are reduced to a single group FC matrix via
    Fisher-z averaging across batch before comparison.

    When *fc_target* is provided (precomputed per-subject or group FC),
    it is used directly instead of computing FC from *ts_target*.
    """

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        fc_target: torch.Tensor | None = None,
        group_size: int = 0,
    ) -> torch.Tensor:
        fc_pred = reshape_for_groups(compute_static_fc(ts_pred), group_size)
        if fc_target is None:
            fc_target = compute_static_fc(ts_target)
        fc_target = reshape_for_groups(fc_target, group_size)
        fc_pred = fisher_batch_average(fc_pred)
        fc_target = fisher_batch_average(fc_target)

        pred_flat = upper_tri_vec(fc_pred, k=1)
        targ_flat = upper_tri_vec(fc_target, k=1)
        pred_c = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
        targ_c = targ_flat - targ_flat.mean(dim=-1, keepdim=True)
        num = (pred_c * targ_c).sum(dim=-1)
        den = torch.sqrt((pred_c ** 2).sum(dim=-1) * (targ_c ** 2).sum(dim=-1)) + 1e-8
        return (1.0 - num / den).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor, group_size: int = 0) -> dict:
        return {"fc_correlation": 1.0 - self(ts_pred, ts_target, group_size=group_size).item()}


class FCMSE(nn.Module):
    """MSE between FC matrices. Metric and loss are the same value.

    ``forward()`` returns the loss. ``evaluate()`` returns ``{"fc_mse": float}``.

    Batched FC tensors are reduced to a single group FC matrix via
    Fisher-z averaging across batch before comparison.

    When *fc_target* is provided (precomputed per-subject or group FC),
    it is used directly instead of computing FC from *ts_target*.
    """

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        fc_target: torch.Tensor | None = None,
        group_size: int = 0,
    ) -> torch.Tensor:
        fc_pred = reshape_for_groups(compute_static_fc(ts_pred), group_size)
        if fc_target is None:
            fc_target = compute_static_fc(ts_target)
        fc_target = reshape_for_groups(fc_target, group_size)
        fc_pred = fisher_batch_average(fc_pred)
        fc_target = fisher_batch_average(fc_target)
        pred_flat = upper_tri_vec(fc_pred, k=1)
        targ_flat = upper_tri_vec(fc_target, k=1)
        return ((pred_flat - targ_flat) ** 2).mean(dim=-1).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor, group_size: int = 0) -> dict:
        return {"fc_mse": self(ts_pred, ts_target, group_size=group_size).item()}
