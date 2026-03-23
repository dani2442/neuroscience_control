"""Functional Connectivity metrics.

FC matrices are real-valued ``(batch, n_rois, n_rois)`` tensors.  If a
complex tensor is passed the real part is used automatically.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# nn.Module metric/loss classes
# ---------------------------------------------------------------------------

class FCCorrelation(nn.Module):
    """1 − Pearson correlation between FC matrices.

    ``forward()`` returns the loss (lower is better).
    ``evaluate()`` returns ``{"fc_correlation": float}`` (higher is better).

    When *fc_target* is provided (precomputed per-subject FC), it is used
    directly instead of computing FC from *ts_target*.
    """

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        fc_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fc_pred = compute_static_fc(ts_pred)
        if fc_target is None:
            fc_target = compute_static_fc(ts_target)

        pred_flat = upper_tri_vec(to_real(fc_pred), k=1)
        targ_flat = upper_tri_vec(to_real(fc_target), k=1)
        pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        targ_c = targ_flat - targ_flat.mean(dim=1, keepdim=True)
        num = (pred_c * targ_c).sum(dim=1)
        den = torch.sqrt((pred_c ** 2).sum(dim=1) * (targ_c ** 2).sum(dim=1)) + 1e-8
        corr = (num / den).mean()
        return 1.0 - corr

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"fc_correlation": 1.0 - self(ts_pred, ts_target).item()}


class FCMSE(nn.Module):
    """MSE between FC matrices. Metric and loss are the same value.

    ``forward()`` returns the loss. ``evaluate()`` returns ``{"fc_mse": float}``.

    When *fc_target* is provided (precomputed per-subject FC), it is used
    directly instead of computing FC from *ts_target*.
    """

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        fc_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fc_pred = compute_static_fc(ts_pred)
        if fc_target is None:
            fc_target = compute_static_fc(ts_target)
        pred_flat = upper_tri_vec(to_real(fc_pred), k=1)
        targ_flat = upper_tri_vec(to_real(fc_target), k=1)
        return ((pred_flat - targ_flat) ** 2).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"fc_mse": self(ts_pred, ts_target).item()}
