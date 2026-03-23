"""Composite loss for neuroscience model training.

Training-specific loss classes (L2Timeseries, AmplitudeLoss, OmegaLoss) and
dataset-level reference helpers live in ``src/metrics/timeseries_metrics``.
All metric/loss classes that correspond to evaluation metrics live in
``src/metrics/``.

CompositeLoss and METRIC_REGISTRY live here.  All classes follow the
nn.Module convention: ``forward(ts_pred, ts_target)`` returns a
differentiable scalar loss tensor.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..metrics.fc_metrics import FCCorrelation, FCMSE
from ..metrics.timeseries_metrics import (
    AutocorrelationDistance,
    PowerSpectrumDistance,
    TemporalCorrelation,
    L2Timeseries,
    AmplitudeLoss,
    OmegaLoss,
    compute_ref_amplitude,
    compute_ref_omega,
)
from ..metrics.dynamics_metrics import FCD, Metastability, PhaseFC, PhFCD


# ---------------------------------------------------------------------------
# FDM (Finite Dimensional Matching) loss
# ---------------------------------------------------------------------------

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Biased RBF kernel matrix k(x,y) = exp(-||x-y||^2 / (2*sigma^2))."""
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True).T
    sq_dist = xx + yy - 2.0 * x @ y.T
    return torch.exp(-sq_dist / (2.0 * sigma ** 2))


def _mmd(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Biased MMD² between sample sets x and y using RBF kernel."""
    kxx = _rbf_kernel(x, x, sigma).mean()
    kxy = _rbf_kernel(x, y, sigma).mean()
    kyy = _rbf_kernel(y, y, sigma).mean()
    return kxx - 2.0 * kxy + kyy


class FDMLoss(nn.Module):
    """FDM loss: MMD between two-time marginals of simulated and real trajectories.

    Based on: Zhang et al., "Efficient Training of Neural SDEs by Matching
    Finite Dimensional Distributions", ICLR 2025.

    Samples random pairs (t1, t2) with |t2-t1| ≤ max_lag, extracts joint
    states [Re(z_{t1}), Im(z_{t1}), Re(z_{t2}), Im(z_{t2})], and computes
    MMD between simulated and empirical joint distributions.

    Args:
        n_pairs: number of (t1, t2) timepoint pairs to sample per batch item
        max_lag: maximum |t2 - t1| in timesteps
        sigma:   RBF bandwidth (scaled internally by sqrt(feature_dim))
    """

    def __init__(self, n_pairs: int = 32, max_lag: int = 50, sigma: float = 1.0):
        super().__init__()
        self.n_pairs = n_pairs
        self.max_lag = max_lag
        self.sigma = sigma

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        if ts_pred.dim() == 2:
            ts_pred = ts_pred.unsqueeze(0)
        if ts_target.dim() == 2:
            ts_target = ts_target.unsqueeze(0)

        B, N, T = ts_pred.shape
        device = ts_pred.device

        t1 = torch.randint(0, T - 1, (self.n_pairs,), device=device)
        lag = torch.randint(1, min(self.max_lag + 1, T), (self.n_pairs,), device=device)
        t2 = torch.clamp(t1 + lag, max=T - 1)

        def _to_real_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            ar = torch.view_as_real(a.contiguous())  # (B, n_pairs, N, 2)
            br = torch.view_as_real(b.contiguous())
            return torch.cat([ar, br], dim=-2).reshape(B * self.n_pairs, -1)

        pred = _to_real_flat(
            ts_pred[:, :, t1].permute(0, 2, 1),
            ts_pred[:, :, t2].permute(0, 2, 1),
        )
        data = _to_real_flat(
            ts_target[:, :, t1].permute(0, 2, 1),
            ts_target[:, :, t2].permute(0, 2, 1),
        )

        sigma_scaled = self.sigma * (pred.shape[1] ** 0.5)
        return _mmd(pred, data, sigma_scaled)


# ---------------------------------------------------------------------------
# Registry of metric/loss classes
# ---------------------------------------------------------------------------

# Maps loss weight key → nn.Module class.
# FCD takes extra kwargs (tr, fcd_win_sec, fcd_step_sec); others take none.
METRIC_REGISTRY: Dict[str, type] = {
    "fc_correlation":       FCCorrelation,
    "fc_mse":               FCMSE,
    "l2":                   L2Timeseries,
    "amplitude":            AmplitudeLoss,
    "omega":                OmegaLoss,
    "power_spectrum":       PowerSpectrumDistance,
    "temporal_correlation": TemporalCorrelation,
    "autocorrelation":      AutocorrelationDistance,
    "fcd":                  FCD,
    "phfcd":                PhFCD,
    "phase_fc_correlation": PhaseFC,
    "metastability":        Metastability,
    "fdm":                  FDMLoss,
}

# Keys whose constructor accepts (tr, fcd_win_sec, fcd_step_sec)
_DYN_KWARGS_KEYS = {"fcd"}

# Keys whose constructor accepts tr only
_TR_KWARGS_KEYS = {"amplitude", "omega"}

# Keys whose constructor accepts (n_pairs, max_lag, sigma)
_FDM_KWARGS_KEYS = {"fdm"}

# Keys whose forward() accepts an optional fc_target kwarg
_FC_KEYS = {"fc_correlation", "fc_mse"}


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """Weighted sum of named metric/loss terms.

    Each term is an ``nn.Module`` whose ``forward(ts_pred, ts_target)``
    returns a scalar loss tensor.  Zero-weight terms are never instantiated
    or called.

    Example::

        loss_fn = CompositeLoss(
            weights={"fc_correlation": 1.0, "fcd": 0.5},
            tr=0.72, fcd_win_sec=30.0, fcd_step_sec=2.0,
        )
        total, components = loss_fn(ts_pred, ts_target)

    Args:
        weights: Mapping of loss term name → scalar weight.  Terms with
            weight 0 are dropped.
        tr: Repetition time (seconds), forwarded to FCD, AmplitudeLoss, OmegaLoss.
        fcd_win_sec: FCD window length in seconds.
        fcd_step_sec: FCD step size in seconds.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        tr: float = 0.72,
        fcd_win_sec: float = 30.0,
        fcd_step_sec: float = 2.0,
        fdm_n_pairs: int = 32,
        fdm_max_lag: int = 50,
        fdm_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        unknown = set(weights) - set(METRIC_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown loss terms: {unknown}. "
                f"Available: {sorted(METRIC_REGISTRY)}"
            )
        # Drop zero-weight terms
        self.weights = {k: v for k, v in weights.items() if v != 0.0}

        modules: Dict[str, nn.Module] = {}
        for name in self.weights:
            cls = METRIC_REGISTRY[name]
            if name in _DYN_KWARGS_KEYS:
                modules[name] = cls(tr=tr, fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec)
            elif name in _TR_KWARGS_KEYS:
                modules[name] = cls(tr=tr)
            elif name in _FDM_KWARGS_KEYS:
                modules[name] = cls(n_pairs=fdm_n_pairs, max_lag=fdm_max_lag, sigma=fdm_sigma)
            else:
                modules[name] = cls()
        self.terms = nn.ModuleDict(modules)

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        fc_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the weighted composite loss.

        Args:
            ts_pred: ``(batch, n_rois, T)`` complex analytic signal.
            ts_target: ``(batch, n_rois, T)`` complex analytic signal.
            fc_target: Optional precomputed ``(batch, n_rois, n_rois)`` FC
                matrices.  When provided, FC-based loss terms use these
                instead of computing FC from the short *ts_target* window.

        Returns:
            total: weighted scalar loss for back-propagation.
            components: ``{"loss_<name>": tensor, ...}`` for logging.
        """
        device = ts_pred.device
        dtype = ts_pred.real.dtype if torch.is_complex(ts_pred) else ts_pred.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        components: Dict[str, torch.Tensor] = {}

        for name, weight in self.weights.items():
            if name in _FC_KEYS and fc_target is not None:
                value = self.terms[name](ts_pred, ts_target, fc_target=fc_target)
            else:
                value = self.terms[name](ts_pred, ts_target)
            components[f"loss_{name}"] = value
            total = total + weight * value

        return total, components

    @property
    def component_names(self) -> list[str]:
        """Return the ``loss_<name>`` keys that will appear in components."""
        return [f"loss_{name}" for name in self.weights]
