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
}

# Keys whose constructor accepts (tr, fcd_win_sec, fcd_step_sec)
_DYN_KWARGS_KEYS = {"fcd"}

# Keys whose constructor accepts tr only
_TR_KWARGS_KEYS = {"amplitude", "omega"}


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
            else:
                modules[name] = cls()
        self.terms = nn.ModuleDict(modules)

    def forward(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the weighted composite loss.

        Args:
            ts_pred: ``(batch, n_rois, T)`` complex analytic signal.
            ts_target: ``(batch, n_rois, T)`` complex analytic signal.

        Returns:
            total: weighted scalar loss for back-propagation.
            components: ``{"loss_<name>": tensor, ...}`` for logging.
        """
        device = ts_pred.device
        dtype = ts_pred.real.dtype if torch.is_complex(ts_pred) else ts_pred.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        components: Dict[str, torch.Tensor] = {}

        for name, weight in self.weights.items():
            value = self.terms[name](ts_pred, ts_target)
            components[f"loss_{name}"] = value
            total = total + weight * value

        return total, components

    @property
    def component_names(self) -> list[str]:
        """Return the ``loss_<name>`` keys that will appear in components."""
        return [f"loss_{name}" for name in self.weights]
