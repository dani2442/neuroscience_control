"""
Differentiable loss functions for neuroscience model training.

Each loss function takes (ts_pred, ts_target, fc_pred, fc_target, **kwargs)
and returns a scalar tensor. Individual terms are composable via
`CompositeLoss`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from ..metrics.dynamics_metrics import (
    _ensure_batch,
    _preprocess_timeseries,
    analytic_signal,
    fcd_mse_loss as _fcd_mse_loss,
    metastability_l1_loss as _metastability_l1_loss,
)
from ..metrics.fc_metrics import fc_correlation, fc_mse


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def loss_fc_mse(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """MSE between predicted and target FC (upper triangle)."""
    return fc_mse(fc_pred, fc_target)


def loss_fc_corr(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """1 − Pearson correlation between FC matrices (upper triangle)."""
    return 1.0 - fc_correlation(fc_pred, fc_target)


def loss_l2_timeseries(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    r"""
    Standard :math:`L^2` error between predicted and target timeseries.

    .. math::
        \mathcal{L} = \frac{1}{B \cdot N \cdot T}
            \sum_{b,n,t} \bigl(x^{\mathrm{pred}}_{b,n,t}
                               - x^{\mathrm{target}}_{b,n,t}\bigr)^2

    Both tensors are expected in shape ``(batch, n_rois, n_timepoints)``.
    If the time dimensions differ the shorter one is used.
    """
    ts_pred = _ensure_batch(ts_pred)
    ts_target = _ensure_batch(ts_target)
    T = min(ts_pred.shape[2], ts_target.shape[2])
    B = min(ts_pred.shape[0], ts_target.shape[0])
    return F.mse_loss(ts_pred[:B, :, :T], ts_target[:B, :, :T])


def _hilbert_amplitude_omega(
    ts: torch.Tensor,
    tr: float,
    f_lo: float,
    f_hi: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-region mean amplitude and mean instantaneous frequency
    from the analytic (Hilbert) signal.

    Args:
        ts: ``(batch, n_rois, n_timepoints)``
        tr, f_lo, f_hi: bandpass parameters

    Returns:
        mean_amplitude: ``(batch, n_rois)``
        mean_omega:     ``(batch, n_rois)``  (rad / s)
    """
    ts = _ensure_batch(ts)
    B, N, T = ts.shape

    amp_list = []
    omega_list = []
    for b in range(B):
        # x: (T, N) after transpose
        x = _preprocess_timeseries(ts[b].transpose(0, 1), tr, f_lo, f_hi)
        z = analytic_signal(x)  # (T, N), complex

        amplitude = z.abs()                        # (T, N)
        phase = torch.angle(z)                     # (T, N)
        # Instantaneous frequency via finite-difference of unwrapped phase.
        # torch has no unwrap; use angle-difference mod 2π instead.
        dphi = torch.diff(phase, dim=0)            # (T-1, N)
        # Wrap to (-π, π]
        dphi = dphi - 2.0 * math.pi * torch.round(dphi / (2.0 * math.pi))
        inst_omega = dphi / tr                     # rad / s, (T-1, N)

        amp_list.append(amplitude.mean(dim=0))      # (N,)
        omega_list.append(inst_omega.mean(dim=0))   # (N,)

    mean_amp = torch.stack(amp_list, dim=0)    # (B, N)
    mean_omega = torch.stack(omega_list, dim=0)  # (B, N)
    return mean_amp, mean_omega


def loss_hilbert_amplitude(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error on the per-region mean envelope amplitude extracted via the
    Hilbert transform.

    .. math::
        \mathcal{L}_{\mathrm{amp}} = \frac{1}{N}
            \sum_n \bigl(\bar{A}_n^{\mathrm{pred}}
                        - \bar{A}_n^{\mathrm{target}}\bigr)^2

    where :math:`\bar{A}_n = \frac{1}{T}\sum_t |z_n(t)|` is the temporal
    mean of the analytic-signal envelope for region *n*.
    """
    amp_pred, _ = _hilbert_amplitude_omega(ts_pred, tr, f_lo, f_hi)
    amp_targ, _ = _hilbert_amplitude_omega(ts_target, tr, f_lo, f_hi)
    B = min(amp_pred.shape[0], amp_targ.shape[0])
    return F.mse_loss(amp_pred[:B], amp_targ[:B])


def loss_hilbert_omega(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error on the per-region mean instantaneous frequency extracted via
    the Hilbert transform.

    .. math::
        \mathcal{L}_{\omega} = \frac{1}{N}
            \sum_n \bigl(\bar{\omega}_n^{\mathrm{pred}}
                        - \bar{\omega}_n^{\mathrm{target}}\bigr)^2
    """
    _, omega_pred = _hilbert_amplitude_omega(ts_pred, tr, f_lo, f_hi)
    _, omega_targ = _hilbert_amplitude_omega(ts_target, tr, f_lo, f_hi)
    B = min(omega_pred.shape[0], omega_targ.shape[0])
    return F.mse_loss(omega_pred[:B], omega_targ[:B])


def loss_fcd(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    fcd_win_sec: float = 60.0,
    fcd_step_sec: float = 2.0,
    **_kwargs,
) -> torch.Tensor:
    """Differentiable FCD loss: MSE between FCD matrices.

    .. note::
        This is a differentiable *surrogate* for the Kolmogorov-Smirnov
        distance reported by the evaluation metric ``fcd_ks``.  MSE between
        FCD matrices is used because the KS statistic is non-differentiable
        and therefore unsuitable for gradient-based optimization.
    """
    return _fcd_mse_loss(
        ts_pred, ts_target,
        tr=tr, f_lo=f_lo, f_hi=f_hi,
        fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec,
    )


def loss_metastability(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    **_kwargs,
) -> torch.Tensor:
    """L1 difference of Kuramoto metastability."""
    return _metastability_l1_loss(
        ts_pred, ts_target,
        tr=tr, f_lo=f_lo, f_hi=f_hi,
    )


# ---------------------------------------------------------------------------
# Registry of named loss terms
# ---------------------------------------------------------------------------

# Each entry maps a short name to a callable with signature:
#   (ts_pred, ts_target, fc_pred, fc_target, **dyn_kwargs) -> scalar tensor
#
# Wrapper lambdas normalise the calling convention so that losses which only
# need a subset of arguments still receive all of them via **kwargs.

LossFn = Callable[..., torch.Tensor]

LOSS_REGISTRY: Dict[str, LossFn] = {
    "fc_mse":       lambda fc_pred, fc_target, **kw: loss_fc_mse(fc_pred, fc_target),
    "fc_corr":      lambda fc_pred, fc_target, **kw: loss_fc_corr(fc_pred, fc_target),
    "l2":           lambda ts_pred, ts_target, **kw: loss_l2_timeseries(ts_pred, ts_target),
    "hilbert_amp":  lambda ts_pred, ts_target, **kw: loss_hilbert_amplitude(ts_pred, ts_target, **kw),
    "hilbert_omega": lambda ts_pred, ts_target, **kw: loss_hilbert_omega(ts_pred, ts_target, **kw),
    "fcd":          lambda ts_pred, ts_target, **kw: loss_fcd(ts_pred, ts_target, **kw),
    "metastability": lambda ts_pred, ts_target, **kw: loss_metastability(ts_pred, ts_target, **kw),
}


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class CompositeLoss:
    """
    Weighted sum of named loss terms.

    Example::

        loss_fn = CompositeLoss(
            weights={"fc_corr": 1.0, "l2": 0.5, "hilbert_amp": 0.2},
            dyn_kwargs={"tr": 0.72, "f_lo": 0.04, "f_hi": 0.07},
        )
        total, components = loss_fn(fc_pred, fc_target, ts_pred, ts_target)
    """

    # Which terms need timeseries (ts) vs functional connectivity (fc) inputs
    _TS_TERMS = {"l2", "hilbert_amp", "hilbert_omega", "fcd", "metastability"}
    _FC_TERMS = {"fc_mse", "fc_corr"}

    def __init__(
        self,
        weights: Dict[str, float],
        dyn_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        unknown = set(weights) - set(LOSS_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown loss terms: {unknown}. "
                f"Available: {sorted(LOSS_REGISTRY)}"
            )
        # Drop zero-weight terms so we never compute them
        self.weights = {k: v for k, v in weights.items() if v != 0.0}
        self.dyn_kwargs = dyn_kwargs or {}

    def __call__(
        self,
        fc_pred: torch.Tensor,
        fc_target: torch.Tensor,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the weighted composite loss.

        Returns:
            total: weighted scalar loss for back-propagation
            components: ``{"loss_<name>": tensor, ...}`` for logging
        """
        device = fc_pred.device
        dtype = fc_pred.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        components: Dict[str, torch.Tensor] = {}
        total = zero.clone()

        for name, weight in self.weights.items():
            fn = LOSS_REGISTRY[name]
            if name in self._FC_TERMS:
                value = fn(fc_pred=fc_pred, fc_target=fc_target, **self.dyn_kwargs)
            else:
                # Convert complex timeseries to real for TS loss terms
                ts_p = ts_pred.real if torch.is_complex(ts_pred) else ts_pred
                ts_t = ts_target.real if torch.is_complex(ts_target) else ts_target
                value = fn(ts_pred=ts_p, ts_target=ts_t, **self.dyn_kwargs)
            components[f"loss_{name}"] = value
            total = total + weight * value

        return total, components

    @property
    def component_names(self) -> list[str]:
        """Return the ``loss_<name>`` keys that will appear in components."""
        return [f"loss_{name}" for name in self.weights]


# ---------------------------------------------------------------------------
# Preset factory (backward-compatible short-hand names)
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Dict[str, float]] = {
    "mse":         {"fc_mse": 1.0},
    "correlation": {"fc_corr": 1.0},
    "combined":    {"fc_mse": 1.0, "fc_corr": 0.5},
    "fc_fcd_meta": {"fc_corr": 1.0, "fcd": 1.0, "metastability": 1.0},
    "full":        {
        "fc_corr": 1.0,
        "l2": 1.0,
        "hilbert_amp": 1.0,
        "hilbert_omega": 1.0,
        "fcd": 1.0,
        "metastability": 1.0,
    },
}


def build_loss(
    name: str,
    weight_overrides: Optional[Dict[str, float]] = None,
    dyn_kwargs: Optional[Dict[str, float]] = None,
) -> CompositeLoss:
    """
    Construct a :class:`CompositeLoss` from a preset name or a custom
    weight dictionary.

    Args:
        name: One of the preset names (``"mse"``, ``"correlation"``,
              ``"combined"``, ``"fc_fcd_meta"``, ``"full"``),
              or ``"custom"`` when *weight_overrides* supplies all weights.
        weight_overrides: Per-term weight overrides applied on top of the
              preset.  For ``"custom"`` this is the full weight dict.
        dyn_kwargs: Dynamics parameters forwarded to Hilbert / FCD /
              metastability terms (``tr``, ``f_lo``, ``f_hi``, …).

    Returns:
        A ready-to-call :class:`CompositeLoss`.
    """
    if name == "custom":
        if not weight_overrides:
            raise ValueError("'custom' loss requires weight_overrides dict")
        weights = dict(weight_overrides)
    elif name in _PRESETS:
        weights = dict(_PRESETS[name])
        if weight_overrides:
            weights.update(weight_overrides)
    else:
        raise ValueError(
            f"Unknown loss preset '{name}'. "
            f"Available: {sorted(list(_PRESETS) + ['custom'])}"
        )

    return CompositeLoss(weights=weights, dyn_kwargs=dyn_kwargs or {})
