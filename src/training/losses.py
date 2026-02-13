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


def loss_fc_correlation(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """1 − Pearson correlation between FC matrices (upper triangle)."""
    return 1.0 - fc_correlation(fc_pred, fc_target)


def loss_fc_corr(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Backward-compatible alias for :func:`loss_fc_correlation`."""
    return loss_fc_correlation(fc_pred, fc_target)


def loss_l2_timeseries(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    r"""
    :math:`L^2` error between predicted and target timeseries.

    Supports both real and complex tensors.  For complex inputs the loss
    is :math:`\frac{1}{B N T}\sum |x^{\mathrm{pred}} - x^{\mathrm{target}}|^2`
    (i.e. the squared modulus, capturing both real and imaginary parts).

    Both tensors are expected in shape ``(batch, n_rois, n_timepoints)``.
    If the time dimensions differ the shorter one is used.
    """
    if ts_pred.ndim == 2:
        ts_pred = ts_pred.unsqueeze(0)
    if ts_target.ndim == 2:
        ts_target = ts_target.unsqueeze(0)
    T = min(ts_pred.shape[2], ts_target.shape[2])
    B = min(ts_pred.shape[0], ts_target.shape[0])
    diff = ts_pred[:B, :, :T] - ts_target[:B, :, :T]
    if torch.is_complex(diff):
        return (diff.real ** 2 + diff.imag ** 2).mean()
    return (diff ** 2).mean()


def _hilbert_amplitude_omega_timeseries(
    ts: torch.Tensor,
    tr: float,
    f_lo: float,
    f_hi: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-region amplitude and instantaneous-frequency timeseries
    from the analytic (Hilbert) signal.

    Args:
        ts: ``(batch, n_rois, n_timepoints)``
        tr, f_lo, f_hi: bandpass parameters

    Returns:
        amplitude: ``(batch, n_rois, n_timepoints)``
        omega:     ``(batch, n_rois, n_timepoints - 1)``  (rad / s)
    """
    ts = _ensure_batch(ts)
    B, N, T = ts.shape

    # Merge batch and ROI dims so helpers see a single (T, B*N) matrix.
    x = ts.reshape(B * N, T).transpose(0, 1)       # (T, B*N)
    x = _preprocess_timeseries(x, tr, f_lo, f_hi)  # (T, B*N)
    z = analytic_signal(x)                          # (T, B*N), complex

    amplitude = z.abs()                             # (T, B*N)
    phase = torch.angle(z)                          # (T, B*N)

    # Instantaneous frequency via finite-difference of unwrapped phase.
    # torch has no unwrap; use angle-difference mod 2π instead.
    dphi = torch.diff(phase, dim=0)                 # (T-1, B*N)
    # Wrap to (-π, π]
    dphi = dphi - 2.0 * math.pi * torch.round(dphi / (2.0 * math.pi))
    inst_omega = dphi / tr                          # rad / s, (T-1, B*N)

    # Reshape back to (B, N, T) / (B, N, T-1)
    amp = amplitude.transpose(0, 1).reshape(B, N, T)
    omega = inst_omega.transpose(0, 1).reshape(B, N, T - 1)
    return amp, omega


def loss_hilbert_amplitude(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error between predicted envelope amplitude and target mean envelope
    amplitude extracted via the Hilbert transform.

    .. math::
        \mathcal{L}_{\mathrm{amp}} =
            \frac{1}{BN}\sum_{b,n}\sum_t
            \bigl(A^{\mathrm{pred}}_{b,n}(t)
                  - \bar{A}^{\mathrm{target}}_{b,n}\bigr)^2

    where :math:`\bar{A}^{\mathrm{target}}_{b,n} = \frac{1}{T_\mathrm{target}}
    \sum_t A^{\mathrm{target}}_{b,n}(t)`.
    """
    amp_pred, _ = _hilbert_amplitude_omega_timeseries(ts_pred, tr, f_lo, f_hi)
    amp_targ, _ = _hilbert_amplitude_omega_timeseries(ts_target, tr, f_lo, f_hi)
    B = min(amp_pred.shape[0], amp_targ.shape[0])
    mean_real_amp = amp_targ[:B].mean(dim=2, keepdim=True)  # (B, N, 1)
    sq_l2_per_series = ((amp_pred[:B] - mean_real_amp) ** 2).mean(dim=2)  # (B, N)
    return sq_l2_per_series.mean()


def loss_hilbert_omega(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error between predicted instantaneous frequency and target mean
    instantaneous frequency extracted via the Hilbert transform.

    .. math::
        \mathcal{L}_{\omega} =
            \frac{1}{BN}\sum_{b,n}\sum_t
            \bigl(\omega^{\mathrm{pred}}_{b,n}(t)
                  - \bar{\omega}^{\mathrm{target}}_{b,n}\bigr)^2
    """
    _, omega_pred = _hilbert_amplitude_omega_timeseries(ts_pred, tr, f_lo, f_hi)
    _, omega_targ = _hilbert_amplitude_omega_timeseries(ts_target, tr, f_lo, f_hi)
    B = min(omega_pred.shape[0], omega_targ.shape[0])
    mean_real_omega = omega_targ[:B].mean(dim=2, keepdim=True)  # (B, N, 1)
    sq_l2_per_series = ((omega_pred[:B] - mean_real_omega) ** 2).mean(dim=2)  # (B, N)
    return sq_l2_per_series.mean()


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
    "fc_correlation": lambda fc_pred, fc_target, **kw: loss_fc_correlation(fc_pred, fc_target),
    "l2":           lambda ts_pred, ts_target, **kw: loss_l2_timeseries(ts_pred, ts_target),
    "hilbert_amp":  lambda ts_pred, ts_target, **kw: loss_hilbert_amplitude(ts_pred, ts_target, **kw),
    "hilbert_omega": lambda ts_pred, ts_target, **kw: loss_hilbert_omega(ts_pred, ts_target, **kw),
    "fcd":          lambda ts_pred, ts_target, **kw: loss_fcd(ts_pred, ts_target, **kw),
    "metastability": lambda ts_pred, ts_target, **kw: loss_metastability(ts_pred, ts_target, **kw),
}

_LOSS_TERM_ALIASES: Dict[str, str] = {
    "fc_corr": "fc_correlation",
}


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------

class CompositeLoss:
    """
    Weighted sum of named loss terms.

    Example::

        loss_fn = CompositeLoss(
            weights={"fc_correlation": 1.0, "l2": 0.5, "hilbert_amp": 0.2},
            dyn_kwargs={"tr": 0.72, "f_lo": 0.04, "f_hi": 0.07},
        )
        total, components = loss_fn(fc_pred, fc_target, ts_pred, ts_target)
    """

    # Which terms need timeseries (ts) vs functional connectivity (fc) inputs
    _TS_TERMS = {"l2", "hilbert_amp", "hilbert_omega", "fcd", "metastability"}
    _FC_TERMS = {"fc_mse", "fc_correlation"}
    # TS terms that accept complex tensors directly (no real conversion)
    _COMPLEX_TS_TERMS = {"l2"}

    def __init__(
        self,
        weights: Dict[str, float],
        dyn_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        normalized_weights: Dict[str, float] = {}
        for raw_name, value in weights.items():
            name = _LOSS_TERM_ALIASES.get(raw_name, raw_name)
            normalized_weights[name] = normalized_weights.get(name, 0.0) + value

        unknown = set(normalized_weights) - set(LOSS_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown loss terms: {unknown}. "
                f"Available: {sorted(LOSS_REGISTRY)}"
            )
        # Drop zero-weight terms so we never compute them
        self.weights = {k: v for k, v in normalized_weights.items() if v != 0.0}
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
            elif name in self._COMPLEX_TS_TERMS:
                # L2 loss operates on complex timeseries directly
                value = fn(ts_pred=ts_pred, ts_target=ts_target, **self.dyn_kwargs)
            else:
                # Hilbert/FCD/metastability: extract real part for spectral analysis
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
    "correlation": {"fc_correlation": 1.0},
    "combined":    {"fc_mse": 1.0, "fc_correlation": 0.5},
    "fc_fcd_meta": {"fc_correlation": 1.0, "fcd": 1.0, "metastability": 1.0},
    "full":        {
        "fc_correlation": 1.0,
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
