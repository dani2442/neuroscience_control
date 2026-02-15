"""
Differentiable loss functions for neuroscience model training.

Each loss function takes (ts_pred, ts_target, fc_pred, fc_target, **kwargs)
and returns a scalar tensor. Individual terms are composable via
``CompositeLoss``.

Both predicted and target timeseries are **complex-valued** analytic signals
(the dataset applies a Hilbert transform at load time and the models output
native complex state). Amplitude and instantaneous frequency are therefore
extracted directly from the complex representation (​|z| and d∠z/dt)
without an additional Hilbert transform.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from ..metrics._utils import ensure_batch
from ..metrics.dynamics_metrics import (
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


def _complex_amplitude_omega(
    ts: torch.Tensor,
    tr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract per-region amplitude and instantaneous-frequency timeseries
    directly from a complex analytic-signal tensor.

    Args:
        ts: ``(batch, n_rois, n_timepoints)``, complex.
        tr: Repetition time (seconds).

    Returns:
        amplitude: ``(batch, n_rois, n_timepoints)``  — |z|
        omega:     ``(batch, n_rois, n_timepoints - 1)``  — rad/s
    """
    ts = ensure_batch(ts)
    if not torch.is_complex(ts):
        raise ValueError(
            "_complex_amplitude_omega expects complex input; "
            "got real tensor.  Ensure the model output is complex."
        )

    amplitude = ts.abs()                              # |z|, (B, N, T)
    phase = torch.angle(ts)                           # ∠z,  (B, N, T)

    # Instantaneous frequency via finite-difference of phase.
    dphi = torch.diff(phase, dim=2)                   # (B, N, T-1)
    # Wrap to (-π, π]
    dphi = dphi - 2.0 * math.pi * torch.round(dphi / (2.0 * math.pi))
    omega = dphi / tr                                 # rad/s

    return amplitude, omega


def loss_amplitude(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error between predicted envelope amplitude and target mean envelope
    amplitude extracted directly from the complex analytic signal.

    .. math::
        \mathcal{L}_{\mathrm{amp}} =
            \frac{1}{BN}\sum_{b,n}\sum_t
            \bigl(|z^{\mathrm{pred}}_{b,n}(t)|
                  - \overline{|z^{\mathrm{target}}_{b,n}|}\bigr)^2
    """
    amp_pred, _ = _complex_amplitude_omega(ts_pred, tr)
    amp_targ, _ = _complex_amplitude_omega(ts_target, tr)
    B = min(amp_pred.shape[0], amp_targ.shape[0])
    mean_real_amp = amp_targ[:B].mean(dim=2, keepdim=True)  # (B, N, 1)
    sq_l2_per_series = ((amp_pred[:B] - mean_real_amp) ** 2).mean(dim=2)  # (B, N)
    return sq_l2_per_series.mean()


def loss_omega(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error between predicted instantaneous frequency and target mean
    instantaneous frequency extracted directly from the complex analytic
    signal.

    .. math::
        \mathcal{L}_{\omega} =
            \frac{1}{BN}\sum_{b,n}\sum_t
            \bigl(\omega^{\mathrm{pred}}_{b,n}(t)
                  - \bar{\omega}^{\mathrm{target}}_{b,n}\bigr)^2
    """
    _, omega_pred = _complex_amplitude_omega(ts_pred, tr)
    _, omega_targ = _complex_amplitude_omega(ts_target, tr)
    B = min(omega_pred.shape[0], omega_targ.shape[0])
    mean_real_omega = omega_targ[:B].mean(dim=2, keepdim=True)  # (B, N, 1)
    sq_l2_per_series = ((omega_pred[:B] - mean_real_omega) ** 2).mean(dim=2)  # (B, N)
    return sq_l2_per_series.mean()


def loss_fcd(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
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
        tr=tr, fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec,
    )


def loss_metastability(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """L1 difference of Kuramoto metastability."""
    return _metastability_l1_loss(ts_pred, ts_target)


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
    "amplitude":    lambda ts_pred, ts_target, **kw: loss_amplitude(ts_pred, ts_target, **kw),
    "omega":        lambda ts_pred, ts_target, **kw: loss_omega(ts_pred, ts_target, **kw),
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
            weights={"fc_correlation": 1.0, "l2": 0.5, "amplitude": 0.2},
            dyn_kwargs={"tr": 0.72},
        )
        total, components = loss_fn(fc_pred, fc_target, ts_pred, ts_target)
    """

    # Which terms need timeseries (ts) vs functional connectivity (fc) inputs
    _TS_TERMS = {"l2", "amplitude", "omega", "fcd", "metastability"}
    _FC_TERMS = {"fc_mse", "fc_correlation"}

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
                # All TS terms receive the complex timeseries directly;
                # each function handles real-conversion internally when needed.
                value = fn(ts_pred=ts_pred, ts_target=ts_target, **self.dyn_kwargs)
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
        "amplitude": 1.0,
        "omega": 1.0,
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
        dyn_kwargs: Dynamics parameters forwarded to amplitude / omega /
              FCD / metastability terms (``tr``, ``fcd_win_sec``, …).

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
