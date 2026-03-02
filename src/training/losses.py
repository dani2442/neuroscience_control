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
    phase_coherence_fc_correlation as _phase_coherence_fc_correlation,
    phfcd_mse_loss as _phfcd_mse_loss,
)
from ..metrics.fc_metrics import fc_correlation, fc_mse


# ---------------------------------------------------------------------------
# Dataset-level reference statistics
# ---------------------------------------------------------------------------

def compute_ref_amplitude(timeseries: torch.Tensor) -> torch.Tensor:
    """Mean envelope amplitude per ROI across the full dataset.

    Args:
        timeseries: ``(n_subjects, n_rois, T)`` complex analytic signal.

    Returns:
        ``(n_rois,)`` — mean |z| over subjects and time.
    """
    ts = timeseries if timeseries.dim() == 3 else timeseries.unsqueeze(0)
    return ts.abs().mean(dim=(0, 2))


def compute_ref_omega(
    timeseries: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
) -> torch.Tensor:
    """Per-ROI intrinsic angular frequency via FFT peak detection.

    Delegates to :func:`~src.dataset.preprocessing.compute_omega_from_timeseries`
    so that the loss reference matches the model's ω initialisation.

    Args:
        timeseries: ``(n_subjects, n_rois, T)`` complex analytic signal.
        tr: Repetition time in seconds.
        f_lo, f_hi: Frequency band of interest (Hz).

    Returns:
        ``(n_rois,)`` — angular frequencies (rad/s).
    """
    from ..dataset.preprocessing import compute_omega_from_timeseries

    return compute_omega_from_timeseries(
        timeseries, dt=tr, f_lo=f_lo, f_hi=f_hi, method="peak",
    )


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
    ref_amplitude: Optional[torch.Tensor] = None,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error between *mean* predicted amplitude and a per-ROI reference.

    When *ref_amplitude* ``(n_rois,)`` is provided (precomputed from the full
    dataset via :func:`compute_ref_amplitude`), it is used as the target.
    Otherwise falls back to the per-window mean of *ts_target*.

    .. math::
        \mathcal{L}_{\mathrm{amp}} =
            \frac{1}{N}\sum_n
            \bigl(\overline{|z^{\mathrm{pred}}_n|}
                  - A^{\mathrm{ref}}_n\bigr)^2
    """
    amp_pred, _ = _complex_amplitude_omega(ts_pred, tr)
    mean_pred = amp_pred.mean(dim=(2))                        # (B, N)
    if ref_amplitude is not None:
        target = ref_amplitude.to(mean_pred.device)              # (B, N)
    else:
        amp_targ, _ = _complex_amplitude_omega(ts_target, tr)
        target = amp_targ.mean(dim=(2))                       # (B, N)
    return ((mean_pred - target) ** 2).mean()


def loss_omega(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    ref_omega: Optional[torch.Tensor] = None,
    **_kwargs,
) -> torch.Tensor:
    r"""
    L² error between *mean* predicted instantaneous frequency and a per-ROI
    reference.

    When *ref_omega* ``(n_rois,)`` is provided (precomputed from the full
    dataset via :func:`compute_ref_omega`), it is used as the target.
    Otherwise falls back to the per-window mean of *ts_target*.

    .. math::
        \mathcal{L}_{\omega} =
            \frac{1}{N}\sum_n
            \bigl(\bar{\omega}^{\mathrm{pred}}_n
                  - \omega^{\mathrm{ref}}_n\bigr)^2
    """
    _, omega_pred = _complex_amplitude_omega(ts_pred, tr)
    mean_pred = omega_pred.mean(dim=(0, 2))                      # (N,)
    if ref_omega is not None:
        target = ref_omega.to(mean_pred.device)                  # (N,)
    else:
        _, omega_targ = _complex_amplitude_omega(ts_target, tr)
        target = omega_targ.mean(dim=(0, 2))                     # (N,)
    return ((mean_pred - target) ** 2).mean()


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


def loss_phfcd(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Differentiable phFCD loss: MSE between phFCD matrices.

    .. note::
        This is a differentiable *surrogate* for the Kolmogorov-Smirnov
        distance reported by the evaluation metric ``phfcd_ks``.  MSE between
        phFCD matrices is used because the KS statistic is non-differentiable.
    """
    return _phfcd_mse_loss(ts_pred, ts_target)


def loss_metastability(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """L1 difference of Kuramoto metastability."""
    return _metastability_l1_loss(ts_pred, ts_target)


def loss_phase_fc_correlation(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    r"""1 − Pearson correlation between phase-coherence FC matrices.

    The phase-coherence FC is the time-averaged instantaneous phase-coherence
    matrix (Eq. 11 in Deco et al. 2019).  This loss converts the similarity
    metric into a minimizable objective analogous to :func:`loss_fc_correlation`.
    """
    corr = _phase_coherence_fc_correlation(ts_pred, ts_target)
    return 1.0 - torch.tensor(corr, device=ts_pred.device, dtype=ts_pred.real.dtype)


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
    "phfcd":        lambda ts_pred, ts_target, **kw: loss_phfcd(ts_pred, ts_target, **kw),
    "phase_fc_correlation": lambda ts_pred, ts_target, **kw: loss_phase_fc_correlation(ts_pred, ts_target, **kw),
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
    _TS_TERMS = {"l2", "amplitude", "omega", "fcd", "phfcd", "phase_fc_correlation", "metastability"}
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
# Preset factory
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Dict[str, float]] = {
    "mse":         {"fc_mse": 1.0},
    "correlation": {"fc_correlation": 1.0},
    "combined":    {"fc_mse": 1.0, "fc_correlation": 0.5},
    "fc_fcd_meta": {"fc_correlation": 1.0, "fcd": 1.0, "metastability": 1.0},
    "fc_phfcd_meta": {"fc_correlation": 1.0, "phfcd": 1.0, "metastability": 1.0},
    "full":        {
        "fc_correlation": 1.0,
        "l2": 1.0,
        "amplitude": 1.0,
        "omega": 1.0,
        "fcd": 1.0,
        "phfcd": 1.0,
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
