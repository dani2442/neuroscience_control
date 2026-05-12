"""Shared rollout and metric evaluation helpers for training and search."""

from __future__ import annotations

import math
import inspect
from typing import Any, Dict, Iterable, Optional

import torch

from ..dataset import fft_bandpass_3d
from ..metrics import (
    FCCorrelation,
    FCMSE,
    FCD,
    Metastability,
    PhFCD,
    PhaseFC,
    PowerSpectrumDistance,
    TemporalCorrelation,
    AutocorrelationDistance,
)


HIGHER_IS_BETTER_METRICS = frozenset({
    "fc_correlation",
    "temporal_correlation",
    "phase_fc_correlation",
})

EVAL_METRIC_KEYS = (
    "fc_correlation",
    "fc_mse",
    "power_spectrum_distance",
    "temporal_correlation",
    "autocorr_distance",
    "fcd_ks",
    "fcd_mse",
    "phfcd_ks",
    "phfcd_mse",
    "phase_fc_correlation",
    "metastability_diff",
)


class MetricAccumulator:
    """Accumulate scalar metrics and optional standard deviations."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.sum_squares: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, Any]) -> None:
        for key, value in metrics.items():
            numeric = to_float_metric(value)
            if numeric is None or math.isnan(numeric):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + numeric
            self.sum_squares[key] = self.sum_squares.get(key, 0.0) + (numeric * numeric)
            self.counts[key] = self.counts.get(key, 0) + 1

    def average(self, key: str) -> float:
        count = self.counts.get(key, 0)
        if count == 0:
            return float("nan")
        return self.sums[key] / count

    def std(self, key: str) -> float:
        count = self.counts.get(key, 0)
        if count == 0:
            return float("nan")
        mean = self.sums[key] / count
        mean_square = self.sum_squares[key] / count
        variance = max(0.0, mean_square - (mean * mean))
        return math.sqrt(variance)

    def summary(
        self,
        *,
        keys: Optional[Iterable[str]] = None,
        include_std: bool = False,
    ) -> Dict[str, float]:
        metric_keys = list(keys) if keys is not None else sorted(self.sums)
        metrics = {key: self.average(key) for key in metric_keys}
        if include_std:
            metrics.update({f"{key}_std": self.std(key) for key in metric_keys})
        return metrics


def to_float_metric(value: Any) -> float | None:
    """Convert supported metric values to float and drop invalid values."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_eval_metrics(
    *,
    tr: float = 0.72,
    fcd_win_sec: float = 30.0,
    fcd_step_sec: float = 2.0,
) -> list[torch.nn.Module]:
    """Build the canonical evaluation metric suite."""
    return [
        FCCorrelation(),
        FCMSE(),
        PowerSpectrumDistance(),
        TemporalCorrelation(),
        AutocorrelationDistance(),
        FCD(tr=tr, fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec),
        PhFCD(),
        Metastability(),
        PhaseFC(),
    ]


def apply_denoise(
    simulated: torch.Tensor,
    dt: float,
    denoise_f_lo: float | None,
    denoise_f_hi: float | None,
) -> torch.Tensor:
    """Apply FFT bandpass denoising to the real component of a trajectory."""
    if denoise_f_lo is None or denoise_f_hi is None:
        return simulated
    if torch.is_complex(simulated):
        filtered_real = fft_bandpass_3d(simulated.real, dt, denoise_f_lo, denoise_f_hi)
        return torch.complex(filtered_real, simulated.imag)
    return fft_bandpass_3d(simulated, dt, denoise_f_lo, denoise_f_hi)


def rollout_model(
    model,
    initial_state: torch.Tensor,
    n_steps: int,
    dt: float,
    *,
    control: torch.Tensor | None = None,
    sde_type: str | None = None,
    method: str | None = None,
    dt_min: float | None = None,
    use_adjoint: bool | None = None,
    adjoint_method: str | None = None,
    denoise_f_lo: float | None = None,
    denoise_f_hi: float | None = None,
) -> torch.Tensor:
    """Forward-simulate a model and apply optional denoising."""
    kwargs: Dict[str, Any] = {
        "initial_state": initial_state,
        "n_steps": n_steps,
        "dt": dt,
    }
    if control is not None:
        kwargs["control"] = control
    if sde_type is not None:
        kwargs["sde_type"] = sde_type
    if method is not None:
        kwargs["method"] = method
    if dt_min is not None:
        kwargs["dt_min"] = dt_min
    if use_adjoint is not None:
        kwargs["use_adjoint"] = use_adjoint
    if adjoint_method is not None:
        kwargs["adjoint_method"] = adjoint_method

    simulated = model.forward(**kwargs)
    return apply_denoise(simulated, dt, denoise_f_lo, denoise_f_hi)


def evaluate_timeseries_metrics(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    eval_metrics: list[torch.nn.Module],
    *,
    return_std: bool = False,
    group_size: int = 0,
) -> Dict[str, float]:
    """Evaluate the canonical metric suite on a prediction/target pair."""
    accumulator = MetricAccumulator()
    accumulate_timeseries_metrics(
        accumulator,
        ts_pred,
        ts_target,
        eval_metrics,
        group_size=group_size,
    )

    all_keys = sorted(set(EVAL_METRIC_KEYS) | set(accumulator.sums.keys()))
    return accumulator.summary(keys=all_keys, include_std=return_std)


def accumulate_timeseries_metrics(
    accumulator: MetricAccumulator,
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    eval_metrics: list[torch.nn.Module],
    *,
    group_size: int = 0,
) -> None:
    """Update an accumulator with metric results.

    *group_size* is forwarded to each metric's ``evaluate()`` method.
    Metrics that aggregate across the batch (e.g. FC via Fisher-z
    averaging) reshape ``(B, …)`` → ``(G, group_size, …)`` internally
    and average over groups — no Python loop over groups is needed here.
    """
    batch_metrics: Dict[str, float] = {}
    for module in eval_metrics:
        evaluate_sig = inspect.signature(module.evaluate)
        if "group_size" in evaluate_sig.parameters:
            module_metrics = module.evaluate(ts_pred, ts_target, group_size=group_size)
        else:
            module_metrics = module.evaluate(ts_pred, ts_target)
        batch_metrics.update(module_metrics)
    accumulator.update(batch_metrics)


def evaluate_fc_metrics(
    ts_pred: torch.Tensor,
    fc_target: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate FC-only metrics against a precomputed target FC matrix."""
    fc_corr = FCCorrelation()
    fc_mse = FCMSE()
    return {
        "fc_correlation": 1.0 - fc_corr(ts_pred, ts_pred, fc_target=fc_target).item(),
        "fc_mse": fc_mse(ts_pred, ts_pred, fc_target=fc_target).item(),
    }


def composite_metric_score(
    metrics: Dict[str, float],
    metric_weights: Dict[str, float],
) -> float:
    """Weighted score from reported metrics for legacy grid-search objectives."""
    score = 0.0
    for key, weight in metric_weights.items():
        val = metrics.get(key, float("nan"))
        if not math.isfinite(val):
            continue
        sign = 1.0 if key in HIGHER_IS_BETTER_METRICS else -1.0
        score += sign * weight * val
    return score
