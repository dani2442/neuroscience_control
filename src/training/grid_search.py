"""Grid search for hyperparameter optimization."""

import torch
import numpy as np
from itertools import product
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from ..models.hopf_model import CoupledHopfModel
from ..models.base_model import BaseNeuroscienceModel
from .evaluation import (
    EVAL_METRIC_KEYS,
    HIGHER_IS_BETTER_METRICS,
    MetricAccumulator,
    build_eval_metrics,
    composite_metric_score,
    evaluate_fc_metrics,
    evaluate_timeseries_metrics,
    rollout_model,
)
from .losses import CompositeLoss

_HIGHER_IS_BETTER = HIGHER_IS_BETTER_METRICS


class GridSearch:
    """Grid search for hyperparameter optimization.

    Evaluates model performance across a grid of hyperparameters
    using batch simulation with data-derived initial states.

    Supports either the same composite loss used by backpropagation or
    a legacy metric-weighted score for FC-only / metric-only searches.
    """

    _composite_score = staticmethod(composite_metric_score)

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        device: str = "cpu",
        save_dir: str = "results/grid_search",
    ):
        self.param_grid = param_grid
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_score: float = -float('inf')
        self.objective_name: str = "fc_correlation"

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def evaluate_params(
        self,
        model: BaseNeuroscienceModel,
        target_fc: torch.Tensor,
        initial_states: torch.Tensor,
        n_timepoints: int,
        dt: float = 0.72,
        target_timeseries: Optional[torch.Tensor] = None,
        tr: float = 0.72,
        fcd_win_sec: float = 60.0,
        fcd_step_sec: float = 2.0,
        control: Optional[torch.Tensor] = None,
        denoise_f_lo: Optional[float] = None,
        denoise_f_hi: Optional[float] = None,
        loss_fn: Optional[CompositeLoss] = None,
        n_simulations: int = 1,
    ) -> Dict[str, float]:
        """Evaluate a model with given initial states.

        Uses the same rollout, denoising, metric suite, and optional
        composite loss as :class:`Trainer`.

        Args:
            model: Model to evaluate.
            target_fc: Target FC ``(n_rois, n_rois)`` — used when
                *target_timeseries* is not provided.
            initial_states: Complex tensor ``(batch, n_rois)``.
            n_timepoints: Simulation length.
            dt: Time step.
            target_timeseries: Optional ``(batch, n_rois, T)`` complex tensor.
                When given it is used as the target for all metrics.
            tr: Repetition time in seconds (for FCD window sizing).
            fcd_win_sec: FCD window length in seconds.
            fcd_step_sec: FCD window step in seconds.
            control: Optional control input.
            denoise_f_lo: Low-frequency cutoff for bandpass denoising.
            denoise_f_hi: High-frequency cutoff for bandpass denoising.
            loss_fn: Optional composite loss built from training loss weights.
            n_simulations: Number of repeated stochastic rollouts.
        """
        batch_size = initial_states.shape[0]
        num_simulations = max(1, n_simulations)
        metrics_accumulator = MetricAccumulator()
        eval_modules = None
        if target_timeseries is not None:
            eval_modules = build_eval_metrics(
                tr=tr,
                fcd_win_sec=fcd_win_sec,
                fcd_step_sec=fcd_step_sec,
            )
            target_ts = target_timeseries[:batch_size, :, :n_timepoints]

        for _ in range(num_simulations):
            with torch.no_grad():
                timeseries = rollout_model(
                    model,
                    initial_states,
                    n_timepoints,
                    dt,
                    control=control,
                    denoise_f_lo=denoise_f_lo,
                    denoise_f_hi=denoise_f_hi,
                )

            with torch.no_grad():
                if target_timeseries is not None and eval_modules is not None:
                    metrics = evaluate_timeseries_metrics(
                        timeseries,
                        target_ts,
                        eval_modules,
                    )
                    if loss_fn is not None:
                        loss, loss_components = loss_fn(timeseries, target_ts)
                        metrics["loss"] = float(loss.item())
                        for name, value in loss_components.items():
                            metrics[name] = float(value.item())
                else:
                    metrics = evaluate_fc_metrics(timeseries, target_fc)

            metrics_accumulator.update(metrics)

        if target_timeseries is not None:
            extra_keys = tuple(loss_fn.component_names) if loss_fn is not None else ()
            all_keys = sorted(
                set(EVAL_METRIC_KEYS)
                | set(("loss",) if loss_fn is not None else ())
                | set(extra_keys)
                | set(metrics_accumulator.sums.keys())
            )
        else:
            all_keys = sorted(set(("fc_correlation", "fc_mse")) | set(metrics_accumulator.sums.keys()))

        return metrics_accumulator.summary(keys=all_keys, include_std=num_simulations > 1)

    def search(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        target_fc: torch.Tensor,
        initial_states: torch.Tensor,
        n_timepoints: int,
        dt: float = 0.72,
        metric: str = "fc_correlation",
        metric_weights: Optional[Dict[str, float]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        eval_kwargs: Optional[Dict[str, Any]] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        combinations = self._generate_param_combinations()
        if not combinations:
            raise ValueError("Parameter grid produced no combinations.")

        self.results = []
        self.best_params = None
        self.best_metrics = None

        if verbose:
            print(f"Grid search over {len(combinations)} parameter combinations")
            combinations = tqdm(combinations)

        _eval_kwargs = dict(eval_kwargs or {})
        eval_n_simulations = int(_eval_kwargs.pop("n_simulations", 1))
        _loss_kwargs = loss_kwargs or {}
        use_loss_objective = loss_weights is not None and _eval_kwargs.get("target_timeseries") is not None
        loss_fn = CompositeLoss(weights=loss_weights, **_loss_kwargs) if use_loss_objective else None
        use_metric_score = metric_weights is not None
        self.best_score = float("inf") if use_loss_objective else -float("inf")

        if use_loss_objective:
            self.objective_name = "loss"
        elif use_metric_score:
            self.objective_name = "metric_score"
        else:
            self.objective_name = metric

        for params in combinations:
            full_kwargs = {**model_kwargs, **params, 'device': self.device}
            model = model_class(**full_kwargs)

            metrics = self.evaluate_params(
                model, target_fc, initial_states, n_timepoints, dt,
                loss_fn=loss_fn,
                n_simulations=eval_n_simulations,
                **_eval_kwargs,
            )
            self.results.append({'params': params, 'metrics': metrics})

            if use_loss_objective:
                if "loss" not in metrics:
                    raise KeyError("Composite loss objective requested but 'loss' was not computed.")
                score = metrics["loss"]
                is_better = self.best_metrics is None or (
                    np.isfinite(score) and (not np.isfinite(self.best_score) or score < self.best_score)
                )
            elif use_metric_score:
                score = composite_metric_score(metrics, metric_weights)
                is_better = self.best_metrics is None or (
                    np.isfinite(score) and (not np.isfinite(self.best_score) or score > self.best_score)
                )
            else:
                if metric not in metrics:
                    raise KeyError(f"Metric '{metric}' not found in evaluated metrics.")
                score = metrics[metric]
                is_better = self.best_metrics is None or (
                    np.isfinite(score) and (not np.isfinite(self.best_score) or score > self.best_score)
                )

            if is_better:
                self.best_score = score
                self.best_params = params
                self.best_metrics = metrics

        if self.best_params is None or self.best_metrics is None:
            raise RuntimeError("Grid search did not find valid best parameters.")

        self._save_results()
        return self.best_params, self.best_metrics

    def _save_results(self):
        results_data = {
            'param_grid': {k: [float(v) if isinstance(v, (int, float)) else v for v in vals]
                           for k, vals in self.param_grid.items()},
            'results': self.results,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'best_score': self.best_score,
            'objective_name': self.objective_name,
        }
        filepath = self.save_dir / "grid_search_results.json"
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"Grid search results saved to {filepath}")


def grid_search_hopf(
    target_fc: torch.Tensor,
    n_rois: int,
    initial_states: torch.Tensor,
    structural_connectivity: Optional[torch.Tensor] = None,
    omega: Optional[torch.Tensor] = None,
    g_values: Optional[List[float]] = None,
    a_values: Optional[List[float]] = None,
    kappa_values: Optional[List[float]] = None,
    n_timepoints: int = 200,
    dt: float = 0.72,
    device: str = "cpu",
    target_timeseries: Optional[torch.Tensor] = None,
    tr: float = 0.72,
    fcd_win_sec: float = 60.0,
    fcd_step_sec: float = 2.0,
    metric_weights: Optional[Dict[str, float]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    noise_sigma: float = 0.0,
    n_control_dims: int = 0,
    control: Optional[torch.Tensor] = None,
    denoise_f_lo: Optional[float] = None,
    denoise_f_hi: Optional[float] = None,
    fdm_n_pairs: int = 32,
    fdm_max_lag: int = 50,
    fdm_sigma: float = 1.0,
) -> Tuple[Dict[str, Any], CoupledHopfModel]:
    """Grid search for Hopf model parameters.

    Args:
        target_fc: Target FC (n_rois, n_rois).
        n_rois: Number of ROIs.
        initial_states: Complex tensor (batch, n_rois) for evaluation.
        structural_connectivity: Optional SC matrix.
        omega: Optional intrinsic frequencies (rad/s).
        g_values, a_values: Search grid values.
        kappa_values: Search grid values for kappa (default: [0.1]).
        n_timepoints: Simulation length.
        dt: Time step.
        device: Device.
        target_timeseries: Optional (batch, n_rois, T) for FCD/meta scoring.
        tr: Repetition time (seconds) for FCD window sizing.
        fcd_win_sec: FCD window length (seconds).
        fcd_step_sec: FCD window step (seconds).
        metric_weights: Optional legacy metric-weighted search objective.
        loss_weights: Optional training loss weights. When provided together
            with *target_timeseries*, grid search ranks candidates by the same
            composite loss used for backpropagation.
        noise_sigma: Noise scale for SDE diffusion (0.0 = deterministic ODE).

    Returns:
        (best_params, fitted_model)
    """
    if g_values is None:
        g_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    if a_values is None:
        a_values = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02]
    if kappa_values is None:
        kappa_values = [0.1]

    param_grid = {'initial_g': g_values, 'initial_a': a_values}
    if len(kappa_values) > 1:
        param_grid['initial_kappa'] = kappa_values

    grid_search = GridSearch(
        param_grid=param_grid,
        device=device,
    )

    model_kwargs = {
        'n_rois': n_rois,
        'structural_connectivity': structural_connectivity,
        'omega': omega,
        'noise_sigma': noise_sigma,
        'learnable_a': False,
        'learnable_g': False,
        'learnable_kappa': False,
        'n_control_dims': n_control_dims,
    }
    # When kappa is not in the grid, pass the single default value
    if len(kappa_values) == 1:
        model_kwargs['initial_kappa'] = kappa_values[0]

    # Build eval kwargs for dynamics metrics when target timeseries provided.
    eval_kwargs: Dict[str, Any] = {}
    if target_timeseries is not None:
        eval_kwargs.update(
            target_timeseries=target_timeseries,
            tr=tr,
            fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec,
        )
    if control is not None:
        eval_kwargs["control"] = control
    if denoise_f_lo is not None and denoise_f_hi is not None:
        eval_kwargs["denoise_f_lo"] = denoise_f_lo
        eval_kwargs["denoise_f_hi"] = denoise_f_hi

    best_params, best_metrics = grid_search.search(
        model_class=CoupledHopfModel,
        model_kwargs=model_kwargs,
        target_fc=target_fc,
        initial_states=initial_states,
        n_timepoints=n_timepoints,
        dt=dt,
        metric_weights=metric_weights,
        loss_weights=loss_weights,
        eval_kwargs=eval_kwargs,
        loss_kwargs={
            "tr": tr,
            "fcd_win_sec": fcd_win_sec,
            "fcd_step_sec": fcd_step_sec,
            "fdm_n_pairs": fdm_n_pairs,
            "fdm_max_lag": fdm_max_lag,
            "fdm_sigma": fdm_sigma,
        },
    )

    print(f"\nBest parameters: {best_params}")
    if "loss" in best_metrics:
        print(f"Best loss: {best_metrics['loss']:.4f}")
    if "fc_correlation" in best_metrics:
        print(f"Best fc_correlation: {best_metrics['fc_correlation']:.4f} "
              f"± {best_metrics.get('fc_correlation_std', float('nan')):.4f}")

    best_model = CoupledHopfModel(
        n_rois=n_rois,
        structural_connectivity=structural_connectivity,
        omega=omega,
        initial_g=best_params['initial_g'],
        initial_a=best_params['initial_a'],
        initial_kappa=best_params.get('initial_kappa', kappa_values[0]),
        noise_sigma=noise_sigma,
        device=device,
        n_control_dims=n_control_dims,
    )

    return best_params, best_model
