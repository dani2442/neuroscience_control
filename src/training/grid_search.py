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
from ..metrics import (
    FCCorrelation, FCMSE,
    FCD, Metastability, PhFCD, PhaseFC,
    PowerSpectrumDistance, TemporalCorrelation, AutocorrelationDistance,
)


# Metrics where a *higher* value is better; all others are lower-is-better.
_HIGHER_IS_BETTER = frozenset({"fc_correlation", "phase_fc_correlation"})

# Evaluation modules that don't need timeseries dynamics (FC + timeseries quality)
_BASE_MODULES = [FCCorrelation(), FCMSE(), PowerSpectrumDistance(), TemporalCorrelation(), AutocorrelationDistance()]


class GridSearch:
    """Grid search for hyperparameter optimization.

    Evaluates model performance across a grid of hyperparameters
    using batch simulation with data-derived initial states.

    Supports composite scoring via ``metric_weights`` in :meth:`search`.
    When weights are provided for FCD and/or metastability the search
    optimises the weighted sum

        score = Σ_k  sign_k × w_k × metric_k

    where *sign_k* is +1 for higher-is-better metrics (FC correlation)
    and −1 for lower-is-better metrics (FCD MSE, metastability difference).
    """

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
    ) -> Dict[str, float]:
        """Evaluate a model with given initial states.

        Uses the same ``module.evaluate()`` pattern as the Trainer.  When
        *target_timeseries* is provided all metrics (FC, timeseries quality,
        and dynamics) are computed; otherwise only FC metrics are returned.

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
        """
        batch_size = initial_states.shape[0]
        with torch.no_grad():
            timeseries = model.forward(
                initial_state=initial_states, n_steps=n_timepoints, dt=dt,
                control=control,
            )

        sums: Dict[str, float] = {}
        sum_sq: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        with torch.no_grad():
            if target_timeseries is not None:
                # All metrics via the same module.evaluate() pattern as Trainer
                target_ts = target_timeseries[:batch_size, :, :n_timepoints]
                dyn_modules = [
                    FCD(tr=tr, fcd_win_sec=fcd_win_sec, fcd_step_sec=fcd_step_sec),
                    PhFCD(), Metastability(), PhaseFC(),
                ]
                eval_modules = _BASE_MODULES + dyn_modules
                for i in range(batch_size):
                    sample_metrics: Dict[str, float] = {}
                    for module in eval_modules:
                        sample_metrics.update(module.evaluate(timeseries[i:i+1], target_ts[i:i+1]))
                    for k, v in sample_metrics.items():
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue
                        sums[k] = sums.get(k, 0.0) + v
                        sum_sq[k] = sum_sq.get(k, 0.0) + v * v
                        counts[k] = counts.get(k, 0) + 1
            else:
                # FC-only: compare simulated FC against provided target_fc
                from ..metrics.fc_metrics import compute_static_fc
                from ..metrics._utils import upper_tri_vec, to_real
                fc_tgt = to_real(target_fc.unsqueeze(0))
                targ_flat = upper_tri_vec(fc_tgt, k=1)
                for i in range(batch_size):
                    fc_sim = to_real(compute_static_fc(timeseries[i:i+1]))
                    pred_flat = upper_tri_vec(fc_sim, k=1)
                    pred_c = pred_flat - pred_flat.mean(dim=1, keepdim=True)
                    targ_c = targ_flat - targ_flat.mean(dim=1, keepdim=True)
                    num = (pred_c * targ_c).sum(dim=1)
                    den = (torch.sqrt((pred_c**2).sum(dim=1) * (targ_c**2).sum(dim=1)) + 1e-8)
                    fc_corr = (num / den).mean().item()
                    fc_mse = ((pred_flat - targ_flat) ** 2).mean().item()
                    for k, v in [("fc_correlation", fc_corr), ("fc_mse", fc_mse)]:
                        sums[k] = sums.get(k, 0.0) + v
                        sum_sq[k] = sum_sq.get(k, 0.0) + v * v
                        counts[k] = counts.get(k, 0) + 1

        metrics: Dict[str, float] = {}
        for k in sums:
            n = counts[k]
            mean = sums[k] / n
            metrics[k] = mean
            if n > 1:
                metrics[f"{k}_std"] = np.sqrt(max(0.0, sum_sq[k] / n - mean * mean))
        return metrics

    @staticmethod
    def _composite_score(
        metrics: Dict[str, float],
        metric_weights: Dict[str, float],
    ) -> float:
        """Compute a weighted composite score.

        Higher-is-better metrics (in ``_HIGHER_IS_BETTER``) contribute
        positively; all other metrics contribute negatively.
        NaN / missing metrics are silently skipped.
        """
        score = 0.0
        for key, weight in metric_weights.items():
            val = metrics.get(key, float("nan"))
            if not np.isfinite(val):
                continue
            sign = 1.0 if key in _HIGHER_IS_BETTER else -1.0
            score += sign * weight * val
        return score

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
        verbose: bool = True,
        eval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        combinations = self._generate_param_combinations()
        if not combinations:
            raise ValueError("Parameter grid produced no combinations.")

        if verbose:
            print(f"Grid search over {len(combinations)} parameter combinations")
            combinations = tqdm(combinations)

        _eval_kwargs = eval_kwargs or {}
        use_composite = metric_weights is not None

        for params in combinations:
            full_kwargs = {**model_kwargs, **params, 'device': self.device}
            model = model_class(**full_kwargs)

            metrics = self.evaluate_params(
                model, target_fc, initial_states, n_timepoints, dt,
                **_eval_kwargs,
            )
            self.results.append({'params': params, 'metrics': metrics})

            if use_composite:
                score = self._composite_score(metrics, metric_weights)
            else:
                if metric not in metrics:
                    raise KeyError(f"Metric '{metric}' not found in evaluated metrics.")
                score = metrics[metric]

            # Always accept the first candidate, then compare finite scores.
            if self.best_metrics is None or (
                np.isfinite(score) and (not np.isfinite(self.best_score) or score > self.best_score)
            ):
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
    noise_sigma: float = 0.0,
    n_control_dims: int = 0,
    control: Optional[torch.Tensor] = None,
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
        metric_weights: Optional dict mapping metric names to weights for
            composite scoring (e.g. ``{"fc_correlation": 1.0,
            "fcd_mse": 0.5, "metastability_diff": 0.5}``).
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

    best_params, best_metrics = grid_search.search(
        model_class=CoupledHopfModel,
        model_kwargs=model_kwargs,
        target_fc=target_fc,
        initial_states=initial_states,
        n_timepoints=n_timepoints,
        dt=dt,
        metric_weights=metric_weights,
        eval_kwargs=eval_kwargs,
    )

    print(f"\nBest parameters: {best_params}")
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
