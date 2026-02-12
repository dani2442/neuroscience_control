"""Grid search for hyperparameter optimization."""

import torch
import numpy as np
from itertools import product
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from src.models.hopf_model import CoupledHopfModel
from src.models.base_model import BaseNeuroscienceModel
from src.metrics import fc_correlation, fc_mse
from src.dataset import NeuroscienceDataset
from src.dataset.preprocessing import compute_omega_from_timeseries


class GridSearch:
    """Grid search for hyperparameter optimization.

    Evaluates model performance across a grid of hyperparameters
    using batch simulation with data-derived initial states.
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
    ) -> Dict[str, float]:
        """Evaluate a model with given initial states.

        Args:
            model: Model to evaluate.
            target_fc: Target FC (n_rois, n_rois).
            initial_states: Complex tensor (batch, n_rois).
            n_timepoints: Simulation length.
            dt: Time step.
        """
        batch_size = initial_states.shape[0]
        with torch.no_grad():
            timeseries = model.forward(
                initial_state=initial_states, n_steps=n_timepoints, dt=dt,
            )
            fc_pred = model.compute_fc(timeseries)

            fc_corrs, fc_mses = [], []
            for i in range(batch_size):
                fc_corrs.append(fc_correlation(fc_pred[i:i+1], target_fc.unsqueeze(0)).item())
                fc_mses.append(fc_mse(fc_pred[i:i+1], target_fc.unsqueeze(0)).item())

        return {
            'fc_correlation_mean': np.mean(fc_corrs),
            'fc_correlation_std': np.std(fc_corrs),
            'fc_mse_mean': np.mean(fc_mses),
            'fc_mse_std': np.std(fc_mses),
        }

    def search(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        target_fc: torch.Tensor,
        initial_states: torch.Tensor,
        n_timepoints: int,
        dt: float = 0.72,
        metric: str = "fc_correlation_mean",
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        combinations = self._generate_param_combinations()

        if verbose:
            print(f"Grid search over {len(combinations)} parameter combinations")
            combinations = tqdm(combinations)

        for params in combinations:
            full_kwargs = {**model_kwargs, **params, 'device': self.device}
            model = model_class(**full_kwargs)

            metrics = self.evaluate_params(model, target_fc, initial_states, n_timepoints, dt)
            self.results.append({'params': params, 'metrics': metrics})

            score = metrics[metric]
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_metrics = metrics

        self._save_results()
        return self.best_params, self.best_metrics

    def _save_results(self):
        results_data = {
            'param_grid': {k: [float(v) if isinstance(v, (int, float)) else v for v in vals]
                           for k, vals in self.param_grid.items()},
            'results': self.results,
            'best_params': self.best_params,
            'best_score': self.best_score,
        }
        filepath = self.save_dir / "grid_search_results.json"
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"Grid search results saved to {filepath}")

    def get_results_dataframe(self):
        return [{**r['params'], **r['metrics']} for r in self.results]


def grid_search_hopf(
    target_fc: torch.Tensor,
    n_rois: int,
    initial_states: torch.Tensor,
    structural_connectivity: Optional[torch.Tensor] = None,
    omega: Optional[torch.Tensor] = None,
    g_values: Optional[List[float]] = None,
    a_values: Optional[List[float]] = None,
    n_timepoints: int = 200,
    dt: float = 0.72,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], CoupledHopfModel]:
    """Grid search for Hopf model parameters.

    Args:
        target_fc: Target FC (n_rois, n_rois).
        n_rois: Number of ROIs.
        initial_states: Complex tensor (batch, n_rois) for evaluation.
        structural_connectivity: Optional SC matrix.
        omega: Optional intrinsic frequencies (rad/s).
        g_values, a_values: Search grid values.
        n_timepoints: Simulation length.
        dt: Time step.
        device: Device.

    Returns:
        (best_params, fitted_model)
    """
    if g_values is None:
        g_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    if a_values is None:
        a_values = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02]

    grid_search = GridSearch(
        param_grid={'initial_g': g_values, 'initial_a': a_values},
        device=device,
    )

    model_kwargs = {
        'n_rois': n_rois,
        'structural_connectivity': structural_connectivity,
        'omega': omega,
        'learnable_a': False,
        'learnable_g': False,
    }

    best_params, best_metrics = grid_search.search(
        model_class=CoupledHopfModel,
        model_kwargs=model_kwargs,
        target_fc=target_fc,
        initial_states=initial_states,
        n_timepoints=n_timepoints,
        dt=dt,
    )

    print(f"\nBest parameters: {best_params}")
    print(f"Best FC correlation: {best_metrics['fc_correlation_mean']:.4f} "
          f"± {best_metrics['fc_correlation_std']:.4f}")

    best_model = CoupledHopfModel(
        n_rois=n_rois,
        structural_connectivity=structural_connectivity,
        omega=omega,
        initial_g=best_params['initial_g'],
        initial_a=best_params['initial_a'],
        device=device,
    )

    return best_params, best_model
