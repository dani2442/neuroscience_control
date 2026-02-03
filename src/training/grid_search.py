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
from ..metrics import fc_correlation, fc_mse, compute_all_fc_metrics
from ..dataset import NeuroscienceDataset


class GridSearch:
    """
    Grid search for hyperparameter optimization.
    
    Evaluates model performance across a grid of hyperparameters
    without gradient-based optimization (useful for Hopf model).
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        n_simulations: int = 10,
        device: str = "cpu",
        save_dir: str = "results/grid_search"
    ):
        """
        Initialize grid search.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            n_simulations: Number of simulations per parameter setting
            device: Device to run on
            save_dir: Directory to save results
        """
        self.param_grid = param_grid
        self.n_simulations = n_simulations
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -float('inf')
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def evaluate_params(
        self,
        model: BaseNeuroscienceModel,
        target_fc: torch.Tensor,
        n_timepoints: int,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Evaluate a model with given parameters.
        
        Args:
            model: Model to evaluate
            target_fc: Target functional connectivity
            n_timepoints: Number of timepoints to simulate
            dt: Time step
            
        Returns:
            Dictionary of metrics
        """
        all_fc_corrs = []
        all_fc_mses = []
        
        for _ in range(self.n_simulations):
            # Simulate
            timeseries = model.forward(
                initial_state=None,
                n_steps=n_timepoints,
                dt=dt,
                batch_size=1
            )
            
            # Compute FC
            fc_pred = model.compute_fc(timeseries)
            
            # Compute metrics
            fc_corr = fc_correlation(fc_pred, target_fc.unsqueeze(0))
            fc_mse_val = fc_mse(fc_pred, target_fc.unsqueeze(0))
            
            all_fc_corrs.append(fc_corr.item())
            all_fc_mses.append(fc_mse_val.item())
        
        return {
            'fc_correlation_mean': np.mean(all_fc_corrs),
            'fc_correlation_std': np.std(all_fc_corrs),
            'fc_mse_mean': np.mean(all_fc_mses),
            'fc_mse_std': np.std(all_fc_mses)
        }
    
    def search(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        target_fc: torch.Tensor,
        n_timepoints: int,
        dt: float = 0.01,
        metric: str = "fc_correlation_mean",
        verbose: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Run grid search.
        
        Args:
            model_class: Class of model to search
            model_kwargs: Fixed kwargs for model
            target_fc: Target functional connectivity
            n_timepoints: Number of timepoints
            dt: Time step
            metric: Metric to optimize
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, best_metrics)
        """
        combinations = self._generate_param_combinations()
        
        if verbose:
            print(f"Grid search over {len(combinations)} parameter combinations")
            combinations = tqdm(combinations)
        
        for params in combinations:
            # Create model with these parameters
            full_kwargs = {**model_kwargs, **params, 'device': self.device}
            model = model_class(**full_kwargs)
            
            # Evaluate
            metrics = self.evaluate_params(model, target_fc, n_timepoints, dt)
            
            # Store results
            result = {
                'params': params,
                'metrics': metrics
            }
            self.results.append(result)
            
            # Check if best
            score = metrics[metric]
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                self.best_metrics = metrics
        
        # Save results
        self._save_results()
        
        return self.best_params, self.best_metrics
    
    def _save_results(self):
        """Save grid search results."""
        results_data = {
            'param_grid': {k: [float(v) if isinstance(v, (int, float)) else v 
                              for v in vals] 
                          for k, vals in self.param_grid.items()},
            'n_simulations': self.n_simulations,
            'results': self.results,
            'best_params': self.best_params,
            'best_score': self.best_score
        }
        
        filepath = self.save_dir / "grid_search_results.json"
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"Grid search results saved to {filepath}")
    
    def get_results_dataframe(self):
        """Convert results to a format suitable for analysis."""
        rows = []
        for result in self.results:
            row = {**result['params'], **result['metrics']}
            rows.append(row)
        return rows


def grid_search_hopf(
    target_fc: torch.Tensor,
    n_rois: int,
    structural_connectivity: Optional[torch.Tensor] = None,
    g_values: List[float] = None,
    a_values: List[float] = None,
    n_timepoints: int = 200,
    n_simulations: int = 10,
    device: str = "cpu"
) -> Tuple[Dict[str, Any], CoupledHopfModel]:
    """
    Grid search specifically for Hopf model.
    
    Args:
        target_fc: Target functional connectivity
        n_rois: Number of ROIs
        structural_connectivity: Optional SC matrix
        g_values: List of global coupling values to try
        a_values: List of bifurcation parameter values to try
        n_timepoints: Number of timepoints to simulate
        n_simulations: Simulations per setting
        device: Device to use
        
    Returns:
        Tuple of (best_params, fitted_model)
    """
    if g_values is None:
        g_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    
    if a_values is None:
        a_values = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02]
    
    param_grid = {
        'initial_g': g_values,
        'initial_a': a_values
    }
    
    grid_search = GridSearch(
        param_grid=param_grid,
        n_simulations=n_simulations,
        device=device
    )
    
    model_kwargs = {
        'n_rois': n_rois,
        'structural_connectivity': structural_connectivity,
        'learnable_a': False,
        'learnable_g': False
    }
    
    best_params, best_metrics = grid_search.search(
        model_class=CoupledHopfModel,
        model_kwargs=model_kwargs,
        target_fc=target_fc,
        n_timepoints=n_timepoints
    )
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best FC correlation: {best_metrics['fc_correlation_mean']:.4f} "
          f"± {best_metrics['fc_correlation_std']:.4f}")
    
    # Create model with best parameters
    best_model = CoupledHopfModel(
        n_rois=n_rois,
        structural_connectivity=structural_connectivity,
        initial_g=best_params['initial_g'],
        initial_a=best_params['initial_a'],
        device=device
    )
    
    return best_params, best_model
