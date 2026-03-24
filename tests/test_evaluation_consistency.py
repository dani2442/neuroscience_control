"""Regression tests for repeated-simulation metric aggregation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.grid_search import GridSearch
from src.utils.evaluation import evaluate_model_loader_metrics


class BatchMeanSquaredMetric:
    """Toy metric for batch-level aggregation tests."""

    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict[str, float]:
        mean = float(ts_pred.real.mean().item())
        return {"toy_metric": mean * mean}


class OffsetRolloutModel:
    """Minimal model whose output changes on each rollout call."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def forward(self, *, initial_state: torch.Tensor, n_steps: int, dt: float, **kwargs) -> torch.Tensor:
        offset = float(self.calls)
        self.calls += 1
        return (initial_state + offset).unsqueeze(-1).expand(-1, -1, n_steps)


class TestEvaluationConsistency(unittest.TestCase):
    def test_loader_evaluation_reports_std_over_repeated_simulations(self) -> None:
        windows = torch.tensor([[[1.0, 1.0]], [[3.0, 3.0]]])
        fc = torch.zeros(2, 1, 1)
        subject_ids = torch.arange(2)
        control = torch.empty(2, 0)
        loader = DataLoader(
            TensorDataset(windows, fc, subject_ids, control),
            batch_size=2,
            shuffle=False,
        )
        cfg = SimpleNamespace(
            tr=0.72,
            fcd_win_sec=30.0,
            fcd_step_sec=2.0,
            sde_type="ito",
            sde_method="euler",
            dt_min=0.1,
            use_adjoint=False,
            adjoint_method=None,
            denoise_f_lo=None,
            denoise_f_hi=None,
        )
        with patch("src.utils.evaluation.build_eval_metrics", return_value=[BatchMeanSquaredMetric()]):
            batch_metrics = evaluate_model_loader_metrics(
                OffsetRolloutModel(),
                loader,
                cfg,
                n_steps=2,
                n_simulations=1,
            )
            repeated_metrics = evaluate_model_loader_metrics(
                OffsetRolloutModel(),
                loader,
                cfg,
                n_steps=2,
                n_simulations=2,
            )
            repeated_metrics_with_std = evaluate_model_loader_metrics(
                OffsetRolloutModel(),
                loader,
                cfg,
                n_steps=2,
                n_simulations=2,
                return_std=True,
            )

        self.assertAlmostEqual(batch_metrics["toy_metric"], 4.0)
        self.assertAlmostEqual(repeated_metrics["toy_metric"], 6.5)
        self.assertAlmostEqual(repeated_metrics_with_std["toy_metric"], 6.5)
        self.assertAlmostEqual(repeated_metrics_with_std["toy_metric_std"], 2.5)

    def test_grid_search_evaluation_reports_std_over_repeated_simulations(self) -> None:
        model = OffsetRolloutModel()
        initial_states = torch.tensor([[1.0], [3.0]])
        target_fc = torch.eye(1)
        target_ts = torch.zeros(2, 1, 2)
        search = GridSearch(param_grid={"dummy": [0]}, device="cpu")

        with patch("src.training.grid_search.build_eval_metrics", return_value=[BatchMeanSquaredMetric()]):
            metrics = search.evaluate_params(
                model,
                target_fc,
                initial_states,
                n_timepoints=2,
                dt=0.72,
                target_timeseries=target_ts,
                n_simulations=2,
            )

        self.assertAlmostEqual(metrics["toy_metric"], 6.5)
        self.assertAlmostEqual(metrics["toy_metric_std"], 2.5)


if __name__ == "__main__":
    unittest.main()
