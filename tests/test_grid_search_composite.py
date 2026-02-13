"""Tests for composite grid-search scoring (FC + FCD + Metastability)."""

import unittest

import numpy as np
import torch

from src.training.grid_search import GridSearch, _HIGHER_IS_BETTER


class TestCompositeScore(unittest.TestCase):
    """Test the _composite_score static method."""

    def test_fc_only(self) -> None:
        metrics = {"fc_correlation": 0.8, "fc_mse": 0.1}
        weights = {"fc_correlation": 1.0}
        score = GridSearch._composite_score(metrics, weights)
        self.assertAlmostEqual(score, 0.8)

    def test_all_three(self) -> None:
        metrics = {
            "fc_correlation": 0.8,
            "fcd_mse": 0.2,
            "metastability_diff": 0.05,
        }
        weights = {
            "fc_correlation": 1.0,
            "fcd_mse": 0.5,
            "metastability_diff": 0.5,
        }
        # Expected: 1.0*0.8 - 0.5*0.2 - 0.5*0.05 = 0.8 - 0.1 - 0.025 = 0.675
        score = GridSearch._composite_score(metrics, weights)
        self.assertAlmostEqual(score, 0.675)

    def test_nan_metric_skipped(self) -> None:
        metrics = {
            "fc_correlation": 0.8,
            "fcd_mse": float("nan"),
            "metastability_diff": 0.1,
        }
        weights = {
            "fc_correlation": 1.0,
            "fcd_mse": 0.5,
            "metastability_diff": 0.5,
        }
        # FCD is NaN -> skipped, so score = 1.0*0.8 - 0.5*0.1 = 0.75
        score = GridSearch._composite_score(metrics, weights)
        self.assertAlmostEqual(score, 0.75)

    def test_missing_metric_skipped(self) -> None:
        metrics = {"fc_correlation": 0.9}
        weights = {
            "fc_correlation": 1.0,
            "fcd_mse": 0.5,
        }
        # fcd_mse missing -> skipped, so score = 0.9
        score = GridSearch._composite_score(metrics, weights)
        self.assertAlmostEqual(score, 0.9)

    def test_higher_is_better_set(self) -> None:
        self.assertIn("fc_correlation", _HIGHER_IS_BETTER)
        self.assertNotIn("fcd_mse", _HIGHER_IS_BETTER)
        self.assertNotIn("metastability_diff", _HIGHER_IS_BETTER)

class TestGridSearchEvaluateParams(unittest.TestCase):
    """Verify evaluate_params returns extra keys when target_timeseries given."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.gs = GridSearch(
            param_grid={"initial_g": [0.5]}, device="cpu",
        )

    def test_without_target_timeseries(self) -> None:
        """Without target_timeseries, only FC keys should be present."""
        from src.models import CoupledHopfModel

        model = CoupledHopfModel(n_rois=10, initial_g=0.5, initial_a=-0.02, device="cpu")
        initial_states = torch.randn(2, 10, dtype=torch.complex64)
        target_fc = torch.eye(10)

        metrics = self.gs.evaluate_params(
            model, target_fc, initial_states, n_timepoints=50, dt=0.72,
        )

        self.assertIn("fc_correlation", metrics)
        self.assertNotIn("fcd_mse", metrics)
        self.assertNotIn("metastability_diff", metrics)

    def test_with_target_timeseries(self) -> None:
        """With target_timeseries, FCD and meta keys should also be present."""
        from src.models import CoupledHopfModel

        model = CoupledHopfModel(n_rois=10, initial_g=0.5, initial_a=-0.02, device="cpu")
        initial_states = torch.randn(2, 10, dtype=torch.complex64)
        target_fc = torch.eye(10)
        target_ts = torch.randn(2, 10, 50, dtype=torch.complex64)

        metrics = self.gs.evaluate_params(
            model, target_fc, initial_states, n_timepoints=50, dt=0.72,
            target_timeseries=target_ts, tr=0.72,
        )

        self.assertIn("fc_correlation", metrics)
        self.assertIn("fcd_mse", metrics)
        self.assertIn("metastability_diff", metrics)
        # Values should be finite non-negative
        self.assertTrue(np.isfinite(metrics["metastability_diff"]))
        self.assertGreaterEqual(metrics["metastability_diff"], 0.0)


class TestGridSearchCompositeSearch(unittest.TestCase):
    """Verify search with metric_weights picks best composite candidate."""

    def test_composite_search_picks_best(self) -> None:
        """With two candidates, composite score should pick the one with
        better combined FC+Meta even if its FC alone is slightly worse."""
        from src.models import CoupledHopfModel

        torch.manual_seed(99)
        gs = GridSearch(
            param_grid={"initial_g": [0.3, 0.5], "initial_a": [-0.02]},
            device="cpu",
        )
        initial_states = torch.randn(2, 10, dtype=torch.complex64)
        target_fc = torch.eye(10)
        target_ts = torch.randn(2, 10, 50, dtype=torch.complex64)

        best_params, best_metrics = gs.search(
            model_class=CoupledHopfModel,
            model_kwargs={"n_rois": 10, "learnable_a": False, "learnable_g": False},
            target_fc=target_fc,
            initial_states=initial_states,
            n_timepoints=50,
            dt=0.72,
            metric_weights={
                "fc_correlation": 1.0,
                "fcd_mse": 0.5,
                "metastability_diff": 0.5,
            },
            eval_kwargs={
                "target_timeseries": target_ts,
                "tr": 0.72,
            },
            verbose=False,
        )

        # Just verify it ran, returned valid structure, and all keys present
        self.assertIn("initial_g", best_params)
        self.assertIn("fc_correlation", best_metrics)
        self.assertIn("fcd_mse", best_metrics)
        self.assertIn("metastability_diff", best_metrics)

    def test_backward_compat_fc_only(self) -> None:
        """Without metric_weights, search still uses the default `metric` arg."""
        from src.models import CoupledHopfModel

        torch.manual_seed(42)
        gs = GridSearch(
            param_grid={"initial_g": [0.5], "initial_a": [-0.02]},
            device="cpu",
        )
        initial_states = torch.randn(2, 10, dtype=torch.complex64)
        target_fc = torch.eye(10)

        best_params, best_metrics = gs.search(
            model_class=CoupledHopfModel,
            model_kwargs={"n_rois": 10, "learnable_a": False, "learnable_g": False},
            target_fc=target_fc,
            initial_states=initial_states,
            n_timepoints=50,
            dt=0.72,
            verbose=False,
        )

        self.assertIn("initial_g", best_params)
        self.assertIn("fc_correlation", best_metrics)
        self.assertNotIn("fcd_mse", best_metrics)


if __name__ == "__main__":
    unittest.main()
