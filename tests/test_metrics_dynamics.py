"""Unit tests for dynamics metric behavior."""

import math
import unittest

import torch

from src.metrics import FCD, PhFCD, Metastability


class TestDynamicsMetrics(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)

    def test_fcd_ks_is_finite_with_standard_settings(self) -> None:
        # Complex inputs (analytic signals) as expected by the pipeline.
        ts_pred = torch.randn(2, 10, 240, dtype=torch.complex64)
        ts_targ = ts_pred + 0.05 * torch.randn(2, 10, 240, dtype=torch.complex64)

        metrics = {}
        metrics.update(FCD(tr=0.72, fcd_win_sec=60.0, fcd_step_sec=2.0).evaluate(ts_pred, ts_targ))
        metrics.update(PhFCD().evaluate(ts_pred, ts_targ))
        metrics.update(Metastability().evaluate(ts_pred, ts_targ))

        self.assertIn("fcd_ks", metrics)
        self.assertIn("phfcd_ks", metrics)
        self.assertIn("metastability_diff", metrics)
        self.assertFalse(math.isnan(metrics["fcd_ks"]))
        self.assertGreaterEqual(metrics["fcd_ks"], 0.0)
        self.assertLessEqual(metrics["fcd_ks"], 1.0)
        self.assertFalse(math.isnan(metrics["phfcd_ks"]))
        self.assertGreaterEqual(metrics["phfcd_ks"], 0.0)
        self.assertLessEqual(metrics["phfcd_ks"], 1.0)
        self.assertFalse(math.isnan(metrics["metastability_diff"]))

    def test_fcd_ks_is_nan_when_windowing_not_feasible(self) -> None:
        ts_pred = torch.randn(2, 10, 80, dtype=torch.complex64)
        ts_targ = torch.randn(2, 10, 80, dtype=torch.complex64)

        metrics = {}
        metrics.update(FCD(tr=0.72, fcd_win_sec=60.0, fcd_step_sec=2.0).evaluate(ts_pred, ts_targ))
        metrics.update(PhFCD().evaluate(ts_pred, ts_targ))
        metrics.update(Metastability().evaluate(ts_pred, ts_targ))

        self.assertTrue(math.isnan(metrics["fcd_ks"]))
        # phfcd_ks should still be valid (no window dependency)
        self.assertFalse(math.isnan(metrics["phfcd_ks"]))
        self.assertFalse(math.isnan(metrics["metastability_diff"]))


if __name__ == "__main__":
    unittest.main()
