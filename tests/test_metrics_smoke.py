"""Smoke tests for metric nn.Module classes."""

import unittest

import torch

from src.metrics import (
    FCCorrelation, FCMSE,
    PowerSpectrumDistance, TemporalCorrelation, AutocorrelationDistance,
    FCD, PhFCD, Metastability, PhaseFC,
)


class TestMetricModules(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.n_rois = 12
        self.n_time = 200
        self.ts_pred = torch.randn(6, self.n_rois, self.n_time, dtype=torch.complex64)
        self.ts_target = torch.randn(6, self.n_rois, self.n_time, dtype=torch.complex64)

    def _check_evaluate_keys(self, module, expected_keys):
        result = module.evaluate(self.ts_pred, self.ts_target)
        self.assertEqual(set(result.keys()), set(expected_keys))

    def test_fc_correlation_keys(self):
        self._check_evaluate_keys(FCCorrelation(), {"fc_correlation"})

    def test_fc_mse_keys(self):
        self._check_evaluate_keys(FCMSE(), {"fc_mse"})

    def test_power_spectrum_distance_keys(self):
        self._check_evaluate_keys(PowerSpectrumDistance(), {"power_spectrum_distance"})

    def test_temporal_correlation_keys(self):
        self._check_evaluate_keys(TemporalCorrelation(), {"temporal_correlation"})

    def test_autocorrelation_distance_keys(self):
        self._check_evaluate_keys(AutocorrelationDistance(), {"autocorr_distance"})

    def test_fcd_keys(self):
        self._check_evaluate_keys(
            FCD(tr=0.72, fcd_win_sec=60.0, fcd_step_sec=2.0),
            {"fcd_mse", "fcd_ks"},
        )

    def test_phfcd_keys(self):
        self._check_evaluate_keys(PhFCD(), {"phfcd_mse", "phfcd_ks"})

    def test_metastability_keys(self):
        self._check_evaluate_keys(Metastability(), {"metastability_diff"})

    def test_phase_fc_keys(self):
        self._check_evaluate_keys(PhaseFC(), {"phase_fc_correlation"})

    def test_fc_and_dynamics_combined(self):
        """Combined FC + dynamics evaluate keys match expected set (like train scripts)."""
        metrics = {}
        metrics.update(FCCorrelation().evaluate(self.ts_pred, self.ts_target))
        metrics.update(FCMSE().evaluate(self.ts_pred, self.ts_target))
        metrics.update(FCD(tr=0.72, fcd_win_sec=60.0, fcd_step_sec=2.0).evaluate(
            self.ts_pred, self.ts_target
        ))
        metrics.update(PhFCD().evaluate(self.ts_pred, self.ts_target))
        metrics.update(PhaseFC().evaluate(self.ts_pred, self.ts_target))
        metrics.update(Metastability().evaluate(self.ts_pred, self.ts_target))
        self.assertEqual(
            set(metrics.keys()),
            {"fc_correlation", "fc_mse", "fcd_ks", "fcd_mse",
             "phfcd_ks", "phfcd_mse", "phase_fc_correlation", "metastability_diff"},
        )

    def test_all_eval_metrics_combined(self):
        """All evaluation metrics combined (like checkpoint eval scripts)."""
        ts_pred = torch.randn(1, 10, self.n_time, dtype=torch.complex64)
        ts_target = torch.randn(1, 10, self.n_time, dtype=torch.complex64)
        metrics = {}
        for module in [
            FCCorrelation(), FCMSE(),
            PowerSpectrumDistance(), TemporalCorrelation(), AutocorrelationDistance(),
            FCD(tr=0.72, fcd_win_sec=60.0, fcd_step_sec=2.0),
            PhFCD(), Metastability(), PhaseFC(),
        ]:
            metrics.update(module.evaluate(ts_pred, ts_target))
        self.assertEqual(
            set(metrics.keys()),
            {
                "fc_correlation", "fc_mse",
                "power_spectrum_distance", "temporal_correlation", "autocorr_distance",
                "fcd_ks", "fcd_mse", "phfcd_ks", "phfcd_mse",
                "phase_fc_correlation", "metastability_diff",
            },
        )

    def test_forward_returns_scalar_tensor(self):
        """forward() returns a differentiable scalar tensor for each module."""
        modules = [
            FCCorrelation(), FCMSE(),
            PowerSpectrumDistance(), TemporalCorrelation(), AutocorrelationDistance(),
            FCD(tr=0.72, fcd_win_sec=60.0, fcd_step_sec=2.0),
            PhFCD(), Metastability(), PhaseFC(),
        ]
        for module in modules:
            out = module(self.ts_pred, self.ts_target)
            self.assertEqual(out.ndim, 0, f"{module.__class__.__name__} forward() should return scalar")
            self.assertFalse(torch.isnan(out), f"{module.__class__.__name__} forward() returned NaN")


if __name__ == "__main__":
    unittest.main()
