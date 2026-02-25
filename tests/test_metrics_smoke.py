"""Smoke tests for script-style metric dictionaries."""

import unittest

import torch

from src.metrics import (
    compute_all_fc_metrics,
    compute_all_timeseries_metrics,
    compute_dynamics_fit_metrics,
)


def _compute_fc(timeseries: torch.Tensor) -> torch.Tensor:
    ts = timeseries.real if torch.is_complex(timeseries) else timeseries
    t = ts.shape[2]
    ts_c = ts - ts.mean(dim=2, keepdim=True)
    ts_n = ts_c / (ts_c.std(dim=2, keepdim=True) + 1e-8)
    return torch.bmm(ts_n, ts_n.transpose(1, 2)) / (t - 1)


class TestScriptMetricSmoke(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def _final_metrics_like_train_scripts(self, n_paths: int) -> dict[str, float]:
        n_rois, n_time = 12, 200
        simulated_ts = torch.randn(n_paths, n_rois, n_time, dtype=torch.complex64)
        target_ts = torch.randn(n_paths, n_rois, n_time, dtype=torch.complex64)
        target_fc = _compute_fc(target_ts).mean(dim=0)
        simulated_fc_mean = _compute_fc(simulated_ts).mean(dim=0)

        metrics = compute_all_fc_metrics(simulated_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
        metrics.update(
            compute_dynamics_fit_metrics(
                simulated_ts,
                target_ts,
                tr=0.72,
                fcd_win_sec=60.0,
                fcd_step_sec=2.0,
                compute_fcd=True,
                compute_metastability=True,
            )
        )
        return metrics

    def test_train_hopf_final_metric_keys(self) -> None:
        metrics = self._final_metrics_like_train_scripts(n_paths=6)
        self.assertEqual(
            set(metrics.keys()),
            {"fc_correlation", "fc_mse", "fcd_ks", "phfcd_ks", "metastability_diff"},
        )

    def test_train_nsde_finetune_final_metric_keys(self) -> None:
        metrics = self._final_metrics_like_train_scripts(n_paths=5)
        self.assertEqual(
            set(metrics.keys()),
            {"fc_correlation", "fc_mse", "fcd_ks", "phfcd_ks", "metastability_diff"},
        )

    def test_checkpoint_eval_per_run_metric_keys(self) -> None:
        n_rois, n_time = 10, 200
        sim_run = torch.randn(1, n_rois, n_time, dtype=torch.complex64)
        real_ts = torch.randn(1, n_rois, n_time, dtype=torch.complex64)
        real_fc_single = _compute_fc(real_ts)
        sim_fc_run = _compute_fc(sim_run)

        metrics = {
            **compute_all_fc_metrics(sim_fc_run, real_fc_single),
            **compute_all_timeseries_metrics(sim_run, real_ts),
            **compute_dynamics_fit_metrics(
                sim_run,
                real_ts,
                tr=0.72,
                fcd_win_sec=60.0,
                fcd_step_sec=2.0,
                compute_fcd=True,
                compute_metastability=True,
            ),
        }

        self.assertEqual(
            set(metrics.keys()),
            {
                "fc_correlation",
                "fc_mse",
                "power_spectrum_distance",
                "temporal_correlation",
                "autocorr_distance",
                "fcd_ks",
                "phfcd_ks",
                "metastability_diff",
            },
        )


if __name__ == "__main__":
    unittest.main()
