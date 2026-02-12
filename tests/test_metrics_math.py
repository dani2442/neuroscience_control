"""Unit tests for FC/FCD metric math."""

import unittest

import torch

from src.metrics.dynamics_metrics import fcd_matrix
from src.metrics.fc_metrics import fc_correlation, fc_mse


def _symmetric_fc(batch: int, n_rois: int) -> torch.Tensor:
    x = torch.randn(batch, n_rois, n_rois)
    x = 0.5 * (x + x.transpose(1, 2))
    idx = torch.arange(n_rois)
    x[:, idx, idx] = 1.0
    return x


class TestFCMetricsMath(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_fc_correlation_matches_reference_upper_triangle(self) -> None:
        batch, n_rois = 4, 7
        fc_pred = _symmetric_fc(batch, n_rois)
        fc_targ = _symmetric_fc(batch, n_rois)
        idx = torch.triu_indices(n_rois, n_rois, offset=1)

        ref_corrs = []
        for b in range(batch):
            p = fc_pred[b, idx[0], idx[1]]
            t = fc_targ[b, idx[0], idx[1]]
            p = p - p.mean()
            t = t - t.mean()
            num = (p * t).sum()
            den = torch.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-8
            ref_corrs.append(num / den)

        ref = torch.stack(ref_corrs).mean()
        got = fc_correlation(fc_pred, fc_targ)
        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=0.0))

    def test_fc_mse_matches_reference_upper_triangle(self) -> None:
        batch, n_rois = 3, 8
        fc_pred = _symmetric_fc(batch, n_rois)
        fc_targ = _symmetric_fc(batch, n_rois)
        idx = torch.triu_indices(n_rois, n_rois, offset=1)

        ref = ((fc_pred[:, idx[0], idx[1]] - fc_targ[:, idx[0], idx[1]]) ** 2).mean()
        got = fc_mse(fc_pred, fc_targ)
        self.assertTrue(torch.allclose(got, ref, atol=1e-7, rtol=0.0))


class TestFCDConstruction(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_fcd_matrix_is_symmetric_with_unit_diagonal(self) -> None:
        ts = torch.randn(240, 10)
        fcd = fcd_matrix(ts, win_len=60, win_step=3)
        self.assertIsNotNone(fcd)
        assert fcd is not None

        self.assertLess((fcd - fcd.transpose(0, 1)).abs().max().item(), 1e-5)
        self.assertLess((fcd.diag() - 1.0).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
