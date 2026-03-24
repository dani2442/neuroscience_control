"""Tests for batched FC metric aggregation."""

import unittest

import torch

from src.metrics import FCCorrelation, FCMSE, compute_static_fc, fisher_batch_average
from src.metrics._utils import upper_tri_vec


def _make_correlation_matrix(a: float, b: float, c: float, dtype: torch.dtype) -> torch.Tensor:
    """Construct a valid 3x3 correlation matrix from triangular loadings."""
    remainder = 1.0 - b * b - c * c
    if remainder <= 0.0:
        raise ValueError("b^2 + c^2 must be < 1.")
    loadings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [a, (1.0 - a * a) ** 0.5, 0.0],
            [b, c, remainder ** 0.5],
        ],
        dtype=dtype,
    )
    return loadings @ loadings.T


def _orthogonal_basis(dtype: torch.dtype) -> torch.Tensor:
    """Centered orthogonal basis with unit sample covariance."""
    basis = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=dtype,
    )
    basis = basis - basis.mean(dim=1, keepdim=True)
    return basis / basis.std(dim=1, keepdim=True)


def _timeseries_from_fc(fc: torch.Tensor) -> torch.Tensor:
    """Build a real timeseries with empirical FC exactly equal to *fc*."""
    basis = _orthogonal_basis(fc.dtype).to(fc.device)
    chol = torch.linalg.cholesky(fc)
    return chol @ basis


def _matrix_corr(fc_pred: torch.Tensor, fc_target: torch.Tensor) -> torch.Tensor:
    p = upper_tri_vec(fc_pred, k=1)
    t = upper_tri_vec(fc_target, k=1)
    pc = p - p.mean()
    tc = t - t.mean()
    return (pc * tc).sum() / (torch.sqrt((pc ** 2).sum() * (tc ** 2).sum()) + 1e-8)


def _legacy_batch_corr(fc_pred: torch.Tensor, fc_target: torch.Tensor) -> torch.Tensor:
    p = upper_tri_vec(fc_pred, k=1)
    t = upper_tri_vec(fc_target, k=1)
    pc = p - p.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    corr = (pc * tc).sum(dim=1) / (torch.sqrt((pc ** 2).sum(dim=1) * (tc ** 2).sum(dim=1)) + 1e-8)
    return corr.mean()


class TestFCMetricsBatchAggregation(unittest.TestCase):
    def setUp(self) -> None:
        dtype = torch.float64
        self.pred_fc = torch.stack(
            [
                _make_correlation_matrix(0.15, -0.25, 0.40, dtype),
                _make_correlation_matrix(0.85, 0.60, -0.10, dtype),
                _make_correlation_matrix(-0.55, 0.35, 0.45, dtype),
            ],
            dim=0,
        )
        self.target_fc = torch.stack(
            [
                _make_correlation_matrix(0.45, 0.20, 0.10, dtype),
                _make_correlation_matrix(0.70, -0.30, 0.55, dtype),
                _make_correlation_matrix(-0.20, 0.65, -0.30, dtype),
            ],
            dim=0,
        )
        self.pred_ts = torch.stack([_timeseries_from_fc(fc) for fc in self.pred_fc], dim=0)
        self.target_ts = torch.stack([_timeseries_from_fc(fc) for fc in self.target_fc], dim=0)

    def test_compute_static_fc_matches_constructed_batch(self) -> None:
        actual = compute_static_fc(self.pred_ts)
        self.assertTrue(torch.allclose(actual, self.pred_fc, atol=1e-7, rtol=1e-7))

    def test_fisher_batch_average_matches_manual_formula(self) -> None:
        clipped = self.pred_fc.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        expected = torch.tanh(torch.atanh(clipped).mean(dim=0))
        diag = self.pred_fc.diagonal(dim1=-2, dim2=-1).mean(dim=0)
        idx = torch.arange(expected.shape[0])
        expected[idx, idx] = diag

        actual = fisher_batch_average(self.pred_fc)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10))
        self.assertFalse(torch.allclose(actual, self.pred_fc.mean(dim=0), atol=1e-4, rtol=1e-4))

    def test_fc_correlation_uses_fisher_pooled_group_fc(self) -> None:
        metric = FCCorrelation()
        loss = metric(self.pred_ts, self.target_ts)

        pooled_pred = fisher_batch_average(self.pred_fc)
        pooled_target = fisher_batch_average(self.target_fc)
        expected_loss = 1.0 - _matrix_corr(pooled_pred, pooled_target)
        legacy_loss = 1.0 - _legacy_batch_corr(self.pred_fc, self.target_fc)

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=7)
        self.assertGreater(abs(loss.item() - legacy_loss.item()), 1e-4)

    def test_fc_mse_uses_fisher_pooled_group_fc(self) -> None:
        metric = FCMSE()
        loss = metric(self.pred_ts, self.target_ts)

        pooled_pred = fisher_batch_average(self.pred_fc)
        pooled_target = fisher_batch_average(self.target_fc)
        expected_loss = ((upper_tri_vec(pooled_pred, k=1) - upper_tri_vec(pooled_target, k=1)) ** 2).mean()
        legacy_loss = ((upper_tri_vec(self.pred_fc, k=1) - upper_tri_vec(self.target_fc, k=1)) ** 2).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=8)
        self.assertGreater(abs(loss.item() - legacy_loss.item()), 1e-4)

    def test_precomputed_fc_targets_support_subjectwise_and_group_inputs(self) -> None:
        corr_metric = FCCorrelation()
        mse_metric = FCMSE()
        target_group_fc = fisher_batch_average(self.target_fc)

        corr_from_subject_fc = corr_metric(self.pred_ts, self.target_ts, fc_target=self.target_fc)
        corr_from_group_fc = corr_metric(self.pred_ts, self.target_ts, fc_target=target_group_fc)
        mse_from_subject_fc = mse_metric(self.pred_ts, self.target_ts, fc_target=self.target_fc)
        mse_from_group_fc = mse_metric(self.pred_ts, self.target_ts, fc_target=target_group_fc)

        self.assertAlmostEqual(corr_from_subject_fc.item(), corr_from_group_fc.item(), places=7)
        self.assertAlmostEqual(mse_from_subject_fc.item(), mse_from_group_fc.item(), places=8)


if __name__ == "__main__":
    unittest.main()
