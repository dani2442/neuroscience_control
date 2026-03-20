"""Unit tests for phase coherence, phFCD, static FC, and phFCD-KSD."""

import math
import unittest

import torch

from src.metrics.dynamics_metrics import (
    ks_distance_2samp,
    phase_coherence_matrix,
    phfcd_distribution,
    phfcd_matrix,
)
from src.metrics.fc_metrics import compute_static_fc
from src.metrics import FCCorrelation, FCMSE, PhFCD


class TestPhaseCoherence(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_output_shape(self) -> None:
        T, N = 100, 5
        phases = torch.randn(T, N)
        P = phase_coherence_matrix(phases)
        self.assertEqual(P.shape, (T, N, N))

    def test_diagonal_is_one(self) -> None:
        """cos(φ_n - φ_n) = cos(0) = 1 for all n, t."""
        T, N = 50, 8
        phases = torch.randn(T, N)
        P = phase_coherence_matrix(phases)
        diag = P[:, torch.arange(N), torch.arange(N)]  # (T, N)
        self.assertTrue(torch.allclose(diag, torch.ones_like(diag), atol=1e-6))

    def test_symmetry(self) -> None:
        """P_nm(t) = P_mn(t) because cos is even."""
        T, N = 50, 6
        phases = torch.randn(T, N)
        P = phase_coherence_matrix(phases)
        self.assertTrue(torch.allclose(P, P.transpose(1, 2), atol=1e-6))

    def test_range(self) -> None:
        """Phase coherence values should be in [-1, 1]."""
        phases = torch.randn(80, 10)
        P = phase_coherence_matrix(phases)
        self.assertTrue((P >= -1.0 - 1e-6).all())
        self.assertTrue((P <= 1.0 + 1e-6).all())

    def test_identical_phases_give_all_ones(self) -> None:
        """If all regions have the same phase at each t, coherence = 1 everywhere."""
        T, N = 30, 4
        # All regions share the same phase at each timepoint
        shared = torch.randn(T, 1).expand(T, N)
        P = phase_coherence_matrix(shared)
        self.assertTrue(torch.allclose(P, torch.ones_like(P), atol=1e-6))


class TestPhFCD(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(99)

    def test_phfcd_matrix_shape(self) -> None:
        T, N = 60, 5
        phases = torch.randn(T, N)
        fcd = phfcd_matrix(phases)
        self.assertIsNotNone(fcd)
        self.assertEqual(fcd.shape, (T, T))  # type: ignore[union-attr]

    def test_phfcd_matrix_symmetry(self) -> None:
        phases = torch.randn(50, 6)
        fcd = phfcd_matrix(phases)
        self.assertIsNotNone(fcd)
        assert fcd is not None
        self.assertTrue(torch.allclose(fcd, fcd.T, atol=1e-5))

    def test_phfcd_matrix_diagonal_is_one(self) -> None:
        """Cosine similarity of a vector with itself = 1."""
        phases = torch.randn(40, 7)
        fcd = phfcd_matrix(phases)
        self.assertIsNotNone(fcd)
        assert fcd is not None
        self.assertTrue(torch.allclose(fcd.diag(), torch.ones(40), atol=1e-5))

    def test_phfcd_matrix_range(self) -> None:
        """Cosine similarities are in [-1, 1]."""
        phases = torch.randn(50, 5)
        fcd = phfcd_matrix(phases)
        assert fcd is not None
        self.assertTrue((fcd >= -1.0 - 1e-5).all())
        self.assertTrue((fcd <= 1.0 + 1e-5).all())

    def test_phfcd_matrix_none_for_single_roi(self) -> None:
        phases = torch.randn(50, 1)
        fcd = phfcd_matrix(phases)
        self.assertIsNone(fcd)

    def test_phfcd_distribution_nonempty(self) -> None:
        phases = torch.randn(30, 5)
        dist = phfcd_distribution(phases)
        expected_len = 30 * 29 // 2
        self.assertEqual(dist.numel(), expected_len)

    def test_phfcd_distribution_empty_for_single_roi(self) -> None:
        phases = torch.randn(30, 1)
        dist = phfcd_distribution(phases)
        self.assertEqual(dist.numel(), 0)


class TestPhFCDKSD(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_identical_signals_give_low_ksd(self) -> None:
        """KSD between identical phFCD distributions should be 0."""
        ts = torch.randn(2, 8, 100, dtype=torch.complex64)
        phases = torch.angle(ts[0]).T   # (T, N)
        dist = phfcd_distribution(phases)
        ksd = ks_distance_2samp(dist, dist)
        self.assertAlmostEqual(ksd.item(), 0.0, places=5)

    def test_different_signals_give_positive_ksd(self) -> None:
        """KSD between different random signals should be > 0."""
        ts1 = torch.randn(1, 6, 80, dtype=torch.complex64)
        ts2 = torch.randn(1, 6, 80, dtype=torch.complex64)
        dist1 = phfcd_distribution(torch.angle(ts1[0]).T)
        dist2 = phfcd_distribution(torch.angle(ts2[0]).T)
        ksd = ks_distance_2samp(dist1, dist2)
        self.assertGreater(ksd.item(), 0.0)
        self.assertLessEqual(ksd.item(), 1.0)

    def test_compute_dynamics_includes_phfcd_ks(self) -> None:
        """Integration: PhFCD.evaluate returns phfcd_ks."""
        ts_pred = torch.randn(2, 8, 200, dtype=torch.complex64)
        ts_targ = torch.randn(2, 8, 200, dtype=torch.complex64)
        metrics = PhFCD().evaluate(ts_pred, ts_targ)
        self.assertIn("phfcd_ks", metrics)
        self.assertFalse(math.isnan(metrics["phfcd_ks"]))
        self.assertGreaterEqual(metrics["phfcd_ks"], 0.0)
        self.assertLessEqual(metrics["phfcd_ks"], 1.0)


class TestStaticFC(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(21)

    def test_output_shape(self) -> None:
        ts = torch.randn(3, 10, 100, dtype=torch.complex64)
        fc = compute_static_fc(ts)
        self.assertEqual(fc.shape, (3, 10, 10))

    def test_2d_input_adds_batch(self) -> None:
        ts = torch.randn(5, 60, dtype=torch.complex64)
        fc = compute_static_fc(ts)
        self.assertEqual(fc.shape, (1, 5, 5))

    def test_symmetry(self) -> None:
        ts = torch.randn(2, 8, 120, dtype=torch.complex64)
        fc = compute_static_fc(ts)
        self.assertTrue(torch.allclose(fc, fc.transpose(1, 2), atol=1e-5))

    def test_diagonal_approx_one(self) -> None:
        ts = torch.randn(2, 6, 200, dtype=torch.complex64)
        fc = compute_static_fc(ts)
        diag = fc[:, torch.arange(6), torch.arange(6)]
        self.assertTrue(torch.allclose(diag, torch.ones_like(diag), atol=1e-3))

    def test_real_input(self) -> None:
        ts = torch.randn(2, 5, 100)
        fc = compute_static_fc(ts)
        self.assertEqual(fc.shape, (2, 5, 5))

    def test_fc_from_timeseries_and_compare(self) -> None:
        ts1 = torch.randn(3, 8, 150, dtype=torch.complex64)
        ts2 = torch.randn(3, 8, 150, dtype=torch.complex64)
        metrics = {**FCCorrelation().evaluate(ts1, ts2), **FCMSE().evaluate(ts1, ts2)}
        self.assertIn("fc_correlation", metrics)
        self.assertIn("fc_mse", metrics)
        self.assertIsInstance(metrics["fc_correlation"], float)
        self.assertIsInstance(metrics["fc_mse"], float)


if __name__ == "__main__":
    unittest.main()
