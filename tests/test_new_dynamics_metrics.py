"""Unit tests for newly added dynamics metrics:
- phase_coherence_fc (grand-average phase-coherence FC, Eq. 11)
- phase_coherence_fc_correlation
- symmetric_kl_divergence (Eq. 6)
- markov_entropy_rate (Eq. 8)
- tpm_entropy_distance
"""

import math
import unittest

import torch

from src.metrics.dynamics_metrics import (
    markov_entropy_rate,
    phase_coherence_fc,
    phase_coherence_fc_correlation,
    symmetric_kl_divergence,
    tpm_entropy_distance,
    compute_dynamics_fit_metrics,
)


# ---------------------------------------------------------------------------
# Grand-average phase-coherence FC (Eq. 11)
# ---------------------------------------------------------------------------


class TestPhaseCoherenceFC(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_output_shape_batched(self) -> None:
        ts = torch.randn(3, 8, 100, dtype=torch.complex64)
        fc = phase_coherence_fc(ts)
        self.assertEqual(fc.shape, (3, 8, 8))

    def test_output_shape_2d(self) -> None:
        """Auto-batch a 2-D input."""
        ts = torch.randn(5, 60, dtype=torch.complex64)
        fc = phase_coherence_fc(ts)
        self.assertEqual(fc.shape, (1, 5, 5))

    def test_diagonal_is_one(self) -> None:
        """cos(φ_i - φ_i) = cos(0) = 1 at every t, so the mean is also 1."""
        ts = torch.randn(2, 6, 200, dtype=torch.complex64)
        fc = phase_coherence_fc(ts)
        N = 6
        diag = fc[:, torch.arange(N), torch.arange(N)]
        self.assertTrue(torch.allclose(diag, torch.ones_like(diag), atol=1e-5))

    def test_symmetry(self) -> None:
        """FC_ij = FC_ji because cos is even."""
        ts = torch.randn(2, 8, 150, dtype=torch.complex64)
        fc = phase_coherence_fc(ts)
        self.assertTrue(torch.allclose(fc, fc.transpose(1, 2), atol=1e-6))

    def test_range(self) -> None:
        """All entries should be in [-1, 1]."""
        ts = torch.randn(3, 10, 120, dtype=torch.complex64)
        fc = phase_coherence_fc(ts)
        self.assertTrue((fc >= -1.0 - 1e-5).all())
        self.assertTrue((fc <= 1.0 + 1e-5).all())

    def test_identical_phases_give_all_ones(self) -> None:
        """If all ROIs share the same phase at every t, FC = 1 everywhere."""
        T, N = 100, 5
        # Construct complex signal where all ROIs share one phase
        phase = torch.randn(1, 1, T)
        z = torch.exp(1j * phase.expand(1, N, T))
        fc = phase_coherence_fc(z)
        self.assertTrue(torch.allclose(fc, torch.ones_like(fc), atol=1e-5))

    def test_raises_on_real_input(self) -> None:
        ts = torch.randn(2, 5, 50)
        with self.assertRaises(TypeError):
            phase_coherence_fc(ts)

    def test_manual_reference(self) -> None:
        """Verify against manual computation for a small case."""
        torch.manual_seed(7)
        ts = torch.randn(1, 3, 20, dtype=torch.complex64)
        phases = torch.angle(ts[0])  # (N, T)
        N, T = phases.shape
        # Manual: FC_ij = mean_t cos(phi_i(t) - phi_j(t))
        fc_ref = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                fc_ref[i, j] = torch.cos(phases[i] - phases[j]).mean()
        fc_got = phase_coherence_fc(ts)[0]
        self.assertTrue(torch.allclose(fc_got, fc_ref, atol=1e-5))


class TestPhaseCoherenceFCCorrelation(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(99)

    def test_identical_signals_give_corr_one(self) -> None:
        ts = torch.randn(2, 8, 150, dtype=torch.complex64)
        corr = phase_coherence_fc_correlation(ts, ts)
        self.assertAlmostEqual(corr, 1.0, places=4)

    def test_returns_float(self) -> None:
        ts1 = torch.randn(2, 6, 100, dtype=torch.complex64)
        ts2 = torch.randn(2, 6, 100, dtype=torch.complex64)
        corr = phase_coherence_fc_correlation(ts1, ts2)
        self.assertIsInstance(corr, float)

    def test_in_dynamics_fit_metrics(self) -> None:
        """compute_dynamics_fit_metrics should include phase_fc_corr."""
        ts = torch.randn(2, 8, 200, dtype=torch.complex64)
        metrics = compute_dynamics_fit_metrics(ts, ts, tr=0.72)
        self.assertIn("phase_fc_corr", metrics)
        self.assertFalse(math.isnan(metrics["phase_fc_corr"]))


# ---------------------------------------------------------------------------
# Symmetrized KL divergence (Eq. 6)
# ---------------------------------------------------------------------------


class TestSymmetricKLDivergence(unittest.TestCase):
    def test_identical_distributions_give_zero(self) -> None:
        p = torch.tensor([0.2, 0.5, 0.3])
        kl = symmetric_kl_divergence(p, p)
        self.assertAlmostEqual(kl.item(), 0.0, places=6)

    def test_symmetry(self) -> None:
        """D_KL^sym(P, Q) = D_KL^sym(Q, P)."""
        p = torch.tensor([0.1, 0.6, 0.3])
        q = torch.tensor([0.3, 0.4, 0.3])
        self.assertAlmostEqual(
            symmetric_kl_divergence(p, q).item(),
            symmetric_kl_divergence(q, p).item(),
            places=6,
        )

    def test_non_negative(self) -> None:
        """Symmetric KL is always >= 0."""
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.assertGreaterEqual(symmetric_kl_divergence(p, q).item(), 0.0)

    def test_known_value(self) -> None:
        """Check against hand-computed value."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.25, 0.75])
        # KL(p||q) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
        #          = 0.5*ln(2) + 0.5*ln(2/3)
        import math as m
        kl_pq = 0.5 * m.log(2) + 0.5 * m.log(2.0 / 3.0)
        kl_qp = 0.25 * m.log(0.25 / 0.5) + 0.75 * m.log(0.75 / 0.5)
        expected = 0.5 * (kl_pq + kl_qp)
        self.assertAlmostEqual(
            symmetric_kl_divergence(p, q).item(), expected, places=5
        )


# ---------------------------------------------------------------------------
# Markov entropy rate (Eq. 8) and TPM entropy distance
# ---------------------------------------------------------------------------


class TestMarkovEntropyRate(unittest.TestCase):
    def test_deterministic_chain_has_zero_entropy(self) -> None:
        """A deterministic TPM (each row has a single 1) -> entropy = 0."""
        T_mat = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        # Technically the entries are 0 or 1; clamp ensures log safety.
        # Entropy rate should be 0 (or very close) since each transition is certain.
        ent = markov_entropy_rate(T_mat)
        self.assertAlmostEqual(ent.item(), 0.0, places=4)

    def test_uniform_chain_has_max_entropy(self) -> None:
        """A uniform TPM (all rows = [1/k, ..., 1/k]) has the maximum entropy."""
        import math as m
        k = 3
        T_mat = torch.ones(k, k) / k
        ent = markov_entropy_rate(T_mat)
        expected = m.log(k)  # max entropy for k states
        self.assertAlmostEqual(ent.item(), expected, places=4)

    def test_stationary_distribution_sums_to_one(self) -> None:
        """Internally the stationary distribution must sum to 1."""
        # We test this indirectly: if entropy rate is computed without error
        # then the stationary distribution was found.
        T_mat = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        ent = markov_entropy_rate(T_mat)
        self.assertFalse(math.isnan(ent.item()))
        self.assertGreater(ent.item(), 0.0)

    def test_non_negative(self) -> None:
        T_mat = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
        ent = markov_entropy_rate(T_mat)
        self.assertGreaterEqual(ent.item(), 0.0)


class TestTPMEntropyDistance(unittest.TestCase):
    def test_identical_tpms_give_zero(self) -> None:
        T_mat = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        dist = tpm_entropy_distance(T_mat, T_mat)
        self.assertAlmostEqual(dist.item(), 0.0, places=5)

    def test_different_tpms_give_positive(self) -> None:
        T1 = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        T2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        dist = tpm_entropy_distance(T1, T2)
        self.assertGreater(dist.item(), 0.0)

    def test_symmetry(self) -> None:
        T1 = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        T2 = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        self.assertAlmostEqual(
            tpm_entropy_distance(T1, T2).item(),
            tpm_entropy_distance(T2, T1).item(),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
