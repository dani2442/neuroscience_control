"""Unit tests for the rewritten low-rank Hybrid Hopf model."""

import unittest

import torch
import torch.nn as nn

from src.models.hybrid_hopf_model import (
    HybridHopfModel,
    KeyNetwork,
    OutputNetwork,
    ComplexLinear,
    PhasePreservingActivation,
    _svd_init,
)


# ── Building blocks ─────────────────────────────────────────────────


class TestComplexLinear(unittest.TestCase):
    def test_output_is_complex(self):
        layer = ComplexLinear(3, 5)
        x = torch.randn(2, 3, dtype=torch.cfloat)
        out = layer(x)
        self.assertTrue(torch.is_complex(out))
        self.assertEqual(out.shape, (2, 5))


class TestPhasePreservingActivation(unittest.TestCase):
    def test_preserves_phase(self):
        act = PhasePreservingActivation()
        z = torch.randn(10, dtype=torch.cfloat) + 1j * torch.randn(10)
        out = act(z)
        # phase should be identical
        self.assertTrue(torch.allclose(z.angle(), out.angle(), atol=1e-5))

    def test_magnitude_bounded(self):
        act = PhasePreservingActivation()
        z = torch.randn(100, dtype=torch.cfloat) * 100
        out = act(z)
        self.assertTrue((out.abs() <= 1.0 + 1e-5).all())


class TestKeyNetwork(unittest.TestCase):
    def test_output_shape(self):
        net = KeyNetwork(d_key=4, hidden_dim=8, n_layers=1)
        z = torch.randn(2, 10, dtype=torch.cfloat)
        out = net(z)
        self.assertEqual(out.shape, (2, 10, 4))
        self.assertTrue(torch.is_complex(out))


class TestOutputNetwork(unittest.TestCase):
    def test_output_shape(self):
        net = OutputNetwork(d_key=4, hidden_dim=8, n_layers=1)
        m = torch.randn(2, 10, 4, dtype=torch.cfloat)
        z = torch.randn(2, 10, dtype=torch.cfloat)
        out = net(m, z)
        self.assertEqual(out.shape, (2, 10))
        self.assertTrue(torch.is_complex(out))


# ── SVD initialisation ──────────────────────────────────────────────


class TestSVDInit(unittest.TestCase):
    def test_reconstruction(self):
        """L R^T should approximate the original matrix."""
        sc = torch.rand(20, 20)
        sc = (sc + sc.T) / 2
        sc.fill_diagonal_(0.0)
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)

        L, R = _svd_init(sc, rank=20)
        recon = L @ R.T
        # Full-rank SVD should reconstruct exactly
        self.assertTrue(torch.allclose(recon, sc.float(), atol=1e-5))

    def test_low_rank_shape(self):
        L, R = _svd_init(torch.rand(50, 50), rank=5)
        self.assertEqual(L.shape, (50, 5))
        self.assertEqual(R.shape, (50, 5))


# ── Low-rank model ──────────────────────────────────────────────────


class TestHybridHopfLowRank(unittest.TestCase):
    """Tests with coupling_rank < n_rois."""

    def setUp(self):
        self.n_rois = 10
        self.rank = 3
        self.sc = torch.rand(self.n_rois, self.n_rois)
        self.sc = (self.sc + self.sc.T) / 2
        self.sc.fill_diagonal_(0.0)

    def _make_model(self, **overrides):
        defaults = dict(
            n_rois=self.n_rois,
            structural_connectivity=self.sc,
            coupling_rank=self.rank,
            d_key=4,
            coupling_hidden_dim=8,
            coupling_n_layers=1,
            noise_sigma=0.02,
            device="cpu",
        )
        defaults.update(overrides)
        return HybridHopfModel(**defaults)

    def test_forward_shape(self):
        model = self._make_model()
        z0 = torch.randn(2, self.n_rois, dtype=torch.cfloat)
        out = model(z0, n_steps=5)
        self.assertEqual(out.shape, (2, self.n_rois, 5))
        self.assertTrue(torch.is_complex(out))

    def test_low_rank_factors_exist(self):
        model = self._make_model()
        self.assertIsNotNone(model.coupling_L)
        self.assertIsNotNone(model.coupling_R)
        self.assertIsNone(model.coupling_full)
        self.assertEqual(model.coupling_L.shape, (self.n_rois, self.rank))
        self.assertEqual(model.coupling_R.shape, (self.n_rois, self.rank))

    def test_coupling_factors_learnable(self):
        model = self._make_model()
        self.assertIsInstance(model.coupling_L, nn.Parameter)
        self.assertIsInstance(model.coupling_R, nn.Parameter)

    def test_gradients_flow(self):
        model = self._make_model()
        z0 = torch.randn(1, self.n_rois, dtype=torch.cfloat)
        out = model(z0, n_steps=3)
        loss = out.real.sum()
        loss.backward()

        # Key + output nets
        for name, p in model.key_net.named_parameters():
            self.assertIsNotNone(p.grad, f"No gradient for key_net.{name}")
        for name, p in model.output_net.named_parameters():
            self.assertIsNotNone(p.grad, f"No gradient for output_net.{name}")

        # Low-rank factors
        self.assertIsNotNone(model.coupling_L.grad)
        self.assertIsNotNone(model.coupling_R.grad)

    def test_param_count_lower_than_full(self):
        low = self._make_model(coupling_rank=3)
        full = self._make_model(coupling_rank=None)
        n_low = sum(p.numel() for p in low.parameters())
        n_full = sum(p.numel() for p in full.parameters())
        self.assertLess(n_low, n_full)


# ── Full-rank model ─────────────────────────────────────────────────


class TestHybridHopfFullRank(unittest.TestCase):
    """Tests with coupling_rank=None (full n×n matrix)."""

    def setUp(self):
        self.n_rois = 8
        self.sc = torch.rand(self.n_rois, self.n_rois)
        self.sc.fill_diagonal_(0.0)

    def _make_model(self, **overrides):
        defaults = dict(
            n_rois=self.n_rois,
            structural_connectivity=self.sc,
            coupling_rank=None,
            d_key=2,
            coupling_hidden_dim=8,
            coupling_n_layers=1,
            noise_sigma=0.02,
            device="cpu",
        )
        defaults.update(overrides)
        return HybridHopfModel(**defaults)

    def test_forward_shape(self):
        model = self._make_model()
        z0 = torch.randn(2, self.n_rois, dtype=torch.cfloat)
        out = model(z0, n_steps=5)
        self.assertEqual(out.shape, (2, self.n_rois, 5))

    def test_full_matrix_exists(self):
        model = self._make_model()
        self.assertIsNotNone(model.coupling_full)
        self.assertIsNone(model.coupling_L)
        self.assertIsNone(model.coupling_R)
        self.assertEqual(model.coupling_full.shape, (self.n_rois, self.n_rois))

    def test_full_matrix_initialized_from_sc(self):
        model = self._make_model()
        sc_norm = self.sc / (self.sc.sum(dim=1, keepdim=True) + 1e-8)
        self.assertTrue(torch.allclose(model.coupling_full.data, sc_norm, atol=1e-5))

    def test_gradients_flow(self):
        model = self._make_model()
        z0 = torch.randn(1, self.n_rois, dtype=torch.cfloat)
        out = model(z0, n_steps=3)
        out.real.sum().backward()
        self.assertIsNotNone(model.coupling_full.grad)


# ── Control inputs ──────────────────────────────────────────────────


class TestHybridHopfControl(unittest.TestCase):
    def test_lowrank_with_control(self):
        n_rois, n_ctrl = 6, 2
        model = HybridHopfModel(
            n_rois=n_rois, coupling_rank=3, d_key=2,
            noise_sigma=0.01, n_control_dims=n_ctrl, device="cpu",
        )
        z0 = torch.randn(2, n_rois, dtype=torch.cfloat)
        u = torch.randn(2, n_ctrl)
        out = model(z0, n_steps=4, control=u)
        self.assertEqual(out.shape, (2, n_rois, 4))

    def test_fullrank_with_control(self):
        n_rois, n_ctrl = 6, 2
        model = HybridHopfModel(
            n_rois=n_rois, coupling_rank=None, d_key=2,
            noise_sigma=0.01, n_control_dims=n_ctrl, device="cpu",
        )
        z0 = torch.randn(2, n_rois, dtype=torch.cfloat)
        u = torch.randn(2, n_ctrl)
        out = model(z0, n_steps=4, control=u)
        self.assertEqual(out.shape, (2, n_rois, 4))

    def test_control_coupling_learnable(self):
        model = HybridHopfModel(
            n_rois=5, n_control_dims=3, device="cpu",
        )
        self.assertIsInstance(model.control_coupling, nn.Parameter)


# ── Config / serialisation ──────────────────────────────────────────


class TestHybridHopfConfig(unittest.TestCase):
    def test_config_roundtrip_lowrank(self):
        model = HybridHopfModel(
            n_rois=10, coupling_rank=5, d_key=3, device="cpu",
        )
        cfg = model.get_model_config()
        self.assertEqual(cfg["coupling_rank"], 5)
        self.assertEqual(cfg["d_key"], 3)
        self.assertEqual(cfg["n_rois"], 10)

    def test_config_roundtrip_fullrank(self):
        model = HybridHopfModel(
            n_rois=8, coupling_rank=None, d_key=4, device="cpu",
        )
        cfg = model.get_model_config()
        self.assertIsNone(cfg["coupling_rank"])

    def test_get_parameters_dict_lowrank(self):
        model = HybridHopfModel(n_rois=6, coupling_rank=2, device="cpu")
        params = model.get_parameters_dict()
        self.assertIn("coupling_L", params)
        self.assertIn("coupling_R", params)
        self.assertNotIn("coupling_matrix", params)

    def test_get_parameters_dict_fullrank(self):
        model = HybridHopfModel(n_rois=6, coupling_rank=None, device="cpu")
        params = model.get_parameters_dict()
        self.assertIn("coupling_matrix", params)
        self.assertNotIn("coupling_L", params)


# ── Memory complexity check ─────────────────────────────────────────


class TestMemoryComplexity(unittest.TestCase):
    """Verify no O(n²) tensor is created in the neural-network path."""

    def test_no_n_squared_network_calls(self):
        n_rois = 15
        model = HybridHopfModel(
            n_rois=n_rois, coupling_rank=3, d_key=4,
            coupling_hidden_dim=8, noise_sigma=0.01, device="cpu",
        )
        key_shapes, out_shapes = [], []
        orig_key = model.key_net.forward
        orig_out = model.output_net.forward

        def track_key(z):
            key_shapes.append(z.shape)
            return orig_key(z)

        def track_out(m, z):
            out_shapes.append(m.shape)
            return orig_out(m, z)

        model.key_net.forward = track_key
        model.output_net.forward = track_out

        z0 = torch.randn(1, n_rois, dtype=torch.cfloat)
        model(z0, n_steps=3)

        for s in key_shapes:
            # key_net input should be (batch, n_rois) — never (batch, n², ...)
            self.assertEqual(len(s), 2)
            self.assertEqual(s[1], n_rois)

        for s in out_shapes:
            # output_net m_i should be (batch, n_rois, d_key)
            self.assertEqual(len(s), 3)
            self.assertEqual(s[1], n_rois)


if __name__ == "__main__":
    unittest.main()
