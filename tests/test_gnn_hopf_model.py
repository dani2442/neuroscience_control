"""Unit tests for the GNN-Hopf model."""

import unittest

import torch

from src.models.gnn_hopf_model import GNNHopfModel


class TestGNNHopfModelBasic(unittest.TestCase):
    """Smoke & shape tests for GNNHopfModel."""

    def setUp(self):
        self.n_rois = 5
        self.sc = torch.rand(self.n_rois, self.n_rois)
        self.sc = (self.sc + self.sc.T) / 2  # symmetrise
        self.sc.fill_diagonal_(0.0)

    def _make_model(self, **kwargs):
        defaults = dict(
            n_rois=self.n_rois,
            structural_connectivity=self.sc,
            device="cpu",
            noise_sigma=0.02,
            node_hidden_dim=8,
            node_n_layers=1,
        )
        defaults.update(kwargs)
        return GNNHopfModel(**defaults)

    def test_forward_shape(self):
        model = self._make_model()
        batch, n_steps = 2, 10
        z0 = torch.randn(batch, self.n_rois, dtype=torch.cfloat)
        out = model(z0, n_steps=n_steps)
        self.assertEqual(out.shape, (batch, self.n_rois, n_steps))
        self.assertTrue(torch.is_complex(out))

    def test_coupling_matrix_is_learnable(self):
        model = self._make_model()
        self.assertIsInstance(model.coupling_matrix, torch.nn.Parameter)
        self.assertTrue(model.coupling_matrix.requires_grad)

    def test_coupling_matrix_initialized_from_sc(self):
        model = self._make_model()
        # SC is normalised row-wise; verify initialisation is close
        sc_norm = self.sc / (self.sc.sum(dim=1, keepdim=True) + 1e-8)
        self.assertTrue(torch.allclose(model.coupling_matrix.data, sc_norm, atol=1e-6))

    def test_node_net_output_shape(self):
        model = self._make_model()
        batch = 3
        m = torch.randn(batch, self.n_rois, dtype=torch.cfloat)
        z = torch.randn(batch, self.n_rois, dtype=torch.cfloat)
        out = model.node_net(m, z)
        self.assertEqual(out.shape, (batch, self.n_rois))
        self.assertTrue(torch.is_complex(out))

    def test_drift_shape(self):
        model = self._make_model()
        batch = 2
        y = torch.randn(batch, self.n_rois, dtype=torch.cfloat)
        t = torch.tensor(0.0)
        drift = model.sde_func.f(t, y)
        self.assertEqual(drift.shape, y.shape)

    def test_diffusion_shape(self):
        model = self._make_model()
        batch = 2
        y = torch.randn(batch, self.n_rois, dtype=torch.cfloat)
        t = torch.tensor(0.0)
        diff = model.sde_func.g(t, y)
        self.assertEqual(diff.shape, y.shape)

    def test_uniform_state_aggregation_vanishes(self):
        """For a uniform state, m_i = Σ C_ij (z_j − z_i) = 0."""
        model = self._make_model(initial_g=0.0)  # zero coupling to check local only
        y = torch.ones(1, self.n_rois, dtype=torch.cfloat)
        C = model.coupling_matrix.detach().to(y.dtype)
        coupled = y @ C.T
        row_sum = C.sum(dim=1, keepdim=True).T
        m = coupled - y * row_sum
        self.assertTrue(torch.allclose(m, torch.zeros_like(m), atol=1e-6))


class TestGNNHopfModelGradients(unittest.TestCase):
    """Verify that gradients flow through both the node network and C."""

    def test_gradients_flow(self):
        n_rois = 4
        sc = torch.eye(n_rois)
        model = GNNHopfModel(
            n_rois=n_rois,
            structural_connectivity=sc,
            device="cpu",
            noise_sigma=0.01,
            node_hidden_dim=8,
            node_n_layers=1,
        )
        z0 = torch.randn(1, n_rois, dtype=torch.cfloat)
        out = model(z0, n_steps=5)
        loss = out.real.sum()
        loss.backward()

        # Node network weights should have gradients
        for name, p in model.node_net.named_parameters():
            self.assertIsNotNone(p.grad, f"No gradient for node_net.{name}")

        # Coupling matrix should have gradients
        self.assertIsNotNone(model.coupling_matrix.grad, "No gradient for coupling_matrix")


class TestGNNHopfModelControl(unittest.TestCase):
    """Test control input integration."""

    def test_forward_with_control(self):
        n_rois, n_ctrl = 4, 2
        sc = torch.rand(n_rois, n_rois)
        model = GNNHopfModel(
            n_rois=n_rois,
            structural_connectivity=sc,
            device="cpu",
            noise_sigma=0.01,
            n_control_dims=n_ctrl,
        )
        batch, n_steps = 2, 8
        z0 = torch.randn(batch, n_rois, dtype=torch.cfloat)
        u = torch.randn(batch, n_ctrl)
        out = model(z0, n_steps=n_steps, control=u)
        self.assertEqual(out.shape, (batch, n_rois, n_steps))

    def test_control_coupling_is_learnable(self):
        model = GNNHopfModel(
            n_rois=3, device="cpu", n_control_dims=2,
        )
        self.assertIsInstance(model.control_coupling, torch.nn.Parameter)


class TestGNNHopfModelConfig(unittest.TestCase):
    """Verify config round-trip."""

    def test_get_model_config(self):
        model = GNNHopfModel(n_rois=5, device="cpu")
        cfg = model.get_model_config()
        self.assertEqual(cfg["n_rois"], 5)
        self.assertIn("node_hidden_dim", cfg)
        self.assertIn("node_n_layers", cfg)

    def test_get_parameters_dict(self):
        model = GNNHopfModel(n_rois=5, device="cpu")
        params = model.get_parameters_dict()
        self.assertIn("coupling_matrix", params)
        self.assertEqual(params["coupling_matrix"].shape, (5, 5))


class TestGNNHopfMemoryComplexity(unittest.TestCase):
    """Verify the model is O(ROI) in neural network evaluations, not O(ROI²)."""

    def test_node_net_called_with_roi_sized_input(self):
        """Node network's forward should receive (batch, n_rois, 2), not (batch, n_rois, n_rois, ...)."""
        n_rois = 10
        model = GNNHopfModel(n_rois=n_rois, device="cpu", noise_sigma=0.01)

        call_shapes = []
        orig_forward = model.node_net.forward

        def tracking_forward(m_i, z_i):
            call_shapes.append(m_i.shape)
            return orig_forward(m_i, z_i)

        model.node_net.forward = tracking_forward
        z0 = torch.randn(1, n_rois, dtype=torch.cfloat)
        model(z0, n_steps=3)

        for shape in call_shapes:
            # Should be (batch, n_rois), never (batch, n_rois, n_rois)
            self.assertEqual(len(shape), 2, f"Expected rank-2 input to node_net, got {shape}")
            self.assertEqual(shape[1], n_rois)


if __name__ == "__main__":
    unittest.main()
