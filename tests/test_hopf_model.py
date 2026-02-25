"""Unit tests for Hopf model utilities."""

import unittest

import torch

from src.models import CoupledHopfModel


class TestHopfSetParameters(unittest.TestCase):
    def test_accepts_scalar_and_sequence_inputs(self) -> None:
        model = CoupledHopfModel(
            n_rois=4,
            device="cpu",
            learnable_a=True,
            learnable_g=True,
            learnable_omega=True,
        )

        model.set_parameters(a=-0.01, g=0.2, omega=[1.0, 2.0, 3.0, 4.0])

        self.assertTrue(torch.isclose(model.a.detach(), torch.tensor(-0.01)))
        self.assertTrue(torch.isclose(model.g.detach(), torch.tensor(0.2)))
        self.assertTrue(
            torch.allclose(model.omega.detach(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )

    def test_scalar_omega_broadcasts(self) -> None:
        model = CoupledHopfModel(
            n_rois=3,
            device="cpu",
            learnable_omega=True,
        )

        model.set_parameters(omega=0.5)
        self.assertTrue(torch.allclose(model.omega.detach(), torch.full((3,), 0.5)))

    def test_diffusive_coupling_vanishes_for_uniform_state(self) -> None:
        sc = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.2, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        model = CoupledHopfModel(
            n_rois=3,
            structural_connectivity=sc,
            initial_a=0.0,
            initial_g=1.0,
            omega=torch.zeros(3),
            device="cpu",
            learnable_a=False,
            learnable_g=False,
            learnable_omega=False,
        )

        y = torch.ones(2, 3, dtype=torch.complex64)
        drift = model.sde_func.f(torch.tensor(0.0), y)
        # local = y*(κa − κ|y|²) = 1*(0 − 0.1*1) = −0.1; coupling = 0 for uniform state
        kappa = model.kappa.item()
        expected_local = -kappa * torch.ones_like(y)
        self.assertTrue(torch.allclose(drift, expected_local, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
