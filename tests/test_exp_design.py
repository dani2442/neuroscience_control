"""Tests for persistent-excitation experiment design."""

import importlib.util
import unittest

import numpy as np

from src.exp_design import FourierBasis, LegendreBasis, PersistentExcitationDesign

HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None


class TestBasisFamilies(unittest.TestCase):
    def test_basis_derivatives_have_expected_shapes(self) -> None:
        grid = np.linspace(0.0, 1.0, 101)
        for basis in (FourierBasis(n_basis=5, horizon=1.0), LegendreBasis(n_basis=5, horizon=1.0)):
            values = basis.evaluate(grid)
            derivative = basis.evaluate_derivative(order=2, grid=grid)

            self.assertEqual(values.shape, (5, 101))
            self.assertEqual(derivative.shape, (5, 101))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.isfinite(derivative).all())


class TestPersistentExcitationDesign(unittest.TestCase):
    def test_derivative_bound_reweights_first_derivative_penalty(self) -> None:
        design = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=5, horizon=1.0),
            pe_order=2,
            sobolev_order=2,
            derivative_bound=2.0,
            n_grid=301,
        )

        expected = design.q_np[(0, 0)] + 0.25 * design.q_np[(1, 1)] + design.q_np[(2, 2)]
        self.assertTrue(np.allclose(design.r_np, expected))
        self.assertTrue(np.allclose(design.constraint_weights_np, np.array([1.0, 0.25, 1.0])))

    def test_backprop_solution_is_feasible(self) -> None:
        design = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=7, horizon=1.0),
            pe_order=3,
            sobolev_order=3,
            n_grid=401,
        )

        result = design.optimize_backprop("e", steps=80, lr=5e-2, restarts=2, seed=0)
        norm_sq = float(result.coefficients @ design.r_np @ result.coefficients)

        self.assertAlmostEqual(norm_sq, 1.0, places=4)
        self.assertEqual(result.gramian.shape, (3, 3))
        self.assertGreaterEqual(result.eigenvalues.min(), -1e-6)

    def test_compare_criteria_returns_requested_methods(self) -> None:
        design = PersistentExcitationDesign(
            basis=LegendreBasis(n_basis=6, horizon=1.0),
            pe_order=2,
            sobolev_order=2,
            n_grid=301,
        )

        results = design.compare_criteria(
            criteria=("e", "d"),
            methods=("backprop",),
            backprop_kwargs={"steps": 40, "lr": 5e-2, "restarts": 1, "seed": 1},
        )

        self.assertEqual(set(results.keys()), {"e", "d"})
        self.assertEqual(set(results["e"].keys()), {"backprop"})
        self.assertEqual(len(results["d"]["backprop"].eigenvalues), 2)

    def test_multidimensional_design_uses_l2_derivative_gramian(self) -> None:
        design = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=5, horizon=1.0),
            signal_dim=2,
            pe_order=3,
            sobolev_order=3,
            n_grid=301,
        )

        result = design.optimize_backprop("e", steps=50, lr=5e-2, restarts=1, seed=2)

        self.assertEqual(result.gramian.shape, (3, 3))
        self.assertEqual(len(result.eigenvalues), 3)
        self.assertEqual(result.waveform.shape, (2, 301))
        self.assertAlmostEqual(design.constraint_value(result.coefficients), 1.0, places=4)
        self.assertGreaterEqual(result.eigenvalues.min(), -1e-6)

    def test_eigenvalue_count_does_not_depend_on_signal_dimension(self) -> None:
        design_scalar = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=5, horizon=1.0),
            signal_dim=1,
            pe_order=3,
            sobolev_order=3,
            n_grid=301,
        )
        design_vector = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=5, horizon=1.0),
            signal_dim=4,
            pe_order=3,
            sobolev_order=3,
            n_grid=301,
        )

        scalar = design_scalar.optimize_backprop("e", steps=40, lr=5e-2, restarts=1, seed=3)
        vector = design_vector.optimize_backprop("e", steps=40, lr=5e-2, restarts=1, seed=3)

        self.assertEqual(scalar.gramian.shape, (3, 3))
        self.assertEqual(vector.gramian.shape, (3, 3))
        self.assertEqual(len(scalar.eigenvalues), len(vector.eigenvalues))

    def test_sign_canonicalization_removes_global_flip(self) -> None:
        design = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=5, horizon=1.0),
            signal_dim=2,
            pe_order=2,
            sobolev_order=2,
            n_grid=301,
        )
        coefficients = np.array([0.2, -0.4, 0.1, 0.3, -0.5, -0.2, 0.4, -0.1, 0.6, -0.3])

        canonical = design.canonicalize_coefficients(coefficients)
        flipped = design.canonicalize_coefficients(-coefficients)

        self.assertTrue(np.allclose(canonical, flipped))
        waveform = np.asarray(design.waveform(canonical), dtype=np.float64)
        dominant = waveform.reshape(-1)[np.argmax(np.abs(waveform))]
        self.assertGreaterEqual(dominant, 0.0)

    @unittest.skipUnless(HAS_CVXPY, "cvxpy extra not installed")
    def test_sdp_solution_is_feasible(self) -> None:
        design = PersistentExcitationDesign(
            basis=FourierBasis(n_basis=5, horizon=1.0),
            pe_order=2,
            sobolev_order=2,
            n_grid=301,
        )

        result = design.solve_sdp_relaxation(
            "d",
            solver="SCS",
            solver_options={"eps": 1e-5, "max_iters": 2_000},
        )
        norm_sq = float(result.coefficients @ design.r_np @ result.coefficients)

        self.assertAlmostEqual(norm_sq, 1.0, places=4)
        self.assertIsNotNone(result.upper_bound)
        self.assertEqual(result.gramian.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
