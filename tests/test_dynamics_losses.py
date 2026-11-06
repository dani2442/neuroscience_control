"""Tests for the sorted-entry MSE surrogate used by the FCD and phFCD losses."""

import unittest

import torch

from src.metrics import FCD, PhFCD
from src.metrics.dynamics_metrics import sorted_entry_mse
from src.metrics._utils import upper_tri_vec


def _random_symmetric(batch: int, m: int, generator: torch.Generator) -> torch.Tensor:
    x = torch.randn(batch, m, m, generator=generator, dtype=torch.float64)
    return (x + x.transpose(-1, -2)) / 2


class TestSortedEntryMSE(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = torch.Generator().manual_seed(0)

    def test_invariant_to_joint_row_column_permutation(self) -> None:
        # Reordering the windows of the target keeps its entry distribution,
        # so the loss must stay zero; the entrywise MSE would not.
        a = _random_symmetric(2, 12, self.gen)
        perm = torch.randperm(12, generator=self.gen)
        b = a[:, perm][:, :, perm]
        self.assertAlmostEqual(sorted_entry_mse(a, b).item(), 0.0, places=12)
        self.assertGreater((a - b).pow(2).mean().item(), 0.0)

    def test_never_exceeds_entrywise_mse_on_upper_triangle(self) -> None:
        a = _random_symmetric(4, 15, self.gen)
        b = _random_symmetric(4, 15, self.gen)
        entrywise = (upper_tri_vec(a) - upper_tri_vec(b)).pow(2).mean(dim=-1).mean()
        self.assertLessEqual(sorted_entry_mse(a, b).item(), entrywise.item() + 1e-12)

    def test_gradients_flow_through_losses(self) -> None:
        pred = torch.randn(2, 8, 120, generator=self.gen, dtype=torch.complex64, requires_grad=True)
        target = torch.randn(2, 8, 120, generator=self.gen, dtype=torch.complex64)
        loss = FCD(tr=0.72, fcd_win_sec=30.0, fcd_step_sec=2.0)(pred, target) + PhFCD()(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(pred.grad.abs().sum().item(), 0.0)

    def test_evaluate_reports_the_training_loss(self) -> None:
        pred = torch.randn(2, 8, 120, generator=self.gen, dtype=torch.complex64)
        target = torch.randn(2, 8, 120, generator=self.gen, dtype=torch.complex64)
        for metric, key in [(FCD(), "fcd_mse"), (PhFCD(), "phfcd_mse")]:
            self.assertAlmostEqual(metric.evaluate(pred, target)[key], metric(pred, target).item(), places=6)


if __name__ == "__main__":
    unittest.main()
