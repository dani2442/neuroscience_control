"""The functional connectome and omega must come from training data only."""

from __future__ import annotations

import argparse
import unittest

import torch

from src.dataset import compute_fc_from_timeseries, compute_split_indices, estimate_split_priors
from src.models import build_model
from src.models.checkpointing import _rename_legacy_keys
from src.training.config import HybridHopfConfig, TrainingConfig


def _dataset(n_subjects: int = 10, n_rois: int = 6, n_timepoints: int = 200, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    ts = torch.randn(n_subjects, n_rois, n_timepoints, generator=gen).to(torch.complex64)
    return argparse.Namespace(
        n_subjects=n_subjects, n_rois=n_rois, n_timepoints=n_timepoints,
        timeseries=ts, dt=0.72, patient_ids=None, n_control_dims=0,
    )


class TestSplitPriors(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = TrainingConfig()
        self.ds = _dataset()
        self.train_idx, _, self.test_idx = compute_split_indices(
            self.ds, self.cfg.train_ratio, self.cfg.val_ratio, self.cfg.seed,
        )

    def test_connectome_is_mean_fc_of_training_segments(self) -> None:
        priors = estimate_split_priors(self.ds, self.train_idx, self.cfg)
        t_split = self.ds.n_timepoints // 2
        expected = compute_fc_from_timeseries(self.ds.timeseries[self.train_idx, :, :t_split]).mean(0)
        self.assertTrue(torch.allclose(priors.connectome, expected))
        self.assertEqual(priors.t_stop, t_split)

    def test_held_out_data_does_not_change_priors(self) -> None:
        before = estimate_split_priors(self.ds, self.train_idx, self.cfg)
        held_out = [i for i in range(self.ds.n_subjects) if i not in set(self.train_idx.tolist())]
        t_split = self.ds.n_timepoints // 2
        perturbed = self.ds.timeseries.clone()
        perturbed[held_out] = 5.0 * torch.randn_like(perturbed[held_out].real).to(perturbed.dtype)
        perturbed[self.train_idx, :, t_split:] = 0.0  # test_intra segments
        self.ds.timeseries = perturbed
        after = estimate_split_priors(self.ds, self.train_idx, self.cfg)
        self.assertTrue(torch.equal(before.connectome, after.connectome))
        self.assertTrue(torch.equal(before.omega, after.omega))
        self.assertEqual(before.fingerprint(), after.fingerprint())

    def test_full_timeseries_mode_uses_whole_training_scans(self) -> None:
        self.cfg.use_full_timeseries = True
        priors = estimate_split_priors(self.ds, self.train_idx, self.cfg)
        self.assertEqual(priors.t_stop, self.ds.n_timepoints)

    def test_hopf_family_requires_explicit_priors(self) -> None:
        with self.assertRaises(ValueError):
            build_model("hybrid_hopf", self.ds, HybridHopfConfig(), "cpu")

    def test_legacy_checkpoint_keys_are_renamed(self) -> None:
        old = {"structural_connectivity": 1, "sde_func.sc": 2, "coupling_L": 3}
        self.assertEqual(
            _rename_legacy_keys(old), {"connectome": 1, "sde_func.connectome": 2, "coupling_L": 3},
        )


if __name__ == "__main__":
    unittest.main()
