"""Tests for public dataset backend dispatch."""

from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

import torch

from examples.cli_args import add_dataset_args
from examples.train_models import _apply_dataset_time_defaults
from src.dataset.data_loader import NeuroscienceDataset, load_dataset
from src.training.config import TrainingConfig


class TestDatasetBackends(unittest.TestCase):
    def _dummy_dataset(self) -> object:
        return argparse.Namespace(
            n_subjects=2,
            n_rois=3,
            n_timepoints=4,
            fc_mean=torch.zeros(3, 3),
            timeseries=torch.zeros(2, 3, 4, dtype=torch.complex64),
            n_control_dims=0,
            dt=2.0,
        )

    def test_cli_dataset_default_leaves_training_config_default(self) -> None:
        parser = argparse.ArgumentParser()
        add_dataset_args(parser)

        args = parser.parse_args([])

        self.assertIsNone(args.dataset_type)
        self.assertIsNone(args.data_path)

    def test_cli_accepts_public_dataset_backends(self) -> None:
        parser = argparse.ArgumentParser()
        add_dataset_args(parser)

        self.assertEqual(parser.parse_args(["--dataset-type", "abide"]).dataset_type, "abide")
        self.assertEqual(parser.parse_args(["--dataset-type", "adhd200"]).dataset_type, "adhd200")

    def test_dataset_time_defaults_use_public_dataset_trs(self) -> None:
        for dataset_type in ("abide", "adhd200"):
            cfg = TrainingConfig(dataset_type=dataset_type)
            cfg.tr = 0.72
            _apply_dataset_time_defaults(cfg, argparse.Namespace(tr=None))
            self.assertEqual(cfg.tr, 2.0)

        cfg = TrainingConfig(dataset_type="abide")
        cfg.tr = 1.5
        _apply_dataset_time_defaults(cfg, argparse.Namespace(tr=1.5))
        self.assertEqual(cfg.tr, 1.5)

    def test_load_dataset_dispatches_abide_backend(self) -> None:
        cfg = TrainingConfig(dataset_type="abide")
        cfg.abide_n_subjects = 2

        with patch.object(
            NeuroscienceDataset,
            "from_abide_pcp",
            return_value=self._dummy_dataset(),
        ) as from_abide:
            load_dataset(cfg, "cpu")

        from_abide.assert_called_once()
        _, kwargs = from_abide.call_args
        self.assertEqual(kwargs["data_dir"], "data/abide")
        self.assertEqual(kwargs["n_subjects"], 2)

    def test_load_dataset_dispatches_adhd200_backend(self) -> None:
        cfg = TrainingConfig(dataset_type="adhd200")
        cfg.adhd200_n_subjects = 3
        cfg.adhd200_local_pattern = "**/sfnwmrda*.nii.gz"

        with patch.object(
            NeuroscienceDataset,
            "from_adhd200",
            return_value=self._dummy_dataset(),
        ) as from_adhd200:
            load_dataset(cfg, "cpu")

        from_adhd200.assert_called_once()
        _, kwargs = from_adhd200.call_args
        self.assertEqual(kwargs["data_dir"], "data/adhd200")
        self.assertEqual(kwargs["n_subjects"], 3)
        self.assertEqual(kwargs["local_pattern"], "**/sfnwmrda*.nii.gz")


if __name__ == "__main__":
    unittest.main()
