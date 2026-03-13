"""Tests for the BOLD -> FFT -> Hilbert visualization helpers."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from src.utils import plot_signal_pipeline, prepare_signal_pipeline


class TestSignalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        time = torch.arange(300, dtype=torch.float32) * 0.72
        subject_a = torch.stack(
            [
                torch.sin(2 * np.pi * 0.03 * time) + 0.1 * torch.sin(2 * np.pi * 0.11 * time),
                torch.cos(2 * np.pi * 0.05 * time) + 0.05 * torch.randn_like(time),
                0.8 * torch.sin(2 * np.pi * 0.06 * time + 0.7),
                0.6 * torch.cos(2 * np.pi * 0.04 * time - 0.4),
            ],
            dim=0,
        )
        subject_b = 0.9 * subject_a + 0.02 * torch.randn_like(subject_a)
        self.timeseries = torch.stack([subject_a, subject_b], dim=0)

    def test_prepare_signal_pipeline_returns_expected_shapes(self) -> None:
        pipeline = prepare_signal_pipeline(
            self.timeseries,
            subject_index=1,
            roi_indices=[0, 2, 3],
            focus_roi=2,
            max_timepoints=128,
        )

        self.assertEqual(pipeline.raw.shape, (3, 128))
        self.assertEqual(pipeline.normalized.shape, (3, 128))
        self.assertEqual(pipeline.filtered.shape, (3, 128))
        self.assertEqual(pipeline.analytic.shape, (128,))
        self.assertEqual(pipeline.envelope.shape, (128,))
        self.assertEqual(pipeline.phase.shape, (128,))
        self.assertEqual(pipeline.focus_roi, 2)
        self.assertEqual(pipeline.roi_indices, [0, 2, 3])
        self.assertTrue(np.allclose(pipeline.envelope, np.abs(pipeline.analytic)))

    def test_plot_signal_pipeline_returns_figure(self) -> None:
        pipeline = prepare_signal_pipeline(self.timeseries, max_timepoints=96)
        fig = plot_signal_pipeline(pipeline)

        self.assertIsInstance(fig, Figure)
        self.assertGreaterEqual(len(fig.axes), 4)

        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
