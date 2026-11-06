"""Group-level quantities (connectome prior, omega) must not see held-out data."""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pytest
import torch

from examples.train_models import _group_view
from examples.train_personalized import _group_prior
from src.dataset import GroupLeakageError, resolve_split, training_view
from src.dataset.data_loader import NeuroscienceDataset
from src.models import build_model
from src.training.config import HopfConfig


def _dataset(n_subjects=20, n_rois=6, n_timepoints=200, seed=0) -> NeuroscienceDataset:
    raw = np.random.default_rng(seed).standard_normal((n_subjects, n_rois, n_timepoints))
    return NeuroscienceDataset._from_raw_timeseries(
        raw, normalize=True, device="cpu", max_subjects=None, dt=0.72,
        fourier_denoise=True, denoise_f_lo=0.008, denoise_f_hi=0.08,
    )


def _scramble_held_out(dataset, held_out_subjects, train_subjects, t_stop):
    """Copy of *dataset* with every held-out value replaced by noise."""
    out = copy.copy(dataset)
    ts = dataset.timeseries.clone()
    gen = torch.Generator().manual_seed(1)
    noise = lambda shape: torch.randn(*shape, generator=gen) + 1j * torch.randn(*shape, generator=gen)
    held = torch.as_tensor(np.asarray(held_out_subjects, dtype=np.int64))
    ts[held] = noise(ts[held].shape).to(ts.dtype)
    train = torch.as_tensor(np.asarray(train_subjects, dtype=np.int64))
    ts[train, :, t_stop:] = noise(ts[train, :, t_stop:].shape).to(ts.dtype)
    out.timeseries = ts
    return out


def test_cohort_prior_and_omega_ignore_held_out_data():
    cfg = HopfConfig()
    dataset = _dataset()
    train_idx, val_idx, test_idx = resolve_split(dataset, cfg.train_ratio, cfg.val_ratio, 42)
    t_stop = dataset.n_timepoints // 2
    scrambled = _scramble_held_out(
        dataset, np.concatenate([val_idx, test_idx]), train_idx, t_stop)

    models = []
    for ds in (dataset, scrambled):
        view = _group_view(ds, cfg, train_idx, val_idx, test_idx)
        assert view.n_subjects == len(train_idx) and view.n_timepoints == t_stop
        models.append(build_model("hopf", view, cfg, "cpu",
                                  structural_connectivity=view.fc_mean))
    a, b = models
    torch.testing.assert_close(a.structural_connectivity, b.structural_connectivity)
    torch.testing.assert_close(a.omega, b.omega)

    # Sanity check: the prior does respond to the training segments.
    changed = _scramble_held_out(dataset, train_idx, [], t_stop)
    view = _group_view(changed, cfg, train_idx, val_idx, test_idx)
    assert not torch.allclose(view.fc_mean, a.structural_connectivity)


def test_training_view_rejects_held_out_subjects():
    dataset = _dataset()
    with pytest.raises(GroupLeakageError):
        training_view(dataset, [0, 1, 2], t_stop=100, held_out_idx=[2, 5])


def test_personalization_prior_excludes_target():
    cfg = HopfConfig()
    dataset = _dataset()
    train_idx, _, _ = resolve_split(dataset, cfg.train_ratio, cfg.val_ratio, 42)
    in_pool = int(train_idx[0])
    outside = min(set(range(dataset.n_subjects)) - set(int(i) for i in train_idx))

    with pytest.raises(SystemExit):
        _group_prior(dataset, cfg, in_pool, split_seed=42)

    view = _group_prior(dataset, cfg, outside, split_seed=42)
    assert outside not in view.group_prior_info["subjects"]
    assert view.group_prior_info["subjects"] == sorted(int(i) for i in train_idx)
    assert view.group_prior_info["t_stop"] == dataset.n_timepoints // 2
