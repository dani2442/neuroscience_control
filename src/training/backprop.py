"""Shared helpers for backpropagation training scripts."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from ..dataset import NeuroscienceDataset, create_data_loaders
from ..metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
from ..metrics.metrics_store import MetricsStore
from ..models import BaseNeuroscienceModel
from .config import TrainingConfig
from .trainer import Trainer


# ---------------------------------------------------------------------------
# Dataset loading (shared across all training scripts)
# ---------------------------------------------------------------------------

def load_dataset(
    cfg: TrainingConfig,
    device: str,
) -> NeuroscienceDataset:
    """Load dataset according to *cfg* and print a summary."""
    dataset = NeuroscienceDataset(
        filepath=cfg.data_path,
        normalize=True,
        device=device,
        max_subjects=cfg.max_subjects,
        dt=cfg.tr,
        fourier_denoise=cfg.fourier_denoise,
        denoise_f_lo=cfg.denoise_f_lo,
        denoise_f_hi=cfg.denoise_f_hi,
    )
    print("Loaded dataset:")
    print(f"  - subjects={dataset.n_subjects}  ROIs={dataset.n_rois}  T={dataset.n_timepoints}")
    print(f"  - FC shape={dataset.fc_mean.shape}  dtype={dataset.timeseries.dtype}  dt={dataset.dt}s")
    return dataset


def create_windowed_loaders(
    dataset: NeuroscienceDataset,
    cfg: TrainingConfig,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Build train/val/test data loaders from random-windowed timeseries."""
    window_size = min(cfg.window_size, dataset.n_timepoints // 4)

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        batch_size=cfg.batch_size,
        n_windows_per_epoch=cfg.n_windows_per_epoch,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        device=device,
    )
    return train_loader, val_loader, test_loader, window_size


def run_backprop_training(
    model: BaseNeuroscienceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    window_size: int,
    cfg: TrainingConfig,
    device: str,
    experiment_name: str | None = None,
    extra_dyn_kwargs: dict[str, object] | None = None,
) -> tuple[Trainer, MetricsStore, dict[str, float]]:
    """
    Train and evaluate a model with the shared :class:`Trainer` workflow.

    Returns:
        (trainer, metrics_store, test_metrics)
    """
    trainer = Trainer(
        model=model,
        lr=cfg.lr,
        loss_fn=cfg.loss_fn,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        experiment_name=experiment_name or cfg.experiment_name,
        cfg=cfg,
        use_wandb=cfg.use_wandb,
        extra_dyn_kwargs=extra_dyn_kwargs,
    )

    metrics_store = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.n_epochs,
        n_steps=window_size,
        dt=cfg.tr,
        early_stopping_patience=cfg.early_stopping_patience,
        verbose=True,
    )

    test_metrics = trainer.test(
        test_loader=test_loader,
        n_steps=window_size,
        dt=cfg.tr,
    )
    return trainer, metrics_store, test_metrics
