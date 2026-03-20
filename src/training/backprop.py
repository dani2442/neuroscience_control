"""Shared helpers for backpropagation training scripts."""

from __future__ import annotations

from torch.utils.data import DataLoader

from ..dataset import NeuroscienceDataset, create_data_loaders
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
    dataset_type = getattr(cfg, "dataset_type", "ts_young")
    dataset_kwargs = dict(
        normalize=True,
        device=device,
        max_subjects=cfg.max_subjects,
        dt=cfg.tr,
        fourier_denoise=cfg.fourier_denoise,
        denoise_f_lo=cfg.denoise_f_lo,
        denoise_f_hi=cfg.denoise_f_hi,
    )

    if dataset_type == "lsd":
        lsd_dir = getattr(cfg, "lsd_data_dir", "data/lsd")
        dataset = NeuroscienceDataset.from_lsd(
            data_dir=lsd_dir,
            **dataset_kwargs,
        )
    elif dataset_type == "nilearn":
        dataset = NeuroscienceDataset.from_nilearn(
            dataset_name=cfg.nilearn_dataset,
            data_dir=cfg.nilearn_data_dir,
            n_subjects=cfg.nilearn_n_subjects,
            reduce_confounds=cfg.nilearn_reduce_confounds,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            **dataset_kwargs,
        )
    elif dataset_type == "openneuro":
        dataset = NeuroscienceDataset.from_openneuro(
            dataset=cfg.openneuro_dataset,
            target_dir=cfg.openneuro_target_dir,
            tag=cfg.openneuro_tag,
            include=cfg.openneuro_include,
            exclude=cfg.openneuro_exclude,
            bids_relative_path=cfg.bids_relative_path,
            derivatives_dir=cfg.bids_derivatives_dir,
            task=cfg.bids_task,
            space=cfg.bids_space,
            desc=cfg.bids_desc,
            subject_ids=cfg.bids_subject_ids,
            n_runs_per_subject=cfg.bids_runs_per_subject,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            atlas_data_dir=cfg.nilearn_data_dir,
            **dataset_kwargs,
        )
    elif dataset_type == "datalad":
        dataset = NeuroscienceDataset.from_datalad(
            source=cfg.datalad_source,
            dataset_dir=cfg.datalad_dataset_dir,
            get_paths=cfg.datalad_get_paths,
            bids_relative_path=cfg.bids_relative_path,
            derivatives_dir=cfg.bids_derivatives_dir,
            task=cfg.bids_task,
            space=cfg.bids_space,
            desc=cfg.bids_desc,
            subject_ids=cfg.bids_subject_ids,
            n_runs_per_subject=cfg.bids_runs_per_subject,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            atlas_data_dir=cfg.nilearn_data_dir,
            **dataset_kwargs,
        )
    elif dataset_type == "bids":
        bids_root = cfg.bids_root
        if not bids_root:
            raise ValueError("dataset_type='bids' requires cfg.bids_root to be set.")
        dataset = NeuroscienceDataset.from_bids(
            bids_root=bids_root,
            derivatives_dir=cfg.bids_derivatives_dir,
            task=cfg.bids_task,
            space=cfg.bids_space,
            desc=cfg.bids_desc,
            subject_ids=cfg.bids_subject_ids,
            n_runs_per_subject=cfg.bids_runs_per_subject,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            atlas_data_dir=cfg.nilearn_data_dir,
            **dataset_kwargs,
        )
    elif dataset_type in ("ts_young", "mat"):
        dataset = NeuroscienceDataset(
            filepath=cfg.data_path,
            **dataset_kwargs,
        )
    else:
        raise ValueError(
            "Unsupported dataset_type. Expected one of "
            "{'ts_young', 'mat', 'lsd', 'nilearn', 'openneuro', 'datalad', 'bids'}, "
            f"got {dataset_type!r}."
        )
    print("Loaded dataset:")
    print(f"  - subjects={dataset.n_subjects}  ROIs={dataset.n_rois}  T={dataset.n_timepoints}")
    print(f"  - FC shape={dataset.fc_mean.shape}  dtype={dataset.timeseries.dtype}  dt={dataset.dt}s")
    print(f"  - control_dims={dataset.n_control_dims}")
    return dataset


def create_windowed_loaders(
    dataset: NeuroscienceDataset,
    cfg: TrainingConfig,
    device: str,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
    """Build train/val/test_inter/test_intra data loaders from random-windowed timeseries."""
    window_size = min(cfg.window_size, dataset.n_timepoints // 4)

    train_loader, val_loader, test_inter_loader, test_intra_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        batch_size=cfg.batch_size,
        n_windows_per_epoch=cfg.n_windows_per_epoch,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        device=device,
    )
    return train_loader, val_loader, test_inter_loader, test_intra_loader, window_size


def run_backprop_training(
    model: BaseNeuroscienceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_inter_loader: DataLoader,
    test_intra_loader: DataLoader,
    window_size: int,
    cfg: TrainingConfig,
    device: str,
    experiment_name: str | None = None,
    extra_dyn_kwargs: dict[str, object] | None = None,
) -> tuple[Trainer, MetricsStore, dict[str, float], dict[str, float]]:
    """
    Train and evaluate a model with the shared :class:`Trainer` workflow.

    Returns:
        (trainer, metrics_store, test_inter_metrics, test_intra_metrics)
    """
    trainer = Trainer(
        model=model,
        lr=cfg.lr,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        experiment_name=experiment_name or cfg.experiment_name,
        cfg=cfg,
        use_wandb=cfg.use_wandb,
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

    test_inter_metrics = trainer.test(
        test_inter_loader,
        n_steps=window_size,
        dt=cfg.tr,
        label="test_inter",
    )
    test_intra_metrics = trainer.test(
        test_intra_loader,
        n_steps=window_size,
        dt=cfg.tr,
        label="test_intra",
    )
    return trainer, metrics_store, test_inter_metrics, test_intra_metrics
