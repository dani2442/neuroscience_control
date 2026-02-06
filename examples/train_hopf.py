#!/usr/bin/env python3
"""
Training script for Coupled Hopf Model.

Supports both:
- Grid search over (G, a)
- Gradient-based training (backpropagation)
"""

import argparse
import dataclasses
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# Set up proxy before importing wandb
os.environ["HTTP_PROXY"] = "http://proxy.nhr.fau.de:80"
os.environ["HTTPS_PROXY"] = "http://proxy.nhr.fau.de:80"

# Import project modules
from src.dataset import NeuroscienceDataset, compute_omega_from_timeseries, create_data_loaders
from src.models import CoupledHopfModel
from src.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
from src.training import Trainer, grid_search_hopf, HopfConfig
from src.utils import (
    plot_fc_comparison,
    plot_realizations,
    plot_timeseries,
    FIGURES_DIR,
)


def setup_device(device: str = "auto") -> str:
    """Set up computation device."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU")
    return device


def init_wandb(cfg: HopfConfig, tags: list[str], run_suffix: str = ""):
    """Initialize wandb with proxy settings for non-Trainer workflows."""
    if not cfg.use_wandb:
        return None

    settings = wandb.Settings(_service_transport="http")
    run_name = cfg.run_name if not run_suffix else f"{cfg.run_name}_{run_suffix}"

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        config=dataclasses.asdict(cfg),
        settings=settings,
        tags=tags,
    )

    print(f"Wandb initialized: {cfg.wandb_project}/{run_name}")
    return run


def load_data(cfg: HopfConfig, device: str):
    """Load and prepare dataset."""
    print(f"\n{'=' * 60}")
    print("STEP 1: Loading and Processing Data")
    print('=' * 60)

    dataset = NeuroscienceDataset(
        filepath=cfg.data_path,
        normalize=True,
        device=device,
        max_subjects=cfg.max_subjects,
        dt=cfg.tr,
    )

    print("Loaded dataset:")
    print(f"  - Number of subjects: {dataset.n_subjects}")
    print(f"  - Number of ROIs: {dataset.n_rois}")
    print(f"  - Number of timepoints: {dataset.n_timepoints}")
    print(f"  - FC matrix shape: {dataset.fc_mean.shape}")
    print(f"  - Time array shape: {dataset.ts.shape}")
    print(f"  - dt (TR): {dataset.dt}s")

    omega = compute_omega_from_timeseries(
        dataset.timeseries,
        dt=dataset.dt,
        f_lo=cfg.f_lo,
        f_hi=cfg.f_hi,
        method="peak",
    )

    print(f"  - Computed omega shape: {omega.shape}")
    print(
        f"  - Omega range: [{omega.min().item() / (2 * 3.14159):.4f}, "
        f"{omega.max().item() / (2 * 3.14159):.4f}] Hz"
    )

    return dataset, omega


def create_loaders(dataset: NeuroscienceDataset, cfg: HopfConfig, device: str):
    """Create windowed train/val/test loaders for backprop."""
    window_size = min(cfg.window_size, dataset.n_timepoints // 4)
    print(f"Window size: {window_size}")

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        stride=cfg.stride or window_size // 2,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        device=device,
    )

    print("Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, window_size


def evaluate_hopf_model(
    hopf_model: CoupledHopfModel,
    dataset: NeuroscienceDataset,
    cfg: HopfConfig,
    n_paths: int = 10,
) -> dict[str, float]:
    """Evaluate FC/FCD/metastability for a trained Hopf model."""
    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)

    with torch.no_grad():
        hopf_ts = hopf_model.forward(n_steps=n_timepoints, dt=cfg.tr, batch_size=n_paths)
        hopf_fc = hopf_model.compute_fc(hopf_ts)
        hopf_fc_mean = hopf_fc.mean(dim=0)

    metrics = compute_all_fc_metrics(hopf_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
    target_ts = dataset.timeseries[: hopf_ts.shape[0]]
    dyn_metrics = compute_dynamics_fit_metrics(
        hopf_ts,
        target_ts,
        tr=cfg.tr,
        f_lo=cfg.f_lo,
        f_hi=cfg.f_hi,
        fcd_win_sec=cfg.fcd_win_sec,
        fcd_step_sec=cfg.fcd_step_sec,
        compute_fcd=cfg.compute_fcd_metrics,
        compute_metastability=cfg.compute_metastability_metrics,
    )
    metrics.update(dyn_metrics)
    return metrics


def train_hopf_grid_search(
    dataset: NeuroscienceDataset,
    omega: torch.Tensor,
    cfg: HopfConfig,
    device: str,
):
    """Train Hopf model using grid search."""
    print(f"\n{'=' * 60}")
    print("STEP 2A: Training Coupled Hopf Model (Grid Search)")
    print('=' * 60)

    run = init_wandb(cfg, tags=["hopf", "grid_search"], run_suffix="grid")

    target_fc = dataset.fc_mean
    n_rois = dataset.n_rois
    n_timepoints = min(dataset.n_timepoints, 200)

    print(f"Grid search over {len(cfg.g_values) * len(cfg.a_values)} parameter combinations")
    print(f"  - G values: {cfg.g_values}")
    print(f"  - a values: {cfg.a_values}")

    best_params, hopf_model = grid_search_hopf(
        target_fc=target_fc,
        n_rois=n_rois,
        omega=omega,
        g_values=cfg.g_values,
        a_values=cfg.a_values,
        n_timepoints=n_timepoints,
        dt=cfg.tr,
        batch_size=cfg.n_simulations,
        device=device,
    )

    metrics = evaluate_hopf_model(hopf_model, dataset, cfg)
    print(f"Grid-search Hopf metrics: {metrics}")

    if cfg.use_wandb and wandb.run is not None:
        wandb.log(
            {
                "best_params/G": best_params.get("initial_g", 0),
                "best_params/a": best_params.get("initial_a", 0),
                **{f"metrics/{k}": v for k, v in metrics.items()},
            }
        )
        wandb.summary.update(metrics)

    return hopf_model, metrics, best_params, run


def train_hopf_backprop(
    dataset: NeuroscienceDataset,
    omega: torch.Tensor,
    cfg: HopfConfig,
    device: str,
    initial_a: float,
    initial_g: float,
):
    """Train Hopf model using backpropagation."""
    print(f"\n{'=' * 60}")
    print("STEP 2B: Training Coupled Hopf Model (Backpropagation)")
    print('=' * 60)

    train_loader, val_loader, test_loader, window_size = create_loaders(dataset, cfg, device)

    hopf_model = CoupledHopfModel(
        n_rois=dataset.n_rois,
        omega=omega,
        initial_a=initial_a,
        initial_g=initial_g,
        noise_sigma=cfg.noise_sigma,
        device=device,
        learnable_a=True,
        learnable_g=True,
        learnable_omega=False,
    )

    print("\nHopf model created for backprop:")
    print(f"  - Learnable parameters: {sum(p.numel() for p in hopf_model.parameters() if p.requires_grad)}")

    trainer = Trainer(
        model=hopf_model,
        lr=cfg.lr,
        loss_fn=cfg.loss_fn,
        device=device,
        experiment_name=f"{cfg.experiment_name}_backprop",
        cfg=cfg,
        use_wandb=cfg.use_wandb,
    )

    print(f"\nTraining for {cfg.n_epochs} epochs...")
    metrics_store = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.n_epochs,
        n_steps=window_size,
        dt=cfg.dt,
        early_stopping_patience=cfg.early_stopping_patience,
        verbose=True,
    )

    test_metrics = trainer.test(test_loader, n_steps=window_size, dt=cfg.dt)
    print(f"\nBackprop test metrics: {test_metrics}")

    return hopf_model, trainer, metrics_store, test_metrics


def save_model_and_figures(
    hopf_model: CoupledHopfModel,
    dataset: NeuroscienceDataset,
    cfg: HopfConfig,
    mode_label: str,
):
    """Save model and generate FC/timeseries/realization figures."""
    print(f"\n{'=' * 60}")
    print(f"STEP 3 ({mode_label.upper()}): Saving Model and Generating Figures")
    print('=' * 60)

    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"hopf_{mode_label}_best_{cfg.run_name}.pt"
    hopf_model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")

    if cfg.use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(f"hopf_model_{mode_label}_{cfg.run_name}", type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)

    n_paths = 6
    with torch.no_grad():
        hopf_ts = hopf_model.forward(n_steps=n_timepoints, dt=cfg.tr, batch_size=n_paths)
        hopf_fc = hopf_model.compute_fc(hopf_ts)
        hopf_fc_mean = hopf_fc.mean(dim=0)

    fig = plot_fc_comparison(
        hopf_fc_mean,
        target_fc,
        title=f"Coupled Hopf ({mode_label}) - FC Comparison",
        default_name=f"hopf_{mode_label}_fc_comparison",
        use_pdf=True,
    )
    if cfg.use_wandb and wandb.run is not None:
        wandb.log({f"figures/{mode_label}/fc_comparison": wandb.Image(fig)})
    plt.close()

    fig = plot_timeseries(
        hopf_ts,
        n_rois=5,
        title=f"Coupled Hopf ({mode_label}) - Simulated Timeseries",
        default_name=f"hopf_{mode_label}_timeseries",
        use_pdf=True,
    )
    if cfg.use_wandb and wandb.run is not None:
        wandb.log({f"figures/{mode_label}/timeseries": wandb.Image(fig)})
    plt.close()

    fig = plot_realizations(
        hopf_ts,
        roi_index=0,
        n_realizations=min(6, hopf_ts.shape[0]),
        title=f"Coupled Hopf ({mode_label}) - Sample Realizations",
        default_name=f"hopf_{mode_label}_realizations",
        use_pdf=True,
    )
    if cfg.use_wandb and wandb.run is not None:
        wandb.log({f"figures/{mode_label}/realizations": wandb.Image(fig)})
    plt.close()

    print(f"Figures saved to {FIGURES_DIR}")
    return checkpoint_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Coupled Hopf Model")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control", help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default="hopf", help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--training-mode", type=str, default="grid", choices=["grid", "backprop", "both"], help="Hopf training strategy")

    # Data / dynamics settings
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit number of subjects (first N)")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds")
    parser.add_argument("--f-lo", type=float, default=0.04, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--f-hi", type=float, default=0.07, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--fcd-win-sec", type=float, default=60.0, help="FCD window length in seconds")
    parser.add_argument("--fcd-step-sec", type=float, default=2.0, help="FCD window step in seconds")
    parser.add_argument("--no-fcd", action="store_true", help="Disable FCD metrics")
    parser.add_argument("--no-metastability", action="store_true", help="Disable metastability metrics")

    # Grid-search settings
    parser.add_argument("--g-values", type=float, nargs="*", default=None, help="Grid values for G")
    parser.add_argument("--a-values", type=float, nargs="*", default=None, help="Grid values for a")

    # Backprop settings
    parser.add_argument("--n-epochs", type=int, default=500, help="Number of training epochs (backprop)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (backprop)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (backprop)")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for training")
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="combined",
        choices=["mse", "correlation", "combined", "fc_fcd_meta"],
        help="Training objective for backprop",
    )
    parser.add_argument("--loss-weight-fc", type=float, default=1.0, help="Weight for FC loss term")
    parser.add_argument("--loss-weight-fcd", type=float, default=1.0, help="Weight for FCD loss term")
    parser.add_argument(
        "--loss-weight-metastability",
        type=float,
        default=1.0,
        help="Weight for metastability loss term",
    )
    parser.add_argument("--initial-a", type=float, default=-0.02, help="Initial bifurcation parameter for backprop")
    parser.add_argument("--initial-g", type=float, default=0.5, help="Initial coupling for backprop")
    parser.add_argument("--noise-sigma", type=float, default=0.1, help="Hopf noise scale")

    args = parser.parse_args()

    print("=" * 60)
    print("COUPLED HOPF MODEL TRAINING")
    print("=" * 60)

    cfg = HopfConfig(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=args.device,
        seed=args.seed,
        dt=0.1,
        n_epochs=args.n_epochs,
        lr=args.lr,
        loss_fn=args.loss_fn,
        loss_weight_fc=args.loss_weight_fc,
        loss_weight_fcd=args.loss_weight_fcd,
        loss_weight_metastability=args.loss_weight_metastability,
        batch_size=args.batch_size,
        window_size=args.window_size,
        max_subjects=args.max_subjects,
        tr=args.tr,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        fcd_win_sec=args.fcd_win_sec,
        fcd_step_sec=args.fcd_step_sec,
        compute_fcd_metrics=not args.no_fcd,
        compute_metastability_metrics=not args.no_metastability,
        noise_sigma=args.noise_sigma,
    )

    if args.g_values:
        cfg.g_values = args.g_values
    if args.a_values:
        cfg.a_values = args.a_values

    device = setup_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset, omega = load_data(cfg, device)

    results = {}

    if args.training_mode in ("grid", "both"):
        hopf_grid_model, grid_metrics, grid_params, grid_run = train_hopf_grid_search(dataset, omega, cfg, device)
        grid_checkpoint = save_model_and_figures(hopf_grid_model, dataset, cfg, mode_label="grid")
        if grid_run is not None:
            wandb.finish()
        results["grid"] = {
            "model": hopf_grid_model,
            "metrics": grid_metrics,
            "params": grid_params,
            "checkpoint": grid_checkpoint,
        }

    if args.training_mode in ("backprop", "both"):
        hopf_bp_model, trainer, metrics_store, test_metrics = train_hopf_backprop(
            dataset,
            omega,
            cfg,
            device,
            initial_a=args.initial_a,
            initial_g=args.initial_g,
        )

        bp_metrics = evaluate_hopf_model(hopf_bp_model, dataset, cfg)
        bp_checkpoint = save_model_and_figures(hopf_bp_model, dataset, cfg, mode_label="backprop")

        trainer.finish()

        results["backprop"] = {
            "model": hopf_bp_model,
            "metrics": bp_metrics,
            "test_metrics": test_metrics,
            "metrics_store": metrics_store,
            "checkpoint": bp_checkpoint,
        }

    print("\n" + "=" * 60)
    print("HOPF TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    if "grid" in results:
        print(f"\n[Grid] best params: {results['grid']['params']}")
        print(f"[Grid] final metrics: {results['grid']['metrics']}")
        print(f"[Grid] checkpoint: {results['grid']['checkpoint']}")

    if "backprop" in results:
        print(f"\n[Backprop] test metrics: {results['backprop']['test_metrics']}")
        print(f"[Backprop] final metrics: {results['backprop']['metrics']}")
        print(f"[Backprop] checkpoint: {results['backprop']['checkpoint']}")

    print(f"Figures saved to: {FIGURES_DIR}")
    return results


if __name__ == "__main__":
    main()
