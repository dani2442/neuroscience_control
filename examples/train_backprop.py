#!/usr/bin/env python3
"""
Unified backpropagation training script for Neural SDE and Coupled Hopf.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure imports work when running this file directly (absolute or relative path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NeuroscienceDataset, compute_omega_from_timeseries
from src.models import CoupledHopfModel, NeuralSDE
from src.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
from src.training import (
    HopfConfig,
    LOSS_REGISTRY,
    NeuralSDEConfig,
    create_windowed_loaders,
    run_backprop_training,
)
from src.utils import (
    FIGURES_DIR,
    ensure_proxy_env,
    plot_fc_comparison,
    plot_realizations,
    plot_timeseries,
    plot_training_curves,
    print_section,
    resolve_device,
    seed_all,
    wandb_log_artifact,
    wandb_log_figure,
    wandb_summary_update,
)


def load_data(cfg: NeuralSDEConfig | HopfConfig, device: str) -> NeuroscienceDataset:
    """Load and summarize dataset."""
    print_section("STEP 1: Loading and Processing Data")

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
    return dataset


def create_loaders(dataset: NeuroscienceDataset, cfg: NeuralSDEConfig | HopfConfig, device: str):
    """Create windowed train/val/test loaders."""
    train_loader, val_loader, test_loader, window_size = create_windowed_loaders(dataset, cfg, device)

    print(f"Window size: {window_size}")
    print("Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, window_size


def build_model(
    model_name: str,
    dataset: NeuroscienceDataset,
    cfg: NeuralSDEConfig | HopfConfig,
    device: str,
    *,
    hidden_dim: int,
    n_layers: int,
    initial_a: float,
    initial_g: float,
):
    """Create model instance for selected mode."""
    if model_name == "nsde":
        model = NeuralSDE(
            n_rois=dataset.n_rois,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            device=device,
        )
        print("\nNeural SDE model created:")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")
        return model

    if model_name == "hopf":
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

        model = CoupledHopfModel(
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

        print("\nCoupled Hopf model created:")
        print(
            f"  - Learnable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def save_model_and_figures(
    model: NeuralSDE | CoupledHopfModel,
    metrics_store,
    dataset: NeuroscienceDataset,
    cfg: NeuralSDEConfig | HopfConfig,
    *,
    model_name: str,
):
    """Save checkpoint, compute final metrics, and produce figures."""
    print_section("STEP 3: Saving Model and Generating Figures")

    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_backprop_best_{cfg.run_name}.pt"
    model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")

    wandb_log_artifact(
        f"{model_name}_backprop_model_{cfg.run_name}",
        checkpoint_path,
        use_wandb=cfg.use_wandb,
    )

    n_paths = 6
    with torch.no_grad():
        simulated_ts = model.forward(n_steps=n_timepoints, dt=cfg.tr, batch_size=n_paths)
        simulated_fc = model.compute_fc(simulated_ts)
        simulated_fc_mean = simulated_fc.mean(dim=0)

    final_metrics = compute_all_fc_metrics(simulated_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
    target_ts = dataset.timeseries[: simulated_ts.shape[0]]
    dyn_metrics = compute_dynamics_fit_metrics(
        simulated_ts,
        target_ts,
        tr=cfg.tr,
        f_lo=cfg.f_lo,
        f_hi=cfg.f_hi,
        fcd_win_sec=cfg.fcd_win_sec,
        fcd_step_sec=cfg.fcd_step_sec,
        compute_fcd=cfg.compute_fcd_metrics,
        compute_metastability=cfg.compute_metastability_metrics,
    )
    final_metrics.update(dyn_metrics)
    print(f"Final metrics: {final_metrics}")

    wandb_summary_update(
        {f"final_{k}": v for k, v in final_metrics.items()},
        use_wandb=cfg.use_wandb,
    )

    model_title = "Neural SDE" if model_name == "nsde" else "Coupled Hopf"

    fig = plot_fc_comparison(
        simulated_fc_mean,
        target_fc,
        title=f"{model_title} (Backprop) - FC Comparison",
        default_name=f"{model_name}_backprop_fc_comparison",
        use_pdf=True,
    )
    wandb_log_figure("figures/fc_comparison", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    fig = plot_timeseries(
        simulated_ts,
        n_rois=5,
        title=f"{model_title} (Backprop) - Simulated Timeseries",
        default_name=f"{model_name}_backprop_timeseries",
        use_pdf=True,
    )
    wandb_log_figure("figures/timeseries", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    fig = plot_realizations(
        simulated_ts,
        roi_index=0,
        n_realizations=min(6, simulated_ts.shape[0]),
        title=f"{model_title} (Backprop) - Sample Realizations",
        default_name=f"{model_name}_backprop_realizations",
        use_pdf=True,
    )
    wandb_log_figure("figures/realizations", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    fig = plot_training_curves(
        metrics_store,
        default_name=f"{model_name}_backprop_training_curves",
        use_pdf=True,
    )
    wandb_log_figure("figures/training_curves", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}")
    return checkpoint_path, final_metrics


def build_config(args: argparse.Namespace):
    """Build typed training config for selected model."""
    common_kwargs = dict(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=args.device,
        seed=args.seed,
        dt=args.dt,
        n_epochs=args.n_epochs,
        lr=args.lr,
        loss_fn=args.loss_fn,
        loss_weight_fc=args.loss_weight_fc,
        loss_weight_fc_mse=args.loss_weight_fc_mse,
        loss_weight_l2=args.loss_weight_l2,
        loss_weight_hilbert_amp=args.loss_weight_hilbert_amp,
        loss_weight_hilbert_omega=args.loss_weight_hilbert_omega,
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
    )

    if args.model == "nsde":
        return NeuralSDEConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            **common_kwargs,
        )

    if args.model == "hopf":
        return HopfConfig(
            noise_sigma=args.noise_sigma,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def main(argv=None):
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train NSDE or Hopf with backpropagation")
    parser.add_argument("--model", type=str, default="hopf", choices=["nsde", "hopf"], help="Model to train")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control", help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Backprop settings
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--window-size", type=int, default=200, help="Window size for training")
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="combined",
        choices=sorted(LOSS_REGISTRY.keys()),
        help="Training objective",
    )

    parser.add_argument("--loss-weight-fc", type=float, default=1.0, help="Weight for FC correlation loss")
    parser.add_argument("--loss-weight-fc-mse", type=float, default=0.0, help="Weight for FC MSE loss")
    parser.add_argument("--loss-weight-l2", type=float, default=1.0, help="Weight for L2 timeseries loss")
    parser.add_argument("--loss-weight-hilbert-amp", type=float, default=1.0, help="Weight for Hilbert amplitude loss")
    parser.add_argument("--loss-weight-hilbert-omega", type=float, default=1.0, help="Weight for Hilbert omega loss")
    parser.add_argument("--loss-weight-fcd", type=float, default=1.0, help="Weight for FCD loss term")
    parser.add_argument(
        "--loss-weight-metastability",
        type=float,
        default=1.0,
        help="Weight for metastability loss term",
    )

    # Dynamics settings
    parser.add_argument("--max-subjects", type=int, default=50, help="Limit number of subjects (first N)")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds")
    parser.add_argument("--dt", type=float, default=0.1, help="Model integration dt during training")
    parser.add_argument("--f-lo", type=float, default=0.04, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--f-hi", type=float, default=0.07, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--fcd-win-sec", type=float, default=60.0, help="FCD window length in seconds")
    parser.add_argument("--fcd-step-sec", type=float, default=2.0, help="FCD window step in seconds")
    parser.add_argument("--no-fcd", action="store_true", help="Disable FCD metrics")
    parser.add_argument("--no-metastability", action="store_true", help="Disable metastability metrics")

    # NSDE settings
    parser.add_argument("--hidden-dim", type=int, default=256, help="NSDE hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="NSDE drift network layers")

    # Hopf settings
    parser.add_argument("--initial-a", type=float, default=-0.02, help="Initial Hopf bifurcation parameter")
    parser.add_argument("--initial-g", type=float, default=0.5, help="Initial Hopf coupling")
    parser.add_argument("--noise-sigma", type=float, default=0.1, help="Hopf noise scale")

    args = parser.parse_args(argv)
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_backprop"

    print_section(f"{args.model.upper()} BACKPROP TRAINING")
    ensure_proxy_env()

    cfg = build_config(args)
    print(f"Config: {dataclasses.asdict(cfg)}")

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    dataset = load_data(cfg, device)
    train_loader, val_loader, test_loader, window_size = create_loaders(dataset, cfg, device)

    print_section("STEP 2: Training Model (Backpropagation)")
    model = build_model(
        args.model,
        dataset,
        cfg,
        device,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        initial_a=args.initial_a,
        initial_g=args.initial_g,
    )

    trainer = None
    try:
        trainer, metrics_store, test_metrics = run_backprop_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            window_size=window_size,
            cfg=cfg,
            device=device,
            experiment_name=cfg.experiment_name,
        )

        checkpoint_path, final_metrics = save_model_and_figures(
            model=model,
            metrics_store=metrics_store,
            dataset=dataset,
            cfg=cfg,
            model_name=args.model,
        )
    finally:
        if trainer is not None:
            trainer.finish()

    print("\n" + "=" * 60)
    print(f"{args.model.upper()} BACKPROP TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nTest metrics: {test_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return {
        "model": model,
        "metrics": final_metrics,
        "test_metrics": test_metrics,
        "checkpoint": checkpoint_path,
    }


if __name__ == "__main__":
    main()
