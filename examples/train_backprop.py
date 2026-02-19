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
import wandb

# Ensure imports work when running this file directly (absolute or relative path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NeuroscienceDataset, compute_omega_from_timeseries
from src.models import CoupledHopfModel, HybridHopfModel, NeuralSDE
from src.training.trainer import Trainer
from src.training import (
    HopfConfig,
    HybridHopfConfig,
    LOSS_REGISTRY,
    NeuralSDEConfig,
    create_windowed_loaders,
    run_backprop_training,
)
from src.training.losses import _PRESETS as LOSS_PRESETS
from src.training.losses import compute_ref_amplitude, compute_ref_omega
from src.utils import (
    FIGURES_DIR,
    ensure_proxy_env,
    extract_val_data,
    generate_fc_figure,
    generate_multigrid_figure,
    log_hopf_best_params,
    print_section,
    resolve_device,
    save_checkpoint,
    seed_all,
    to_float_metric,
    wandb_log,
    wandb_summary_update,
)


def log_train_validation_metrics(metrics_store, *, use_wandb: bool) -> None:
    """Log every train/validation metric from MetricsStore with consistent notation."""
    if not use_wandb or wandb.run is None:
        return

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("validation/*", step_metric="epoch")
    n_epochs = max(len(metrics_store.train_metrics), len(metrics_store.val_metrics))
    for idx in range(n_epochs):
        train_entry = metrics_store.train_metrics[idx] if idx < len(metrics_store.train_metrics) else {}
        val_entry = metrics_store.val_metrics[idx] if idx < len(metrics_store.val_metrics) else {}
        epoch = int(train_entry.get("epoch", val_entry.get("epoch", idx)))
        log_data = {"epoch": epoch}

        for key, value in train_entry.items():
            if key == "epoch":
                continue
            numeric = to_float_metric(value)
            if numeric is not None:
                log_data[f"train/{key}"] = numeric

        for key, value in val_entry.items():
            if key == "epoch":
                continue
            numeric = to_float_metric(value)
            if numeric is not None:
                log_data[f"validation/{key}"] = numeric

        # Avoid explicit step rewinds when replaying metrics after training.
        wandb.log(log_data)


def load_data(cfg: NeuralSDEConfig | HopfConfig, device: str) -> NeuroscienceDataset:
    """Load and summarize dataset."""
    print_section("STEP 1: Loading and Processing Data")

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
    print(f"  - Number of subjects: {dataset.n_subjects}")
    print(f"  - Number of ROIs: {dataset.n_rois}")
    print(f"  - Number of timepoints: {dataset.n_timepoints}")
    print(f"  - FC matrix shape: {dataset.fc_mean.shape}")
    print(f"  - Timeseries dtype: {dataset.timeseries.dtype}")
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
    initial_kappa: float, 
    structural_connectivity: torch.Tensor | None = None,
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
            structural_connectivity=structural_connectivity,
            omega=omega,
            initial_a=initial_a,
            initial_g=initial_g,
            initial_kappa=initial_kappa,
            noise_sigma=cfg.noise_sigma,
            device=device,
            learnable_a=cfg.learnable_a,
            learnable_g=cfg.learnable_g,
            learnable_kappa=cfg.learnable_kappa,
            learnable_omega=False,
        )

        print("\nCoupled Hopf model created:")
        print(
            f"  - Learnable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        return model

    if model_name == "hybrid_hopf":
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

        model = HybridHopfModel(
            n_rois=dataset.n_rois,
            structural_connectivity=structural_connectivity,
            omega=omega,
            initial_a=initial_a,
            initial_g=initial_g,
            initial_kappa=initial_kappa,
            noise_sigma=cfg.noise_sigma,
            coupling_hidden_dim=cfg.coupling_hidden_dim,
            coupling_n_layers=cfg.coupling_n_layers,
            device=device,
            learnable_a=cfg.learnable_a,
            learnable_g=cfg.learnable_g,
            learnable_kappa=cfg.learnable_kappa,
            learnable_omega=cfg.learnable_omega,
        )

        print("\nHybrid Hopf model created:")
        print(
            f"  - Learnable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        print(f"  - Coupling network: hidden_dim={cfg.coupling_hidden_dim}, n_layers={cfg.coupling_n_layers}")
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def save_model_and_figures(
    model: NeuralSDE | CoupledHopfModel | HybridHopfModel,
    trainer: Trainer,
    metrics_store,
    dataset: NeuroscienceDataset,
    val_loader,
    cfg: NeuralSDEConfig | HopfConfig | HybridHopfConfig,
    *,
    model_name: str,
    window_size: int,
    skip_figures: bool = False,
):
    """Save checkpoint, compute final metrics, and produce figures."""
    print_section("STEP 3: Saving Model and Generating Figures")

    val_timeseries, val_fc_matrices, target_fc, n_timepoints = extract_val_data(val_loader)

    checkpoint_path = save_checkpoint(
        model,
        checkpoint_name=f"{model_name}_backprop_best_{cfg.run_name}.pt",
        artifact_name=f"{model_name}_backprop_model_{cfg.run_name}",
        checkpoint_dir=cfg.checkpoint_dir,
        use_wandb=cfg.use_wandb,
    )

    # Keep "final" summary aligned with loader-based epoch/test metric semantics.
    val_eval_metrics = trainer.validate(
        val_loader=val_loader,
        n_steps=window_size,
        dt=cfg.tr,
        verbose=False,
    )
    # Log ALL final evaluation metrics (not just a fixed subset).
    final_metrics = {
        k: v for k, v in val_eval_metrics.items()
        if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.dim() == 0)
    }
    print(f"Final metrics: {final_metrics}")

    wandb_summary_update(
        {f"final_{k}": to_float_metric(v) for k, v in final_metrics.items() if to_float_metric(v) is not None},
        use_wandb=cfg.use_wandb,
    )

    if not skip_figures:
        model_titles = {"nsde": "Neural SDE", "hopf": "Coupled Hopf", "hybrid_hopf": "Hybrid Hopf"}
        model_title = model_titles.get(model_name, model_name)

        generate_fc_figure(
            model, val_timeseries, target_fc, n_timepoints, cfg.tr,
            sde_type=cfg.sde_type,
            method=cfg.sde_method,
            dt_min=cfg.dt_min,
            use_adjoint=False,
            adjoint_method=cfg.adjoint_method,
            title=f"{model_title} (Backprop) - FC Comparison",
            default_name=f"{model_name}_backprop_fc_comparison",
            use_wandb=cfg.use_wandb,
        )

        generate_multigrid_figure(
            model, val_timeseries, n_timepoints, cfg.tr,
            n_simulations=cfg.n_simulations,
            sde_type=cfg.sde_type,
            method=cfg.sde_method,
            dt_min=cfg.dt_min,
            use_adjoint=False,
            adjoint_method=cfg.adjoint_method,
            n_rois=12,
            n_cols=4,
            title=f"{model_title} (Backprop) - Real vs Simulated",
            default_name=f"{model_name}_backprop_real_vs_sim_multigrid",
            use_wandb=cfg.use_wandb,
        )
    else:
        print("Skipping figure generation (--skip-figures).")

    if model_name in ("hopf", "hybrid_hopf"):
        log_hopf_best_params(model, use_wandb=cfg.use_wandb)

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
        n_epochs=args.n_epochs,
        lr=args.lr,
        loss_fn=args.loss_fn,
        loss_weight_fc=args.loss_weight_fc,
        loss_weight_fc_mse=args.loss_weight_fc_mse,
        loss_weight_l2=args.loss_weight_l2,
        loss_weight_amplitude=args.loss_weight_amplitude,
        loss_weight_omega=args.loss_weight_omega,
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
        fourier_denoise=args.fourier_denoise,
        denoise_f_lo=args.denoise_f_lo,
        denoise_f_hi=args.denoise_f_hi,
        n_windows_per_epoch=args.n_windows_per_epoch,
        n_simulations=args.n_simulations,
        sde_type=args.sde_type,
        sde_method=args.sde_method,
        dt_min=args.dt_min,
        use_adjoint=args.use_adjoint,
        adjoint_method=args.adjoint_method,
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
            learnable_a=args.learnable_a,
            learnable_g=args.learnable_g,
            learnable_kappa=args.learnable_kappa,
            **common_kwargs,
        )

    if args.model == "hybrid_hopf":
        return HybridHopfConfig(
            noise_sigma=args.noise_sigma,
            learnable_a=args.learnable_a,
            learnable_g=args.learnable_g,
            learnable_kappa=args.learnable_kappa,
            learnable_omega=args.learnable_omega,
            coupling_hidden_dim=args.coupling_hidden_dim,
            coupling_n_layers=args.coupling_n_layers,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def main(argv=None):
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train NSDE or Hopf with backpropagation")
    parser.add_argument("--model", type=str, default="nsde", choices=["nsde", "hopf", "hybrid_hopf"], help="Model to train")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control", help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Backprop settings
    parser.add_argument("--n-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for training")
    _VALID_LOSS_NAMES = sorted(set(LOSS_REGISTRY.keys()) | set(LOSS_PRESETS.keys()) | {"custom"})
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="combined",
        choices=_VALID_LOSS_NAMES,
        help="Training objective (preset name or individual term)",
    )

    parser.add_argument(
        "--loss-weight-fc-correlation",
        dest="loss_weight_fc",
        type=float,
        default=1.0,
        help="Weight for `loss_fc_correlation` (overrides preset)",
    )
    parser.add_argument("--loss-weight-fc", dest="loss_weight_fc", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--loss-weight-fc-mse", type=float, default=0., help="Weight for `loss_fc_mse` (overrides preset)")
    parser.add_argument("--loss-weight-l2", type=float, default=None, help="Weight for L2 timeseries loss (overrides preset)")
    parser.add_argument("--loss-weight-amplitude", type=float, default=1., help="Weight for amplitude loss (|z| envelope; overrides preset)")
    parser.add_argument("--loss-weight-omega", type=float, default=1., help="Weight for instantaneous-frequency loss (overrides preset)")
    parser.add_argument("--loss-weight-fcd", type=float, default=None, help="Weight for `loss_fcd` (overrides preset)")
    parser.add_argument(
        "--loss-weight-metastability",
        type=float,
        default=None,
        help="Weight for `loss_metastability` (overrides preset)",
    )

    # Dynamics settings
    parser.add_argument("--max-subjects", type=int, default=50, help="Limit number of subjects (first N)")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time / simulation dt (seconds)")
    parser.add_argument("--sde-type", type=str, default="ito", choices=["ito", "stratonovich"], help="SDE interpretation")
    parser.add_argument("--sde-method", type=str, default="euler", help="SDE solver method")
    parser.add_argument("--dt-min", type=float, default=0.1, help="SDE solver sub-step passed as torchsde `dt`")
    parser.add_argument("--use-adjoint", dest="use_adjoint", action="store_true", help="Use torchsde adjoint solver")
    parser.add_argument("--no-adjoint", dest="use_adjoint", action="store_false", help="Disable torchsde adjoint solver")
    # parser.set_defaults(False)
    parser.add_argument("--adjoint-method", type=str, default="euler", help="Adjoint solver method")
    parser.add_argument("--f-lo", type=float, default=0.008, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--f-hi", type=float, default=0.08, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--fcd-win-sec", type=float, default=60.0, help="Sliding-window length for FCD metrics/losses (seconds)")
    parser.add_argument("--fcd-step-sec", type=float, default=2.0, help="Sliding-window step for FCD metrics/losses (seconds)")
    parser.add_argument("--n-simulations", type=int, default=5, help="Number of stochastic simulations for final multigrid figure")
    parser.add_argument("--skip-figures", action="store_true", help="Skip final figure generation for faster smoke tests")
    parser.add_argument("--no-fcd-ks", dest="no_fcd", action="store_true", help="Disable `fcd_ks` metric computation")
    parser.add_argument("--no-fcd", dest="no_fcd", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-metastability-diff",
        dest="no_metastability",
        action="store_true",
        help="Disable `metastability_diff` metric computation",
    )
    parser.add_argument("--no-metastability", dest="no_metastability", action="store_true", help=argparse.SUPPRESS)

    # Preprocessing
    parser.add_argument("--fourier-denoise", action="store_true", default=True, help="Apply FFT bandpass denoising")
    parser.add_argument("--denoise-f-lo", type=float, default=0.008, help="Denoising low cutoff (Hz)")
    parser.add_argument("--denoise-f-hi", type=float, default=0.08, help="Denoising high cutoff (Hz)")
    parser.add_argument("--n-windows-per-epoch", type=int, default=512, help="Random windows per epoch")

    # NSDE settings
    parser.add_argument("--hidden-dim", type=int, default=256, help="NSDE hidden dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="NSDE drift network layers")

    # Hopf settings
    parser.add_argument("--initial-a", type=float, default=-0.02, help="Initial Hopf bifurcation parameter")
    parser.add_argument("--initial-g", type=float, default=0.05, help="Initial Hopf coupling")
    parser.add_argument("--initial-kappa", type=float, default=0.1, help="Initial kappa Hopf")
    parser.add_argument("--noise-sigma", type=float, default=0.2, help="Hopf noise scale (0.0 = deterministic)")

    # HybridHopf settings
    parser.add_argument("--coupling-hidden-dim", type=int, default=32, help="HybridHopf coupling network hidden dimension")
    parser.add_argument("--coupling-n-layers", type=int, default=2, help="HybridHopf coupling network layers")
    parser.add_argument("--learnable-a", action="store_true", default=True, help="Make bifurcation param a learnable")
    parser.add_argument("--no-learnable-a", dest="learnable_a", action="store_false", help="Freeze bifurcation param a")
    parser.add_argument("--learnable-g", action="store_true", default=True, help="Make global coupling g learnable")
    parser.add_argument("--no-learnable-g", dest="learnable_g", action="store_false", help="Freeze global coupling g")
    parser.add_argument("--learnable-kappa", action="store_true", default=False, help="Make kappa learnable")
    parser.add_argument("--no-learnable-kappa", dest="learnable_kappa", action="store_false", help="Freeze kappa")
    parser.add_argument("--learnable-omega", action="store_true", default=False, help="Make frequencies omega learnable")

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
    structural_connectivity = dataset.fc_mean
    model = build_model(
        args.model,
        dataset,
        cfg,
        device,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        initial_a=args.initial_a,
        initial_g=args.initial_g,
        initial_kappa=args.initial_kappa,
        structural_connectivity=structural_connectivity,
    )

    # Precompute dataset-level amplitude/omega references for losses.
    ref_amplitude = compute_ref_amplitude(dataset.timeseries)
    ref_omega = compute_ref_omega(
        dataset.timeseries, tr=cfg.tr, f_lo=cfg.f_lo, f_hi=cfg.f_hi,
    )
    print(f"  - ref_amplitude range: [{ref_amplitude.min():.4f}, {ref_amplitude.max():.4f}]")
    print(f"  - ref_omega range: [{ref_omega.min() / (2 * 3.14159):.4f}, {ref_omega.max() / (2 * 3.14159):.4f}] Hz")

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
            extra_dyn_kwargs={
                "ref_amplitude": ref_amplitude,
                "ref_omega": ref_omega,
            },
        )

        checkpoint_path, final_metrics = save_model_and_figures(
            model=model,
            trainer=trainer,
            metrics_store=metrics_store,
            dataset=dataset,
            val_loader=val_loader,
            cfg=cfg,
            model_name=args.model,
            window_size=window_size,
            skip_figures=args.skip_figures,
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
