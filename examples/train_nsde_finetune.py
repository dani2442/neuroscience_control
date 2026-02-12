#!/usr/bin/env python3
"""
Fine-tune a pretrained Neural SDE checkpoint.
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

from src.dataset import NeuroscienceDataset
from src.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
from src.models import NeuralSDE, load_model_from_checkpoint
from src.training import FineTuner, NeuralSDEConfig, Trainer, create_windowed_loaders
from src.utils import (
    FIGURES_DIR,
    ensure_proxy_env,
    finish_wandb_run,
    init_wandb_run,
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


def load_data(cfg: NeuralSDEConfig, device: str) -> NeuroscienceDataset:
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


def create_loaders(dataset: NeuroscienceDataset, cfg: NeuralSDEConfig, device: str):
    """Create windowed train/val/test loaders."""
    train_loader, val_loader, test_loader, window_size = create_windowed_loaders(dataset, cfg, device)

    print(f"Window size: {window_size}")
    print("Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, window_size


def load_nsde_checkpoint(checkpoint_path: Path, device: str) -> NeuralSDE:
    """Restore a pretrained Neural SDE from checkpoint."""
    model, model_name, _ = load_model_from_checkpoint(checkpoint_path, device=device)
    if model_name != "NeuralSDE" or not isinstance(model, NeuralSDE):
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains {model_name}, expected NeuralSDE"
        )
    return model


def fine_tune_model(
    model: NeuralSDE,
    train_loader,
    val_loader,
    window_size: int,
    cfg: NeuralSDEConfig,
    device: str,
):
    """Run fine-tuning from pretrained checkpoint."""
    print_section("STEP 2: Fine-tuning Neural SDE Model")

    fine_tuner = FineTuner(model=model, device=device)
    print(f"Fine-tuning for {cfg.fine_tune_epochs} epochs at lr={cfg.fine_tune_lr}...")

    metrics_store = fine_tuner.fine_tune(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.fine_tune_epochs,
        lr=cfg.fine_tune_lr,
        warmup_epochs=cfg.warmup_epochs,
        n_steps=window_size,
        dt=cfg.tr,
        experiment_name=f"{cfg.experiment_name}_finetuned",
    )
    return model, metrics_store


def evaluate_on_test(
    model: NeuralSDE,
    test_loader,
    window_size: int,
    cfg: NeuralSDEConfig,
    device: str,
) -> dict[str, float]:
    """Compute loader-based test metrics after fine-tuning."""
    evaluator = Trainer(
        model=model,
        lr=cfg.fine_tune_lr,
        loss_fn=cfg.loss_fn,
        device=device,
        experiment_name=f"{cfg.experiment_name}_finetune_eval",
        cfg=cfg,
        use_wandb=False,
    )
    metrics = evaluator.test(test_loader=test_loader, n_steps=window_size, dt=cfg.tr)
    evaluator.finish()
    return metrics


def save_model_and_figures(
    model: NeuralSDE,
    metrics_store,
    dataset: NeuroscienceDataset,
    cfg: NeuralSDEConfig,
):
    """Save finetuned model checkpoint and figures."""
    print_section("STEP 3: Saving Model and Generating Figures")

    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"nsde_finetuned_{cfg.run_name}.pt"
    model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")

    wandb_log_artifact(
        f"nsde_finetuned_model_{cfg.run_name}",
        checkpoint_path,
        use_wandb=cfg.use_wandb,
    )

    n_paths = min(5, dataset.n_subjects)
    initial_states = dataset.timeseries[:n_paths, :, 0]
    with torch.no_grad():
        sde_ts = model.forward(initial_state=initial_states, n_steps=n_timepoints, dt=cfg.tr)
        sde_fc = model.compute_fc(sde_ts)
        sde_fc_mean = sde_fc.mean(dim=0)

    final_metrics = compute_all_fc_metrics(sde_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
    target_ts = dataset.timeseries[: sde_ts.shape[0], :, :n_timepoints]
    dyn_metrics = compute_dynamics_fit_metrics(
        sde_ts,
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

    fig = plot_fc_comparison(
        sde_fc_mean,
        target_fc,
        title="Neural SDE (Finetuned) - FC Comparison",
        default_name="nsde_finetuned_fc_comparison",
        use_pdf=True,
    )
    wandb_log_figure("figures/fc_comparison", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    fig = plot_timeseries(
        sde_ts.real,
        n_rois=5,
        title="Neural SDE (Finetuned) - Simulated Timeseries",
        default_name="nsde_finetuned_timeseries",
        use_pdf=True,
    )
    wandb_log_figure("figures/timeseries", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    fig = plot_realizations(
        sde_ts.real,
        roi_index=0,
        n_realizations=min(6, sde_ts.shape[0]),
        title="Neural SDE (Finetuned) - Sample Realizations",
        default_name="nsde_finetuned_realizations",
        use_pdf=True,
    )
    wandb_log_figure("figures/realizations", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    fig = plot_training_curves(
        metrics_store,
        default_name="nsde_finetuned_training_curves",
        use_pdf=True,
    )
    wandb_log_figure("figures/training_curves", fig, use_wandb=cfg.use_wandb)
    plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}")
    return checkpoint_path, final_metrics


def main(argv=None):
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained Neural SDE checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained NSDE checkpoint (*.pt)")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control", help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default="neural_sde_finetune", help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--window-size", type=int, default=100, help="Window size")
    parser.add_argument("--max-subjects", type=int, default=50, help="Limit number of subjects (first N)")

    parser.add_argument("--loss-fn", type=str, default="combined", help="Loss objective for evaluation")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time / simulation dt (seconds)")
    parser.add_argument("--f-lo", type=float, default=0.04, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--f-hi", type=float, default=0.07, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--fcd-win-sec", type=float, default=60.0, help="FCD window length in seconds")
    parser.add_argument("--fcd-step-sec", type=float, default=2.0, help="FCD window step in seconds")
    parser.add_argument("--no-fcd", action="store_true", help="Disable FCD metrics")
    parser.add_argument("--no-metastability", action="store_true", help="Disable metastability metrics")

    parser.add_argument("--fine-tune-epochs", type=int, default=20, help="Number of fine-tuning epochs")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs for fine-tuning scheduler")

    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print_section("NEURAL SDE MODEL FINE-TUNING")
    ensure_proxy_env()

    cfg = NeuralSDEConfig(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=args.device,
        seed=args.seed,
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
        loss_fn=args.loss_fn,
        fine_tune=True,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        warmup_epochs=args.warmup_epochs,
    )

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    init_wandb_run(
        use_wandb=cfg.use_wandb,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        run_name=cfg.run_name,
        config=dataclasses.asdict(cfg),
        tags=["nsde", "fine_tune"],
    )

    try:
        dataset = load_data(cfg, device)
        train_loader, val_loader, test_loader, window_size = create_loaders(dataset, cfg, device)

        model = load_nsde_checkpoint(args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")

        model, metrics_store = fine_tune_model(
            model,
            train_loader,
            val_loader,
            window_size,
            cfg,
            device,
        )

        test_metrics = evaluate_on_test(model, test_loader, window_size, cfg, device)
        checkpoint_path, final_metrics = save_model_and_figures(model, metrics_store, dataset, cfg)
    finally:
        finish_wandb_run()

    print("\n" + "=" * 60)
    print("NEURAL SDE FINE-TUNING COMPLETED SUCCESSFULLY")
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
