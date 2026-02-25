#!/usr/bin/env python3
"""
Fine-tune a pretrained Neural SDE checkpoint.

All hyper-parameters live as defaults in the config dataclasses
(see src/training/config.py).

    python examples/train_nsde_finetune.py --checkpoint <path>
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
from src.models import NeuralSDE, load_model_from_checkpoint
from src.training import FineTuner, NeuralSDEConfig, Trainer, create_windowed_loaders, load_dataset
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained Neural SDE checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained NSDE checkpoint (*.pt)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # ── Build config ────────────────────────────────────────────────────
    cfg = NeuralSDEConfig(experiment_name="neural_sde_finetune")
    cfg.fine_tune = True
    if args.no_wandb:
        cfg.use_wandb = False
    if args.device is not None:
        cfg.device = args.device

    print_section("NEURAL SDE MODEL FINE-TUNING")
    ensure_proxy_env()

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
        # ── Data ────────────────────────────────────────────────────────
        print_section("STEP 1: Loading and Processing Data")
        dataset = load_dataset(cfg, device)
        train_loader, val_loader, test_loader, window_size = create_windowed_loaders(dataset, cfg, device)
        print(f"  window_size={window_size}  train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")

        # ── Load checkpoint ─────────────────────────────────────────────
        model, model_name, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        if model_name != "NeuralSDE" or not isinstance(model, NeuralSDE):
            raise ValueError(f"Checkpoint contains {model_name}, expected NeuralSDE")
        print(f"Loaded checkpoint: {args.checkpoint}")

        # ── Fine-tune ───────────────────────────────────────────────────
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

        # ── Test evaluation ─────────────────────────────────────────────
        evaluator = Trainer(
            model=model, lr=cfg.fine_tune_lr, loss_fn=cfg.loss_fn,
            device=device, experiment_name=f"{cfg.experiment_name}_finetune_eval",
            cfg=cfg, use_wandb=False,
        )
        test_metrics = evaluator.test(test_loader=test_loader, n_steps=window_size, dt=cfg.tr)
        evaluator.finish()

        # ── Save & figures ──────────────────────────────────────────────
        print_section("STEP 3: Saving Model and Generating Figures")
        target_fc = dataset.fc_mean
        n_timepoints = min(dataset.n_timepoints, 200)

        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"nsde_finetuned_{cfg.run_name}.pt"
        model.save(str(checkpoint_path))
        print(f"Model saved to {checkpoint_path}")
        wandb_log_artifact(f"nsde_finetuned_model_{cfg.run_name}", checkpoint_path, use_wandb=cfg.use_wandb)

        n_paths = min(5, dataset.n_subjects)
        initial_states = dataset.timeseries[:n_paths, :, 0]
        with torch.no_grad():
            sde_ts = model.forward(initial_state=initial_states, n_steps=n_timepoints, dt=cfg.tr)
            sde_fc = model.compute_fc(sde_ts)
            sde_fc_mean = sde_fc.mean(dim=0)

        final_metrics = compute_all_fc_metrics(sde_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
        target_ts = dataset.timeseries[: sde_ts.shape[0], :, :n_timepoints]
        final_metrics.update(compute_dynamics_fit_metrics(
            sde_ts, target_ts, tr=cfg.tr,
            fcd_win_sec=cfg.fcd_win_sec, fcd_step_sec=cfg.fcd_step_sec,
            compute_fcd=cfg.compute_fcd_metrics, compute_metastability=cfg.compute_metastability_metrics,
        ))
        print(f"Final metrics: {final_metrics}")
        wandb_summary_update({f"final_{k}": v for k, v in final_metrics.items()}, use_wandb=cfg.use_wandb)

        for name, fig in [
            ("figures/fc_comparison", plot_fc_comparison(sde_fc_mean, target_fc, title="Neural SDE (Finetuned) - FC", default_name="nsde_finetuned_fc_comparison", use_pdf=True)),
            ("figures/timeseries", plot_timeseries(sde_ts.real, n_rois=5, title="Neural SDE (Finetuned) - Timeseries", default_name="nsde_finetuned_timeseries", use_pdf=True)),
            ("figures/realizations", plot_realizations(sde_ts.real, roi_index=0, n_realizations=min(6, sde_ts.shape[0]), title="Neural SDE (Finetuned) - Realizations", default_name="nsde_finetuned_realizations", use_pdf=True)),
            ("figures/training_curves", plot_training_curves(metrics_store, default_name="nsde_finetuned_training_curves", use_pdf=True)),
        ]:
            wandb_log_figure(name, fig, use_wandb=cfg.use_wandb)
            plt.close(fig)
    finally:
        finish_wandb_run()

    # ── Done ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NEURAL SDE FINE-TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nTest metrics: {test_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return {"model": model, "metrics": final_metrics, "test_metrics": test_metrics, "checkpoint": checkpoint_path}


if __name__ == "__main__":
    main()
