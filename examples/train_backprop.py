#!/usr/bin/env python3
"""
Unified backpropagation training script for Neural SDE and Coupled Hopf.

All hyper-parameters live as defaults in the config dataclasses
(see src/training/config.py).  Only model choice, device, and a few
runtime flags are exposed as CLI args.

    python examples/train_backprop.py --model hopf
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

# Ensure imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training import (
    HopfConfig,
    HybridHopfConfig,
    NeuralSDEConfig,
    create_windowed_loaders,
    load_dataset,
    run_backprop_training,
)
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
    wandb_summary_update,
)

# Map model name → config class
_CONFIG_CLS = {
    "nsde": NeuralSDEConfig,
    "hopf": HopfConfig,
    "hybrid_hopf": HybridHopfConfig,
}

_MODEL_TITLES = {
    "nsde": "Neural SDE",
    "hopf": "Coupled Hopf",
    "hybrid_hopf": "Hybrid Hopf",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train NSDE or Hopf with backpropagation")
    parser.add_argument("--model", type=str, default="hopf", choices=list(_CONFIG_CLS), help="Model to train")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    parser.add_argument("--skip-figures", action="store_true", help="Skip final figure generation")
    args = parser.parse_args(argv)

    # ── Build config ────────────────────────────────────────────────────
    cfg = _CONFIG_CLS[args.model]()
    cfg.experiment_name = f"{args.model}_backprop"
    if args.no_wandb:
        cfg.use_wandb = False
    if args.device is not None:
        cfg.device = args.device

    print_section(f"{args.model.upper()} BACKPROP TRAINING")
    ensure_proxy_env()
    print(f"Config: {dataclasses.asdict(cfg)}")

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    # ── Data ────────────────────────────────────────────────────────────
    print_section("STEP 1: Loading and Processing Data")
    dataset = load_dataset(cfg, device)
    train_loader, val_loader, test_loader, window_size = create_windowed_loaders(dataset, cfg, device)
    print(f"  window_size={window_size}  train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")

    # ── Model ───────────────────────────────────────────────────────────
    print_section("STEP 2: Training Model (Backpropagation)")
    model = build_model(args.model, dataset, cfg, device, structural_connectivity=dataset.fc_mean)

    ref_amplitude = compute_ref_amplitude(dataset.timeseries)
    ref_omega = compute_ref_omega(dataset.timeseries, tr=cfg.tr, f_lo=cfg.f_lo, f_hi=cfg.f_hi)
    print(f"  ref_amplitude=[{ref_amplitude.min():.4f}, {ref_amplitude.max():.4f}]")
    print(f"  ref_omega=[{ref_omega.min() / 6.2832:.4f}, {ref_omega.max() / 6.2832:.4f}] Hz")

    # ── Training ────────────────────────────────────────────────────────
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
            extra_dyn_kwargs={"ref_amplitude": ref_amplitude, "ref_omega": ref_omega},
        )

        # ── Save & figures ──────────────────────────────────────────────
        print_section("STEP 3: Saving Model and Generating Figures")
        val_timeseries, _, target_fc, n_timepoints = extract_val_data(val_loader)

        checkpoint_path = save_checkpoint(
            model,
            checkpoint_name=f"{args.model}_backprop_best_{cfg.run_name}.pt",
            artifact_name=f"{args.model}_backprop_model_{cfg.run_name}",
            checkpoint_dir=cfg.checkpoint_dir,
            use_wandb=cfg.use_wandb,
        )

        val_eval = trainer.validate(val_loader=val_loader, n_steps=window_size, dt=cfg.tr, verbose=False)
        final_metrics = {
            k: v for k, v in val_eval.items()
            if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.dim() == 0)
        }
        print(f"Final metrics: {final_metrics}")
        wandb_summary_update(
            {f"final_{k}": to_float_metric(v) for k, v in final_metrics.items() if to_float_metric(v) is not None},
            use_wandb=cfg.use_wandb,
        )

        if not args.skip_figures:
            title = _MODEL_TITLES.get(args.model, args.model)
            generate_fc_figure(
                model, val_timeseries, target_fc, n_timepoints, cfg.tr,
                sde_type=cfg.sde_type, method=cfg.sde_method, dt_min=cfg.dt_min,
                title=f"{title} (Backprop) - FC Comparison",
                default_name=f"{args.model}_backprop_fc_comparison",
                use_wandb=cfg.use_wandb,
            )
            generate_multigrid_figure(
                model, val_timeseries, n_timepoints, cfg.tr,
                n_simulations=cfg.n_simulations,
                sde_type=cfg.sde_type, method=cfg.sde_method, dt_min=cfg.dt_min,
                n_rois=12, n_cols=4,
                title=f"{title} (Backprop) - Real vs Simulated",
                default_name=f"{args.model}_backprop_real_vs_sim_multigrid",
                use_wandb=cfg.use_wandb,
            )
        else:
            print("Skipping figure generation (--skip-figures).")

        if args.model in ("hopf", "hybrid_hopf"):
            log_hopf_best_params(model, use_wandb=cfg.use_wandb)
    finally:
        if trainer is not None:
            trainer.finish()

    # ── Done ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{args.model.upper()} BACKPROP TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nTest metrics: {test_metrics}")
    print(f"Final metrics: {final_metrics}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return {"model": model, "metrics": final_metrics, "test_metrics": test_metrics, "checkpoint": checkpoint_path}


if __name__ == "__main__":
    main()
