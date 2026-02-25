#!/usr/bin/env python3
"""
Grid-search training script for Coupled Hopf Model.

All hyper-parameters live as defaults in the config dataclasses
(see src/training/config.py).

    python examples/train_hopf.py
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
import wandb
from torch.utils.data import DataLoader

# Ensure imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NeuroscienceDataset, RandomWindowDataset, compute_omega_from_timeseries
from src.models import CoupledHopfModel
from src.training import HopfConfig, grid_search_hopf, load_dataset
from src.utils import (
    FIGURES_DIR,
    ensure_proxy_env,
    evaluate_hopf_model,
    evaluate_hopf_loader_metrics,
    extract_val_data,
    finish_wandb_run,
    generate_fc_figure,
    generate_multigrid_figure,
    init_wandb_run,
    log_hopf_best_params,
    prefixed_metrics,
    print_section,
    resolve_device,
    save_checkpoint,
    seed_all,
    split_subject_indices,
    to_float_metric,
    wandb_log,
    wandb_summary_update,
)


def _create_validation_loader(
    dataset: NeuroscienceDataset,
    cfg: HopfConfig,
    device: str,
    val_idx: torch.Tensor,
) -> DataLoader:
    """Build a validation loader from the deterministic validation subject split."""
    window_size = min(cfg.window_size, dataset.n_timepoints // 4)
    n_val_win = max(cfg.batch_size, cfg.n_windows_per_epoch // 4)
    val_idx = torch.as_tensor(val_idx, device=dataset.timeseries.device, dtype=torch.long)
    val_dataset = RandomWindowDataset(
        dataset.timeseries[val_idx],
        dataset.fc_matrices[val_idx],
        window_size=window_size,
        n_windows=n_val_win,
        device=device,
    )
    return DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Coupled Hopf Model via grid search")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    args = parser.parse_args(argv)

    # ── Build config ────────────────────────────────────────────────────
    cfg = HopfConfig(experiment_name="hopf_grid")
    if args.no_wandb:
        cfg.use_wandb = False
    if args.device is not None:
        cfg.device = args.device

    print_section("COUPLED HOPF MODEL TRAINING (GRID SEARCH)")
    ensure_proxy_env()
    print(f"Config: {dataclasses.asdict(cfg)}")

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    # ── Data ────────────────────────────────────────────────────────────
    print_section("STEP 1: Loading and Processing Data")
    dataset = load_dataset(cfg, device)
    omega = compute_omega_from_timeseries(
        dataset.timeseries, dt=dataset.dt, f_lo=cfg.f_lo, f_hi=cfg.f_hi, method="peak",
    )
    print(
        f"  omega: shape={omega.shape}, "
        f"range=[{omega.min().item() / 6.2832:.4f}, {omega.max().item() / 6.2832:.4f}] Hz"
    )
    structural_connectivity = dataset.fc_mean

    train_idx, val_idx = split_subject_indices(cfg, dataset.n_subjects)
    val_loader = _create_validation_loader(dataset, cfg, device, val_idx)

    # ── Grid search ─────────────────────────────────────────────────────
    print_section("STEP 2: Training Coupled Hopf Model (Grid Search)")

    init_wandb_run(
        use_wandb=cfg.use_wandb,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        run_name=f"{cfg.run_name}_grid",
        config=dataclasses.asdict(cfg),
        tags=["hopf", "grid_search"],
    )
    if cfg.use_wandb and wandb.run is not None:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")

    n_timepoints = min(dataset.n_timepoints, 200)
    train_fc = dataset.fc_matrices[train_idx].mean(dim=0)
    n_eval = min(cfg.n_simulations, len(train_idx))
    eval_train_idx = train_idx[:n_eval]
    initial_states = dataset.timeseries[eval_train_idx, :, 0]
    target_ts = dataset.timeseries[eval_train_idx, :, :n_timepoints]

    n_combos = len(cfg.g_values) * len(cfg.a_values) * len(cfg.kappa_values)
    print(f"Grid search over {n_combos} parameter combinations")
    print(f"  Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")

    metric_weights = {}
    if cfg.weight_fc:
        metric_weights["fc_correlation"] = cfg.weight_fc
    if cfg.weight_fcd:
        metric_weights["fcd_mse"] = cfg.weight_fcd
    if cfg.weight_meta:
        metric_weights["metastability_diff"] = cfg.weight_meta
    if cfg.weight_phfcd:
        metric_weights["phfcd_mse"] = cfg.weight_phfcd

    best_params, hopf_model = grid_search_hopf(
        target_fc=train_fc,
        n_rois=dataset.n_rois,
        initial_states=initial_states,
        structural_connectivity=structural_connectivity,
        omega=omega,
        g_values=cfg.g_values,
        a_values=cfg.a_values,
        kappa_values=cfg.kappa_values,
        n_timepoints=n_timepoints,
        dt=cfg.tr,
        device=device,
        target_timeseries=target_ts,
        tr=cfg.tr,
        fcd_win_sec=cfg.fcd_win_sec,
        fcd_step_sec=cfg.fcd_step_sec,
        metric_weights=metric_weights or None,
        noise_sigma=cfg.noise_sigma,
    )

    n_eval_paths = max(10, cfg.n_simulations)
    train_metrics = evaluate_hopf_model(hopf_model, dataset, cfg, n_paths=n_eval_paths, subject_indices=train_idx)
    val_metrics = evaluate_hopf_model(hopf_model, dataset, cfg, n_paths=n_eval_paths, subject_indices=val_idx)
    print(f"Train metrics: {train_metrics}")
    print(f"Validation metrics: {val_metrics}")

    wandb_log(
        {
            "epoch": 0,
            "best_params/G": float(best_params.get("initial_g", 0.0)),
            "best_params/a": float(best_params.get("initial_a", 0.0)),
            "best_params/kappa": float(best_params.get("initial_kappa", best_params.get("kappa", 0.0))),
            **prefixed_metrics("train", train_metrics),
            **prefixed_metrics("validation", val_metrics),
        },
        use_wandb=cfg.use_wandb,
    )

    # ── Save & figures ──────────────────────────────────────────────────
    print_section("STEP 3: Saving Model and Generating Figures")

    checkpoint = save_checkpoint(
        hopf_model,
        checkpoint_name=f"hopf_grid_best_{cfg.run_name}.pt",
        artifact_name=f"hopf_model_grid_{cfg.run_name}",
        checkpoint_dir=cfg.checkpoint_dir,
        use_wandb=cfg.use_wandb,
    )

    val_timeseries, _, target_fc, n_tp = extract_val_data(val_loader)
    generate_fc_figure(
        hopf_model, val_timeseries, target_fc, n_tp, cfg.tr,
        title="Coupled Hopf (Grid) - FC Comparison",
        default_name="hopf_grid_fc_comparison",
        use_wandb=cfg.use_wandb,
    )
    generate_multigrid_figure(
        hopf_model, val_timeseries, n_tp, cfg.tr,
        n_simulations=3, n_rois=3, n_cols=3,
        title="Coupled Hopf (Grid) - Real vs Simulated",
        default_name="hopf_grid_real_vs_sim",
        use_wandb=cfg.use_wandb,
    )

    val_window_size = getattr(getattr(val_loader, "dataset", None), "window_size", None)
    final_metrics_all = evaluate_hopf_loader_metrics(hopf_model, val_loader, cfg, n_steps=val_window_size)
    final_metrics = {k: v for k, v in final_metrics_all.items() if isinstance(v, (int, float))}
    wandb_summary_update({f"final_{k}": v for k, v in final_metrics.items()}, use_wandb=cfg.use_wandb)
    log_hopf_best_params(hopf_model, use_wandb=cfg.use_wandb)
    finish_wandb_run()

    # ── Done ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HOPF GRID SEARCH COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nBest params: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Final metrics (val loader): {final_metrics}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return {
        "model": hopf_model,
        "metrics": final_metrics,
        "train_metrics": train_metrics,
        "params": best_params,
        "checkpoint": checkpoint,
    }


if __name__ == "__main__":
    main()
