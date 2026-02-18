#!/usr/bin/env python3
"""
Grid-search training script for Coupled Hopf Model.
"""

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import wandb
from torch.utils.data import DataLoader

# Ensure imports work when running this file directly (absolute or relative path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NeuroscienceDataset, RandomWindowDataset, compute_omega_from_timeseries
from src.models import CoupledHopfModel
from src.metrics import compute_all_fc_metrics, compute_dynamics_fit_metrics
from src.training import HopfConfig, grid_search_hopf
from src.utils import (
    FIGURES_DIR,
    ensure_proxy_env,
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
    to_float_metric,
    wandb_log,
    wandb_summary_update,
)


def load_data(cfg: HopfConfig, device: str):
    """Load and prepare dataset."""
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

    return dataset, omega, dataset.fc_mean


def split_subject_indices(cfg: HopfConfig, n_subjects: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce deterministic subject split for train/validation reporting."""
    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(n_subjects, generator=generator)
    n_train = max(1, int(cfg.train_ratio * n_subjects))
    n_val = max(1, int(cfg.val_ratio * n_subjects))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    if val_idx.numel() == 0:
        val_idx = train_idx[:1]
    return train_idx, val_idx


def create_validation_loader(
    dataset: NeuroscienceDataset,
    cfg: HopfConfig,
    device: str,
    val_idx: torch.Tensor,
) -> DataLoader:
    """Build a validation loader using the deterministic validation subject split."""
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


def evaluate_hopf_loader_metrics(
    hopf_model: CoupledHopfModel,
    loader: DataLoader,
    cfg: HopfConfig,
    *,
    n_steps: int | None = None,
) -> dict[str, float]:
    """Evaluate FC/FCD/metastability with loader-based batch aggregation."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    sampled_batches = 0
    metrics_limit = cfg.metrics_sample_batches

    for batch_idx, batch in enumerate(loader):
        windows, fc_targets, _ = batch
        windows = windows.to(hopf_model.device)
        fc_targets = fc_targets.to(hopf_model.device)

        batch_steps = windows.shape[2]
        n_sim_steps = min(batch_steps, n_steps) if n_steps is not None else batch_steps
        target_window = windows[:, :, :n_sim_steps]
        initial_state = target_window[:, :, 0]

        if metrics_limit is None:
            compute_expensive = True
        elif metrics_limit <= 0:
            compute_expensive = False
        else:
            compute_expensive = batch_idx < metrics_limit
        if compute_expensive:
            sampled_batches += 1

        with torch.no_grad():
            simulated = hopf_model.forward(initial_state=initial_state, n_steps=n_sim_steps, dt=cfg.tr)
            fc_pred = hopf_model.compute_fc(simulated)
            batch_metrics = compute_all_fc_metrics(fc_pred, fc_targets)
            if compute_expensive:
                batch_metrics.update(
                    compute_dynamics_fit_metrics(
                        simulated,
                        target_window,
                        tr=cfg.tr,
                        fcd_win_sec=cfg.fcd_win_sec,
                        fcd_step_sec=cfg.fcd_step_sec,
                        compute_fcd=cfg.compute_fcd_metrics,
                        compute_metastability=cfg.compute_metastability_metrics,
                    )
                )

        for key, value in batch_metrics.items():
            numeric = to_float_metric(value)
            if numeric is None or math.isnan(numeric):
                continue
            sums[key] = sums.get(key, 0.0) + numeric
            counts[key] = counts.get(key, 0) + 1

    metric_keys = ("fc_correlation", "fc_mse", "fcd_ks", "metastability_diff")
    metrics = {
        key: (sums[key] / counts[key]) if counts.get(key, 0) > 0 else float("nan")
        for key in metric_keys
    }
    # metrics["metrics_sampled_batches"] = len(loader) if metrics_limit is None else sampled_batches
    return metrics


def evaluate_hopf_model(
    hopf_model: CoupledHopfModel,
    dataset: NeuroscienceDataset,
    cfg: HopfConfig,
    n_paths: int = 10,
    subject_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate FC/FCD/metastability for a trained Hopf model."""
    if subject_indices is None:
        subject_indices = torch.arange(dataset.n_subjects, device=dataset.timeseries.device)
    else:
        subject_indices = torch.as_tensor(subject_indices, device=dataset.timeseries.device, dtype=torch.long)

    if subject_indices.numel() == 0:
        raise ValueError("subject_indices must contain at least one subject.")

    target_fc = dataset.fc_matrices[subject_indices].mean(dim=0)
    n_timepoints = min(dataset.n_timepoints, 200)
    n_paths = min(n_paths, subject_indices.numel())
    eval_indices = subject_indices[:n_paths]
    initial_states = dataset.timeseries[eval_indices, :, 0]

    with torch.no_grad():
        hopf_ts = hopf_model.forward(initial_state=initial_states, n_steps=n_timepoints, dt=cfg.tr)
        hopf_fc = hopf_model.compute_fc(hopf_ts)
        hopf_fc_mean = hopf_fc.mean(dim=0)

    metrics = compute_all_fc_metrics(hopf_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
    target_ts = dataset.timeseries[eval_indices[: hopf_ts.shape[0]], :, :n_timepoints]
    dyn_metrics = compute_dynamics_fit_metrics(
        hopf_ts,
        target_ts,
        tr=cfg.tr,
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
    structural_connectivity: torch.Tensor,
    cfg: HopfConfig,
    device: str,
):
    """Train Hopf model using grid search."""
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
        wandb.define_metric("metrics/*", step_metric="epoch")
        wandb.define_metric("best_params/*", step_metric="epoch")

    # ---- Train / validation split BEFORE grid search (no data leakage) ----
    train_idx, val_idx = split_subject_indices(cfg, dataset.n_subjects)
    n_rois = dataset.n_rois
    n_timepoints = min(dataset.n_timepoints, 200)

    # Grid search uses only the training subjects.
    train_fc = dataset.fc_matrices[train_idx].mean(dim=0)
    n_eval = min(cfg.n_simulations, len(train_idx))
    eval_train_idx = train_idx[:n_eval]
    initial_states = dataset.timeseries[eval_train_idx, :, 0]
    target_ts = dataset.timeseries[eval_train_idx, :, :n_timepoints]

    print(f"Grid search over {len(cfg.g_values) * len(cfg.a_values) * len(cfg.kappa_values)} parameter combinations")
    print(f"  - G values: {cfg.g_values}")
    print(f"  - a values: {cfg.a_values}")
    print(f"  - kappa values: {cfg.kappa_values}")
    print(f"  - Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")

    # Build composite metric weights from CLI.
    metric_weights = {}
    if cfg.weight_fc:
        metric_weights["fc_correlation"] = cfg.weight_fc
    if cfg.weight_fcd:
        metric_weights["fcd_mse"] = cfg.weight_fcd
    if cfg.weight_meta:
        metric_weights["metastability_diff"] = cfg.weight_meta

    best_params, hopf_model = grid_search_hopf(
        target_fc=train_fc,
        n_rois=n_rois,
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
        metric_weights=metric_weights if metric_weights else None,
        noise_sigma=cfg.noise_sigma,
    )

    # Evaluate on train and validation splits with consistent methodology.
    n_eval_paths = max(10, cfg.n_simulations)
    train_metrics = evaluate_hopf_model(
        hopf_model,
        dataset,
        cfg,
        n_paths=n_eval_paths,
        subject_indices=train_idx,
    )
    val_metrics = evaluate_hopf_model(
        hopf_model,
        dataset,
        cfg,
        n_paths=n_eval_paths,
        subject_indices=val_idx,
    )

    print(f"Train metrics: {train_metrics}")
    print(f"Validation metrics: {val_metrics}")

    log_payload = {
        "epoch": 0,
        "best_params/G": float(best_params.get("initial_g", 0.0)),
        "best_params/a": float(best_params.get("initial_a", 0.0)),
        "best_params/kappa": float(best_params.get("initial_kappa", best_params.get("kappa", 0.0))),
        **prefixed_metrics("train", train_metrics),
        **prefixed_metrics("validation", val_metrics),
    }
    wandb_log(log_payload, use_wandb=cfg.use_wandb)
    wandb_summary_update(
        {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        },
        use_wandb=cfg.use_wandb,
    )

    return hopf_model, train_metrics, best_params


def save_model_and_figures(
    hopf_model: CoupledHopfModel,
    dataset: NeuroscienceDataset,
    val_loader,
    cfg: HopfConfig,
):
    """Save model and generate FC/timeseries/realization figures."""
    print_section("STEP 3: Saving Model and Generating Figures")

    val_timeseries, val_fc_matrices, target_fc, n_timepoints = extract_val_data(val_loader)

    checkpoint_path = save_checkpoint(
        hopf_model,
        checkpoint_name=f"hopf_grid_best_{cfg.run_name}.pt",
        artifact_name=f"hopf_model_grid_{cfg.run_name}",
        checkpoint_dir=cfg.checkpoint_dir,
        use_wandb=cfg.use_wandb,
    )

    generate_fc_figure(
        hopf_model, val_timeseries, target_fc, n_timepoints, cfg.tr,
        title="Coupled Hopf (Grid) - FC Comparison",
        default_name="hopf_grid_fc_comparison",
        use_wandb=cfg.use_wandb,
    )

    generate_multigrid_figure(
        hopf_model, val_timeseries, n_timepoints, cfg.tr,
        n_simulations=3,
        n_rois=3,
        n_cols=3,
        title="Coupled Hopf (Grid) - Real vs Simulated",
        default_name="hopf_grid_real_vs_sim",
        use_wandb=cfg.use_wandb,
    )

    print(f"Figures saved to {FIGURES_DIR}")
    return checkpoint_path


def main(argv=None):
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Coupled Hopf Model via grid search")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control", help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default="hopf_grid", help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data / dynamics settings
    parser.add_argument("--max-subjects", type=int, default=90, help="Limit number of subjects (first N)")
    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time in seconds")
    parser.add_argument("--f-lo", type=float, default=0.04, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--f-hi", type=float, default=0.07, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--fcd-win-sec", type=float, default=60.0, help="Sliding-window length for FCD metrics/losses (seconds)")
    parser.add_argument("--fcd-step-sec", type=float, default=2.0, help="Sliding-window step for FCD metrics/losses (seconds)")
    parser.add_argument("--no-fcd-ks", dest="no_fcd", action="store_true", help="Disable `fcd_ks` metric computation")
    parser.add_argument("--no-fcd", dest="no_fcd", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-metastability-diff", dest="no_metastability", action="store_true", help="Disable `metastability_diff` metric computation")
    parser.add_argument("--no-metastability", dest="no_metastability", action="store_true", help=argparse.SUPPRESS)

    # Preprocessing
    parser.add_argument("--fourier-denoise", action="store_true", default=True, help="Apply FFT bandpass denoising")
    parser.add_argument("--denoise-f-lo", type=float, default=0.008, help="Denoising low cutoff (Hz)")
    parser.add_argument("--denoise-f-hi", type=float, default=0.08, help="Denoising high cutoff (Hz)")

    # Grid-search settings
    parser.add_argument("--g-values", type=float, nargs="*", default=None, help="Grid values for G")
    parser.add_argument("--a-values", type=float, nargs="*", default=None, help="Grid values for a")
    parser.add_argument("--kappa-values", type=float, nargs="*", default=None, help="Grid values for noise sigma (kappa)")
    parser.add_argument("--n-simulations", type=int, default=10, help="Number of stochastic simulations per grid point")
    parser.add_argument("--weight-fc-correlation", dest="weight_fc", type=float, default=1.0, help="Weight for `fc_correlation` in grid-search composite score")
    parser.add_argument("--weight-fc", dest="weight_fc", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--weight-fcd-mse", dest="weight_fcd", type=float, default=None, help="Weight for `fcd_mse` in grid-search composite score")
    parser.add_argument("--weight-fcd", default=1., dest="weight_fcd", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--weight-metastability-diff", dest="weight_meta", type=float, default=None, help="Weight for `metastability_diff` in grid-search composite score")
    parser.add_argument("--weight-meta", default=1., dest="weight_meta", type=float, help=argparse.SUPPRESS)

    # Hopf settings
    # parser.add_argument("--initial-a", type=float, default=0.0, help="Initial Hopf bifurcation parameter")
    # parser.add_argument("--initial-g", type=float, default=0.0, help="Initial Hopf coupling")
    parser.add_argument("--noise-sigma", type=float, default=0.2, help="Hopf noise scale")

    args = parser.parse_args(argv)

    print_section("COUPLED HOPF MODEL TRAINING (GRID SEARCH)")
    ensure_proxy_env()

    cfg = HopfConfig(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=args.device,
        seed=args.seed,
        max_subjects=args.max_subjects,
        tr=args.tr,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        fcd_win_sec=args.fcd_win_sec,
        fcd_step_sec=args.fcd_step_sec,
        compute_fcd_metrics=not args.no_fcd,
        compute_metastability_metrics=not args.no_metastability,
        n_simulations=args.n_simulations,
        fourier_denoise=args.fourier_denoise,
        denoise_f_lo=args.denoise_f_lo,
        denoise_f_hi=args.denoise_f_hi,
        weight_fc=args.weight_fc,
        weight_fcd=args.weight_fcd,
        weight_meta=args.weight_meta,
        noise_sigma=args.noise_sigma,
    )

    if args.g_values:
        cfg.g_values = args.g_values
    if args.a_values:
        cfg.a_values = args.a_values
    if args.kappa_values:
        cfg.kappa_values = args.kappa_values

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    try:
        dataset, omega, structural_connectivity = load_data(cfg, device)
        _, val_idx = split_subject_indices(cfg, dataset.n_subjects)
        val_loader = create_validation_loader(dataset, cfg, device, val_idx)
        hopf_model, train_metrics, best_params = train_hopf_grid_search(dataset, omega, structural_connectivity, cfg, device)
        checkpoint = save_model_and_figures(hopf_model, dataset, val_loader, cfg)
        val_window_size = getattr(getattr(val_loader, "dataset", None), "window_size", None)
        final_metrics_all = evaluate_hopf_loader_metrics(
            hopf_model,
            val_loader,
            cfg,
            n_steps=val_window_size,
        )
        # Log ALL final evaluation metrics (not just a fixed subset).
        final_metrics = {
            k: v for k, v in final_metrics_all.items()
            if isinstance(v, (int, float))
        }
        wandb_summary_update(
            {f"final_{k}": v for k, v in final_metrics.items()},
            use_wandb=cfg.use_wandb,
        )
        log_hopf_best_params(hopf_model, use_wandb=cfg.use_wandb)
    finally:
        finish_wandb_run()

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
