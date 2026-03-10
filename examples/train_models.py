#!/usr/bin/env python3
"""
Unified training entry point for Hopf grid-search and backprop models.

Examples
--------
    python examples/train_models.py backprop --model nsde
    python examples/train_models.py hopf-grid
    python examples/train_models.py paper
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
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
from src.models import build_model
from src.training import (
    GNNHopfConfig,
    HopfConfig,
    HybridHopfConfig,
    NeuralSDEConfig,
    create_windowed_loaders,
    grid_search_hopf,
    load_dataset,
    run_backprop_training,
)
from src.training.losses import compute_ref_amplitude, compute_ref_omega
from src.utils import (
    FIGURES_DIR,
    EVAL_METRIC_KEYS,
    ensure_proxy_env,
    evaluate_hopf_model,
    evaluate_model_loader_metrics,
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

# Map model name → config class
_CONFIG_CLS = {
    "nsde": NeuralSDEConfig,
    "hopf": HopfConfig,
    "hybrid_hopf": HybridHopfConfig,
    "gnn_hopf": GNNHopfConfig,
}

_MODEL_TITLES = {
    "nsde": "Neural SDE",
    "hopf": "Coupled Hopf",
    "hybrid_hopf": "Hybrid Hopf",
    "gnn_hopf": "GNN Hopf",
}

_DATASET_ARG_NAMES = (
    "dataset_type",
    "data_path",
    "lsd_data_dir",
    "max_subjects",
    "nilearn_dataset",
    "nilearn_data_dir",
    "nilearn_n_subjects",
    "openneuro_dataset",
    "openneuro_tag",
    "openneuro_target_dir",
    "openneuro_include",
    "openneuro_exclude",
    "datalad_source",
    "datalad_dataset_dir",
    "datalad_get_paths",
    "bids_root",
    "bids_relative_path",
    "bids_derivatives_dir",
    "bids_task",
    "bids_space",
    "bids_desc",
    "bids_subject_ids",
    "bids_runs_per_subject",
    "atlas_n_rois",
    "atlas_yeo_networks",
    "atlas_resolution_mm",
    "atlas_smoothing_fwhm",
)


def _apply_common_cfg_overrides(cfg, args: argparse.Namespace) -> None:
    if args.no_wandb:
        cfg.use_wandb = False
    if args.device is not None:
        cfg.device = args.device
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs


def _apply_dataset_cfg_overrides(cfg, args: argparse.Namespace) -> None:
    for name in _DATASET_ARG_NAMES:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "no_nilearn_reduce_confounds", False):
        cfg.nilearn_reduce_confounds = False


def _make_backprop_namespace_from_paper_args(args: argparse.Namespace, model_name: str) -> argparse.Namespace:
    payload = {
        "model": model_name,
        "no_wandb": args.no_wandb,
        "device": args.device,
        "skip_figures": args.skip_figures,
        "n_epochs": args.n_epochs,
        "no_nilearn_reduce_confounds": getattr(args, "no_nilearn_reduce_confounds", False),
    }
    for name in _DATASET_ARG_NAMES:
        payload[name] = getattr(args, name, None)
    return argparse.Namespace(**payload)


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
    ctrl = getattr(dataset, "control", None)
    val_ctrl = ctrl[val_idx] if ctrl is not None else None
    val_dataset = RandomWindowDataset(
        dataset.timeseries[val_idx],
        dataset.fc_matrices[val_idx],
        window_size=window_size,
        n_windows=n_val_win,
        device=device,
        control=val_ctrl,
    )
    return DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)


def _as_numeric(metrics: dict[str, object]) -> dict[str, float]:
    """Keep only scalar numeric metrics."""
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        cast = to_float_metric(value)
        if cast is not None:
            numeric[key] = cast
    return numeric


def _format_metrics_mean_std(metrics: dict[str, object]) -> dict[str, str]:
    """Format metrics as `mean ± std` when `<metric>_std` is present."""
    numeric = _as_numeric(metrics)
    formatted: dict[str, str] = {}
    for key in sorted(numeric):
        if key.endswith("_std"):
            continue
        std_key = f"{key}_std"
        if std_key in numeric:
            formatted[key] = f"{numeric[key]:.6f} ± {numeric[std_key]:.6f}"
        else:
            formatted[key] = f"{numeric[key]:.6f}"
    return formatted


def _run_backprop(args: argparse.Namespace) -> dict[str, object]:
    cfg = _CONFIG_CLS[args.model]()
    _apply_common_cfg_overrides(cfg, args)
    _apply_dataset_cfg_overrides(cfg, args)
    # Include dataset type in experiment / run names so checkpoints are identifiable.
    ds_tag = cfg.dataset_type
    cfg.experiment_name = f"{args.model}_backprop_{ds_tag}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.run_name = f"{cfg.experiment_name}_{timestamp}"

    print_section(f"{args.model.upper()} BACKPROP TRAINING")
    ensure_proxy_env()
    print(f"Config: {dataclasses.asdict(cfg)}")

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    print_section("STEP 1: Loading and Processing Data")
    dataset = load_dataset(cfg, device)
    train_loader, val_loader, test_loader, window_size = create_windowed_loaders(dataset, cfg, device)
    print(f"  window_size={window_size}  train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}")

    print_section("STEP 2: Training Model (Backpropagation)")
    model = build_model(args.model, dataset, cfg, device, structural_connectivity=dataset.fc_mean,
                        n_control_dims=dataset.n_control_dims)
    ref_amplitude = compute_ref_amplitude(dataset.timeseries)
    ref_omega = compute_ref_omega(dataset.timeseries, tr=cfg.tr, f_lo=cfg.f_lo, f_hi=cfg.f_hi)
    print(f"  ref_amplitude=[{ref_amplitude.min():.4f}, {ref_amplitude.max():.4f}]")
    print(f"  ref_omega=[{ref_omega.min() / 6.2832:.4f}, {ref_omega.max() / 6.2832:.4f}] Hz")

    trainer = None
    final_metrics: dict[str, float] = {}
    checkpoint_path: Path | None = None
    test_metrics: dict[str, float] = {}
    try:
        trainer, _metrics_store, test_metrics = run_backprop_training(
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

        print_section("STEP 3: Saving Model and Generating Figures")
        val_timeseries, _, target_fc, n_timepoints = extract_val_data(val_loader)

        checkpoint_path = save_checkpoint(
            model,
            checkpoint_name=f"{args.model}_backprop_{ds_tag}_best_{cfg.run_name}.pt",
            artifact_name=f"{args.model}_backprop_{ds_tag}_model_{cfg.run_name}",
            checkpoint_dir=cfg.checkpoint_dir,
            use_wandb=cfg.use_wandb,
        )

        final_metrics = evaluate_model_loader_metrics(model, val_loader, cfg, n_steps=window_size)
        print(f"Final metrics (full val loader): {final_metrics}")
        wandb_summary_update(
            {f"final_{k}": v for k, v in final_metrics.items()},
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
                n_simulations=3, n_rois=3, n_cols=3,
                sde_type=cfg.sde_type, method=cfg.sde_method, dt_min=cfg.dt_min,
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

    print("\n" + "=" * 60)
    print(f"{args.model.upper()} BACKPROP TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nTest metrics (mean ± std): {_format_metrics_mean_std(test_metrics)}")
    print(f"Final metrics: {final_metrics}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return {
        "run": f"{args.model}_backprop",
        "model": model,
        "paper_metrics": final_metrics,
        "test_metrics": _as_numeric(test_metrics),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def _run_hopf_grid(args: argparse.Namespace) -> dict[str, object]:
    cfg = HopfConfig()
    _apply_common_cfg_overrides(cfg, args)
    _apply_dataset_cfg_overrides(cfg, args)
    # Include dataset type in experiment / run names so checkpoints are identifiable.
    ds_tag = cfg.dataset_type
    cfg.experiment_name = f"hopf_grid_{ds_tag}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.run_name = f"{cfg.experiment_name}_{timestamp}"

    print_section("COUPLED HOPF MODEL TRAINING (GRID SEARCH)")
    ensure_proxy_env()
    print(f"Config: {dataclasses.asdict(cfg)}")

    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

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

    train_idx, val_idx, test_idx = split_subject_indices(cfg, dataset.n_subjects)
    val_loader = _create_validation_loader(dataset, cfg, device, val_idx)
    test_loader = _create_validation_loader(dataset, cfg, device, test_idx)

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

    checkpoint: Path | None = None
    best_params: dict[str, float] = {}
    final_metrics: dict[str, float] = {}
    train_metrics: dict[str, float] = {}
    try:
        n_timepoints = min(dataset.n_timepoints, 200)
        train_fc = dataset.fc_matrices[train_idx].mean(dim=0)
        n_eval = min(cfg.n_simulations, len(train_idx))
        eval_train_idx = train_idx[:n_eval]
        initial_states = dataset.timeseries[eval_train_idx, :, 0]
        target_ts = dataset.timeseries[eval_train_idx, :, :n_timepoints]

        n_combos = len(cfg.g_values) * len(cfg.a_values) * len(cfg.kappa_values)
        print(f"Grid search over {n_combos} parameter combinations")
        print(f"  Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}, Test subjects: {len(test_idx)}")

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
            n_control_dims=dataset.n_control_dims,
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

        print_section("STEP 3: Saving Model and Generating Figures")
        checkpoint = save_checkpoint(
            hopf_model,
            checkpoint_name=f"hopf_grid_{ds_tag}_best_{cfg.run_name}.pt",
            artifact_name=f"hopf_model_grid_{ds_tag}_{cfg.run_name}",
            checkpoint_dir=cfg.checkpoint_dir,
            use_wandb=cfg.use_wandb,
        )

        val_timeseries, _, target_fc, n_tp = extract_val_data(val_loader)
        if not args.skip_figures:
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
        else:
            print("Skipping figure generation (--skip-figures).")

        val_window_size = getattr(getattr(val_loader, "dataset", None), "window_size", None)
        final_metrics = evaluate_model_loader_metrics(hopf_model, val_loader, cfg, n_steps=val_window_size)
        wandb_summary_update({f"final_{k}": v for k, v in final_metrics.items()}, use_wandb=cfg.use_wandb)
        log_hopf_best_params(hopf_model, use_wandb=cfg.use_wandb)
        
        # Evaluate on test set with std for paper reporting
        test_window_size = getattr(getattr(test_loader, "dataset", None), "window_size", None)
        test_metrics = evaluate_model_loader_metrics(hopf_model, test_loader, cfg, n_steps=test_window_size, return_std=True)
        print(f"Test metrics (mean ± std): {_format_metrics_mean_std(test_metrics)}")
    finally:
        finish_wandb_run()

    print("\n" + "=" * 60)
    print("HOPF GRID SEARCH COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nBest params: {best_params}")
    print(f"Train metrics: {train_metrics}")
    print(f"Final metrics (full val loader): {final_metrics}")
    print(f"Test metrics (mean ± std): {_format_metrics_mean_std(test_metrics)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return {
        "run": "hopf_grid",
        "paper_metrics": _as_numeric(test_metrics),  # Use test_metrics for paper consistency
        "train_metrics": _as_numeric(train_metrics),
        "params": {k: float(v) for k, v in best_params.items()},
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    }


def _format_metric(value: float) -> str:
    if torch.isnan(torch.tensor(value)):
        return "nan"
    return f"{value:.6f}"


def _print_paper_report(report: dict[str, dict[str, float]]) -> None:
    metric_keys = list(EVAL_METRIC_KEYS)
    header = ["model"] + metric_keys
    col_widths = [max(len(key), 12) for key in header]

    def _line(parts: list[str]) -> str:
        return " | ".join(part.ljust(col_widths[idx]) for idx, part in enumerate(parts))

    print_section("PAPER METRIC REPORT")
    print(_line(header))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for model_name, metrics in report.items():
        row = [model_name] + [_format_metric(metrics.get(key, float("nan"))) for key in metric_keys]
        print(_line(row))


def _save_paper_report(report: dict[str, dict[str, float]], output: Path | None, dataset_type: str = "ts_young") -> Path:
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("results") / f"{dataset_type}_paper_metrics_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Include dataset_type in the JSON for table routing
    report_with_meta = {"dataset_type": dataset_type, "metrics": report}
    with output.open("w", encoding="utf-8") as fh:
        json.dump(report_with_meta, fh, indent=2, sort_keys=True)
    print(f"Paper metric report saved to: {output}")
    return output


def _run_paper_suite(args: argparse.Namespace) -> dict[str, object]:
    print_section("PAPER COMPARISON SUITE")
    report: dict[str, dict[str, float]] = {}

    grid_result = _run_hopf_grid(args)
    report["hopf_grid"] = grid_result["paper_metrics"]
    
    # Get dataset_type from the grid config to use in filename
    cfg_temp = HopfConfig()
    _apply_dataset_cfg_overrides(cfg_temp, args)
    dataset_type = cfg_temp.dataset_type

    for model_name in ("hopf", "nsde", "hybrid_hopf", "gnn_hopf"):
        backprop_args = _make_backprop_namespace_from_paper_args(args, model_name)
        backprop_result = _run_backprop(backprop_args)
        # Use test_metrics (which include std) for paper reporting
        report[f"{model_name}_backprop"] = backprop_result.get("test_metrics", backprop_result["paper_metrics"])

    _print_paper_report(report)
    output_path = _save_paper_report(report, args.output_json, dataset_type=dataset_type)

    return {"paper_metrics": report, "output_json": str(output_path), "dataset_type": dataset_type}


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=["ts_young", "mat", "lsd", "nilearn", "openneuro", "datalad", "bids"],
        help="Dataset backend to use.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to local .mat dataset.")
    parser.add_argument("--lsd-data-dir", type=str, default=None, help="Directory containing LSD .mat files.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit number of subjects loaded (None = all).")
    parser.add_argument(
        "--nilearn-dataset",
        type=str,
        default=None,
        choices=["development_fmri", "adhd"],
        help="nilearn fetcher dataset name.",
    )
    parser.add_argument("--nilearn-data-dir", type=str, default=None, help="Cache directory for nilearn data.")
    parser.add_argument("--nilearn-n-subjects", type=int, default=None, help="Limit nilearn fetched subjects.")
    parser.add_argument(
        "--no-nilearn-reduce-confounds",
        action="store_true",
        help="Disable nilearn fetch_development_fmri confound reduction.",
    )
    parser.add_argument("--openneuro-dataset", type=str, default=None, help="OpenNeuro dataset id (e.g. ds000030).")
    parser.add_argument("--openneuro-tag", type=str, default=None, help="OpenNeuro dataset tag/revision.")
    parser.add_argument("--openneuro-target-dir", type=str, default=None, help="Directory for OpenNeuro download.")
    parser.add_argument(
        "--openneuro-include",
        nargs="+",
        default=None,
        help="OpenNeuro include glob patterns.",
    )
    parser.add_argument(
        "--openneuro-exclude",
        nargs="+",
        default=None,
        help="OpenNeuro exclude glob patterns.",
    )
    parser.add_argument("--datalad-source", type=str, default=None, help="DataLad dataset source URL/path.")
    parser.add_argument("--datalad-dataset-dir", type=str, default=None, help="Local DataLad dataset checkout dir.")
    parser.add_argument(
        "--datalad-get-paths",
        nargs="+",
        default=None,
        help="DataLad paths/globs to materialize with `datalad get`.",
    )
    parser.add_argument("--bids-root", type=str, default=None, help="Root path for direct BIDS loading.")
    parser.add_argument("--bids-relative-path", type=str, default=None, help="Subpath under downloaded dataset root.")
    parser.add_argument(
        "--bids-derivatives-dir",
        type=str,
        default=None,
        help="Relative derivatives directory containing preprocessed BOLD files.",
    )
    parser.add_argument("--bids-task", type=str, default=None, help="BIDS task label filter.")
    parser.add_argument("--bids-space", type=str, default=None, help="BIDS space label filter.")
    parser.add_argument("--bids-desc", type=str, default=None, help="BIDS desc label filter.")
    parser.add_argument(
        "--bids-subject-ids",
        nargs="+",
        default=None,
        help="Optional subject list (sub-XXX or XXX).",
    )
    parser.add_argument(
        "--bids-runs-per-subject",
        type=int,
        default=None,
        help="Number of BOLD runs to use per subject.",
    )
    parser.add_argument("--atlas-n-rois", type=int, default=None, help="Schaefer atlas ROI count.")
    parser.add_argument("--atlas-yeo-networks", type=int, default=None, help="Schaefer atlas network count.")
    parser.add_argument("--atlas-resolution-mm", type=int, default=None, help="Schaefer atlas resolution.")
    parser.add_argument(
        "--atlas-smoothing-fwhm",
        type=float,
        default=None,
        help="Optional smoothing passed to nilearn masker.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified training script for Hopf grid-search and backprop models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backprop = subparsers.add_parser("backprop", help="Train NSDE/Hopf/Hybrid Hopf with backpropagation")
    backprop.add_argument("--model", type=str, default="nsde", choices=list(_CONFIG_CLS), help="Model to train")
    backprop.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    backprop.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    backprop.add_argument("--skip-figures", action="store_true", help="Skip final figure generation")
    backprop.add_argument("--n-epochs", type=int, default=None, help="Override number of training epochs")
    _add_dataset_args(backprop)

    hopf_grid = subparsers.add_parser("hopf-grid", help="Train Coupled Hopf with grid search")
    hopf_grid.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    hopf_grid.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    hopf_grid.add_argument("--skip-figures", action="store_true", help="Skip final figure generation")
    hopf_grid.add_argument("--n-epochs", type=int, default=None, help=argparse.SUPPRESS)
    _add_dataset_args(hopf_grid)

    paper = subparsers.add_parser("paper", help="Run Hopf-grid + all backprop models and report paper metrics")
    paper.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    paper.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    paper.add_argument("--skip-figures", action="store_true", help="Skip final figure generation")
    paper.add_argument("--n-epochs", type=int, default=None, help="Override backprop training epochs")
    _add_dataset_args(paper)
    paper.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the aggregated paper metric report JSON",
    )

    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "backprop":
        return _run_backprop(args)
    if args.command == "hopf-grid":
        return _run_hopf_grid(args)
    if args.command == "paper":
        return _run_paper_suite(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
