#!/usr/bin/env python3
"""
Training entry point for Hopf grid-search and backprop models.

# Single model training:
    python examples/train_models.py backprop --model nsde
    python examples/train_models.py hopf-grid

# Paper suite — trains all models and runs the post-training pipeline:
    python examples/train_models.py paper \\
        --dataset-type lsd --lsd-data-dir data/lsd
    python examples/train_models.py paper \\
        --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat

Post-training commands (update tables, compare models, compare conditions)
are in examples/postprocess.py.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.cli_args import add_dataset_args, DATASET_ARG_NAMES
from src.dataset import compute_omega_from_timeseries, load_dataset
from src.dataset import compute_split_indices, create_data_loaders
from src.metrics import fisher_batch_average
from src.models import build_model
from src.training import (
    GNNHopfConfig,
    HopfConfig,
    HybridHopfConfig,
    HybridNeuralConfig,
    NeuralSDEConfig,
    grid_search_hopf,
    Trainer,
)
from src.training.losses import compute_ref_amplitude, compute_ref_omega
from src.utils import (
    FIGURES_DIR,
    EVAL_METRIC_KEYS,
    as_numeric_metrics,
    ensure_proxy_env,
    evaluate_model_loader_metrics,
    extract_val_data,
    finish_wandb_run,
    format_metrics_mean_std,
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

_CONFIG_CLS = {
    "nsde": NeuralSDEConfig,
    "hopf": HopfConfig,
    "hybrid_hopf": HybridHopfConfig,
    "gnn_hopf": GNNHopfConfig,
    "hybrid_neural": HybridNeuralConfig,
}

_MODEL_TITLES = {
    "nsde": "Neural SDE",
    "hopf": "Coupled Hopf",
    "hybrid_hopf": "Hybrid Hopf",
    "gnn_hopf": "GNN Hopf",
    "hybrid_neural": "Hybrid Hopf+Neural",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_common_cfg_overrides(cfg, args: argparse.Namespace) -> None:
    if args.no_wandb:
        cfg.use_wandb = False
    if args.device is not None:
        cfg.device = args.device
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.group_size is not None:
        cfg.group_size = args.group_size


def _apply_dataset_cfg_overrides(cfg, args: argparse.Namespace) -> None:
    for name in DATASET_ARG_NAMES:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "no_nilearn_reduce_confounds", False):
        cfg.nilearn_reduce_confounds = False


def _setup_run(cfg_cls, args: argparse.Namespace, exp_prefix: str):
    """Create config, apply overrides, set names, resolve device and seed."""
    cfg = cfg_cls()
    _apply_common_cfg_overrides(cfg, args)
    _apply_dataset_cfg_overrides(cfg, args)
    ds_tag = cfg.dataset_type
    cfg.experiment_name = f"{exp_prefix}_{ds_tag}"
    cfg.run_name = f"{cfg.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_proxy_env()
    print(f"Config: {dataclasses.asdict(cfg)}")
    device = resolve_device(cfg.device)
    seed_all(cfg.seed)
    return cfg, device, ds_tag


def _make_backprop_namespace_from_paper_args(args: argparse.Namespace, model_name: str) -> argparse.Namespace:
    payload = {
        "model": model_name,
        "no_wandb": args.no_wandb,
        "device": args.device,
        "skip_figures": args.skip_figures,
        "n_epochs": args.n_epochs,
        "no_nilearn_reduce_confounds": getattr(args, "no_nilearn_reduce_confounds", False),
    }
    for name in DATASET_ARG_NAMES:
        payload[name] = getattr(args, name, None)
    return argparse.Namespace(**payload)


# ---------------------------------------------------------------------------
# Shared training helpers
# ---------------------------------------------------------------------------


def _save_figures(model, val_loader, cfg, ds_tag: str, name_prefix: str, title: str,
                  args: argparse.Namespace, **sde_kwargs) -> None:
    """Generate and save FC + timeseries figures (no-op when --skip-figures)."""
    if args.skip_figures:
        print("Skipping figure generation (--skip-figures).")
        return
    val_timeseries, _, target_fc, n_tp = extract_val_data(val_loader)
    generate_fc_figure(
        model, val_timeseries, target_fc, n_tp, cfg.tr,
        title=f"{title} - FC Comparison",
        default_name=f"{ds_tag}/{name_prefix}_fc",
        use_wandb=cfg.use_wandb,
        wandb_key=f"figures/{ds_tag}/{name_prefix}_fc",
        **sde_kwargs,
    )
    generate_multigrid_figure(
        model, val_timeseries, n_tp, cfg.tr,
        n_simulations=3, n_rois=3, n_cols=3,
        title=f"{title} - Real vs Simulated",
        default_name=f"{ds_tag}/{name_prefix}_timeseries",
        use_wandb=cfg.use_wandb,
        wandb_key=f"figures/{ds_tag}/{name_prefix}_timeseries",
        **sde_kwargs,
    )


# ---------------------------------------------------------------------------
# Shared post-training evaluation
# ---------------------------------------------------------------------------


def _post_training(
    model,
    model_key: str,
    title: str,
    cfg,
    ds_tag: str,
    train_loader,
    val_loader,
    test_inter_loader,
    test_intra_loader,
    window_size: int,
    args: argparse.Namespace,
    *,
    is_hopf: bool = False,
    best_params: dict[str, float] | None = None,
    precomputed_train_metrics: dict[str, float] | None = None,
    precomputed_val_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    """Shared evaluation, save, figure-gen, and reporting for every model.

    Both grid-search and backprop paths call this after training is finished
    so that losses, metrics, saving, and reporting are identical.
    """
    print_section("Evaluating & Saving")

    # --- evaluate all splits consistently ---
    train_metrics = precomputed_train_metrics or evaluate_model_loader_metrics(
        model, train_loader, cfg, n_steps=window_size,
    )
    val_metrics = precomputed_val_metrics or evaluate_model_loader_metrics(
        model, val_loader, cfg, n_steps=window_size,
    )
    test_inter_metrics = evaluate_model_loader_metrics(
        model, test_inter_loader, cfg, n_steps=window_size, return_std=True,
    )
    test_intra_metrics = evaluate_model_loader_metrics(
        model, test_intra_loader, cfg, n_steps=window_size, return_std=True,
    )

    # --- wandb summary ---
    wandb_summary_update(
        {f"final_{k}": v for k, v in val_metrics.items()},
        use_wandb=cfg.use_wandb,
    )

    # --- save checkpoint ---
    checkpoint_path = save_checkpoint(
        model,
        checkpoint_name=f"{ds_tag}_{model_key}.pt",
        artifact_name=f"{ds_tag}_{model_key}",
        checkpoint_dir=cfg.checkpoint_dir,
        use_wandb=cfg.use_wandb,
    )

    # --- figures ---
    _save_figures(
        model, val_loader, cfg, ds_tag, model_key, title, args,
        sde_type=cfg.sde_type, method=cfg.sde_method, dt_min=cfg.dt_min,
        denoise_f_lo=cfg.denoise_f_lo, denoise_f_hi=cfg.denoise_f_hi,
    )
    if is_hopf:
        log_hopf_best_params(model, use_wandb=cfg.use_wandb)

    # --- save individual JSON ---
    _save_individual_metrics(model_key, test_inter_metrics, test_intra_metrics, ds_tag)

    # --- console report ---
    print(f"\n{'='*60}")
    print(f"  {title.upper()} — COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    if best_params:
        print(f"Best params: {best_params}")
    print(f"Train metrics: {format_metrics_mean_std(train_metrics)}")
    print(f"Val   metrics: {format_metrics_mean_std(val_metrics)}")
    print(f"Test inter (mean ± std): {format_metrics_mean_std(test_inter_metrics)}")
    print(f"Test intra (mean ± std): {format_metrics_mean_std(test_intra_metrics)}")
    print(f"Checkpoint: {checkpoint_path}  |  Figures: {FIGURES_DIR / ds_tag}")

    return {
        "run": model_key,
        "model": model,
        "train_metrics": as_numeric_metrics(train_metrics),
        "val_metrics": as_numeric_metrics(val_metrics),
        "test_inter_metrics": as_numeric_metrics(test_inter_metrics),
        "test_intra_metrics": as_numeric_metrics(test_intra_metrics),
        "params": {k: float(v) for k, v in best_params.items()} if best_params else None,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
    }


# ---------------------------------------------------------------------------
# Training runners
# ---------------------------------------------------------------------------

def _run_backprop(args: argparse.Namespace) -> dict[str, object]:
    fdm_weight = getattr(args, "fdm", 0.0) or 0.0
    exp_prefix = f"{args.model}_backprop"
    title = _MODEL_TITLES.get(args.model, args.model)
    print_section(f"{args.model.upper()} BACKPROP TRAINING")
    cfg, device, ds_tag = _setup_run(_CONFIG_CLS[args.model], args, exp_prefix)

    if fdm_weight > 0.0:
        cfg.loss_weights["fdm"] = fdm_weight

    print_section("STEP 1: Loading and Processing Data")
    dataset = load_dataset(cfg, device)
    window_size = min(cfg.window_size, dataset.n_timepoints // 2)
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
    print(f"  window_size={window_size}  train={len(train_loader)}  val={len(val_loader)}"
          f"  test_inter={len(test_inter_loader)}  test_intra={len(test_intra_loader)}")

    print_section("STEP 2: Training Model (Backpropagation)")
    model = build_model(args.model, dataset, cfg, device,
                        structural_connectivity=dataset.fc_mean,
                        n_control_dims=dataset.n_control_dims)
    ref_amplitude = compute_ref_amplitude(dataset.timeseries)
    ref_omega = compute_ref_omega(dataset.timeseries, tr=cfg.tr, f_lo=cfg.f_lo, f_hi=cfg.f_hi)
    print(f"  ref_amplitude=[{ref_amplitude.min():.4f}, {ref_amplitude.max():.4f}]")
    print(f"  ref_omega=[{ref_omega.min() / 6.2832:.4f}, {ref_omega.max() / 6.2832:.4f}] Hz")

    trainer = None
    try:
        trainer = Trainer(
            model=model, lr=cfg.lr, device=device,
            checkpoint_dir=cfg.checkpoint_dir, experiment_name=cfg.experiment_name,
            cfg=cfg, use_wandb=cfg.use_wandb,
        )
        trainer.train(
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=cfg.n_epochs, n_steps=window_size, dt=cfg.tr,
            early_stopping_patience=cfg.early_stopping_patience, verbose=True,
        )
    finally:
        if trainer is not None:
            trainer.finish()

    is_hopf = args.model in ("hopf", "hybrid_hopf", "hybrid_neural", "gnn_hopf")
    return _post_training(
        model, args.model, f"{title} (Backprop)", cfg, ds_tag,
        train_loader, val_loader, test_inter_loader, test_intra_loader,
        window_size, args, is_hopf=is_hopf,
    )


def _run_hopf_grid(args: argparse.Namespace) -> dict[str, object]:
    print_section("COUPLED HOPF MODEL TRAINING (GRID SEARCH)")
    cfg, device, ds_tag = _setup_run(HopfConfig, args, "hopf_grid")

    print_section("STEP 1: Loading and Processing Data")
    dataset = load_dataset(cfg, device)
    omega = compute_omega_from_timeseries(
        dataset.timeseries, dt=dataset.dt, f_lo=cfg.f_lo, f_hi=cfg.f_hi, method="peak",
    )
    print(f"  omega: shape={omega.shape}, "
          f"range=[{omega.min().item() / 6.2832:.4f}, {omega.max().item() / 6.2832:.4f}] Hz")

    # Use the same split as backprop (patient-aware when applicable)
    window_size = min(cfg.window_size, dataset.n_timepoints // 2)
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
    # Also get raw train indices for grid_search_hopf inputs
    train_idx, _, _ = compute_split_indices(
        dataset, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed,
    )
    print(f"  window_size={window_size}  train={len(train_loader)}  val={len(val_loader)}"
          f"  test_inter={len(test_inter_loader)}  test_intra={len(test_intra_loader)}")

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

    best_params: dict[str, float] = {}
    try:
        n_timepoints = min(dataset.n_timepoints, 200)
        train_fc = fisher_batch_average(dataset.fc_matrices[train_idx])
        eval_train_idx = train_idx[:min(cfg.n_simulations, len(train_idx))]
        initial_states = dataset.timeseries[eval_train_idx, :, 0]
        target_ts = dataset.timeseries[eval_train_idx, :, :n_timepoints]

        n_combos = len(cfg.g_values) * len(cfg.a_values) * len(cfg.kappa_values)
        print(f"Grid search over {n_combos} combinations  (train={len(train_idx)} timeseries)")

        best_params, hopf_model = grid_search_hopf(
            target_fc=train_fc,
            n_rois=dataset.n_rois,
            initial_states=initial_states,
            structural_connectivity=dataset.fc_mean,
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
            loss_weights=cfg.loss_weights,
            noise_sigma=cfg.noise_sigma,
            n_control_dims=dataset.n_control_dims,
            denoise_f_lo=cfg.denoise_f_lo,
            denoise_f_hi=cfg.denoise_f_hi,
            fdm_n_pairs=cfg.fdm_n_pairs,
            fdm_max_lag=cfg.fdm_max_lag,
            fdm_sigma=cfg.fdm_sigma,
        )

        # Log grid-search-specific wandb info before _post_training
        train_metrics_quick = evaluate_model_loader_metrics(
            hopf_model,
            train_loader,
            cfg,
            n_steps=window_size
        )
        val_metrics_quick = evaluate_model_loader_metrics(
            hopf_model,
            val_loader,
            cfg,
            n_steps=window_size
        )
        wandb_log(
            {
                "epoch": 0,
                "best_params/G": float(best_params.get("initial_g", 0.0)),
                "best_params/a": float(best_params.get("initial_a", 0.0)),
                "best_params/kappa": float(best_params.get("initial_kappa", best_params.get("kappa", 0.0))),
                **prefixed_metrics("train", train_metrics_quick),
                **prefixed_metrics("validation", val_metrics_quick),
            },
            use_wandb=cfg.use_wandb,
        )
    finally:
        finish_wandb_run()

    return _post_training(
        hopf_model, "hopf_grid", "Coupled Hopf (Grid Search)", cfg, ds_tag,
        train_loader, val_loader, test_inter_loader, test_intra_loader,
        window_size, args, is_hopf=True, best_params=best_params,
        precomputed_train_metrics=train_metrics_quick,
        precomputed_val_metrics=val_metrics_quick,
    )


# ---------------------------------------------------------------------------
# Paper suite
# ---------------------------------------------------------------------------

def _format_metric(value: float) -> str:
    return "nan" if torch.isnan(torch.tensor(value)) else f"{value:.6f}"


def _print_paper_report(report: dict[str, dict[str, float]]) -> None:
    metric_keys = list(EVAL_METRIC_KEYS)
    header = ["model"] + metric_keys
    col_widths = [max(len(k), 12) for k in header]

    def _line(parts: list[str]) -> str:
        return " | ".join(p.ljust(col_widths[i]) for i, p in enumerate(parts))

    print_section("PAPER METRIC REPORT")
    print(_line(header))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for model_name, metrics in report.items():
        row = [model_name] + [_format_metric(metrics.get(k, float("nan"))) for k in metric_keys]
        print(_line(row))


def _save_individual_metrics(
    model_key: str,
    test_inter: dict[str, object],
    test_intra: dict[str, object],
    ds_tag: str,
) -> Path:
    """Save both test split metrics to results/{ds_tag}_{model_key}.json."""
    output = Path("results") / f"{ds_tag}_{model_key}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_type": ds_tag,
        "model": model_key,
        "metrics": {
            "test_inter": {k: float(v) for k, v in as_numeric_metrics(test_inter).items()},
            "test_intra": {k: float(v) for k, v in as_numeric_metrics(test_intra).items()},
        },
    }
    with output.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    print(f"Metrics saved to: {output}")
    return output


def _run_paper_suite(args: argparse.Namespace) -> dict[str, object]:
    print_section("PAPER COMPARISON SUITE")
    report: dict[str, dict[str, float]] = {}

    grid_result = _run_hopf_grid(args)
    report["hopf_grid"] = grid_result["test_inter_metrics"]

    cfg_temp = HopfConfig()
    _apply_dataset_cfg_overrides(cfg_temp, args)
    dataset_type = cfg_temp.dataset_type

    for model_name in ("hopf", "nsde", "hybrid_hopf", "gnn_hopf", "hybrid_neural"):
        backprop_args = _make_backprop_namespace_from_paper_args(args, model_name)
        result = _run_backprop(backprop_args)
        report[model_name] = result["test_inter_metrics"]

    _print_paper_report(report)
    return {"paper_metrics": report, "dataset_type": dataset_type}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Hopf (grid) and backprop models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
        p.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
        p.add_argument("--skip-figures", action="store_true", help="Skip final figure generation")
        p.add_argument("--n-epochs", type=int, default=None, help="Override number of training epochs")
        p.add_argument("--group-size", type=int, default=None, help="Samples per group for metric evaluation (0 = full batch)")
        add_dataset_args(p)

    backprop = subparsers.add_parser("backprop", help="Train NSDE/Hopf/Hybrid Hopf with backpropagation")
    backprop.add_argument("--model", type=str, default="nsde", choices=list(_CONFIG_CLS))
    backprop.add_argument("--fdm", type=float, default=0.0,
                          help="FDM loss weight (0 = disabled, suggested: 0.25)")
    _add_common(backprop)

    hopf_grid = subparsers.add_parser("hopf-grid", help="Train Coupled Hopf with grid search")
    _add_common(hopf_grid)

    paper = subparsers.add_parser("paper", help="Run Hopf-grid + all backprop models and report paper metrics")
    _add_common(paper)

    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "backprop": _run_backprop,
        "hopf-grid": _run_hopf_grid,
        "paper": _run_paper_suite,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    main()
