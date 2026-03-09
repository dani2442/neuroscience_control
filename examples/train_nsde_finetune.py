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


def _apply_dataset_overrides(cfg: NeuralSDEConfig, args: argparse.Namespace) -> None:
    override_names = (
        "dataset_type",
        "data_path",
        "lsd_data_dir",
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
    for name in override_names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.no_nilearn_reduce_confounds:
        cfg.nilearn_reduce_confounds = False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained Neural SDE checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained NSDE checkpoint (*.pt)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=["ts_young", "mat", "lsd", "nilearn", "openneuro", "datalad", "bids"],
        help="Dataset backend to use.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to local .mat dataset.")
    parser.add_argument("--lsd-data-dir", type=str, default=None, help="Directory containing LSD .mat files.")
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
    parser.add_argument("--openneuro-dataset", type=str, default=None, help="OpenNeuro dataset id.")
    parser.add_argument("--openneuro-tag", type=str, default=None, help="OpenNeuro dataset tag/revision.")
    parser.add_argument("--openneuro-target-dir", type=str, default=None, help="Directory for OpenNeuro download.")
    parser.add_argument("--openneuro-include", nargs="+", default=None, help="OpenNeuro include glob patterns.")
    parser.add_argument("--openneuro-exclude", nargs="+", default=None, help="OpenNeuro exclude glob patterns.")
    parser.add_argument("--datalad-source", type=str, default=None, help="DataLad dataset source URL/path.")
    parser.add_argument("--datalad-dataset-dir", type=str, default=None, help="Local DataLad checkout dir.")
    parser.add_argument("--datalad-get-paths", nargs="+", default=None, help="DataLad paths/globs to fetch.")
    parser.add_argument("--bids-root", type=str, default=None, help="Root path for direct BIDS loading.")
    parser.add_argument("--bids-relative-path", type=str, default=None, help="Subpath under downloaded dataset root.")
    parser.add_argument("--bids-derivatives-dir", type=str, default=None, help="BIDS derivatives dir.")
    parser.add_argument("--bids-task", type=str, default=None, help="BIDS task label filter.")
    parser.add_argument("--bids-space", type=str, default=None, help="BIDS space label filter.")
    parser.add_argument("--bids-desc", type=str, default=None, help="BIDS desc label filter.")
    parser.add_argument("--bids-subject-ids", nargs="+", default=None, help="Optional subject list.")
    parser.add_argument("--bids-runs-per-subject", type=int, default=None, help="Runs per subject.")
    parser.add_argument("--atlas-n-rois", type=int, default=None, help="Schaefer atlas ROI count.")
    parser.add_argument("--atlas-yeo-networks", type=int, default=None, help="Schaefer atlas network count.")
    parser.add_argument("--atlas-resolution-mm", type=int, default=None, help="Schaefer atlas resolution.")
    parser.add_argument("--atlas-smoothing-fwhm", type=float, default=None, help="Masker smoothing FWHM.")
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
    _apply_dataset_overrides(cfg, args)

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
