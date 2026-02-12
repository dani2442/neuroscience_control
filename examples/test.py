#!/usr/bin/env python3
"""Checkpoint evaluation script for Hopf and Neural SDE models."""

import argparse
import json
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

from src.dataset import NeuroscienceDataset, create_data_loaders
from src.metrics import (
    compute_all_fc_metrics,
    compute_all_timeseries_metrics,
    compute_dynamics_fit_metrics,
)
from src.models import load_model_from_checkpoint
from src.training import Trainer, TrainingConfig
from src.utils import (
    FIGURES_DIR,
    plot_real_vs_sim_multigrid,
    print_section,
    resolve_device,
    seed_all,
)

def format_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(f"  - {k}: {v:.6f}" for k, v in sorted(metrics.items()))


def main(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    seed_all(args.seed)

    print_section("CHECKPOINT EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    dataset = NeuroscienceDataset(
        filepath=args.data_path,
        normalize=True,
        device=device,
        max_subjects=args.max_subjects,
        dt=args.tr,
        fourier_denoise=args.fourier_denoise,
        denoise_f_lo=args.denoise_f_lo,
        denoise_f_hi=args.denoise_f_hi,
    )
    print("\nDataset loaded:")
    print(f"  - subjects: {dataset.n_subjects}")
    print(f"  - rois: {dataset.n_rois}")
    print(f"  - timepoints: {dataset.n_timepoints}")

    model, model_name, _ = load_model_from_checkpoint(checkpoint_path, device=device)
    print(f"\nModel restored: {model_name}")

    window_size = min(args.window_size, dataset.n_timepoints // 4)
    _, _, test_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        batch_size=args.batch_size,
        n_windows_per_epoch=args.n_windows_per_epoch,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=device,
    )

    cfg = TrainingConfig(
        experiment_name=f"eval_{checkpoint_path.stem}",
        use_wandb=False,
        tr=args.tr,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        fcd_win_sec=args.fcd_win_sec,
        fcd_step_sec=args.fcd_step_sec,
        compute_fcd_metrics=not args.no_fcd,
        compute_metastability_metrics=not args.no_metastability,
        metrics_sample_batches=None,
    )
    trainer = Trainer(
        model=model,
        lr=1e-3,
        loss_fn=args.loss_fn,
        device=device,
        experiment_name=f"eval_{checkpoint_path.stem}",
        cfg=cfg,
        use_wandb=False,
    )

    test_metrics = trainer.test(test_loader=test_loader, n_steps=window_size, dt=args.model_dt)

    subject_idx = int(args.subject_index)
    if not 0 <= subject_idx < dataset.n_subjects:
        raise ValueError(f"subject-index must be in [0, {dataset.n_subjects - 1}], got {subject_idx}")

    real_ts = dataset.timeseries[subject_idx:subject_idx + 1]
    n_steps = min(args.sim_steps, real_ts.shape[2])
    real_ts = real_ts[:, :, :n_steps]
    initial_state = real_ts[:, :, 0]

    with torch.no_grad():
        sim_ts = model.forward(initial_state=initial_state, n_steps=n_steps, dt=args.tr)

    sim_fc = model.compute_fc(sim_ts)
    real_fc = model.compute_fc(real_ts)
    comparison_metrics = {
        **compute_all_fc_metrics(sim_fc, real_fc),
        **compute_all_timeseries_metrics(sim_ts, real_ts),
        **compute_dynamics_fit_metrics(
            sim_ts,
            real_ts,
            tr=args.tr,
            f_lo=args.f_lo,
            f_hi=args.f_hi,
            fcd_win_sec=args.fcd_win_sec,
            fcd_step_sec=args.fcd_step_sec,
            compute_fcd=not args.no_fcd,
            compute_metastability=not args.no_metastability,
        ),
    }

    fig = plot_real_vs_sim_multigrid(
        real_timeseries=real_ts.real,
        simulated_timeseries=sim_ts.real,
        n_rois=args.n_rois_plot,
        n_cols=args.grid_cols,
        max_timepoints=n_steps,
        title=f"{model_name} - Real vs Simulated (subject {subject_idx})",
        save_path=args.figure_path,
        default_name=f"{checkpoint_path.stem}_real_vs_sim_multigrid",
        use_pdf=True,
    )
    plt.close(fig)

    print("\nTest metrics (loader-based):")
    print(format_metrics(test_metrics))
    print("\nReal-vs-sim metrics (single-subject trajectory):")
    print(format_metrics(comparison_metrics))
    if args.figure_path is None:
        print(f"\nFigure saved under: {FIGURES_DIR}")

    if args.print_json:
        print("\nJSON:")
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "model": model_name,
                    "test_metrics": test_metrics,
                    "real_vs_sim_metrics": comparison_metrics,
                },
                indent=2,
            )
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Hopf/NeuralSDE checkpoint and visualize real-vs-sim trajectories."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (*.pt)")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to .mat dataset")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--max-subjects", type=int, default=50, help="Limit subjects for evaluation")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for test loader")
    parser.add_argument("--n-windows-per-epoch", type=int, default=256, help="Random windows per epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for test loader")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")

    parser.add_argument("--loss-fn", type=str, default="combined", help="Loss function used by Trainer.test")
    parser.add_argument("--model-dt", type=float, default=0.1, help="Integration dt for Trainer.test")

    parser.add_argument("--tr", type=float, default=0.72, help="Repetition time for dataset and dynamics metrics")
    parser.add_argument("--f-lo", type=float, default=0.04, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--f-hi", type=float, default=0.07, help="Bandpass high cutoff (Hz)")
    parser.add_argument("--fcd-win-sec", type=float, default=60.0, help="FCD window length (seconds)")
    parser.add_argument("--fcd-step-sec", type=float, default=2.0, help="FCD step (seconds)")
    parser.add_argument("--no-fcd", action="store_true", help="Disable FCD metric computation")
    parser.add_argument("--no-metastability", action="store_true", help="Disable metastability metric computation")

    # Preprocessing
    parser.add_argument("--fourier-denoise", action="store_true", help="Apply FFT bandpass denoising")
    parser.add_argument("--denoise-f-lo", type=float, default=0.01, help="Denoising low cutoff (Hz)")
    parser.add_argument("--denoise-f-hi", type=float, default=0.1, help="Denoising high cutoff (Hz)")

    parser.add_argument("--subject-index", type=int, default=0, help="Subject index for real-vs-sim trajectory plot")
    parser.add_argument("--sim-steps", type=int, default=200, help="Timepoints to simulate for real-vs-sim comparison")
    parser.add_argument("--n-rois-plot", type=int, default=12, help="Number of ROIs in multigrid visualization")
    parser.add_argument("--grid-cols", type=int, default=4, help="Columns in multigrid visualization")
    parser.add_argument("--figure-path", type=str, default=None, help="Optional figure output path")
    parser.add_argument("--print-json", action="store_true", help="Print metrics as JSON")

    args = parser.parse_args()
    main(args)
