#!/usr/bin/env python3
"""Measure runtime cost of every model: training s/epoch, grid-search s/trial,
and inference s.

This script is measurement-only. It never writes checkpoints and never touches
the canonical files under ``results/`` or ``checkpoints/`` -- trained models are
loaded **read-only** so that timing reflects converged (stable) weights.

Usage (mirrors the training commands, GPU via SLURM):
    sbatch -M tinygpu --gres=gpu:1 --time=00:40:00 --wrap=\\
        ".venv/bin/python examples/benchmark_timing.py \\
            --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat"

Results are written to results/timing_benchmark.json.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    compute_omega_from_timeseries,
    compute_split_indices,
    create_data_loaders,
    load_dataset,
)
from src.metrics import fisher_batch_average
from src.models import build_model
from src.models.hopf_model import CoupledHopfModel
from src.training import (
    GNNHopfConfig,
    GridSearch,
    HopfConfig,
    HybridHopfConfig,
    HybridNeuralConfig,
    NeuralSDEConfig,
    Trainer,
)
from src.utils import ensure_proxy_env, resolve_device, seed_all
from src.utils.evaluation import _forward_for_metrics

_CONFIG_CLS = {
    "hopf": HopfConfig,
    "nsde": NeuralSDEConfig,
    "hybrid_hopf": HybridHopfConfig,
    "gnn_hopf": GNNHopfConfig,
    "hybrid_neural": HybridNeuralConfig,
}

_MODEL_TITLES = {
    "hopf": "Coupled Hopf",
    "nsde": "Neural SDE",
    "hybrid_hopf": "Hybrid Hopf",
    "gnn_hopf": "GNN-Hopf",
    "hybrid_neural": "Hopf+Neural",
}

# Order matches the supplementary full-comparison table.
_MODEL_ORDER = ["hopf", "nsde", "hybrid_hopf", "gnn_hopf", "hybrid_neural"]


def _sync(cuda: bool) -> None:
    if cuda:
        torch.cuda.synchronize()


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


# ---------------------------------------------------------------------------
# Hardware / software specs
# ---------------------------------------------------------------------------

def collect_specs(device: str) -> dict[str, object]:
    specs: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": device,
        "cpu_logical_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
    }

    # CPU model name from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    specs["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        specs["cpu_model"] = None

    specs["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        specs["gpu_name"] = torch.cuda.get_device_name(0)
        specs["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        specs["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        specs["gpu_multiprocessor_count"] = props.multi_processor_count
        specs["cuda_version"] = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        specs["cudnn_version"] = cudnn_version
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            specs["nvidia_driver_version"] = driver.stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, OSError, IndexError):
            specs["nvidia_driver_version"] = None
    return specs


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_key: str,
    dataset,
    base_cfg,
    device: str,
    train_loader,
    val_loader,
    window_size: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    cuda = device.startswith("cuda")
    title = _MODEL_TITLES[model_key]
    print(f"\n{'=' * 70}\n  {title.upper()} ({model_key})\n{'=' * 70}")

    cfg = _CONFIG_CLS[model_key]()
    cfg.use_wandb = False
    cfg.device = device
    cfg.dataset_type = base_cfg.dataset_type
    cfg.data_path = base_cfg.data_path
    cfg.experiment_name = f"timing_benchmark/{model_key}"

    seed_all(cfg.seed)
    model = build_model(
        model_key, dataset, cfg, device,
        structural_connectivity=dataset.fc_mean,
        n_control_dims=dataset.n_control_dims,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- load trained weights read-only (stable timing, no file writes) ---
    ckpt_path = Path(base_cfg.checkpoint_dir) / f"{base_cfg.dataset_type}_{model_key}.pt"
    checkpoint_loaded = False
    if ckpt_path.exists():
        try:
            model.load(str(ckpt_path))
            checkpoint_loaded = True
            print(f"  Loaded weights (read-only) from {ckpt_path}")
        except Exception as exc:  # noqa: BLE001 - fall back to fresh weights
            print(f"  WARNING: could not load {ckpt_path}: {exc}\n"
                  f"  Continuing with freshly initialized weights.")
    else:
        print(f"  WARNING: {ckpt_path} not found; using freshly initialized weights.")

    # --- inference timing (eval mode, no grad) ---
    windows, _fc, _extra, control = next(iter(train_loader))
    windows = windows.to(device)
    control = control.to(device)
    ctrl_arg = control if control.shape[-1] > 0 else None
    n_steps = min(window_size, windows.shape[2])
    initial_state = windows[:, :, :n_steps][:, :, 0]
    infer_batch = initial_state.shape[0]

    model.eval()
    with torch.no_grad():
        for _ in range(args.inference_warmup):
            _forward_for_metrics(model, initial_state, n_steps, cfg, control=ctrl_arg)
        _sync(cuda)
        infer_times: list[float] = []
        for _ in range(args.inference_reps):
            t0 = time.perf_counter()
            _forward_for_metrics(model, initial_state, n_steps, cfg, control=ctrl_arg)
            _sync(cuda)
            infer_times.append(time.perf_counter() - t0)
    infer_stats = _stats(infer_times)
    print(f"  Inference: {infer_stats['mean']:.4f} s "
          f"(+/- {infer_stats['std']:.4f}) per batch of {infer_batch} windows x {n_steps} steps")

    # --- training timing (full forward+backward+step epoch) ---
    trainer = Trainer(
        model=model, lr=cfg.lr, device=device,
        checkpoint_dir=cfg.checkpoint_dir, experiment_name=cfg.experiment_name,
        cfg=cfg, use_wandb=False,
    )
    for _ in range(args.warmup_epochs):
        trainer.train_epoch(train_loader, n_steps=window_size, dt=cfg.tr, verbose=False)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    _sync(cuda)
    epoch_times: list[float] = []
    for i in range(args.measure_epochs):
        t0 = time.perf_counter()
        trainer.train_epoch(train_loader, n_steps=window_size, dt=cfg.tr, verbose=False)
        _sync(cuda)
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        print(f"  train epoch {i + 1}/{args.measure_epochs}: {dt:.3f} s")
    epoch_stats = _stats(epoch_times)
    peak_mem_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if cuda else None

    n_train_batches = len(train_loader)
    print(f"  Training: {epoch_stats['mean']:.3f} s/epoch "
          f"(+/- {epoch_stats['std']:.3f}); {n_train_batches} batches/epoch")

    return {
        "model_key": model_key,
        "title": title,
        "checkpoint_loaded": checkpoint_loaded,
        "n_parameters": n_params,
        "n_trainable_parameters": n_trainable,
        "train_s_per_epoch": epoch_stats,
        "train_batches_per_epoch": n_train_batches,
        "train_peak_gpu_memory_gb": peak_mem_gb,
        "inference_s": infer_stats,
        "inference_batch_size": infer_batch,
        "inference_n_steps": n_steps,
        "inference_s_per_window": infer_stats["mean"] / infer_batch,
    }


# ---------------------------------------------------------------------------
# Grid-search benchmark
# ---------------------------------------------------------------------------

def benchmark_grid_search(
    dataset,
    device: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    print(f"\n{'=' * 70}\n  COUPLED HOPF GRID SEARCH\n{'=' * 70}")
    cuda = device.startswith("cuda")
    cfg = HopfConfig()
    cfg.use_wandb = False
    cfg.device = device

    seed_all(cfg.seed)
    omega = compute_omega_from_timeseries(
        dataset.timeseries, dt=dataset.dt, f_lo=cfg.f_lo, f_hi=cfg.f_hi, method="peak",
    )
    train_idx, _, _ = compute_split_indices(
        dataset, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed,
    )
    n_timepoints = min(dataset.n_timepoints, 200)
    train_fc = fisher_batch_average(dataset.fc_matrices[train_idx])
    eval_train_idx = train_idx[:min(cfg.n_simulations, len(train_idx))]
    initial_states = dataset.timeseries[eval_train_idx, :, 0]
    target_ts = dataset.timeseries[eval_train_idx, :, :n_timepoints]

    g_values, a_values, kappa_values = cfg.g_values, cfg.a_values, cfg.kappa_values
    param_grid: dict[str, list] = {"initial_g": g_values, "initial_a": a_values}
    if len(kappa_values) > 1:
        param_grid["initial_kappa"] = kappa_values
    n_combos = 1
    for vals in param_grid.values():
        n_combos *= len(vals)

    model_kwargs: dict[str, object] = {
        "n_rois": dataset.n_rois,
        "structural_connectivity": dataset.fc_mean,
        "omega": omega,
        "noise_sigma": cfg.noise_sigma,
        "learnable_a": False,
        "learnable_g": False,
        "learnable_kappa": False,
        "n_control_dims": dataset.n_control_dims,
    }
    if len(kappa_values) == 1:
        model_kwargs["initial_kappa"] = kappa_values[0]

    eval_kwargs: dict[str, object] = {
        "target_timeseries": target_ts,
        "tr": cfg.tr,
        "fcd_win_sec": cfg.fcd_win_sec,
        "fcd_step_sec": cfg.fcd_step_sec,
    }
    if cfg.denoise_f_lo is not None and cfg.denoise_f_hi is not None:
        eval_kwargs["denoise_f_lo"] = cfg.denoise_f_lo
        eval_kwargs["denoise_f_hi"] = cfg.denoise_f_hi
    loss_kwargs = {
        "tr": cfg.tr,
        "fcd_win_sec": cfg.fcd_win_sec,
        "fcd_step_sec": cfg.fcd_step_sec,
        "fdm_n_pairs": cfg.fdm_n_pairs,
        "fdm_max_lag": cfg.fdm_max_lag,
        "fdm_sigma": cfg.fdm_sigma,
    }

    # Isolated save_dir so the canonical results/grid_search/ is untouched.
    grid = GridSearch(
        param_grid=param_grid, device=device,
        save_dir="results/timing_benchmark/grid_search",
    )

    print(f"  Grid: {n_combos} trials "
          f"(G={len(g_values)} x a={len(a_values)} x kappa={len(kappa_values)})")
    _sync(cuda)
    t0 = time.perf_counter()
    grid.search(
        model_class=CoupledHopfModel,
        model_kwargs=model_kwargs,
        target_fc=train_fc,
        initial_states=initial_states,
        n_timepoints=n_timepoints,
        dt=cfg.tr,
        loss_weights=cfg.loss_weights,
        verbose=False,
        eval_kwargs=eval_kwargs,
        loss_kwargs=loss_kwargs,
    )
    _sync(cuda)
    total_s = time.perf_counter() - t0
    s_per_trial = total_s / n_combos
    print(f"  Grid search: {total_s:.3f} s total, {s_per_trial:.4f} s/trial")

    return {
        "n_combinations": n_combos,
        "total_s": total_s,
        "s_per_trial": s_per_trial,
        "g_values": list(g_values),
        "a_values": list(a_values),
        "kappa_values": list(kappa_values),
        "n_eval_timeseries": int(initial_states.shape[0]),
        "n_timepoints": n_timepoints,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark model runtimes.")
    parser.add_argument("--dataset-type", type=str, default="ts_young")
    parser.add_argument("--data-path", type=str,
                        default="data/ts_young/ts_young_TR0.72.mat")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Unmeasured training epochs before timing.")
    parser.add_argument("--measure-epochs", type=int, default=4,
                        help="Measured training epochs averaged into s/epoch.")
    parser.add_argument("--inference-warmup", type=int, default=5)
    parser.add_argument("--inference-reps", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/timing_benchmark.json")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    import dataclasses
    import json

    args = _build_parser().parse_args(argv)
    ensure_proxy_env()

    base_cfg = HopfConfig()
    base_cfg.use_wandb = False
    base_cfg.dataset_type = args.dataset_type
    base_cfg.data_path = args.data_path
    base_cfg.device = args.device

    device = resolve_device(base_cfg.device)
    seed_all(base_cfg.seed)
    specs = collect_specs(device)
    print("Hardware/software specs:")
    for key, value in specs.items():
        print(f"  {key}: {value}")

    print("\nLoading dataset (shared across all models)...")
    dataset = load_dataset(base_cfg, device)
    window_size = min(base_cfg.window_size, dataset.n_timepoints // 2)
    train_loader, val_loader, _test_inter, _test_intra = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        batch_size=base_cfg.batch_size,
        n_windows_per_epoch=base_cfg.n_windows_per_epoch,
        train_ratio=base_cfg.train_ratio,
        val_ratio=base_cfg.val_ratio,
        seed=base_cfg.seed,
        device=device,
        use_full_timeseries=base_cfg.use_full_timeseries,
    )
    print(f"  window_size={window_size}  train_batches={len(train_loader)}  "
          f"batch_size={base_cfg.batch_size}")

    model_results: dict[str, object] = {}
    for model_key in _MODEL_ORDER:
        model_results[model_key] = benchmark_model(
            model_key, dataset, base_cfg, device,
            train_loader, val_loader, window_size, args,
        )

    grid_result = benchmark_grid_search(dataset, device, args)

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_type": args.dataset_type,
            "data_path": args.data_path,
            "n_subjects": int(dataset.n_subjects),
            "n_rois": int(dataset.n_rois),
            "n_timepoints": int(dataset.n_timepoints),
            "tr": float(base_cfg.tr),
            "window_size": window_size,
            "batch_size": base_cfg.batch_size,
            "n_windows_per_epoch": base_cfg.n_windows_per_epoch,
            "warmup_epochs": args.warmup_epochs,
            "measure_epochs": args.measure_epochs,
            "inference_warmup": args.inference_warmup,
            "inference_reps": args.inference_reps,
        },
        "hardware": specs,
        "models": model_results,
        "grid_search": grid_result,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(f"\nResults written to {out_path}")

    # Clean up empty MetricsStore directories created by Trainer construction.
    import shutil
    shutil.rmtree("results/metrics/timing_benchmark", ignore_errors=True)

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    print(f"{'Model':<16}{'params':>12}{'s/epoch':>14}{'inference s':>16}")
    for key in _MODEL_ORDER:
        r = model_results[key]
        print(f"{r['title']:<16}{r['n_parameters']:>12}"
              f"{r['train_s_per_epoch']['mean']:>14.3f}"
              f"{r['inference_s']['mean']:>16.4f}")
    print(f"Grid search: {grid_result['s_per_trial']:.4f} s/trial "
          f"({grid_result['n_combinations']} trials, {grid_result['total_s']:.2f} s total)")
    return output


if __name__ == "__main__":
    main()
