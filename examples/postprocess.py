#!/usr/bin/env python3
"""
Post-training pipeline entry point.

Consolidates all post-training steps that produce publication artefacts:

1. **update-tables** – patches LaTeX model-comparison tables from a
   metrics JSON produced by ``train_models.py paper``.
2. **compare** – generates FC/timeseries comparison figures for the
   best Hopf and Neural SDE checkpoints.
3. **compare-conditions** *(LSD only)* – simulates all trained models
   under pharmacological conditions (u=0 Placebo, u=1 LSD+Ketanserin,
   u=2 LSD) and produces FC grids, ΔFC plots, bar charts, a metrics
   JSON and a LaTeX table fragment.
4. **pipeline** – runs all three steps above in sequence.

# Run the full post-training pipeline (tables + compare + conditions):
    python examples/postprocess.py pipeline \\
        --dataset-type lsd --lsd-data-dir data/lsd

# Update LaTeX tables from a metrics JSON:
    python examples/postprocess.py update-tables \\
        --metrics results/lsd_paper_metrics_<timestamp>.json

# Compare Hopf vs Neural SDE (figures):
    python examples/postprocess.py compare \\
        --hopf-checkpoint checkpoints/best_hopf_backprop_ts_young.pt \\
        --nsde-checkpoint checkpoints/best_nsde_backprop_ts_young.pt \\
        --data-path data/ts_young/ts_young_TR0.72.mat

# Compare LSD control conditions (u=0, u=1, u=2):
    python examples/postprocess.py compare-conditions \\
        --lsd-data-dir data/lsd
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.cli_args import add_dataset_args
from src.metrics import fisher_batch_average
from src.metrics.metrics_store import MetricsStore
from src.utils.runtime import (
    ensure_proxy_env,
    managed_wandb_run,
    print_section,
    resolve_device,
    seed_all,
    wandb_log,
    wandb_log_figure,
    wandb_summary_update,
)
from src.utils.evaluation import (
    EVAL_METRIC_KEYS,
    as_numeric_metrics,
    evaluate_model_loader_metrics,
    format_metrics_mean_std,
    to_float_metric,
)
from src.utils.visualization import (
    FIGURES_DIR,
    create_comparison_report,
    plot_fc_comparison,
    plot_timeseries,
)


# ===================================================================
# Checkpoint discovery
# ===================================================================

#: Canonical checkpoint patterns per dataset tag.
_BEST_CKPT_PATTERNS: dict[str, list[str]] = {
    "ts_young": [
        "ts_young_hopf_grid.pt",
        "ts_young_hopf.pt",
        "ts_young_nsde.pt",
        "ts_young_hybrid_hopf.pt",
        "ts_young_gnn_hopf.pt",
        "ts_young_hybrid_neural.pt",
    ],
    "lsd": [
        "lsd_hopf_grid.pt",
        "lsd_hopf.pt",
        "lsd_nsde.pt",
        "lsd_hybrid_hopf.pt",
        "lsd_gnn_hopf.pt",
        "lsd_hybrid_neural.pt",
    ],
}


def _discover_checkpoints(
    dataset_type: str,
    checkpoint_dir: str = "checkpoints",
    explicit_checkpoints: dict[str, str] | None = None,
) -> list[str]:
    """Return a list of existing checkpoint paths for the given dataset.

    *explicit_checkpoints* maps model tags (e.g. ``"hopf_grid"``) to paths and
    takes precedence over auto-discovery.
    """
    ckpt_dir = Path(checkpoint_dir)
    found: list[str] = []

    if explicit_checkpoints:
        for _tag, path in explicit_checkpoints.items():
            if path and Path(path).exists():
                found.append(str(path))

    if not found:
        patterns = _BEST_CKPT_PATTERNS.get(dataset_type, [])
        for name in patterns:
            p = ckpt_dir / name
            if p.exists():
                found.append(str(p))

    if not found and ckpt_dir.exists():
        globbed = sorted(ckpt_dir.glob(f"*{dataset_type}*.pt"))
        found = [str(p) for p in globbed]

    return found


# ===================================================================
# 1. update_paper_tables
# ===================================================================

METRIC_COLS: list[tuple[str, str]] = [
    ("fc_correlation",          r"FC corr $\uparrow$"),
    ("fc_mse",                  r"FC MSE $\downarrow$"),
    ("phase_fc_correlation",    r"phFC corr $\uparrow$"),
    ("fcd_ks",                  r"FCD KS $\downarrow$"),
    ("phfcd_ks",                r"phFCD KS $\downarrow$"),
    ("metastability_diff",      r"Meta $|\Delta|$ $\downarrow$"),
    ("temporal_correlation",    r"TS corr $\uparrow$"),
    ("power_spectrum_distance", r"TS PSD $\downarrow$"),
    ("autocorr_distance",       r"Autocorr $\downarrow$"),
]

MODEL_DISPLAY: dict[str, str] = {
    "hopf_grid":     "Coupled Hopf (grid search)",
    "hopf":          "Coupled Hopf",
    "nsde":          "Neural SDE",
    "hybrid_hopf":   "Hybrid Hopf",
    "gnn_hopf":      "GNN Hopf",
    "hybrid_neural": "Hybrid Hopf+Neural",
}

HIGHER_BETTER: dict[str, bool] = {
    "fc_correlation": True,
    "fc_mse": False,
    "phase_fc_correlation": True,
    "fcd_ks": False,
    "phfcd_ks": False,
    "metastability_diff": False,
    "temporal_correlation": True,
    "power_spectrum_distance": False,
    "autocorr_distance": False,
}

TABLE_TARGET_BY_LABEL: dict[str, Path] = {
    "tab:model_comparison": Path("paper/sections/ts_young_model_table.tex"),
    "tab:lsd_model_comparison": Path("paper/sections/lsd_model_table.tex"),
}


def _fmt_table(val: float | None, key: str, std: float | None = None) -> str:
    """Format a metric value, optionally with std in smaller font."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r"\textemdash"
    if key == "metastability_diff" and abs(val) < 0.01:
        exp = math.floor(math.log10(max(abs(val), 1e-12)))
        mantissa = val / 10**exp
        base = rf"${mantissa:.1f}\times 10^{{{exp}}}$"
        if std is not None and not math.isnan(std):
            std_exp = math.floor(math.log10(max(abs(std), 1e-12)))
            std_mantissa = std / 10**std_exp
            base += rf" {{\scriptsize $\pm$ {std_mantissa:.1f}$\times 10^{{{std_exp}}}$}}"
        return base
    base = f"{val:.3f}"
    if std is not None and not math.isnan(std):
        base += rf" {{\scriptsize $\pm$ {std:.3f}}}"
    return base


def _bold(s: str) -> str:
    return rf"\bf{{{s}}}"


def _active_metric_cols(metrics: dict[str, dict[str, float]]) -> list[tuple[str, str]]:
    """Return metric columns that are present in the provided metrics."""
    active: list[tuple[str, str]] = []
    for metric_key, header in METRIC_COLS:
        for model_metrics in metrics.values():
            val = model_metrics.get(metric_key)
            if val is None:
                continue
            try:
                if math.isnan(float(val)):
                    continue
            except (TypeError, ValueError):
                continue
            active.append((metric_key, header))
            break
    return active


def _build_table_body(
    metrics: dict[str, dict[str, float]],
    metric_cols: list[tuple[str, str]],
) -> str:
    """Return LaTeX table rows (\\midrule … \\bottomrule exclusive)."""
    col_keys = [mk for mk, _ in metric_cols]

    best: dict[str, float] = {}
    for mk in col_keys:
        vals = [
            m.get(mk) for m in metrics.values()
            if m.get(mk) is not None and not math.isnan(m.get(mk, float("nan")))
        ]
        if not vals:
            continue
        best[mk] = max(vals) if HIGHER_BETTER.get(mk, True) else min(vals)

    rows: list[str] = []
    for model_key, display_name in MODEL_DISPLAY.items():
        if model_key not in metrics:
            continue
        m = metrics[model_key]
        cells = [display_name]
        for mk in col_keys:
            val = m.get(mk)
            std = m.get(f"{mk}_std")
            formatted = _fmt_table(val, mk, std)
            if val is not None and not math.isnan(float(val)) and best.get(mk) is not None:
                if abs(float(val) - best[mk]) < 1e-6:
                    formatted = _bold(formatted)
            cells.append(formatted)
        rows.append("    " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def _build_tabular(metrics: dict[str, dict[str, float]]) -> str:
    """Build a complete LaTeX tabular fragment matching the active metrics."""
    metric_cols = _active_metric_cols(metrics)
    body = _build_table_body(metrics, metric_cols)
    header = "Model"
    if metric_cols:
        header += " & " + " & ".join(label for _, label in metric_cols)
    header += r" \\"
    col_spec = "l" + (" " + " ".join("c" for _ in metric_cols) if metric_cols else "")
    return (
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\toprule" "\n"
        + header + "\n"
        + r"\midrule" "\n"
        + body + "\n"
        + r"\bottomrule" "\n"
        + r"\end{tabular}" "\n"
    )


def _replace_first_tabular(tex: str, new_tabular: str) -> str:
    """Replace the first tabular environment, preserving surrounding TeX."""
    tabular_pattern = re.compile(
        r"\\begin\{tabular\}\{.*?\}.*?\\end\{tabular\}\n?",
        re.DOTALL,
    )
    patched, n_subs = tabular_pattern.subn(new_tabular, tex, count=1)
    if n_subs == 0:
        print("  Warning: could not find a tabular environment in target file.", file=sys.stderr)
    return patched


def _aggregate_individual_metrics(
    results_dir: Path,
    dataset_type: str | None = None,
) -> tuple[str | None, dict | None]:
    """Aggregate per-model metrics files written by individual training runs.

    Each file has the shape::

        {"dataset_type": "ts_young", "model": "nsde",
         "metrics": {"test_inter": {<metric_key>: <value>, ...},
                     "test_intra": {<metric_key>: <value>, ...}}}

    If *dataset_type* is given, only files whose ``dataset_type`` field matches
    are included (prevents ts_young files from overwriting lsd files when both
    live in the same results directory).

    Returns the inferred dataset_type and a combined metrics dict keyed by
    model name (using test_inter metrics), or ``(None, None)`` when no
    matching files are found.
    """
    combined: dict[str, dict] = {}
    inferred_type: str | None = None
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        model_key = data.get("model")
        test_inter = data.get("metrics", {}).get("test_inter")
        if not model_key or not test_inter:
            continue
        file_dataset_type = data.get("dataset_type")
        if dataset_type is not None and file_dataset_type != dataset_type:
            continue
        combined[model_key] = test_inter
        if inferred_type is None:
            inferred_type = file_dataset_type

    return inferred_type or dataset_type, combined or None


def run_update_tables(
    metrics_json: str | Path | None = None,
    tex_target: str | Path | None = None,
    table_label: str | None = None,
    dry_run: bool = False,
) -> None:
    """Patch paper LaTeX tables with fresh metrics."""
    if metrics_json is None:
        # Infer expected dataset_type from table_label so we only pick up
        # matching JSON files (avoids ts_young files overwriting lsd ones).
        expected_dataset_type: str | None = None
        if table_label == "tab:lsd_model_comparison":
            expected_dataset_type = "lsd"
        elif table_label == "tab:model_comparison":
            expected_dataset_type = "ts_young"
        dataset_type, metrics = _aggregate_individual_metrics(
            Path("results"), dataset_type=expected_dataset_type
        )
        if not metrics:
            print("No metrics found in results/. Please run training first.")
            return
        dataset_type = dataset_type or "ts_young"
        print(f"Aggregated metrics for {len(metrics)} models (dataset: {dataset_type})")
    else:
        metrics_json = Path(metrics_json)
        if not metrics_json.exists():
            print(f"Metrics file not found: {metrics_json}")
            return
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        metrics = data.get("metrics", data)
        dataset_type = data.get("dataset_type") or "ts_young"
        print(f"Loaded metrics for: {list(metrics.keys())} (dataset: {dataset_type})")

    if table_label:
        pass
    elif dataset_type == "lsd":
        table_label = "tab:lsd_model_comparison"
    else:
        table_label = "tab:model_comparison"

    print(f"Target table: {table_label}")

    tex_path: Path
    if tex_target is not None:
        tex_path = Path(tex_target)
    else:
        tex_path = TABLE_TARGET_BY_LABEL.get(table_label, Path("paper/sections/03_results.tex"))
    print(f"Target file: {tex_path}")

    tabular = _build_tabular(metrics)
    print("\nGenerated table:")
    print(tabular)

    if dry_run:
        return

    if not tex_path.exists():
        print(f"LaTeX file not found: {tex_path}")
        return

    original = tex_path.read_text(encoding="utf-8")
    stripped_original = original.strip()
    if stripped_original.startswith(r"\begin{tabular}") and stripped_original.endswith(r"\end{tabular}"):
        patched = tabular
    else:
        patched = _replace_first_tabular(original, tabular)
    if patched == original:
        print("No changes made (table not found or already up to date).")
    else:
        tex_path.write_text(patched, encoding="utf-8")
        print(f"Updated {tex_path}")


# ===================================================================
# 2. compare_models
# ===================================================================

def _load_all_checkpoints(
    checkpoints: list[str],
    n_rois: int,
    device: str,
) -> dict[str, Any]:
    """Load all model checkpoints, returning {label: model}."""
    from src.models import load_model_from_checkpoint

    print_section("Loading Models")
    models: dict[str, Any] = {}
    for ckpt_path in checkpoints:
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  Warning: checkpoint not found: {p}")
            continue
        try:
            model, mtype, _ = load_model_from_checkpoint(str(p), device=device)
            label = _short_model_label(p.stem)
            if model.n_rois != n_rois:
                print(f"  Warning: {label} ROI mismatch ({model.n_rois} != {n_rois}); skipping.")
            else:
                models[label] = model
                print(f"  Loaded '{label}' ({mtype}) from {p.name}")
        except Exception as exc:
            print(f"  Warning: failed to load {p.name}: {exc}")
    return models


_as_numeric = as_numeric_metrics  # backward-compat alias used below


def _fc_corr_and_mse(fc_pred: "torch.Tensor", fc_target: "torch.Tensor"):
    """Upper-triangle Pearson correlation and MSE between two FC matrices."""
    from src.metrics._utils import upper_tri_vec
    p = upper_tri_vec(fc_pred, k=1)
    t = upper_tri_vec(fc_target, k=1)
    pc, tc = p - p.mean(dim=-1, keepdim=True), t - t.mean(dim=-1, keepdim=True)
    corr = ((pc * tc).sum(dim=-1) / (
        (pc**2).sum(dim=-1).sqrt() * (tc**2).sum(dim=-1).sqrt() + 1e-8
    )).mean().item()
    mse = ((p - t) ** 2).mean().item()
    return corr, mse


def _generate_comparison_figures(
    models: dict,
    dataset,
    target_fc,
    n_timepoints: int,
    use_wandb: bool = False,
    save_dir: Path | None = None,
):
    """Generate all comparison figures (FC, timeseries, report)."""
    print_section("Generating Comparison Figures")
    out_dir = save_dir if save_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Any] = {}
    n_paths = min(10, dataset.timeseries.shape[0])
    initial_state = dataset.timeseries[:n_paths, :, 0]

    for name, model in models.items():
        with torch.no_grad():
            ts = model.forward(initial_state=initial_state, n_steps=n_timepoints)
            fc_pred = model.compute_fc(ts)  # (n_paths, n_rois, n_rois) — fisher_batch_average applied in plot_fc_comparison

        stem = f"{name.lower().replace(' ', '_')}_fc_comparison"
        fig = plot_fc_comparison(
            fc_pred, target_fc,
            title=f"{name} - FC Comparison",
            save_path=str(out_dir / f"{stem}.pdf"),
            use_pdf=True,
        )
        figures[f"{name}_fc"] = fig
        wandb_log_figure(f"figures/{name}_fc_comparison", fig, use_wandb=use_wandb)
        plt.close()

        stem = f"{name.lower().replace(' ', '_')}_timeseries"
        fig = plot_timeseries(
            ts[0].real, n_rois=5,
            title=f"{name} - Simulated Timeseries",
            save_path=str(out_dir / f"{stem}.pdf"),
            use_pdf=True,
        )
        figures[f"{name}_ts"] = fig
        wandb_log_figure(f"figures/{name}_timeseries", fig, use_wandb=use_wandb)
        plt.close()

    fig = create_comparison_report(
        models=models, target_fc=target_fc, n_timepoints=n_timepoints,
        initial_state=initial_state, save_dir=str(out_dir), use_pdf=True,
    )
    figures["report"] = fig
    wandb_log_figure("figures/comparison_report", fig, use_wandb=use_wandb)
    plt.close()

    print(f"All figures saved to {out_dir}")
    return figures


def _save_comparison_results(
    test_inter_results: dict[str, dict[str, float]],
    test_intra_results: dict[str, dict[str, float]],
    save_path: str,
) -> None:
    """Save comparison results (both splits) to JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "test_inter": {
            mn: {k: float(v) for k, v in metrics.items()}
            for mn, metrics in test_inter_results.items()
        },
        "test_intra": {
            mn: {k: float(v) for k, v in metrics.items()}
            for mn, metrics in test_intra_results.items()
        },
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {save_path}")


def _print_comparison_table(
    results: dict[str, dict[str, float]],
    split_label: str,
) -> None:
    """Print a full metric comparison table for one test split."""
    metric_keys = list(EVAL_METRIC_KEYS)
    print(f"\n{'=' * 60}")
    print(f"  {split_label}")
    print(f"{'=' * 60}")

    col_w = 14
    header = f"{'Model':<25}" + "".join(f"{k:>{col_w}}" for k in metric_keys)
    print(header)
    print("-" * len(header))

    for name, m in results.items():
        numeric = _as_numeric(m)
        cells = f"{name:<25}"
        for k in metric_keys:
            val = numeric.get(k, float("nan"))
            std = numeric.get(f"{k}_std")
            if std is not None and not math.isnan(std):
                cells += f"{val:>{col_w}.4f}"
            elif math.isnan(val):
                cells += f"{'nan':>{col_w}}"
            else:
                cells += f"{val:>{col_w}.4f}"
        print(cells)

    # Print best model by fc_correlation
    fc_vals = {n: _as_numeric(m).get("fc_correlation", float("nan")) for n, m in results.items()}
    fc_vals = {n: v for n, v in fc_vals.items() if not math.isnan(v)}
    if fc_vals:
        best = max(fc_vals, key=fc_vals.get)
        print(f"\n  Best model (by fc_correlation): {best} = {fc_vals[best]:.4f}")


def run_compare_models(
    dataset_type: str = "ts_young",
    checkpoint_dir: str = "checkpoints",
    checkpoints: list[str] | None = None,
    device: str = "auto",
    no_wandb: bool = True,
    wandb_project: str = "neuroscience-control",
    run_name: str = "model_comparison",
    output_path: str = "results/comparison_results.json",
    save_dir: Path | None = None,
    skip_figures: bool = False,
    # Dataset config overrides (forwarded to load_dataset via cfg)
    data_path: str | None = None,
    lsd_data_dir: str | None = None,
) -> dict | None:
    """Compare trained models using the same evaluation as train_models.py.

    Loads data with ``load_dataset`` + ``create_data_loaders`` and evaluates
    every discovered checkpoint with ``evaluate_model_loader_metrics`` on both
    the inter-patient and intra-patient test loaders.
    """
    from src.dataset import load_dataset, create_data_loaders
    from src.training import HopfConfig

    print_section("MODEL COMPARISON")
    ensure_proxy_env()
    device = resolve_device(device)
    seed_all(42)
    use_wandb = not no_wandb
    figures_dir = save_dir if save_dir is not None else FIGURES_DIR / dataset_type

    # Build a config for dataset loading (same defaults as training)
    cfg = HopfConfig()
    cfg.dataset_type = dataset_type
    if data_path is not None:
        cfg.data_path = data_path
    if lsd_data_dir is not None:
        cfg.lsd_data_dir = lsd_data_dir
    cfg.use_wandb = use_wandb

    print_section("STEP 1: Loading Data")
    dataset = load_dataset(cfg, device)
    window_size = min(cfg.window_size, dataset.n_timepoints // 2)
    _, _, test_inter_loader, test_intra_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        batch_size=cfg.batch_size,
        n_windows_per_epoch=cfg.n_windows_per_epoch,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        device=device,
    )
    print(f"  window_size={window_size}  "
          f"test_inter={len(test_inter_loader)}  test_intra={len(test_intra_loader)}")

    # Discover and load checkpoints
    if not checkpoints:
        checkpoints = _discover_checkpoints(dataset_type, checkpoint_dir)
    if not checkpoints:
        print("Error: No checkpoints found. Please check checkpoint paths.")
        return None

    models = _load_all_checkpoints(checkpoints, n_rois=dataset.n_rois, device=device)
    if not models:
        print("Error: No models could be loaded. Please check checkpoint paths.")
        return None

    print_section("STEP 2: Evaluating Models")
    test_inter_results: dict[str, dict[str, float]] = {}
    test_intra_results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        print(f"\n  Evaluating {name}...")
        inter_metrics = evaluate_model_loader_metrics(
            model, test_inter_loader, cfg, n_steps=window_size, return_std=True,
        )
        intra_metrics = evaluate_model_loader_metrics(
            model, test_intra_loader, cfg, n_steps=window_size, return_std=True,
        )
        test_inter_results[name] = _as_numeric(inter_metrics)
        test_intra_results[name] = _as_numeric(intra_metrics)
        print(f"    Inter: {format_metrics_mean_std(inter_metrics)}")
        print(f"    Intra: {format_metrics_mean_std(intra_metrics)}")

    with managed_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project,
        run_name=run_name,
        tags=["comparison", "evaluation"],
    ):
        for model_name, metrics in test_inter_results.items():
            wandb_log({f"test_inter/{model_name}/{k}": v for k, v in metrics.items()}, use_wandb=use_wandb)
        for model_name, metrics in test_intra_results.items():
            wandb_log({f"test_intra/{model_name}/{k}": v for k, v in metrics.items()}, use_wandb=use_wandb)

        if not skip_figures:
            _generate_comparison_figures(
                models, dataset, dataset.fc_matrices,
                min(dataset.n_timepoints, 200),
                use_wandb=use_wandb, save_dir=figures_dir,
            )

    _save_comparison_results(test_inter_results, test_intra_results, output_path)
    _print_comparison_table(test_inter_results, "INTER-PATIENT TEST SET")
    _print_comparison_table(test_intra_results, "INTRA-PATIENT TEST SET")

    print_section("COMPARISON COMPLETED SUCCESSFULLY")
    print(f"\nResults saved to: {output_path}")
    print(f"Figures saved to: {figures_dir}")
    return {"test_inter": test_inter_results, "test_intra": test_intra_results}


# ===================================================================
# 3. compare_control_conditions
# ===================================================================

_MODEL_TAGS = [
    ("gnn_hopf", "GNN Hopf"),
    ("hybrid_neural", "Hybrid+Neural"),
    ("hybrid_hopf", "Hybrid Hopf"),
    ("nsde", "Neural SDE"),
    ("hopf", "Hopf"),
]


def _short_model_label(stem: str) -> str:
    """Extract a concise model tag from a checkpoint file stem."""
    lower = stem.lower()
    for tag, label in _MODEL_TAGS:
        if tag in lower:
            grid_suffix = " (Grid)" if "grid" in lower else ""
            return f"{label}{grid_suffix}"
    return stem


CONDITION_LABELS: dict[int, str] = {
    0: "Placebo (u=0)",
    1: "LSD+Ketanserin (u=1)",
    2: "LSD (u=2)",
}

CONDITION_COLORS: dict[int, str] = {
    0: "#4878CF",
    1: "#D65F5F",
    2: "#6ACC65",
}


def _simulate_fc(model, initial_states, ctrl_val: int, n_steps: int, dt: float, device: str) -> torch.Tensor:
    """Simulate *model* with constant control u=ctrl_val and return mean FC."""
    n_paths = initial_states.shape[0]
    if getattr(model, "n_control_dims", 0) > 0:
        ctrl = torch.full((n_paths, 1), float(ctrl_val), dtype=torch.float32, device=device)
    else:
        ctrl = None

    fwd_kwargs: dict[str, Any] = dict(initial_state=initial_states, n_steps=n_steps, dt=dt)
    if ctrl is not None:
        fwd_kwargs["control"] = ctrl

    with torch.no_grad():
        ts = model.forward(**fwd_kwargs)
        fc_per_subject = model.compute_fc(ts)
    return fisher_batch_average(fc_per_subject)


def _fc_to_numpy(fc: torch.Tensor) -> np.ndarray:
    arr = fc.detach().cpu().numpy()
    return arr.real if np.iscomplexobj(arr) else arr


def _plot_fc_grid(fc_per_model, empirical_fc_per_ctrl, *, save_path: Path) -> None:
    model_names = list(fc_per_model.keys())
    ctrl_vals = sorted(CONDITION_LABELS.keys())
    n_models, n_conds = len(model_names), len(ctrl_vals)
    n_rows = n_models + 1
    fig, axes = plt.subplots(n_rows, n_conds, figsize=(4 * n_conds, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_conds == 1:
        axes = axes[:, np.newaxis]
    vmin, vmax = -1.0, 1.0

    for col, cv in enumerate(ctrl_vals):
        ax = axes[0, col]
        emp_fc = empirical_fc_per_ctrl.get(cv)
        if emp_fc is not None:
            im = ax.imshow(_fc_to_numpy(emp_fc), cmap="coolwarm", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(CONDITION_LABELS[cv], fontsize=9)
        if col == 0:
            ax.set_ylabel("Empirical", fontsize=9)

    for row, mname in enumerate(model_names):
        for col, cv in enumerate(ctrl_vals):
            ax = axes[row + 1, col]
            fc = fc_per_model[mname].get(cv)
            if fc is not None:
                im = ax.imshow(_fc_to_numpy(fc), cmap="coolwarm", vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            if col == 0:
                ax.set_ylabel(mname, fontsize=9)

    plt.suptitle("FC under pharmacological conditions (rows=model, cols=condition)", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved FC grid → {save_path}")


def _plot_fc_diff(fc_per_ctrl, model_name: str, *, baseline_ctrl: int = 0, save_path: Path) -> None:
    ctrl_vals = [cv for cv in sorted(CONDITION_LABELS.keys()) if cv != baseline_ctrl]
    if not ctrl_vals:
        return
    fig, axes = plt.subplots(1, len(ctrl_vals), figsize=(5 * len(ctrl_vals), 4.5))
    if len(ctrl_vals) == 1:
        axes = [axes]
    baseline_fc = fc_per_ctrl.get(baseline_ctrl)
    if baseline_fc is None:
        plt.close(fig)
        return
    base_np = _fc_to_numpy(baseline_fc)
    for ax, cv in zip(axes, ctrl_vals):
        fc = fc_per_ctrl.get(cv)
        if fc is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        diff = _fc_to_numpy(fc) - base_np
        vmax_d = max(abs(diff.min()), abs(diff.max()), 1e-6)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"ΔFC: {CONDITION_LABELS[cv]} − {CONDITION_LABELS[baseline_ctrl]}", fontsize=9)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
    plt.suptitle(f"{model_name} — FC change under drug conditions", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved ΔFC plot → {save_path}")


def _plot_fc_corr_bar(metrics, *, save_path: Path) -> None:
    model_names = list(metrics.keys())
    ctrl_vals = sorted(CONDITION_LABELS.keys())
    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(model_names)), 4))
    for i, cv in enumerate(ctrl_vals):
        corrs = [metrics[m].get(cv, {}).get("fc_corr", float("nan")) for m in model_names]
        ax.bar(x + (i - 1) * width, corrs, width, label=CONDITION_LABELS[cv], color=CONDITION_COLORS[cv], alpha=0.85)
    ax.set_ylabel("FC Pearson correlation ↑")
    ax.set_title("FC fit per model and pharmacological condition")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved FC corr bar chart → {save_path}")


def _fmt_cond(val: float | None, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def _build_lsd_latex_table(metrics) -> str:
    ctrl_vals = sorted(CONDITION_LABELS.keys())
    short = {0: "Placebo", 1: "LSD+Ket.", 2: "LSD"}
    col_header = " & ".join(
        [f"FC corr $\\uparrow$ ({short[cv]}) & FC MSE $\\downarrow$ ({short[cv]})" for cv in ctrl_vals]
    )
    header = f"Model & {col_header} \\\\\n\\midrule\n"
    body = ""
    for mname, cond_metrics in metrics.items():
        row = mname.replace("_", " ")
        for cv in ctrl_vals:
            m = cond_metrics.get(cv, {})
            row += f" & {_fmt_cond(m.get('fc_corr'))} & {_fmt_cond(m.get('fc_mse'))}"
        body += row + " \\\\\n"
    col_spec = "l" + "".join([" c c" for _ in ctrl_vals])
    return (
        "\\begin{tabular}{" + col_spec + "}\n\\toprule\n"
        + header + body + "\\bottomrule\n\\end{tabular}\n"
    )


def run_compare_conditions(
    checkpoints: list[str] | None = None,
    lsd_data_dir: str = "data/lsd",
    n_steps: int = 240,
    n_paths: int = 10,
    device: str | None = None,
    model_labels: list[str] | None = None,
    ctrl_values: list[int] | None = None,
    output_dir: str | None = None,
) -> None:
    """Compare model-simulated FC under different LSD control conditions."""
    from src.dataset import NeuroscienceDataset
    from src.models import load_model_from_checkpoint

    device = resolve_device(device)
    seed_all(42)

    ctrl_values = ctrl_values or sorted(CONDITION_LABELS.keys())
    figures_dir = Path(output_dir) if output_dir else FIGURES_DIR
    results_dir = Path("results")
    latex_dir = Path("paper/sections")

    print_section("Loading LSD dataset")
    dataset = NeuroscienceDataset.from_lsd(data_dir=lsd_data_dir, normalize=True, device=device)
    print(f"  subjects={dataset.n_subjects}  ROIs={dataset.n_rois}  T={dataset.n_timepoints}")

    empirical_fc_per_ctrl: dict[int, torch.Tensor] = {}
    for cv in ctrl_values:
        mask = dataset.control[:, 0] == cv
        if mask.sum() == 0:
            print(f"  Warning: no subjects for ctrl={cv}, skipping.")
            continue
        empirical_fc_per_ctrl[cv] = fisher_batch_average(dataset.fc_matrices[mask])
        print(f"  Empirical FC ctrl={cv}: {mask.sum().item()} subjects")

    n_p = min(n_paths, dataset.n_subjects)
    initial_states = dataset.timeseries[:n_p, :, 0]

    print_section("Loading model checkpoints")
    if not checkpoints:
        ckpt_dir = Path("checkpoints")
        found = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        if not found:
            print("No checkpoints specified or found. Exiting.")
            return
        checkpoints = [str(p) for p in found]
        print(f"  Auto-discovered {len(checkpoints)} checkpoint(s).")

    labels = model_labels or [_short_model_label(Path(c).stem) for c in checkpoints]
    if len(labels) < len(checkpoints):
        labels += [_short_model_label(Path(c).stem) for c in checkpoints[len(labels):]]

    models: dict[str, Any] = {}
    for ckpt_path, label in zip(checkpoints, labels):
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  Warning: checkpoint not found: {p}")
            continue
        try:
            model, mtype, _ = load_model_from_checkpoint(str(p), device=device)
            models[label] = model
            print(f"  Loaded '{label}' ({mtype}) from {p.name}")
        except Exception as exc:
            print(f"  Warning: failed to load {p.name}: {exc}")

    if not models:
        print("No models loaded. Exiting.")
        return

    print_section("Simulating FC per condition")
    fc_per_model: dict[str, dict[int, torch.Tensor]] = {}
    cond_metrics: dict[str, dict[int, dict[str, float]]] = {}

    for mname, model in models.items():
        fc_per_model[mname] = {}
        cond_metrics[mname] = {}
        for cv in ctrl_values:
            if cv not in empirical_fc_per_ctrl:
                continue
            print(f"  {mname}: simulating ctrl={cv} …", end=" ", flush=True)
            fc_sim = _simulate_fc(model, initial_states, cv, n_steps=n_steps, dt=dataset.dt, device=device)
            fc_emp = empirical_fc_per_ctrl[cv]
            corr, mse = _fc_corr_and_mse(fc_sim.unsqueeze(0), fc_emp.unsqueeze(0))
            fc_per_model[mname][cv] = fc_sim
            cond_metrics[mname][cv] = {"fc_corr": corr, "fc_mse": mse}
            print(f"FC corr={corr:.3f}  MSE={mse:.4f}")

    print_section("Generating figures")
    _plot_fc_grid(fc_per_model, empirical_fc_per_ctrl, save_path=figures_dir / "lsd" / "control_fc_grid.pdf")
    for mname in models:
        _plot_fc_diff(
            fc_per_model[mname], model_name=mname, baseline_ctrl=0,
            save_path=figures_dir / "lsd" / f"fc_delta_{mname.lower().replace(' ', '_')}.pdf",
        )
    _plot_fc_diff(
        empirical_fc_per_ctrl, model_name="Empirical", baseline_ctrl=0,
        save_path=figures_dir / "lsd" / "fc_delta_empirical.pdf",
    )
    _plot_fc_corr_bar(cond_metrics, save_path=figures_dir / "lsd" / "control_fc_corr.pdf")

    print_section("Saving metrics and LaTeX table")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "lsd_control_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(cond_metrics, fh, indent=2, sort_keys=True)
    print(f"  Metrics → {metrics_path}")

    latex = _build_lsd_latex_table(cond_metrics)
    latex_dir.mkdir(parents=True, exist_ok=True)
    latex_path = latex_dir / "lsd_control_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"  LaTeX table → {latex_path}")

    print_section("Results summary")
    ctrl_short = {0: "Placebo", 1: "LSD+Ket.", 2: "LSD"}
    header = f"{'Model':<35}" + "".join(
        f"  {ctrl_short[cv]:>14}" for cv in ctrl_values if cv in empirical_fc_per_ctrl
    )
    print(header + "  (FC corr ↑)")
    print("-" * len(header))
    for mname in models:
        row = f"{mname:<35}"
        for cv in ctrl_values:
            if cv not in empirical_fc_per_ctrl:
                continue
            corr = cond_metrics[mname].get(cv, {}).get("fc_corr", float("nan"))
            row += f"  {corr:>14.4f}"
        print(row)


# ===================================================================
# 4. plot_metrics_over_epochs
# ===================================================================

#: Metrics to include in the epoch-curve figure (key, y-axis label).
_EPOCH_METRICS: list[tuple[str, str]] = [
    ("loss",                  "Loss"),
    ("fc_correlation",        "FC Corr ↑"),
    ("fc_mse",                "FC MSE ↓"),
    ("fcd_ks",                "FCD KS ↓"),
    ("phase_fc_correlation",  "phFC Corr ↑"),
    ("phfcd_ks",              "phFCD KS ↓"),
    ("metastability_diff",    "Meta |Δ| ↓"),
    ("temporal_correlation",  "TS Corr ↑"),
    ("power_spectrum_distance", "TS PSD ↓"),
    ("autocorr_distance",     "Autocorr ↓"),
]


def _plot_metrics_epochs(
    dataset_type: str,
    metrics_dir: str = "results/metrics",
    save_dir: Path | None = None,
) -> None:
    """Load the latest metrics JSON for each model matching *dataset_type*
    and plot every key metric over epochs (log y-axis where applicable).
    """
    metrics_root = Path(metrics_dir)
    out_dir = save_dir if save_dir is not None else FIGURES_DIR / dataset_type
    out_dir.mkdir(parents=True, exist_ok=True)

    model_stores: dict[str, MetricsStore] = {}
    for subdir in sorted(metrics_root.iterdir()):
        if not subdir.is_dir() or dataset_type not in subdir.name:
            continue
        jsons = sorted(subdir.glob("*.json"))
        if not jsons:
            continue
        latest = jsons[-1]
        try:
            store = MetricsStore.load(str(latest))
        except Exception as exc:
            print(f"  Warning: could not load {latest}: {exc}")
            continue
        label = _short_model_label(subdir.name)
        model_stores[label] = store
        print(f"  Loaded metrics for '{label}' from {latest.name} "
              f"({len(store.val_metrics)} epochs)")

    if not model_stores:
        print(f"  No metrics found for dataset_type='{dataset_type}' in {metrics_root}")
        return

    n_metrics = len(_EPOCH_METRICS)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten()

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax_idx, (mkey, mlabel) in enumerate(_EPOCH_METRICS):
        ax = axes_flat[ax_idx]
        has_data = False
        all_pos = True
        for ci, (label, store) in enumerate(model_stores.items()):
            val_vals = store.get_metric_history(mkey, "val")
            if not val_vals:
                continue
            epochs = [e["epoch"] for e in store.val_metrics if mkey in e]
            clean = [(ep, v) for ep, v in zip(epochs, val_vals) if v is not None and not math.isnan(v)]
            if not clean:
                continue
            ep_arr, v_arr = zip(*clean)
            ax.plot(ep_arr, v_arr, label=label, color=colors[ci % len(colors)], linewidth=1.5)
            has_data = True
            if any(v <= 0 for v in v_arr):
                all_pos = False

        ax.set_xlabel("Epoch")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        if has_data and all_pos:
            ax.set_yscale("log")

    for ax_idx in range(n_metrics, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.suptitle(f"Validation metrics over epochs — {dataset_type}", fontsize=13)
    plt.tight_layout()
    save_path = out_dir / "metrics_over_epochs.pdf"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved metrics over epochs → {save_path}")


def run_plot_metrics(
    dataset_type: str = "ts_young",
    metrics_dir: str = "results/metrics",
    save_dir: Path | None = None,
) -> None:
    """Entry point for the ``plot-metrics`` CLI sub-command."""
    print_section("PLOT METRICS OVER EPOCHS")
    _plot_metrics_epochs(dataset_type=dataset_type, metrics_dir=metrics_dir, save_dir=save_dir)


# ===================================================================
# Orchestrator: pipeline
# ===================================================================

def run_paper_pipeline(
    metrics_json: str | Path | None = None,
    dataset_type: str = "ts_young",
    device: str | None = None,
    no_wandb: bool = True,
    checkpoint_dir: str = "checkpoints",
    explicit_checkpoints: dict[str, str] | None = None,
    data_path: str | None = None,
    lsd_data_dir: str | None = None,
    output_path: str = "results/comparison_results.json",
    n_steps: int = 240,
    n_paths: int = 10,
) -> dict[str, Any]:
    """Run the full post-training paper pipeline."""
    print_section("POST-TRAINING PAPER PIPELINE")
    results: dict[str, Any] = {}

    if dataset_type == "lsd":
        table_label = "tab:lsd_model_comparison"
    else:
        table_label = "tab:model_comparison"
    dataset_figures_dir = FIGURES_DIR / dataset_type

    print_section("Step 1/3: Updating LaTeX tables")
    try:
        run_update_tables(metrics_json=metrics_json, table_label=table_label)
        results["update_tables"] = "ok"
    except Exception as exc:
        print(f"  Warning: update_paper_tables failed: {exc}")
        results["update_tables"] = f"error: {exc}"

    print_section("Step 2/3: Comparing models (inter + intra patient)")
    checkpoints = _discover_checkpoints(dataset_type, checkpoint_dir, explicit_checkpoints)

    if checkpoints:
        try:
            run_compare_models(
                dataset_type=dataset_type,
                checkpoint_dir=checkpoint_dir,
                checkpoints=checkpoints,
                device=device or "auto",
                no_wandb=no_wandb,
                output_path=output_path,
                save_dir=dataset_figures_dir,
                data_path=data_path,
                lsd_data_dir=lsd_data_dir,
            )
            results["compare_models"] = "ok"
        except Exception as exc:
            print(f"  Warning: compare_models failed: {exc}")
            results["compare_models"] = f"error: {exc}"
    else:
        print("  Skipped: no checkpoints found.")
        results["compare_models"] = "skipped"

    if dataset_type == "lsd":
        print_section("Step 3/3: Comparing control conditions (LSD)")
        if checkpoints:
            try:
                run_compare_conditions(
                    checkpoints=checkpoints,
                    lsd_data_dir=lsd_data_dir or "data/lsd",
                    n_steps=n_steps,
                    n_paths=n_paths,
                    device=device,
                )
                results["compare_control_conditions"] = "ok"
            except Exception as exc:
                print(f"  Warning: compare_control_conditions failed: {exc}")
                results["compare_control_conditions"] = f"error: {exc}"
        else:
            print("  Skipped: no LSD checkpoints found.")
            results["compare_control_conditions"] = "skipped"
    else:
        print_section("Step 3/3: Compare control conditions — skipped (not LSD)")
        results["compare_control_conditions"] = "skipped (not LSD)"

    print_section("PAPER PIPELINE COMPLETED")
    for step, status in results.items():
        print(f"  {step}: {status}")

    return results


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    result = run_paper_pipeline(
        metrics_json=getattr(args, "metrics", None),
        dataset_type=getattr(args, "dataset_type", None) or "ts_young",
        device=args.device,
        no_wandb=args.no_wandb,
        checkpoint_dir=getattr(args, "checkpoint_dir", "checkpoints"),
        data_path=getattr(args, "data_path", None),
        lsd_data_dir=getattr(args, "lsd_data_dir", None),
        output_path=getattr(args, "output_path", "results/comparison_results.json"),
        n_steps=getattr(args, "n_steps", 240),
        n_paths=getattr(args, "n_paths", 10),
    )
    return {"pipeline_results": result}


def _run_update_tables(args: argparse.Namespace) -> dict[str, object]:
    run_update_tables(
        metrics_json=getattr(args, "metrics", None),
        tex_target=getattr(args, "tex", None),
        table_label=getattr(args, "table_label", None),
        dry_run=getattr(args, "dry_run", False),
    )
    return {"update_tables": "done"}


def _run_compare(args: argparse.Namespace) -> dict[str, object]:
    result = run_compare_models(
        dataset_type=getattr(args, "dataset_type", None) or "ts_young",
        checkpoint_dir=getattr(args, "checkpoint_dir", "checkpoints"),
        checkpoints=getattr(args, "checkpoints", None),
        device=args.device or "auto",
        no_wandb=args.no_wandb,
        wandb_project=getattr(args, "wandb_project", "neuroscience-control"),
        run_name=getattr(args, "run_name", "model_comparison"),
        output_path=getattr(args, "output_path", "results/comparison_results.json"),
        skip_figures=getattr(args, "skip_figures", False),
        data_path=getattr(args, "data_path", None),
        lsd_data_dir=getattr(args, "lsd_data_dir", None),
    )
    return {"compare_results": result}


def _run_plot_metrics(args: argparse.Namespace) -> dict[str, object]:
    run_plot_metrics(
        dataset_type=getattr(args, "dataset_type", None) or "ts_young",
        metrics_dir=getattr(args, "metrics_dir", "results/metrics"),
        save_dir=Path(args.output_dir) if getattr(args, "output_dir", None) else None,
    )
    return {"plot_metrics": "done"}


def _run_compare_conditions(args: argparse.Namespace) -> dict[str, object]:
    run_compare_conditions(
        checkpoints=getattr(args, "checkpoints", None),
        lsd_data_dir=getattr(args, "lsd_data_dir", None) or "data/lsd",
        n_steps=getattr(args, "n_steps", 240),
        n_paths=getattr(args, "n_paths", 10),
        device=args.device,
        model_labels=getattr(args, "model_labels", None),
        ctrl_values=getattr(args, "ctrl_values", None),
        output_dir=getattr(args, "output_dir", None),
    )
    return {"compare_conditions": "done"}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    add_dataset_args(p)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-training pipeline for paper artefact generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # pipeline: run all post-training steps
    pipeline = subparsers.add_parser(
        "pipeline",
        help="Run post-training pipeline (update tables, compare models, compare conditions)",
    )
    _add_pipeline_args(pipeline)
    pipeline.add_argument("--metrics", type=str, default=None, help="Path to paper metrics JSON")
    pipeline.add_argument("--output-path", type=str, default="results/comparison_results.json")
    pipeline.add_argument("--n-steps", type=int, default=100, help="Simulation length for compare_conditions")
    pipeline.add_argument("--n-paths", type=int, default=32, help="Initial conditions for compare_conditions")

    # update-tables
    update_tables = subparsers.add_parser("update-tables", help="Patch LaTeX tables from metrics JSON")
    update_tables.add_argument("--metrics", type=str, default=None, help="Path to paper metrics JSON")
    update_tables.add_argument("--tex", type=str, default=None, help="Explicit .tex file to patch")
    update_tables.add_argument("--table-label", type=str, default=None, help="LaTeX table label")
    update_tables.add_argument("--dry-run", action="store_true", help="Preview without writing")

    # compare
    compare = subparsers.add_parser("compare", help="Compare all models (inter + intra patient tables)")
    compare.add_argument("--checkpoints", nargs="+", type=str, default=None,
                         help="Explicit checkpoint paths (auto-discovers if omitted)")
    compare.add_argument("--output-path", type=str, default="results/comparison_results.json")
    compare.add_argument("--wandb-project", type=str, default="neuroscience-control")
    compare.add_argument("--run-name", type=str, default="model_comparison")
    compare.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    _add_pipeline_args(compare)

    # plot-metrics
    plot_metrics = subparsers.add_parser("plot-metrics", help="Plot validation metrics over epochs for each model")
    plot_metrics.add_argument("--metrics-dir", type=str, default="results/metrics")
    plot_metrics.add_argument("--output-dir", type=str, default=None)
    add_dataset_args(plot_metrics)

    # compare-conditions
    compare_cond = subparsers.add_parser("compare-conditions", help="Compare model FC under LSD control conditions")
    compare_cond.add_argument("--checkpoints", nargs="+", type=str, default=None)
    compare_cond.add_argument("--model-labels", nargs="+", type=str, default=None)
    compare_cond.add_argument("--n-steps", type=int, default=240)
    compare_cond.add_argument("--n-paths", type=int, default=10)
    compare_cond.add_argument("--ctrl-values", nargs="+", type=int, default=None)
    compare_cond.add_argument("--output-dir", type=str, default=None)
    _add_pipeline_args(compare_cond)

    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = _build_parser()

    if len(sys.argv) == 1:
        sys.argv.append("pipeline")
    args = parser.parse_args(argv)

    

    dispatch = {
        "pipeline": _run_pipeline,
        "update-tables": _run_update_tables,
        "compare": _run_compare,
        "compare-conditions": _run_compare_conditions,
        "plot-metrics": _run_plot_metrics,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    main()
