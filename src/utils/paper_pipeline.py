"""
Post-training paper pipeline.

Consolidates all post-training steps that produce publication artefacts:

1. **update_paper_tables** – patches LaTeX model-comparison tables from a
   metrics JSON produced by ``train_models.py paper``.
2. **compare_models** – generates FC/timeseries comparison figures for the
   best Hopf and Neural SDE checkpoints.
3. **compare_control_conditions** *(LSD only)* – simulates all trained models
   under pharmacological conditions (u=0 Placebo, u=1 LSD+Ketanserin, u=2 LSD)
   and produces FC grids, ΔFC plots, bar charts, a metrics JSON and a LaTeX
   table fragment.

All logic lives here; standalone example scripts delegate to this module.

Usage from Python
-----------------
>>> from src.utils import run_paper_pipeline
>>> run_paper_pipeline(
...     metrics_json="results/lsd_paper_metrics_20260311.json",
...     dataset_type="lsd",
...     device="cuda",
...     lsd_data_dir="data/lsd",
... )

Called automatically at the end of ``train_models.py paper``, or via the
``pipeline``, ``update-tables``, ``compare``, ``compare-conditions``
sub-commands.
"""

from __future__ import annotations

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

# Ensure project root is importable when called from any CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.runtime import print_section, resolve_device, seed_all
from src.utils.visualization import (
    FIGURES_DIR,
    plot_fc_comparison,
    plot_model_comparison,
    plot_timeseries,
    create_comparison_report,
)
from src.utils.runtime import (
    ensure_proxy_env,
    managed_wandb_run,
    wandb_log,
    wandb_log_figure,
    wandb_summary_update,
)


# ===================================================================
# Checkpoint discovery
# ===================================================================

#: Canonical "best" checkpoint patterns per dataset tag.
_BEST_CKPT_PATTERNS: dict[str, list[str]] = {
    "ts_young": [
        "best_hopf_grid_ts_young.pt",
        "best_hopf_backprop_ts_young.pt",
        "best_nsde_backprop_ts_young.pt",
        "best_hybrid_hopf_backprop_ts_young.pt",
        "best_gnn_hopf_backprop_ts_young.pt",
    ],
    "lsd": [
        "best_hopf_grid_lsd.pt",
        "best_hopf_backprop_lsd.pt",
        "best_nsde_backprop_lsd.pt",
        "best_hybrid_hopf_backprop_lsd.pt",
        "best_gnn_hopf_backprop_lsd.pt",
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

    # Prefer explicitly provided paths (from the training run).
    if explicit_checkpoints:
        for _tag, path in explicit_checkpoints.items():
            if path and Path(path).exists():
                found.append(str(path))

    # Fall back to well-known best checkpoint names.
    if not found:
        patterns = _BEST_CKPT_PATTERNS.get(dataset_type, [])
        for name in patterns:
            p = ckpt_dir / name
            if p.exists():
                found.append(str(p))

    # Last resort: glob for any matching checkpoint.
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
    ("phfcd_ks",                r"phFCD KS $\downarrow$"),
    ("metastability_diff",      r"Meta $|\Delta|$ $\downarrow$"),
    ("temporal_correlation",    r"TS corr $\uparrow$"),
    ("power_spectrum_distance", r"TS PSD $\downarrow$"),
]

MODEL_DISPLAY: dict[str, str] = {
    "hopf_grid":            "Coupled Hopf (grid search)",
    "hopf_backprop":        "Coupled Hopf (gradient opt.)",
    "hybrid_hopf_backprop": "Hybrid Hopf (gradient opt.)",
    "nsde_backprop":        "Neural SDE (gradient opt.)",
    "gnn_hopf_backprop":    "GNN Hopf (gradient opt.)",
}

HIGHER_BETTER: dict[str, bool] = {
    "fc_correlation": True,
    "fc_mse": False,
    "phase_fc_correlation": True,
    "phfcd_ks": False,
    "metastability_diff": False,
    "temporal_correlation": True,
    "power_spectrum_distance": False,
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


def _build_table_body(metrics: dict[str, dict[str, float]]) -> str:
    """Return LaTeX table rows (\\midrule … \\bottomrule exclusive)."""
    col_keys = [mk for mk, _ in METRIC_COLS]

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


def _patch_tabular_body(tex: str, new_body: str) -> str:
    """Replace row block between \\midrule and \\bottomrule in a tabular snippet."""
    tabular_pattern = re.compile(
        r"(\\midrule\n)(.*?)(\n\s*\\bottomrule)",
        re.DOTALL,
    )
    def _replacement(m: re.Match) -> str:
        return m.group(1) + new_body + m.group(3)
    patched, n_subs = tabular_pattern.subn(_replacement, tex)
    if n_subs == 0:
        print("  Warning: could not find \\midrule/\\bottomrule block in target file.", file=sys.stderr)
    return patched


def run_update_tables(
    metrics_json: str | Path | None = None,
    tex_target: str | Path | None = None,
    table_label: str | None = None,
    dry_run: bool = False,
) -> None:
    """Patch paper LaTeX tables with fresh metrics.

    Parameters
    ----------
    metrics_json
        Path to the paper metrics JSON.  If *None*, auto-discovers the latest
        ``results/*_paper_metrics_*.json``.
    tex_target
        Explicit ``.tex`` file to patch.  Inferred from *table_label* or
        *dataset_type* in the JSON if not given.
    table_label
        E.g. ``"tab:model_comparison"`` or ``"tab:lsd_model_comparison"``.
    dry_run
        Print the generated rows without writing to disk.
    """
    if metrics_json is None:
        results_dir = Path("results")
        candidates = sorted(
            results_dir.glob("*_paper_metrics_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No metrics JSON found in results/. Please specify --metrics or run training first.")
            return
        metrics_json = candidates[0]
        print(f"Auto-detected latest metrics file: {metrics_json}")

    metrics_json = Path(metrics_json)
    if not metrics_json.exists():
        print(f"Metrics file not found: {metrics_json}")
        return

    with metrics_json.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if "metrics" in data and "dataset_type" in data:
        metrics = data["metrics"]
        dataset_type = data["dataset_type"]
        print(f"Dataset type: {dataset_type}")
    else:
        metrics = data
        dataset_type = None
        print("Warning: Old JSON format detected (no dataset_type metadata)")

    print(f"Loaded metrics for: {list(metrics.keys())}")

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

    body = _build_table_body(metrics)
    print("\nGenerated table rows:")
    print(body)

    if dry_run:
        return

    if not tex_path.exists():
        print(f"LaTeX file not found: {tex_path}")
        return

    original = tex_path.read_text(encoding="utf-8")
    patched = _patch_tabular_body(original, body)
    if patched == original:
        print("No changes made (table not found or already up to date).")
    else:
        tex_path.write_text(patched, encoding="utf-8")
        print(f"Updated {tex_path}")


# ===================================================================
# 2. compare_models
# ===================================================================

def _load_data(data_path: str, device: str):
    """Load dataset from a .mat file."""
    from src.dataset import NeuroscienceDataset

    print_section("Loading Data")
    dataset = NeuroscienceDataset(filepath=data_path, normalize=True, device=device)
    print(f"Loaded dataset:")
    print(f"  - Number of subjects: {dataset.n_subjects}")
    print(f"  - Number of ROIs: {dataset.n_rois}")
    print(f"  - Number of timepoints: {dataset.n_timepoints}")
    return dataset


def _load_models(hopf_checkpoint: str | None, nsde_checkpoint: str | None, n_rois: int, device: str):
    """Load trained models from checkpoints."""
    from src.models import load_model_from_checkpoint

    print_section("Loading Models")
    models: dict[str, Any] = {}

    for label, ckpt, tag in [
        ("Coupled Hopf", hopf_checkpoint, "Hopf"),
        ("Neural SDE", nsde_checkpoint, "Neural SDE"),
    ]:
        if not ckpt or not Path(ckpt).exists():
            print(f"Warning: {tag} checkpoint not found at {ckpt}")
            continue
        print(f"Loading {tag} model from {ckpt}")
        try:
            model, model_name, _ = load_model_from_checkpoint(ckpt, device=device)
            if model.n_rois != n_rois:
                print(f"Warning: {model_name} ROI mismatch ({model.n_rois} != {n_rois}); skipping.")
            else:
                models[label] = model
                print(f"  - {tag} model loaded successfully")
        except Exception as exc:
            print(f"Warning: Failed to load {tag} checkpoint: {exc}")
    return models


def _evaluate_models(models: dict, dataset, target_fc, n_timepoints: int, n_simulations: int = 10):
    """Evaluate all models and compute FC metrics."""
    from src.metrics import compute_all_fc_metrics

    print_section("Evaluating Models")
    all_results: dict[str, dict[str, float]] = {}
    n_eval = min(n_simulations, dataset.n_subjects)
    initial_states = dataset.timeseries[:n_eval, :, 0]

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        with torch.no_grad():
            ts = model.forward(initial_state=initial_states, n_steps=n_timepoints)
            fc_pred = model.compute_fc(ts)
            all_fc_corrs, all_fc_mses = [], []
            for i in range(n_eval):
                m = compute_all_fc_metrics(fc_pred[i:i+1], target_fc.unsqueeze(0))
                all_fc_corrs.append(m["fc_correlation"])
                all_fc_mses.append(m["fc_mse"])

        results = {
            "fc_correlation": float(np.mean(all_fc_corrs)),
            "fc_correlation_std": float(np.std(all_fc_corrs)),
            "fc_mse": float(np.mean(all_fc_mses)),
            "fc_mse_std": float(np.std(all_fc_mses)),
        }
        all_results[name] = results
        print(f"  fc_correlation: {results['fc_correlation']:.4f} ± {results['fc_correlation_std']:.4f}")
        print(f"  fc_mse: {results['fc_mse']:.4f} ± {results['fc_mse_std']:.4f}")
    return all_results


def _generate_comparison_figures(
    models: dict,
    dataset,
    target_fc,
    results: dict,
    n_timepoints: int,
    use_wandb: bool = False,
):
    """Generate all comparison figures (FC, timeseries, bar, report)."""
    print_section("Generating Comparison Figures")
    figures: dict[str, Any] = {}
    initial_state = dataset.timeseries[:1, :, 0]

    for name, model in models.items():
        with torch.no_grad():
            ts = model.forward(initial_state=initial_state, n_steps=n_timepoints)
            fc_pred = model.compute_fc(ts)[0]

        fig = plot_fc_comparison(
            fc_pred, target_fc,
            title=f"{name} - FC Comparison",
            default_name=f"{name.lower().replace(' ', '_')}_fc_comparison",
            use_pdf=True,
        )
        figures[f"{name}_fc"] = fig
        wandb_log_figure(f"figures/{name}_fc_comparison", fig, use_wandb=use_wandb)
        plt.close()

        fig = plot_timeseries(
            ts[0].real, n_rois=5,
            title=f"{name} - Simulated Timeseries",
            default_name=f"{name.lower().replace(' ', '_')}_timeseries",
            use_pdf=True,
        )
        figures[f"{name}_ts"] = fig
        wandb_log_figure(f"figures/{name}_timeseries", fig, use_wandb=use_wandb)
        plt.close()

    fig = plot_model_comparison(
        results, metric_names=["fc_correlation", "fc_mse"],
        default_name="model_comparison_metrics", use_pdf=True,
    )
    figures["comparison"] = fig
    wandb_log_figure("figures/model_comparison", fig, use_wandb=use_wandb)
    plt.close()

    fig = create_comparison_report(
        models=models, target_fc=target_fc, n_timepoints=n_timepoints,
        initial_state=initial_state, save_dir=str(FIGURES_DIR), use_pdf=True,
    )
    figures["report"] = fig
    wandb_log_figure("figures/comparison_report", fig, use_wandb=use_wandb)
    plt.close()

    print(f"All figures saved to {FIGURES_DIR}")
    return figures


def _save_comparison_results(results: dict, save_path: str) -> None:
    """Save comparison results to JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        mn: {k: float(v) for k, v in metrics.items()}
        for mn, metrics in results.items()
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {save_path}")


def _print_comparison_summary(results: dict) -> None:
    """Print comparison summary table."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60 + "\n" + "-" * 40)
    print(f"{'Model':<20} {'fc_correlation':<15} {'fc_mse':<15}")
    print("-" * 40)
    for name, m in results.items():
        fc_corr = f"{m['fc_correlation']:.4f} ± {m['fc_correlation_std']:.4f}"
        fc_mse = f"{m['fc_mse']:.4f} ± {m['fc_mse_std']:.4f}"
        print(f"{name:<20} {fc_corr:<15} {fc_mse:<15}")
    print("-" * 40)
    best = max(results, key=lambda x: results[x]["fc_correlation"])
    print(f"\nBest model (by fc_correlation): {best}")
    print(f"  fc_correlation: {results[best]['fc_correlation']:.4f}")


def run_compare_models(
    data_path: str = "data/ts_young/ts_young_TR0.72.mat",
    hopf_checkpoint: str | None = "checkpoints/hopf_best.pt",
    nsde_checkpoint: str | None = "checkpoints/nsde_best.pt",
    device: str = "auto",
    no_wandb: bool = True,
    wandb_project: str = "neuroscience-control",
    run_name: str = "model_comparison",
    n_simulations: int = 10,
    output_path: str = "results/comparison_results.json",
) -> dict | None:
    """Compare trained Hopf and Neural SDE models.

    Returns the results dict or *None* if no models were loaded.
    """
    print_section("MODEL COMPARISON")
    ensure_proxy_env()
    device = resolve_device(device)
    seed_all(42)
    use_wandb = not no_wandb

    with managed_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project,
        run_name=run_name,
        tags=["comparison", "evaluation"],
    ):
        dataset = _load_data(data_path, device)
        target_fc = dataset.fc_mean
        n_timepoints = min(dataset.n_timepoints, 200)

        models = _load_models(hopf_checkpoint, nsde_checkpoint, n_rois=dataset.n_rois, device=device)
        if not models:
            print("Error: No models could be loaded. Please check checkpoint paths.")
            return None

        results = _evaluate_models(models, dataset, target_fc, n_timepoints, n_simulations=n_simulations)

        for model_name, metrics in results.items():
            wandb_log({f"{model_name}/{k}": v for k, v in metrics.items()}, use_wandb=use_wandb)
        wandb_summary_update(
            {"best_model": max(results, key=lambda x: results[x]["fc_correlation"])},
            use_wandb=use_wandb,
        )

        _generate_comparison_figures(models, dataset, target_fc, results, n_timepoints, use_wandb=use_wandb)
        _save_comparison_results(results, output_path)
        _print_comparison_summary(results)

        print_section("COMPARISON COMPLETED SUCCESSFULLY")
        print(f"\nResults saved to: {output_path}")
        print(f"Figures saved to: {FIGURES_DIR}")
        return results


# ===================================================================
# 3. compare_control_conditions
# ===================================================================

_MODEL_TAGS = [
    ("gnn_hopf", "GNN Hopf"),
    ("hybrid_hopf", "Hybrid Hopf"),
    ("nsde", "Neural SDE"),
    ("hopf", "Hopf"),
]


def _short_model_label(stem: str) -> str:
    """Extract a concise model tag from a checkpoint file stem."""
    lower = stem.lower()
    for tag, label in _MODEL_TAGS:
        if tag in lower:
            suffix = " (Grid)" if "grid" in lower else ""
            return f"{label}{suffix}"
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
    return fc_per_subject.mean(dim=0)


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
    from src.metrics import fc_correlation
    from src.models import load_model_from_checkpoint

    device = resolve_device(device)
    seed_all(42)

    ctrl_values = ctrl_values or sorted(CONDITION_LABELS.keys())
    figures_dir = Path(output_dir) if output_dir else FIGURES_DIR
    results_dir = Path("results")
    latex_dir = Path("paper/sections")

    # 1. Load LSD dataset
    print_section("Loading LSD dataset")
    dataset = NeuroscienceDataset.from_lsd(data_dir=lsd_data_dir, normalize=True, device=device)
    print(f"  subjects={dataset.n_subjects}  ROIs={dataset.n_rois}  T={dataset.n_timepoints}")

    empirical_fc_per_ctrl: dict[int, torch.Tensor] = {}
    for cv in ctrl_values:
        mask = dataset.control[:, 0] == cv
        if mask.sum() == 0:
            print(f"  Warning: no subjects for ctrl={cv}, skipping.")
            continue
        empirical_fc_per_ctrl[cv] = dataset.fc_matrices[mask].mean(dim=0)
        print(f"  Empirical FC ctrl={cv}: {mask.sum().item()} subjects")

    n_p = min(n_paths, dataset.n_subjects)
    initial_states = dataset.timeseries[:n_p, :, 0]

    # 2. Load checkpoints
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

    # 3. Simulate FC per condition
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
            corr = fc_correlation(fc_sim.unsqueeze(0), fc_emp.unsqueeze(0)).item()
            fc_sim_r = fc_sim.real if torch.is_complex(fc_sim) else fc_sim
            fc_emp_r = fc_emp.real if torch.is_complex(fc_emp) else fc_emp
            idx = torch.triu_indices(fc_sim_r.shape[0], fc_sim_r.shape[1], offset=1)
            mse = ((fc_sim_r[idx[0], idx[1]] - fc_emp_r[idx[0], idx[1]]) ** 2).mean().item()
            fc_per_model[mname][cv] = fc_sim
            cond_metrics[mname][cv] = {"fc_corr": corr, "fc_mse": mse}
            print(f"FC corr={corr:.3f}  MSE={mse:.4f}")

    # 4. Figures
    print_section("Generating figures")
    _plot_fc_grid(fc_per_model, empirical_fc_per_ctrl, save_path=figures_dir / "lsd" / "control_fc_grid.pdf")
    for mname in models:
        _plot_fc_diff(
            fc_per_model[mname], model_name=mname, baseline_ctrl=0,
            save_path=figures_dir / "lsd" / f"fc_delta_{mname.lower().replace(' ', '_')}.pdf",
        )
    _plot_fc_corr_bar(cond_metrics, save_path=figures_dir / "lsd" / "control_fc_corr.pdf")

    # 5. Save metrics JSON and LaTeX table
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

    # Summary
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
# Orchestrator: run_paper_pipeline
# ===================================================================

def run_paper_pipeline(
    metrics_json: str | Path | None = None,
    dataset_type: str = "ts_young",
    device: str | None = None,
    no_wandb: bool = True,
    checkpoint_dir: str = "checkpoints",
    explicit_checkpoints: dict[str, str] | None = None,
    # ts_young-specific
    data_path: str | None = None,
    # LSD-specific
    lsd_data_dir: str | None = None,
    # compare_models options
    n_simulations: int = 10,
    output_path: str = "results/comparison_results.json",
    # compare_control_conditions options
    n_steps: int = 240,
    n_paths: int = 10,
) -> dict[str, Any]:
    """Run the full post-training paper pipeline.

    Parameters
    ----------
    metrics_json
        Path to the paper metrics JSON (output of ``train_models.py paper``).
        If *None*, the latest file in ``results/`` matching
        ``*_paper_metrics_*.json`` is used.
    dataset_type
        ``"ts_young"`` or ``"lsd"``.
    device
        Compute device (``"auto"``, ``"cuda"``, ``"cpu"``).
    no_wandb
        Disable wandb logging for the comparison steps.
    checkpoint_dir
        Directory containing trained model checkpoints.
    explicit_checkpoints
        Dict mapping model tag → checkpoint path.  Overrides auto-discovery.
    data_path
        Path to the ``.mat`` data file (ts_young).
    lsd_data_dir
        Directory with LSD ``.mat`` files.
    n_simulations
        Number of simulations for ``compare_models``.
    output_path
        Where ``compare_models`` saves its JSON results.
    n_steps
        Simulation length for ``compare_control_conditions``.
    n_paths
        Number of initial conditions for ``compare_control_conditions``.

    Returns
    -------
    dict
        Summary with keys ``"update_tables"``, ``"compare_models"``,
        ``"compare_control_conditions"`` indicating success / skipped status.
    """
    print_section("POST-TRAINING PAPER PIPELINE")
    results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Update LaTeX tables
    # ------------------------------------------------------------------
    print_section("Step 1/3: Updating LaTeX tables")
    try:
        run_update_tables(metrics_json=metrics_json)
        results["update_tables"] = "ok"
    except Exception as exc:
        print(f"  Warning: update_paper_tables failed: {exc}")
        results["update_tables"] = f"error: {exc}"

    # ------------------------------------------------------------------
    # 2. Compare models (FC / timeseries figures)
    # ------------------------------------------------------------------
    print_section("Step 2/3: Comparing models (figures)")
    checkpoints = _discover_checkpoints(dataset_type, checkpoint_dir, explicit_checkpoints)

    hopf_ckpt = next(
        (c for c in checkpoints
         if "hopf" in Path(c).stem and "nsde" not in Path(c).stem
         and "hybrid" not in Path(c).stem and "gnn" not in Path(c).stem),
        None,
    )
    nsde_ckpt = next((c for c in checkpoints if "nsde" in Path(c).stem), None)

    if hopf_ckpt or nsde_ckpt:
        try:
            resolve_data_path = data_path
            if dataset_type == "lsd" and lsd_data_dir and not data_path:
                mat_files = sorted(Path(lsd_data_dir).glob("*.mat"))
                if mat_files:
                    resolve_data_path = str(mat_files[0])

            if resolve_data_path:
                run_compare_models(
                    data_path=resolve_data_path,
                    hopf_checkpoint=hopf_ckpt,
                    nsde_checkpoint=nsde_ckpt,
                    device=device or "auto",
                    no_wandb=no_wandb,
                    n_simulations=n_simulations,
                    output_path=output_path,
                )
                results["compare_models"] = "ok"
            else:
                print("  Skipped: no data path resolved.")
                results["compare_models"] = "skipped (no data path)"
        except Exception as exc:
            print(f"  Warning: compare_models failed: {exc}")
            results["compare_models"] = f"error: {exc}"
    else:
        print("  Skipped: no Hopf or Neural SDE checkpoints found.")
        results["compare_models"] = "skipped"

    # ------------------------------------------------------------------
    # 3. Compare control conditions (LSD only)
    # ------------------------------------------------------------------
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
