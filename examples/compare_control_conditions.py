#!/usr/bin/env python3
"""
Compare model-simulated FC under different pharmacological control conditions.

Loads one or more trained model checkpoints (trained on the LSD dataset with
n_control_dims=1) and simulates trajectories for each condition value
(u=0: Placebo, u=1: LSD+Ketanserin, u=2: LSD).  The script produces:

  1. A 3×3 grid of FC matrices (row = model, col = condition)   → lsd_control_fc_grid.pdf
  2. Per-condition FC difference heat-maps (Δ vs Placebo)       → lsd_control_fc_diff_{model}.pdf
  3. A metrics table (FC correlation, FC MSE vs empirical)      → results/lsd_control_metrics.json
  4. A ready-to-paste LaTeX table fragment                       → results/lsd_control_table.tex

Usage
-----
    python examples/compare_control_conditions.py \
        --checkpoints checkpoints/hopf_backprop_best_*.pt \
                      checkpoints/nsde_backprop_best_*.pt \
        --lsd-data-dir data/lsd

You can also pass a single checkpoint and a readable --model-label.
"""

from __future__ import annotations

import argparse
import json
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

from src.dataset import NeuroscienceDataset
from src.dataset.data_loader import LSD_CONDITION_MAP
from src.metrics import fc_correlation
from src.models import load_model_from_checkpoint
from src.utils import FIGURES_DIR, print_section, resolve_device, seed_all


# ---------------------------------------------------------------------------
# Condition metadata
# ---------------------------------------------------------------------------

# Sorted by control value for consistent display ordering.
CONDITION_LABELS: dict[int, str] = {
    0: "Placebo (u=0)",
    1: "LSD+Ketanserin (u=1)",
    2: "LSD (u=2)",
}

CONDITION_COLORS: dict[int, str] = {
    0: "#4878CF",  # blue
    1: "#D65F5F",  # red
    2: "#6ACC65",  # green
}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_fc(
    model: Any,
    initial_states: torch.Tensor,
    ctrl_val: int,
    n_steps: int,
    dt: float,
    device: str,
) -> torch.Tensor:
    """Simulate *model* with constant control u=ctrl_val and return mean FC."""
    n_paths = initial_states.shape[0]
    # Build control tensor: (batch, n_control_dims)
    if getattr(model, "n_control_dims", 0) > 0:
        ctrl = torch.full((n_paths, 1), float(ctrl_val), dtype=torch.float32, device=device)
    else:
        ctrl = None

    fwd_kwargs: dict[str, Any] = dict(
        initial_state=initial_states,
        n_steps=n_steps,
        dt=dt,
    )
    if ctrl is not None:
        fwd_kwargs["control"] = ctrl

    with torch.no_grad():
        ts = model.forward(**fwd_kwargs)
        fc_per_subject = model.compute_fc(ts)  # (batch, n_rois, n_rois)
    return fc_per_subject.mean(dim=0)  # (n_rois, n_rois)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _fc_to_numpy(fc: torch.Tensor) -> np.ndarray:
    arr = fc.detach().cpu().numpy()
    if np.iscomplexobj(arr):
        arr = arr.real
    return arr


def plot_fc_grid(
    fc_per_model: dict[str, dict[int, torch.Tensor]],
    empirical_fc_per_ctrl: dict[int, torch.Tensor],
    *,
    save_path: Path,
) -> None:
    """Plot a grid of FC matrices: rows=models, cols=conditions."""
    model_names = list(fc_per_model.keys())
    ctrl_vals = sorted(CONDITION_LABELS.keys())
    n_models = len(model_names)
    n_conds = len(ctrl_vals)

    # Extra row for empirical FCs
    n_rows = n_models + 1
    fig, axes = plt.subplots(n_rows, n_conds, figsize=(4 * n_conds, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_conds == 1:
        axes = axes[:, np.newaxis]

    vmin, vmax = -1.0, 1.0

    # Empirical row (first row)
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

    # Model rows
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


def plot_fc_diff(
    fc_per_ctrl: dict[int, torch.Tensor],
    model_name: str,
    *,
    baseline_ctrl: int = 0,
    save_path: Path,
) -> None:
    """Plot FC difference matrices (condition − baseline)."""
    ctrl_vals = [cv for cv in sorted(CONDITION_LABELS.keys()) if cv != baseline_ctrl]
    n_diffs = len(ctrl_vals)
    if n_diffs == 0:
        return

    fig, axes = plt.subplots(1, n_diffs, figsize=(5 * n_diffs, 4.5))
    if n_diffs == 1:
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


def plot_fc_corr_bar(
    metrics: dict[str, dict[int, dict[str, float]]],
    *,
    save_path: Path,
) -> None:
    """Bar plot of FC correlation per model per condition."""
    model_names = list(metrics.keys())
    ctrl_vals = sorted(CONDITION_LABELS.keys())
    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(model_names)), 4))
    for i, cv in enumerate(ctrl_vals):
        corrs = [metrics[m].get(cv, {}).get("fc_corr", float("nan")) for m in model_names]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, corrs, width,
            label=CONDITION_LABELS[cv],
            color=CONDITION_COLORS[cv],
            alpha=0.85,
        )

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


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _fmt(val: float | None, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def build_latex_table(
    metrics: dict[str, dict[int, dict[str, float]]],
) -> str:
    """Return a LaTeX tabular fragment summarising FC metrics per condition."""
    ctrl_vals = sorted(CONDITION_LABELS.keys())
    short_labels = {0: "Placebo", 1: "LSD+Ket.", 2: "LSD"}

    col_header = " & ".join(
        [f"FC corr $\\uparrow$ ({short_labels[cv]}) & FC MSE $\\downarrow$ ({short_labels[cv]})" for cv in ctrl_vals]
    )
    header = f"Model & {col_header} \\\\\n\\midrule\n"
    body = ""
    for mname, cond_metrics in metrics.items():
        row = mname.replace("_", " ")
        for cv in ctrl_vals:
            m = cond_metrics.get(cv, {})
            row += f" & {_fmt(m.get('fc_corr'))} & {_fmt(m.get('fc_mse'))}"
        body += row + " \\\\\n"
    n_cols = 1 + 2 * len(ctrl_vals)
    col_spec = "l" + "".join([" c c" for _ in ctrl_vals])
    table = (
        "\\begin{tabular}{" + col_spec + "}\n"
        "\\toprule\n"
        + header
        + body
        + "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare model FC under LSD conditions (u=0,1,2)."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default=[],
        help="Paths to trained model checkpoints (.pt files).",
    )
    parser.add_argument(
        "--lsd-data-dir",
        type=str,
        default="data/lsd",
        help="Directory containing LSD .mat files.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=240,
        help="Simulation length (timepoints).",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=10,
        help="Number of initial conditions to average over per condition.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device.",
    )
    parser.add_argument(
        "--model-labels",
        nargs="+",
        type=str,
        default=None,
        help="Human-readable names for each checkpoint (same order as --checkpoints).",
    )
    parser.add_argument(
        "--ctrl-values",
        nargs="+",
        type=int,
        default=None,
        help="Control values to compare (default: 0 1 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output figures/metrics (default: paper/images and results/).",
    )
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    seed_all(42)

    ctrl_values: list[int] = args.ctrl_values if args.ctrl_values else sorted(CONDITION_LABELS.keys())
    figures_dir = Path(args.output_dir) if args.output_dir else FIGURES_DIR
    results_dir = Path("results")
    latex_dir = Path("paper/sections")

    # ------------------------------------------------------------------
    # 1. Load LSD dataset to get empirical FCs
    # ------------------------------------------------------------------
    print_section("Loading LSD dataset")
    dataset = NeuroscienceDataset.from_lsd(
        data_dir=args.lsd_data_dir,
        normalize=True,
        device=device,
    )
    print(f"  subjects={dataset.n_subjects}  ROIs={dataset.n_rois}  T={dataset.n_timepoints}")

    # Empirical FC per condition (averaged over subjects in that condition)
    empirical_fc_per_ctrl: dict[int, torch.Tensor] = {}
    for cv in ctrl_values:
        mask = (dataset.control[:, 0] == cv)
        if mask.sum() == 0:
            print(f"  Warning: no subjects for ctrl={cv}, skipping.")
            continue
        empirical_fc_per_ctrl[cv] = dataset.fc_matrices[mask].mean(dim=0)
        print(f"  Empirical FC ctrl={cv}: {mask.sum().item()} subjects")

    # Pool initial states (use first args.n_paths subjects for each run)
    n_paths = min(args.n_paths, dataset.n_subjects)
    initial_states = dataset.timeseries[:n_paths, :, 0]

    # ------------------------------------------------------------------
    # 2. Load model checkpoints
    # ------------------------------------------------------------------
    print_section("Loading model checkpoints")
    if not args.checkpoints:
        # Auto-discover from checkpoints dir
        ckpt_dir = Path("checkpoints")
        found = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        if not found:
            print("No checkpoints specified or found. Exiting.")
            return
        args.checkpoints = [str(p) for p in found]
        print(f"  Auto-discovered {len(args.checkpoints)} checkpoint(s).")

    labels = args.model_labels or [Path(c).stem for c in args.checkpoints]
    if len(labels) < len(args.checkpoints):
        labels += [Path(c).stem for c in args.checkpoints[len(labels):]]

    models: dict[str, Any] = {}
    for ckpt_path, label in zip(args.checkpoints, labels):
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

    # ------------------------------------------------------------------
    # 3. Simulate FC for each model × condition
    # ------------------------------------------------------------------
    print_section("Simulating FC per condition")
    fc_per_model: dict[str, dict[int, torch.Tensor]] = {}
    metrics: dict[str, dict[int, dict[str, float]]] = {}

    for mname, model in models.items():
        fc_per_model[mname] = {}
        metrics[mname] = {}
        for cv in ctrl_values:
            if cv not in empirical_fc_per_ctrl:
                continue
            print(f"  {mname}: simulating ctrl={cv} …", end=" ", flush=True)
            fc_sim = _simulate_fc(
                model, initial_states, cv,
                n_steps=args.n_steps, dt=dataset.dt, device=device,
            )
            fc_emp = empirical_fc_per_ctrl[cv]
            corr = fc_correlation(fc_sim.unsqueeze(0), fc_emp.unsqueeze(0)).item()

            fc_sim_r = fc_sim.real if torch.is_complex(fc_sim) else fc_sim
            fc_emp_r = fc_emp.real if torch.is_complex(fc_emp) else fc_emp
            # Upper-triangular MSE
            idx = torch.triu_indices(fc_sim_r.shape[0], fc_sim_r.shape[1], offset=1)
            mse = ((fc_sim_r[idx[0], idx[1]] - fc_emp_r[idx[0], idx[1]]) ** 2).mean().item()

            fc_per_model[mname][cv] = fc_sim
            metrics[mname][cv] = {"fc_corr": corr, "fc_mse": mse}
            print(f"FC corr={corr:.3f}  MSE={mse:.4f}")

    # ------------------------------------------------------------------
    # 4. Generate figures
    # ------------------------------------------------------------------
    print_section("Generating figures")

    plot_fc_grid(
        fc_per_model, empirical_fc_per_ctrl,
        save_path=figures_dir / "lsd_control_fc_grid.pdf",
    )

    for mname in models:
        plot_fc_diff(
            fc_per_model[mname],
            model_name=mname,
            baseline_ctrl=0,
            save_path=figures_dir / f"lsd_fc_delta_{mname.lower().replace(' ', '_')}.pdf",
        )

    plot_fc_corr_bar(
        metrics,
        save_path=figures_dir / "lsd_control_fc_corr_bar.pdf",
    )

    # ------------------------------------------------------------------
    # 5. Save metrics JSON and LaTeX table
    # ------------------------------------------------------------------
    print_section("Saving metrics and LaTeX table")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "lsd_control_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print(f"  Metrics → {metrics_path}")

    latex = build_latex_table(metrics)
    latex_dir.mkdir(parents=True, exist_ok=True)
    latex_path = latex_dir / "lsd_control_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"  LaTeX table → {latex_path}")

    # Print summary table to stdout
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
            corr = metrics[mname].get(cv, {}).get("fc_corr", float("nan"))
            row += f"  {corr:>14.4f}"
        print(row)


if __name__ == "__main__":
    main()
