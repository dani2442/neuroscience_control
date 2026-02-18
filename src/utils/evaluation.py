"""Shared evaluation, checkpointing, and figure-generation utilities.

These helpers consolidate boiler-plate that was previously duplicated across the
training scripts (train_hopf, train_backprop, train_nsde_finetune).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .visualization import (
    FIGURES_DIR,
    plot_fc_comparison,
    plot_simulation_multigrid,
)
from .runtime import (
    print_section,
    wandb_log,
    wandb_log_artifact,
    wandb_log_figure,
    wandb_summary_update,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def to_float_metric(value: Any) -> float | None:
    """Convert supported metric values (Tensor, int, …) to float for logging."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Attach a namespace *prefix/* and keep only numeric entries."""
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = to_float_metric(value)
        if numeric is not None:
            payload[f"{prefix}/{key}"] = numeric
    return payload


# ---------------------------------------------------------------------------
# FC comparison figure (common across all scripts)
# ---------------------------------------------------------------------------

def generate_fc_figure(
    model: Any,
    val_timeseries: torch.Tensor,
    target_fc: torch.Tensor,
    n_timepoints: int,
    dt: float,
    sde_type: str = "ito",
    method: str = "euler",
    dt_min: float | None = 0.1,
    use_adjoint: bool = False,
    adjoint_method: str | None = "adjoint_euler",
    *,
    title: str,
    default_name: str,
    use_wandb: bool = False,
    wandb_key: str = "figures/fc_comparison",
) -> plt.Figure:
    """Simulate a few subjects for aggregate FC and plot comparison."""
    n_fc_paths = min(6, val_timeseries.shape[0])
    fc_initial_states = val_timeseries[:n_fc_paths, :, 0]
    with torch.no_grad():
        fc_ts = model.forward(
            initial_state=fc_initial_states,
            n_steps=n_timepoints,
            dt=dt,
            sde_type=sde_type,
            method=method,
            dt_min=dt_min,
            use_adjoint=use_adjoint,
            adjoint_method=adjoint_method,
        )
        fc_pred = model.compute_fc(fc_ts)
        fc_mean = fc_pred.mean(dim=0)

    fig = plot_fc_comparison(
        fc_mean,
        target_fc,
        title=title,
        default_name=default_name,
        use_pdf=True,
    )
    wandb_log_figure(wandb_key, fig, use_wandb=use_wandb)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Multigrid figure: one real path, n_simulations stochastic runs from same IC
# ---------------------------------------------------------------------------

def generate_multigrid_figure(
    model: Any,
    val_timeseries: torch.Tensor,
    n_timepoints: int,
    dt: float,
    n_simulations: int = 10,
    sde_type: str = "ito",
    method: str = "euler",
    dt_min: float | None = 0.1,
    use_adjoint: bool = False,
    adjoint_method: str | None = "adjoint_euler",
    *,
    n_rois: int = 12,
    n_cols: int = 4,
    title: str = "Real vs Simulated",
    default_name: str = "real_vs_sim_multigrid",
    use_wandb: bool = False,
    wandb_key: str = "figures/real_vs_sim_multigrid",
) -> plt.Figure:
    """Simulate *n_simulations* stochastic paths from a single subject's IC.

    The first validation subject is used as the real reference.  Its initial
    condition is repeated ``n_simulations`` times and forward-simulated so the
    plot shows the mean ± std envelope of stochastic realizations from the
    **same** starting point.
    """
    real_ts = val_timeseries[0]  # (n_rois, T)  or complex
    ic = real_ts[:, 0]  # (n_rois,)
    initial_states_repeated = ic.unsqueeze(0).expand(n_simulations, -1)  # (n_sim, n_rois)

    with torch.no_grad():
        sim_ts = model.forward(
            initial_state=initial_states_repeated,
            n_steps=n_timepoints,
            dt=dt,
            sde_type=sde_type,
            method=method,
            dt_min=dt_min,
            use_adjoint=use_adjoint,
            adjoint_method=adjoint_method,
        )
    # sim_ts: (n_simulations, n_rois, n_timepoints)

    fig = plot_simulation_multigrid(
        real_timeseries=real_ts.real,
        simulated_runs=sim_ts.real,
        n_rois=n_rois,
        n_cols=n_cols,
        max_timepoints=n_timepoints,
        title=title,
        default_name=default_name,
        use_pdf=True,
    )
    wandb_log_figure(wandb_key, fig, use_wandb=use_wandb)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# save_checkpoint: save model + artifact
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Any,
    checkpoint_name: str,
    artifact_name: str,
    *,
    checkpoint_dir: str = "checkpoints",
    use_wandb: bool = False,
) -> Path:
    """Save model checkpoint and optionally log a W&B artifact."""
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cp_dir / checkpoint_name
    model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")

    wandb_log_artifact(artifact_name, checkpoint_path, use_wandb=use_wandb)
    return checkpoint_path


# ---------------------------------------------------------------------------
# Log best Hopf params (shared between backprop and grid-search Hopf)
# ---------------------------------------------------------------------------

def log_hopf_best_params(model: Any, *, use_wandb: bool = False) -> None:
    """Log ``a`` and ``G`` for a trained Hopf model."""
    if not (hasattr(model, "a") and hasattr(model, "g")):
        return
    best_a = model.a.item() if model.a.dim() == 0 else model.a.mean().item()
    best_G = model.g.item() if model.g.dim() == 0 else model.g.mean().item()
    wandb_log({"best_params/a": best_a, "best_params/G": best_G}, use_wandb=use_wandb)
    wandb_summary_update({"best_a": best_a, "best_G": best_G}, use_wandb=use_wandb)
    print(f"Best Hopf params — a: {best_a:.6f}, G: {best_G:.6f}")


# ---------------------------------------------------------------------------
# extract_val_data: pull timeseries / FC from a validation DataLoader
# ---------------------------------------------------------------------------

def extract_val_data(
    val_loader,
    max_timepoints: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return ``(val_timeseries, val_fc_matrices, target_fc, n_timepoints)``."""
    val_dataset = getattr(val_loader, "dataset", None)
    if (
        val_dataset is None
        or not hasattr(val_dataset, "timeseries")
        or not hasattr(val_dataset, "fc_matrices")
    ):
        raise ValueError(
            "val_loader must expose dataset.timeseries and dataset.fc_matrices for evaluation."
        )

    val_timeseries = val_dataset.timeseries
    val_fc_matrices = val_dataset.fc_matrices
    if val_timeseries.shape[0] == 0:
        raise ValueError("Validation loader dataset is empty; cannot generate final figures.")

    target_fc = val_fc_matrices.mean(dim=0)
    n_timepoints = min(val_timeseries.shape[2], max_timepoints)
    return val_timeseries, val_fc_matrices, target_fc, n_timepoints
