"""Shared evaluation, checkpointing, and figure-generation utilities."""

from __future__ import annotations

import math
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
    control: torch.Tensor | None = None,
) -> plt.Figure:
    """Simulate a few subjects for aggregate FC and plot comparison."""
    n_fc_paths = min(6, val_timeseries.shape[0])
    fc_initial_states = val_timeseries[:n_fc_paths, :, 0]
    fwd_kwargs: dict[str, Any] = dict(
        initial_state=fc_initial_states,
        n_steps=n_timepoints,
        dt=dt,
        sde_type=sde_type,
        method=method,
        dt_min=dt_min,
        use_adjoint=use_adjoint,
        adjoint_method=adjoint_method,
    )
    if control is not None:
        fwd_kwargs["control"] = control[:n_fc_paths]
    with torch.no_grad():
        fc_ts = model.forward(**fwd_kwargs)
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
    control: torch.Tensor | None = None,
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

    fwd_kwargs: dict[str, Any] = dict(
        initial_state=initial_states_repeated,
        n_steps=n_timepoints,
        dt=dt,
        sde_type=sde_type,
        method=method,
        dt_min=dt_min,
        use_adjoint=use_adjoint,
        adjoint_method=adjoint_method,
    )
    if control is not None:
        # Repeat the first subject's control for all simulations
        fwd_kwargs["control"] = control[:1].expand(n_simulations, -1)
    with torch.no_grad():
        sim_ts = model.forward(**fwd_kwargs)
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
    """Log ``a``, ``G``, and ``kappa`` for a trained Hopf model."""
    if not (hasattr(model, "a") and hasattr(model, "g")):
        return
    best_a = model.a.item() if model.a.dim() == 0 else model.a.mean().item()
    best_G = model.g.item() if model.g.dim() == 0 else model.g.mean().item()
    payload = {"best_params/a": best_a, "best_params/G": best_G}
    summary = {"best_a": best_a, "best_G": best_G}
    if hasattr(model, "kappa"):
        kappa_val = model.kappa
        if isinstance(kappa_val, torch.Tensor):
            best_kappa = kappa_val.item() if kappa_val.dim() == 0 else kappa_val.mean().item()
        else:
            best_kappa = float(kappa_val)
        payload["best_params/kappa"] = best_kappa
        summary["best_kappa"] = best_kappa
    else:
        best_kappa = None
    wandb_log(payload, use_wandb=use_wandb)
    wandb_summary_update(summary, use_wandb=use_wandb)
    kappa_str = f", kappa: {best_kappa:.6f}" if best_kappa is not None else ""
    print(f"Best Hopf params — a: {best_a:.6f}, G: {best_G:.6f}{kappa_str}")


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


# ---------------------------------------------------------------------------
# Hopf evaluation helpers (moved from train_hopf.py)
# ---------------------------------------------------------------------------

def split_subject_indices(cfg, n_subjects: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic train/validation/test subject split."""
    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(n_subjects, generator=generator)
    n_train = max(1, int(cfg.train_ratio * n_subjects))
    n_val = max(1, int(cfg.val_ratio * n_subjects))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    if val_idx.numel() == 0:
        val_idx = train_idx[:1]
    if test_idx.numel() == 0:
        test_idx = val_idx[:1]
    return train_idx, val_idx, test_idx


def _forward_for_metrics(
    model,
    initial_state: torch.Tensor,
    n_steps: int,
    cfg,
    control: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward helper used to keep metric simulation settings identical."""
    kwargs: dict[str, Any] = dict(
        initial_state=initial_state,
        n_steps=n_steps,
        dt=cfg.tr,
        sde_type=getattr(cfg, "sde_type", "ito"),
        method=getattr(cfg, "sde_method", "euler"),
        dt_min=getattr(cfg, "dt_min", 0.1),
        use_adjoint=getattr(cfg, "use_adjoint", False),
        adjoint_method=getattr(cfg, "adjoint_method", None),
    )
    if control is not None:
        kwargs["control"] = control
    return model.forward(**kwargs)


EVAL_METRIC_KEYS = (
    "fc_correlation",
    "fc_mse",
    "temporal_correlation",
    "power_spectrum_distance",
    "autocorr_distance",
    "fcd_ks",
    "phfcd_ks",
    "phase_fc_correlation",
    "metastability_diff",
)


def evaluate_model_loader_metrics(
    model,
    loader,
    cfg,
    *,
    n_steps: int | None = None,
    return_std: bool = False,
) -> dict[str, float]:
    """
    Evaluate a model on the provided data loader.

    Uses the same ``module.evaluate()`` pattern as :class:`Trainer`.
    When *return_std* is True, metrics are computed **per sample** (not per
    batch) so that the reported standard deviations reflect true cross-sample
    variability rather than the much smaller variability between batch-level
    means.

    Args:
        model: Model to evaluate
        loader: DataLoader
        cfg: Config containing simulation settings
        n_steps: Optional simulation length override
        return_std: If True, also return <metric>_std keys computed across
                    individual samples

    Returns:
        Dictionary of metric_name → mean (and optionally metric_name_std → std)
    """
    from ..metrics import (
        FCCorrelation, FCMSE,
        FCD, PhFCD, Metastability, PhaseFC,
        PowerSpectrumDistance, TemporalCorrelation, AutocorrelationDistance,
    )

    eval_modules = [
        FCCorrelation(), FCMSE(),
        PowerSpectrumDistance(), TemporalCorrelation(), AutocorrelationDistance(),
        FCD(tr=cfg.tr, fcd_win_sec=cfg.fcd_win_sec, fcd_step_sec=cfg.fcd_step_sec),
        PhFCD(), Metastability(), PhaseFC(),
    ]

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    sample_values: dict[str, list[float]] = {}  # per-sample values for std

    for batch in loader:
        if len(batch) == 4:
            windows, _, _, control = batch
        else:
            windows, _, _ = batch
            control = None
        windows = windows.to(model.device)
        if control is not None:
            control = control.to(model.device)
            if control.shape[-1] == 0:
                control = None

        batch_steps = windows.shape[2]
        n_sim_steps = min(batch_steps, n_steps) if n_steps is not None else batch_steps
        target_window = windows[:, :, :n_sim_steps]
        initial_state = target_window[:, :, 0]

        with torch.no_grad():
            simulated = _forward_for_metrics(model, initial_state, n_sim_steps, cfg, control=control)

            if return_std:
                # Per-sample metrics for proper std computation
                B = simulated.shape[0]
                for i in range(B):
                    sim_i = simulated[i : i + 1]
                    tgt_i = target_window[i : i + 1]
                    sample_metrics: dict[str, float] = {}
                    for module in eval_modules:
                        sample_metrics.update(module.evaluate(sim_i, tgt_i))
                    for key, value in sample_metrics.items():
                        numeric = to_float_metric(value)
                        if numeric is None or math.isnan(numeric):
                            continue
                        sums[key] = sums.get(key, 0.0) + numeric
                        counts[key] = counts.get(key, 0) + 1
                        sample_values.setdefault(key, []).append(numeric)
            else:
                # Batch-level aggregate (faster, no std needed)
                batch_metrics: dict[str, float] = {}
                for module in eval_modules:
                    batch_metrics.update(module.evaluate(simulated, target_window))
                for key, value in batch_metrics.items():
                    numeric = to_float_metric(value)
                    if numeric is None or math.isnan(numeric):
                        continue
                    sums[key] = sums.get(key, 0.0) + numeric
                    counts[key] = counts.get(key, 0) + 1

    # Return all computed metrics (use EVAL_METRIC_KEYS for formatted reports).
    all_keys = sorted(set(EVAL_METRIC_KEYS) | set(sums.keys()))
    result = {
        key: (sums[key] / counts[key]) if counts.get(key, 0) > 0 else float("nan")
        for key in all_keys
    }

    if return_std:
        for key in all_keys:
            vals = sample_values.get(key, [])
            if len(vals) > 1:
                result[f"{key}_std"] = torch.tensor(vals).std().item()
            else:
                result[f"{key}_std"] = float("nan")

    return result
