"""Plotting helpers for PE design comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .design import DesignResult


def _criterion_colors(criteria: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Assign a stable color to each criterion."""
    cmap = plt.get_cmap("tab10")
    return {criterion: cmap(index % cmap.N) for index, criterion in enumerate(criteria)}


def _method_style(method: str) -> tuple[str, str]:
    """Return line and marker style for one optimization method."""
    if method == "backprop":
        return "-", "o"
    if method == "sdp":
        return "--", "s"
    if method == "sdp+backprop":
        return "-.", "^"
    return ":", "d"


def _waveform_matrix(waveform: np.ndarray) -> np.ndarray:
    """Return waveform samples with shape ``(signal_dim, n_grid)``."""
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim == 1:
        return values[None, :]
    if values.ndim != 2:
        raise ValueError(f"Expected waveform with 1 or 2 dimensions, got shape {values.shape}.")
    return values


def _canonicalize_waveform_sign(waveform: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Fix the global sign ambiguity so the dominant sample is positive."""
    values = _waveform_matrix(waveform)
    flat = values.reshape(-1)
    index = int(np.argmax(np.abs(flat)))
    if flat.size == 0 or abs(float(flat[index])) <= eps:
        return values
    sign = 1.0 if float(flat[index]) >= 0.0 else -1.0
    return sign * values


def plot_design_summary(
    results: dict[str, dict[str, DesignResult]],
    grid: np.ndarray,
    *,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot eigenvalues and optimized waveforms in a single summary figure."""
    criteria = list(results.keys())
    colors = _criterion_colors(criteria)
    floor = np.finfo(np.float64).tiny
    first_result = next(iter(next(iter(results.values())).values()))
    signal_dim = _waveform_matrix(first_result.waveform).shape[0]
    figsize = figsize or (4.8 * (signal_dim + 1), 4.8)

    fig, axes = plt.subplots(1, signal_dim + 1, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    ax_eigs = axes_flat[0]
    wave_axes = list(axes_flat[1:])

    max_eig_count = 0
    for criterion in criteria:
        for method, result in results[criterion].items():
            linestyle, marker = _method_style(method)
            label = f"{criterion.upper()} ({method})"

            eigenvalues = np.sort(np.asarray(result.eigenvalues, dtype=np.float64))
            max_eig_count = max(max_eig_count, eigenvalues.size)
            ax_eigs.plot(
                np.arange(1, eigenvalues.size + 1, dtype=int),
                np.maximum(eigenvalues, floor),
                color=colors[criterion],
                linestyle=linestyle,
                marker=marker,
                label=label,
                linewidth=2.0,
            )

            waveform = _canonicalize_waveform_sign(result.waveform)
            for component, ax_wave in enumerate(wave_axes):
                ax_wave.plot(
                    grid,
                    waveform[component],
                    color=colors[criterion],
                    linestyle=linestyle,
                    linewidth=2.0,
                )

    ax_eigs.set_title("PE Gramian Eigenvalues")
    ax_eigs.set_xlabel("Eigenvalue index")
    ax_eigs.set_ylabel("Eigenvalue")
    ax_eigs.set_yscale("log")
    if max_eig_count > 0:
        ax_eigs.set_xticks(np.arange(1, max_eig_count + 1, dtype=int))
    ax_eigs.grid(True, which="both", alpha=0.3)
    ax_eigs.legend(loc="upper left")

    for component, ax_wave in enumerate(wave_axes, start=1):
        title = "Optimized Waveforms" if signal_dim == 1 else f"Waveform Component {component}"
        ylabel = "u(t)" if signal_dim == 1 else rf"$u_{{{component}}}(t)$"
        ax_wave.set_title(title)
        ax_wave.set_xlabel("Time")
        ax_wave.set_ylabel(ylabel)
        ax_wave.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig
