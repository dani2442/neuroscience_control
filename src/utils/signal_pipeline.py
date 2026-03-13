"""Helpers to visualize the BOLD -> FFT -> Hilbert processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..dataset import fft_bandpass_3d, hilbert_transform


@dataclass(slots=True)
class SignalPipelineResult:
    """Container with all stages of a single-subject signal pipeline."""

    time: np.ndarray
    frequencies: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray
    filtered: np.ndarray
    analytic: np.ndarray
    envelope: np.ndarray
    phase: np.ndarray
    raw_spectrum: np.ndarray
    filtered_spectrum: np.ndarray
    roi_indices: list[int]
    focus_roi: int
    subject_index: int
    dt: float
    f_lo: float
    f_hi: float


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def _resolve_subject(timeseries: torch.Tensor | np.ndarray) -> torch.Tensor:
    ts = torch.as_tensor(timeseries)
    if torch.is_complex(ts):
        ts = ts.real
    ts = ts.to(dtype=torch.float32)
    if ts.dim() == 2:
        ts = ts.unsqueeze(0)
    if ts.dim() != 3:
        raise ValueError("timeseries must have shape (subjects, rois, timepoints) or (rois, timepoints).")
    return ts


def _validate_roi_indices(roi_indices: Sequence[int], n_rois: int) -> list[int]:
    if not roi_indices:
        raise ValueError("roi_indices must contain at least one ROI.")
    cleaned: list[int] = []
    seen: set[int] = set()
    for roi in roi_indices:
        roi_int = int(roi)
        if roi_int < 0 or roi_int >= n_rois:
            raise ValueError(f"ROI index {roi_int} is out of bounds for {n_rois} ROIs.")
        if roi_int not in seen:
            cleaned.append(roi_int)
            seen.add(roi_int)
    return cleaned


def prepare_signal_pipeline(
    timeseries: torch.Tensor | np.ndarray,
    *,
    subject_index: int = 0,
    roi_indices: Optional[Sequence[int]] = None,
    focus_roi: Optional[int] = None,
    dt: float = 0.72,
    f_lo: float = 0.008,
    f_hi: float = 0.08,
    normalize: bool = True,
    max_timepoints: Optional[int] = 240,
) -> SignalPipelineResult:
    """Compute explicit BOLD, FFT-bandpassed, and Hilbert stages for one subject."""

    ts = _resolve_subject(timeseries)
    n_subjects, n_rois, n_timepoints = ts.shape

    if subject_index < 0 or subject_index >= n_subjects:
        raise ValueError(f"subject_index must be in [0, {n_subjects - 1}], got {subject_index}.")
    if n_timepoints < 4:
        raise ValueError("Need at least 4 timepoints to visualize the signal pipeline.")
    if dt <= 0.0:
        raise ValueError("dt must be strictly positive.")
    if f_lo < 0.0 or f_hi <= f_lo:
        raise ValueError("Expected 0 <= f_lo < f_hi for the band-pass filter.")

    if roi_indices is None:
        n_default = min(3, n_rois)
        roi_indices = np.linspace(0, n_rois - 1, n_default, dtype=int).tolist()
    selected_rois = _validate_roi_indices(roi_indices, n_rois)

    if focus_roi is None:
        focus_roi = selected_rois[len(selected_rois) // 2]
    focus_roi = int(focus_roi)
    if focus_roi < 0 or focus_roi >= n_rois:
        raise ValueError(f"focus_roi must be in [0, {n_rois - 1}], got {focus_roi}.")
    if focus_roi not in selected_rois:
        selected_rois = [*selected_rois, focus_roi]

    subject_ts = ts[subject_index : subject_index + 1]
    if normalize:
        mean = subject_ts.mean(dim=2, keepdim=True)
        std = subject_ts.std(dim=2, keepdim=True) + 1e-8
        normalized = (subject_ts - mean) / std
    else:
        normalized = subject_ts.clone()

    filtered = fft_bandpass_3d(normalized, dt=dt, f_lo=f_lo, f_hi=f_hi)
    analytic = hilbert_transform(filtered)

    t_limit = n_timepoints if max_timepoints is None else min(int(max_timepoints), n_timepoints)
    if t_limit < 2:
        raise ValueError("max_timepoints must leave at least 2 samples for plotting.")

    focus_real = _to_numpy(normalized[0, focus_roi])
    focus_filtered = _to_numpy(filtered[0, focus_roi])
    focus_analytic = _to_numpy(analytic[0, focus_roi, :t_limit])

    return SignalPipelineResult(
        time=np.arange(t_limit, dtype=np.float32) * dt,
        frequencies=np.fft.rfftfreq(n_timepoints, d=dt),
        raw=_to_numpy(subject_ts[0, selected_rois, :t_limit]),
        normalized=_to_numpy(normalized[0, selected_rois, :t_limit]),
        filtered=_to_numpy(filtered[0, selected_rois, :t_limit]),
        analytic=focus_analytic,
        envelope=np.abs(focus_analytic),
        phase=np.unwrap(np.angle(focus_analytic)),
        raw_spectrum=np.abs(np.fft.rfft(focus_real)),
        filtered_spectrum=np.abs(np.fft.rfft(focus_filtered)),
        roi_indices=selected_rois,
        focus_roi=focus_roi,
        subject_index=subject_index,
        dt=dt,
        f_lo=f_lo,
        f_hi=f_hi,
    )


def _stack_signals(
    ax: plt.Axes,
    time: np.ndarray,
    signals: np.ndarray,
    roi_indices: Sequence[int],
    focus_roi: int,
    *,
    title: str,
    ylabel: str,
) -> None:
    palette = ["#345995", "#03cea4", "#fb4d3d", "#ca1551", "#eac435"]
    spacing = max(2.5, 1.4 * float(np.max(np.ptp(signals, axis=1))))

    for idx, roi in enumerate(roi_indices):
        trace = signals[idx]
        offset = (len(roi_indices) - idx - 1) * spacing
        color = "#111111" if roi == focus_roi else palette[idx % len(palette)]
        line_width = 2.0 if roi == focus_roi else 1.5
        ax.plot(time, trace + offset, color=color, linewidth=line_width)
        ax.text(
            time[0],
            offset + np.mean(trace[: min(10, trace.shape[0])]),
            f"ROI {roi}",
            color=color,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.grid(True, alpha=0.25)


def _add_pipeline_arrow(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes, label: str) -> None:
    to_fig = fig.transFigure.inverted().transform
    left_point = to_fig(left_ax.transAxes.transform((1.02, 0.55)))
    right_point = to_fig(right_ax.transAxes.transform((-0.02, 0.55)))
    arrow = FancyArrowPatch(
        left_point,
        right_point,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color="#4c566a",
    )
    fig.add_artist(arrow)
    fig.text(
        (left_point[0] + right_point[0]) / 2.0,
        max(left_point[1], right_point[1]) + 0.035,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4c566a",
    )


def _plot_hilbert_trajectory(ax: plt.Axes, analytic: np.ndarray, time: np.ndarray, focus_roi: int) -> None:
    x = analytic.real
    y = analytic.imag

    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    line = LineCollection(
        segments,
        cmap="viridis",
        norm=Normalize(vmin=float(time[0]), vmax=float(time[-1])),
        linewidths=2.0,
    )
    line.set_array(time[:-1])
    ax.add_collection(line)

    x_pad = max(0.15, 0.08 * float(np.ptp(x)))
    y_pad = max(0.15, 0.08 * float(np.ptp(y)))
    ax.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
    ax.set_ylim(float(y.min() - y_pad), float(y.max() + y_pad))
    ax.scatter(x[0], y[0], color="#03cea4", s=40, label="start", zorder=3)
    ax.scatter(x[-1], y[-1], color="#fb4d3d", s=40, label="end", zorder=3)
    ax.set_title(f"Hilbert Analytic Signal\n(ROI {focus_roi})", fontsize=12, pad=10)
    ax.set_xlabel("Re{z(t)}")
    ax.set_ylabel("Im{z(t)}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def plot_signal_pipeline(
    pipeline: SignalPipelineResult,
    *,
    title: str = "Data Pipeline: BOLD -> FFT band-pass -> Hilbert transform",
    figsize: tuple[float, float] = (19.0, 5.8),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the full data pipeline in a single left-to-right figure."""

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("white")

    _stack_signals(
        axes[0],
        pipeline.time,
        pipeline.raw,
        pipeline.roi_indices,
        pipeline.focus_roi,
        title="1. BOLD Signal",
        ylabel="Stacked ROIs",
    )

    raw_spec = pipeline.raw_spectrum / (pipeline.raw_spectrum.max() + 1e-8)
    filtered_spec = pipeline.filtered_spectrum / (pipeline.filtered_spectrum.max() + 1e-8)
    axes[1].plot(pipeline.frequencies, raw_spec, color="#8d99ae", linewidth=2.0, label="Normalized BOLD")
    axes[1].plot(
        pipeline.frequencies,
        filtered_spec,
        color="#d62828",
        linewidth=2.0,
        label="Band-passed",
    )
    axes[1].axvspan(pipeline.f_lo, pipeline.f_hi, color="#ffd166", alpha=0.25, label="Pass band")
    axes[1].set_title(f"2. FFT / Band Selection\n[{pipeline.f_lo:.3f}, {pipeline.f_hi:.3f}] Hz", fontsize=12, pad=10)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative amplitude")
    axes[1].set_xlim(left=0.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)

    _stack_signals(
        axes[2],
        pipeline.time,
        pipeline.filtered,
        pipeline.roi_indices,
        pipeline.focus_roi,
        title="3. Filtered BOLD",
        ylabel="Stacked ROIs",
    )

    _plot_hilbert_trajectory(axes[3], pipeline.analytic, pipeline.time, pipeline.focus_roi)

    envelope_ax = inset_axes(axes[3], width="49%", height="28%", loc="upper left", borderpad=1.1)
    envelope_ax.plot(pipeline.time, pipeline.envelope, color="#ff9f1c", linewidth=1.4)
    envelope_ax.fill_between(pipeline.time, pipeline.envelope, color="#ff9f1c", alpha=0.15)
    envelope_ax.set_title("Envelope", fontsize=8, pad=4)
    envelope_ax.tick_params(labelsize=7)
    envelope_ax.grid(True, alpha=0.2)

    phase_ax = inset_axes(axes[3], width="49%", height="28%", loc="lower left", borderpad=1.1)
    phase_ax.plot(pipeline.time, pipeline.phase, color="#6a4c93", linewidth=1.4)
    phase_ax.set_title("Phase", fontsize=8, pad=4)
    phase_ax.tick_params(labelsize=7)
    phase_ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{title}  |  subject={pipeline.subject_index}, focus ROI={pipeline.focus_roi}",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.14, top=0.82, wspace=0.34)
    fig.canvas.draw()

    _add_pipeline_arrow(fig, axes[0], axes[1], "FFT")
    _add_pipeline_arrow(fig, axes[1], axes[2], "Band-pass")
    _add_pipeline_arrow(fig, axes[2], axes[3], "Hilbert")

    if save_path is not None:
        final_path = Path(save_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(final_path, dpi=dpi, bbox_inches="tight")

    return fig


__all__ = ["SignalPipelineResult", "prepare_signal_pipeline", "plot_signal_pipeline"]
