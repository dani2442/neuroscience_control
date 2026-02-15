import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline

# Ensure imports work when running this file directly (absolute or relative path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import CoupledHopfModel, NeuralSDE
from src.dataset import NeuroscienceDataset, compute_omega_from_timeseries

# ── Config ──────────────────────────────────────────────────────────────────
TAIL_LEN = 80          # number of segments visible in the comet tail
INTERP_FACTOR = 8      # cubic-spline up-sampling for both traces
N_STEPS =100          # SDE evaluation points (dt=0.72, dt_min=0.05)
SIM_DT = 0.72          # keep original dt to avoid Brownian tree overflow
SIM_DT_MIN = 0.05      # sub-step for the SDE solver
FPS = 20               # gif frame rate
FRAME_SKIP = 2         # render every N-th interpolated point
OUTPUT = "paper/images/trajectory3.gif"

# ── Data & model ────────────────────────────────────────────────────────────
dataset = NeuroscienceDataset(
    filepath="data/ts_young/ts_young_TR0.72.mat",
    normalize=True, max_subjects=2, fourier_denoise=True,
)
omega = compute_omega_from_timeseries(
    dataset.timeseries, dt=dataset.dt, f_lo=0.04, f_hi=0.07, method="peak",
)

patient = 0
ROIS = [0, 25, 50]  # three ROIs to compare side-by-side

model = CoupledHopfModel(
    n_rois=100, initial_a=-0.02, initial_g=1.5, initial_kappa=0.1,
    noise_sigma=0.2, omega=omega, structural_connectivity=dataset.fc_mean,
)

# ── Simulate with small dt ──────────────────────────────────────────────────
with torch.no_grad():
    z0 = dataset.timeseries[:, :, 0]
    zs = model.forward(initial_state=z0, n_steps=N_STEPS, dt=SIM_DT, dt_min=SIM_DT_MIN)

# ── Per-ROI interpolated trajectories ───────────────────────────────────────
sim_xs, sim_ys, data_xs, data_ys = [], [], [], []

for roi in ROIS:
    # Simulation
    srx = zs[patient, roi].real.cpu().numpy()
    sry = zs[patient, roi].imag.cpu().numpy()
    t_sim = np.arange(len(srx))
    t_sim_fine = np.linspace(0, len(srx) - 1, len(srx) * INTERP_FACTOR)
    sim_xs.append(CubicSpline(t_sim, srx)(t_sim_fine))
    sim_ys.append(CubicSpline(t_sim, sry)(t_sim_fine))

    # Recorded data
    d_raw = dataset.timeseries[patient, roi].cpu().numpy()
    t_orig = np.arange(len(d_raw))
    t_fine = np.linspace(0, len(d_raw) - 1, len(d_raw) * INTERP_FACTOR)
    data_xs.append(CubicSpline(t_orig, d_raw.real)(t_fine))
    data_ys.append(CubicSpline(t_orig, d_raw.imag)(t_fine))

# Trim all to same frame count
n_anim_frames = min(
    *(len(sx) // FRAME_SKIP for sx in sim_xs),
    *(len(dx) // FRAME_SKIP for dx in data_xs),
)

# ── Colour helpers ──────────────────────────────────────────────────────────

def make_trail_segments(x, y, head, tail_len):
    """Return LineCollection segments + per-segment RGBA for a comet tail."""
    start = max(0, head - tail_len)
    xs, ys = x[start:head + 1], y[start:head + 1]
    if len(xs) < 2:
        return None, None
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    # alpha ramp: 0 → 1 along the tail
    alphas = np.linspace(0.0, 1.0, n)
    return segs, alphas


# ── Figure setup (dark theme, 1×3 grid) ─────────────────────────────────────
plt.style.use("dark_background")
n_cols = len(ROIS)
fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), facecolor="#0d1117")
if n_cols == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.tick_params(colors="#444")
    for spine in ax.spines.values():
        spine.set_color("#222")
    ax.grid(True, color="#1a1f29", linewidth=0.5)
    ax.set_title(f"ROI {ROIS[i]}", color="#c9d1d9", fontsize=12, pad=10)

# Colormaps for the two trails
cmap_sim = plt.cm.inferno   # warm fire tones
cmap_dat = plt.cm.cool      # cool cyan-magenta

# Colours for head dots / legend
COLOR_SIM = "#ffcc33"
COLOR_DAT = "#66ffee"

# Per-axis artists
lc_sims, lc_dats = [], []
head_sims, head_dats = [], []
glow_sims, glow_dats = [], []

for ax in axes:
    lc_s = LineCollection([], linewidths=2, cmap=cmap_sim)
    lc_d = LineCollection([], linewidths=2, cmap=cmap_dat)
    ax.add_collection(lc_s)
    ax.add_collection(lc_d)
    lc_sims.append(lc_s)
    lc_dats.append(lc_d)

    (hs,) = ax.plot([], [], "o", color=COLOR_SIM, markersize=7, zorder=5)
    (hd,) = ax.plot([], [], "o", color=COLOR_DAT, markersize=7, zorder=5)
    head_sims.append(hs)
    head_dats.append(hd)

    (gs,) = ax.plot([], [], "o", color=COLOR_SIM, markersize=16, alpha=0.25, zorder=4)
    (gd,) = ax.plot([], [], "o", color=COLOR_DAT, markersize=16, alpha=0.25, zorder=4)
    glow_sims.append(gs)
    glow_dats.append(gd)

# ── Shared legend (placed once on the figure) ──────────────────────────────
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLOR_SIM, lw=2, marker="o", markersize=5, label="Simulation (Hopf SDE)"),
    Line2D([0], [0], color=COLOR_DAT, lw=2, marker="o", markersize=5, label="Recorded data"),
]
fig.legend(
    handles=legend_elements, loc="lower center", ncol=2,
    frameon=False, fontsize=10, labelcolor="#c9d1d9",
)
fig.subplots_adjust(bottom=0.13)

# Timer label (inside last axes so it survives blit)
timer_text = axes[-1].text(
    0.98, 0.96, "t = 0.00 s", ha="right", va="top",
    fontsize=11, color="#c9d1d9", family="monospace",
    transform=axes[-1].transAxes, zorder=10,
)


# ── Animation ───────────────────────────────────────────────────────────────

def _update(frame):
    artists = []
    for k in range(n_cols):
        idx = frame * FRAME_SKIP

        # simulation trail
        segs, alphas = make_trail_segments(sim_xs[k], sim_ys[k], idx, TAIL_LEN * FRAME_SKIP)
        if segs is not None:
            n = len(segs)
            colors = cmap_sim(np.linspace(0.25, 1.0, n))
            colors[:, 3] = alphas
            widths = np.linspace(0.5, 3.0, n)
            lc_sims[k].set_segments(segs)
            lc_sims[k].set_color(colors)
            lc_sims[k].set_linewidths(widths)

        head_sims[k].set_data([sim_xs[k][idx]], [sim_ys[k][idx]])
        glow_sims[k].set_data([sim_xs[k][idx]], [sim_ys[k][idx]])

        # data trail
        segs_d, alphas_d = make_trail_segments(data_xs[k], data_ys[k], idx, TAIL_LEN * FRAME_SKIP)
        if segs_d is not None:
            n = len(segs_d)
            colors_d = cmap_dat(np.linspace(0.25, 1.0, n))
            colors_d[:, 3] = alphas_d
            widths_d = np.linspace(0.5, 3.0, n)
            lc_dats[k].set_segments(segs_d)
            lc_dats[k].set_color(colors_d)
            lc_dats[k].set_linewidths(widths_d)

        head_dats[k].set_data([data_xs[k][idx]], [data_ys[k][idx]])
        glow_dats[k].set_data([data_xs[k][idx]], [data_ys[k][idx]])

        artists += [lc_sims[k], lc_dats[k], head_sims[k], head_dats[k],
                    glow_sims[k], glow_dats[k]]

    # Update timer: convert interpolated index back to real seconds
    t_seconds = (frame * FRAME_SKIP / INTERP_FACTOR) * SIM_DT
    timer_text.set_text(f"t = {t_seconds:6.2f} s")
    artists.append(timer_text)

    return artists


print(f"Rendering {n_anim_frames} frames …")
ani = animation.FuncAnimation(
    fig, _update, frames=n_anim_frames, interval=1000 // FPS, blit=True,
)
ani.save(OUTPUT, writer="pillow", fps=FPS, dpi=120)
print(f"Saved → {OUTPUT}")
plt.close(fig)