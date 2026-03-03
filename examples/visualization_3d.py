"""
3-D brain-surface animation of Coupled Hopf simulation over time.

Maps per-ROI amplitude onto the Schaefer 100-ROI atlas projected onto
fsaverage, producing:
  1. A GIF animation  (paper/images/brain_activity_3d.gif)
     – both hemispheres merged, with face colours that update each frame.
  2. An interactive HTML (paper/images/brain_activity_3d.html)
     – both hemispheres, with a time-slider + auto-play button.

Requires: torch, nilearn, nibabel, matplotlib, Pillow, plotly
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for rendering

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import nibabel as nib
import torch
import plotly.graph_objects as go
from nilearn import datasets, surface
from nilearn.surface import load_surf_mesh
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import CoupledHopfModel
from src.dataset import NeuroscienceDataset, compute_omega_from_timeseries

# ── Config ──────────────────────────────────────────────────────────────────
N_STEPS = 50           # number of simulation time-steps
SIM_DT = 0.72          # TR in seconds
PATIENT = 0            # subject index
N_ROIS = 100           # Schaefer atlas parcellation size
FPS = 6                # gif frame rate
OUTPUT_GIF = PROJECT_ROOT / "paper/images/brain_activity_3d.gif"
OUTPUT_HTML = PROJECT_ROOT / "paper/images/brain_activity_3d.html"

# ── Data & model ────────────────────────────────────────────────────────────
print("Loading dataset …")
dataset = NeuroscienceDataset(
    filepath="data/ts_young/ts_young_TR0.72.mat",
    normalize=True,
    max_subjects=2,
    fourier_denoise=True,
    denoise_f_lo=0.008,
    denoise_f_hi=0.1,
)
omega = compute_omega_from_timeseries(
    dataset.timeseries, dt=dataset.dt, f_lo=0.04, f_hi=0.07, method="peak",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CoupledHopfModel(
    n_rois=N_ROIS,
    initial_a=-0.02,
    initial_g=0.4,
    initial_kappa=0.1,
    noise_sigma=0.1,
    omega=omega,
    structural_connectivity=dataset.fc_mean,
    device=device,
)

print("Running simulation …")
with torch.no_grad():
    z0 = dataset.timeseries[:, :, 0]
    timeseries = model.forward(initial_state=z0, n_steps=N_STEPS, dt=SIM_DT)

# timeseries: complex (batch, n_rois, n_steps) → brain activity (real part)
activity = timeseries[PATIENT].real.cpu().numpy()  # (n_rois, n_steps)
print(f"Simulated timeseries shape: {timeseries.shape}")
print(f"Activity range: [{activity.min():.4f}, {activity.max():.4f}]")


# ── Atlas + surface setup ──────────────────────────────────────────────────
print("Fetching Schaefer atlas & fsaverage surfaces …")
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=N_ROIS, yeo_networks=7, resolution_mm=2,
)
labels_img = nib.load(atlas["maps"])
lab_data = labels_img.get_fdata().astype(int)

fsavg = datasets.fetch_surf_fsaverage()


def roi_vals_to_surf_textures(roi_vals: np.ndarray):
    """Map a length-N_ROIS vector into left/right fsaverage surface textures."""
    vol = np.zeros(lab_data.shape, dtype=np.float32)
    for i, v in enumerate(roi_vals, start=1):  # Schaefer labels are 1..N
        vol[lab_data == i] = v
    stat_img = nib.Nifti1Image(vol, labels_img.affine)
    tex_l = surface.vol_to_surf(stat_img, fsavg.pial_left)
    tex_r = surface.vol_to_surf(stat_img, fsavg.pial_right)
    return tex_l, tex_r


# Pre-compute global colour limits (symmetric around 0 for diverging cmap)
abs_max = float(np.abs(activity).max())
vmin, vmax = -abs_max, abs_max
norm = Normalize(vmin=vmin, vmax=vmax)

# ── Load and merge meshes (both hemispheres) ───────────────────────────────
mesh_l = load_surf_mesh(fsavg.infl_left)
mesh_r = load_surf_mesh(fsavg.infl_right)

coords_l, faces_l = np.asarray(mesh_l[0]), np.asarray(mesh_l[1])
coords_r, faces_r = np.asarray(mesh_r[0]), np.asarray(mesh_r[1])

# Separate hemispheres so they sit side-by-side without intersection.
# Both inflated meshes are centred near x=0, so we push each outward.
coords_l = coords_l.copy()
coords_r = coords_r.copy()

# Compute the shift needed: each hemisphere's medial edge should clear x=0
# then add extra gap between them.
hemi_gap = 5.0   # mm of empty space between the two hemispheres
shift_l = coords_l[:, 0].max() + hemi_gap / 2  # move left hemi so its max_x ≈ -gap/2
shift_r = coords_r[:, 0].min() - hemi_gap / 2  # move right hemi so its min_x ≈ +gap/2
coords_l[:, 0] -= shift_l   # left hemisphere entirely in x < 0
coords_r[:, 0] -= shift_r   # right hemisphere entirely in x > 0  (subtract negative = add)

# Offset indices for the right hemisphere so faces reference merged coords
n_verts_l = coords_l.shape[0]
coords_both = np.vstack([coords_l, coords_r])
faces_both = np.vstack([faces_l, faces_r + n_verts_l])

# Pre-compute textures for every frame
print("Pre-computing surface textures for all frames …")
all_tex_l, all_tex_r = [], []
for t in range(N_STEPS):
    tl, tr = roi_vals_to_surf_textures(activity[:, t])
    all_tex_l.append(tl)
    all_tex_r.append(tr)

all_tex_l = np.array(all_tex_l)   # (N_STEPS, n_verts_l)
all_tex_r = np.array(all_tex_r)   # (N_STEPS, n_verts_r)


def merged_vertex_intensity(t_idx: int) -> np.ndarray:
    """Concatenated vertex intensity for both hemispheres at time t_idx."""
    return np.concatenate([all_tex_l[t_idx], all_tex_r[t_idx]])


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Interactive HTML with Plotly  (both hemispheres, slider + play)
# ═══════════════════════════════════════════════════════════════════════════
print("Building interactive HTML …")

x, y, z = coords_both[:, 0], coords_both[:, 1], coords_both[:, 2]
i_f, j_f, k_f = faces_both[:, 0], faces_both[:, 1], faces_both[:, 2]

# Initial mesh (t=0)
intensity_0 = merged_vertex_intensity(0)

mesh_trace = go.Mesh3d(
    x=x, y=y, z=z,
    i=i_f, j=j_f, k=k_f,
    intensity=intensity_0,
    intensitymode="vertex",
    colorscale="RdBu_r",
    cmin=vmin, cmax=vmax,
    colorbar=dict(title="Brain Activity", len=0.6),
    lighting=dict(ambient=0.5, diffuse=0.7, specular=0.2, roughness=0.5),
    flatshading=False,
    hoverinfo="skip",
)

# One frame per time-step: only update `intensity`
frames = []
for t in range(N_STEPS):
    t_sec = t * SIM_DT
    frames.append(go.Frame(
        data=[go.Mesh3d(intensity=merged_vertex_intensity(t))],
        name=str(t),
        layout=go.Layout(title_text=f"Coupled Hopf – Brain Activity   t = {t_sec:.2f} s"),
    ))

# Slider + play/pause buttons
sliders = [dict(
    active=0,
    currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=14)),
    pad=dict(t=40),
    steps=[
        dict(args=[[str(t)], dict(frame=dict(duration=1000 // FPS, redraw=True), mode="immediate")],
             label=f"{t * SIM_DT:.1f}",
             method="animate")
        for t in range(N_STEPS)
    ],
)]

updatemenus = [dict(
    type="buttons",
    showactive=False,
    y=-0.05,
    x=0.5,
    xanchor="center",
    buttons=[
        dict(label="▶ Play",
             method="animate",
             args=[None, dict(frame=dict(duration=1000 // FPS, redraw=True),
                              fromcurrent=True, mode="immediate")]),
        dict(label="⏸ Pause",
             method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                mode="immediate")]),
    ],
)]

layout = go.Layout(
    title="Coupled Hopf – Brain Activity   t = 0.00 s",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        camera=dict(eye=dict(x=1.8, y=0.3, z=0.3)),
    ),
    sliders=sliders,
    updatemenus=updatemenus,
    margin=dict(l=0, r=0, t=50, b=60),
    width=1000,
    height=700,
)

fig_plotly = go.Figure(data=[mesh_trace], layout=layout, frames=frames)
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig_plotly.write_html(str(OUTPUT_HTML), auto_play=False)
print(f"  → {OUTPUT_HTML}")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  GIF animation (matplotlib – both hemispheres merged)
# ═══════════════════════════════════════════════════════════════════════════
print(f"Rendering {N_STEPS}-frame GIF …")

fig_mpl = plt.figure(figsize=(12, 7), facecolor="white")
ax = fig_mpl.add_subplot(111, projection="3d")

# Pre-compute fixed axis limits
_margin = 8
_xlim = (coords_both[:, 0].min() - _margin, coords_both[:, 0].max() + _margin)
_ylim = (coords_both[:, 1].min() - _margin, coords_both[:, 1].max() + _margin)
_zlim = (coords_both[:, 2].min() - _margin, coords_both[:, 2].max() + _margin)


def _render_frame(t_idx: int):
    """Draw merged brain with face colours driven by brain activity."""
    ax.clear()

    intensity = merged_vertex_intensity(t_idx)

    # Build polygon collection from faces
    verts = coords_both[faces_both]  # (n_faces, 3, 3)
    poly = Poly3DCollection(verts, linewidths=0)

    # Per-face colour = mean vertex intensity → RdBu_r (diverging)
    face_vals = intensity[faces_both].mean(axis=1)
    face_colors = cm.RdBu_r(norm(face_vals))
    poly.set_facecolor(face_colors)
    poly.set_edgecolor("none")
    ax.add_collection3d(poly)

    ax.set_xlim(*_xlim)
    ax.set_ylim(*_ylim)
    ax.set_zlim(*_zlim)
    # Oblique top-front view for a visually pleasing perspective
    ax.view_init(elev=25, azim=-120)
    ax.set_axis_off()
    ax.dist = 6.5  # zoom in slightly

    t_seconds = t_idx * SIM_DT
    fig_mpl.suptitle(
        f"Coupled Hopf – Brain Activity   t = {t_seconds:.2f} s",
        fontsize=14, fontweight="bold", y=0.95,
    )


ani = animation.FuncAnimation(
    fig_mpl, lambda f: _render_frame(f), frames=N_STEPS, interval=1000 // FPS,
)
OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)
ani.save(str(OUTPUT_GIF), writer="pillow", fps=FPS, dpi=120)
print(f"  → {OUTPUT_GIF}")
plt.close(fig_mpl)

print("Done.")