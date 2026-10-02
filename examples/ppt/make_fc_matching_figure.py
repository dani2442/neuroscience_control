"""Empirical vs simulated FC matrices for the loss slide.

Empirical FC comes from the HCP young cohort (Schaefer-100, TR = 0.72 s),
preprocessed exactly as in the talk: standardise, band-pass [0.008, 0.08] Hz.
The simulated FC is produced by the repo's own CoupledHopfModel, with omega
estimated from the data and (a, G) picked by a small sweep on FC correlation.
"""

import os
import sys

import numpy as np
import scipy.io as sio
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

ROOT = "/home/dani/projects/neuroscience_control"
OUT = f"{ROOT}/paper_submission/images/ppt"
sys.path.insert(0, ROOT)
os.makedirs(OUT, exist_ok=True)

from src.models.hopf_model import CoupledHopfModel          # noqa: E402
from src.dataset.preprocessing import compute_omega_from_timeseries  # noqa: E402

TR = 0.72
F_LO, F_HI = 0.008, 0.08
SEED = 0
torch.manual_seed(SEED)


def bandpass(x, dt, f_lo, f_hi):
    b, a = butter(2, [f_lo, f_hi], btype="band", fs=1.0 / dt)
    return filtfilt(b, a, x, axis=-1)


def fc(ts):
    """Pearson FC of (n_rois, T) or mean FC over (n_sub, n_rois, T)."""
    ts = np.asarray(ts)
    if ts.ndim == 2:
        ts = ts[None]
    z = (ts - ts.mean(-1, keepdims=True)) / (ts.std(-1, keepdims=True) + 1e-8)
    return np.mean(z @ z.transpose(0, 2, 1) / z.shape[-1], axis=0)


def offdiag(m):
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


CACHE = f"{OUT}/fc_pair.npz"

if os.path.exists(CACHE):
    d = np.load(CACHE)
    FC_emp, FC_sim, r = d["FC_emp"], d["FC_sim"], float(d["r"])
    print(f"cached FC pair (FC corr = {r:.3f}) -- delete {CACHE} to recompute")
else:
    # ---------------------------------------------------------------- empirical
    raw = sio.loadmat(f"{ROOT}/data/ts_young/ts_young_TR0.72.mat")["timeseries_all"]
    raw = np.transpose(raw, (2, 0, 1))                     # (n_sub, 100, 1200)
    n_sub, n_rois, T = raw.shape
    print(f"data: {n_sub} subjects, {n_rois} ROIs, {T} timepoints")

    emp = (raw - raw.mean(-1, keepdims=True)) / (raw.std(-1, keepdims=True) + 1e-8)
    emp = np.ascontiguousarray(bandpass(emp, TR, F_LO, F_HI))
    FC_emp = fc(emp)

    omega = compute_omega_from_timeseries(
        torch.tensor(emp, dtype=torch.float32), dt=TR, f_lo=0.04, f_hi=0.07, method="peak"
    )

    # ---------------------------------------------------------------- simulated
    # Sparse coupling matrix from the functional connectome: keep the strongest
    # 10% of connections.  A dense coupling matrix is dominated by its
    # mean-field term and washes out all network structure.
    C = np.abs(FC_emp).copy()
    np.fill_diagonal(C, 0.0)
    C[C < np.percentile(C, 90)] = 0.0
    conn = torch.tensor(C, dtype=torch.float32)


    def simulate(a, g, kappa=1.0, sigma=0.1, n_batch=8):
        torch.manual_seed(SEED)
        model = CoupledHopfModel(
            n_rois=n_rois, connectome=conn, initial_a=a, initial_g=g,
            initial_kappa=kappa, omega=omega, noise_sigma=sigma, device="cpu",
        ).eval()
        z0 = 0.1 * torch.randn(n_batch, n_rois, dtype=torch.cfloat)
        with torch.no_grad():
            z = model(z0, n_steps=T, dt=TR, dt_min=TR / 2)
        x = z.real.numpy()
        x = np.ascontiguousarray(bandpass(x, TR, F_LO, F_HI))
        return fc(x)


    A, G = -0.05, 1.0          # best of a coarse sweep on FC correlation
    FC_sim = simulate(A, G)
    r = np.corrcoef(offdiag(FC_emp), offdiag(FC_sim))[0, 1]
    print(f"a={A:+.2f} G={G:.1f}  FC corr = {r:.3f}")
    np.savez(f"{OUT}/fc_pair.npz", FC_emp=FC_emp, FC_sim=FC_sim, a=A, g=G, r=r)

# ------------------------------------------------------------------ panels
# Two bare matrices; titles and the connecting arrow are drawn in TikZ.
plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm"})
VMIN, VMAX = -0.3, 1.0

for name, M in (("fc_emp", FC_emp), ("fc_sim", FC_sim)):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(M, cmap="RdBu_r", vmin=VMIN, vmax=VMAX, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.55"); sp.set_linewidth(0.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(f"{OUT}/{name}.pdf", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf")
