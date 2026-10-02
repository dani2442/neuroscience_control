"""Animated 'analytic signal' panel for the presentation.

One real BOLD trace on the left, the same instant as a point of the analytic
signal z(t) = x(t) + i H[x](t) in the complex plane on the right.
Output: PNG frames for beamer's \\animategraphics.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import butter, filtfilt, hilbert

ROOT = "/home/dani/projects/neuroscience_control"
OUT = f"{ROOT}/paper_submission/images/ppt/frames_analytic"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
    "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREY = "#2B6CB0", "#E4572E", "#8A8A8A"

# ---------------------------------------------------------------- data
TR = 0.72
ts = sio.loadmat(f"{ROOT}/data/ts_young/ts_young_TR0.72.mat")["timeseries_all"][:, :, 0]
b, a_ = butter(2, [0.008, 0.08], btype="band", fs=1 / TR)
x = filtfilt(b, a_, (ts[7] - ts[7].mean()) / ts[7].std())
x = x[150:400]
t = np.arange(x.size) * TR
z = hilbert(x)

# every other sample, played at 12 fps  ->  ~10 s per loop
STRIDE = 2
KS = list(range(0, x.size, STRIDE))
TRAIL = 60                                   # samples kept in the complex-plane trail
lim = 1.05 * max(np.abs(z).max(), np.abs(x).max())

for n, k in enumerate(KS):
    fig = plt.figure(figsize=(13.0, 4.6), dpi=96)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.75, 1.0], wspace=0.22,
                           left=0.055, right=0.985, top=0.90, bottom=0.135)

    # ---- left: the real trace and its Hilbert transform
    ax = fig.add_subplot(gs[0])
    ax.plot(t, x, color="k", lw=1.0, alpha=0.18)
    ax.plot(t, z.imag, color=BLUE, lw=1.0, alpha=0.15)
    ax.plot(t[:k + 1], x[:k + 1], color="k", lw=2.0, label=r"$x(t)$")
    ax.plot(t[:k + 1], z.imag[:k + 1], color=BLUE, lw=1.6,
            label=r"$\mathcal{H}[x](t)$")
    ax.axvline(t[k], color=GREY, lw=1.0, ls="--")
    ax.plot(t[k], x[k], "o", color="k", ms=9, zorder=5)
    ax.plot(t[k], z.imag[k], "o", color=BLUE, ms=9, zorder=5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(-lim, lim)
    ax.set_xlabel("time  [s]")
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", ncol=1, fontsize=17, frameon=False,
              handlelength=1.6, labelspacing=0.4)

    # ---- right: the same instant in the complex plane
    ax = fig.add_subplot(gs[1])
    ax.plot(z.real, z.imag, color=GREY, lw=0.5, alpha=0.22)
    lo = max(0, k - TRAIL)
    seg = z[lo:k + 1]
    if seg.size > 1:
        pts = np.stack([seg.real, seg.imag], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        alphas = np.linspace(0.05, 0.9, len(segs))
        lc = matplotlib.collections.LineCollection(
            segs, colors=[(0.0, 0.0, 0.0, al) for al in alphas], linewidths=1.4)
        ax.add_collection(lc)
    ax.plot([0, z.real[k]], [0, z.imag[k]], color=RED, lw=2.2, zorder=4)
    ax.plot(z.real[k], z.imag[k], "o", color=RED, ms=10, zorder=5)
    # projections onto the two axes = the two traces on the left
    ax.plot([z.real[k], z.real[k]], [0, z.imag[k]], color=GREY, lw=0.9, ls=":")
    ax.plot([0, z.real[k]], [z.imag[k], z.imag[k]], color=GREY, lw=0.9, ls=":")
    ax.plot(z.real[k], 0, "o", color="k", ms=6, zorder=5)
    ax.plot(0, z.imag[k], "o", color=BLUE, ms=6, zorder=5)
    ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(r"$z(t)=x+i\,\mathcal{H}[x]$")
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(f"{OUT}/f-{n:03d}.jpg", dpi=96, facecolor="white",
                pil_kwargs={"quality": 88})
    plt.close(fig)
    if n % 25 == 0:
        print("frame", n, flush=True)

print(f"done -> {OUT}  ({len(KS)} frames, last index {len(KS) - 1:03d})")
