"""Animated 'what is fMRI' panel for the presentation.

Paints the Schaefer-100 parcellation with the empirical HCP BOLD signal at each
time point, renders it as a glass brain, and shows the same instant as a column
of the n x T data matrix.  Portrait-ish layout so it sits in one slide column.
Output: PNG frames for beamer's \animategraphics.
"""

import os
import numpy as np
import scipy.io as sio
import nibabel as nib
import matplotlib
from scipy.signal import butter, filtfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from nilearn import plotting, image

ROOT = "/home/dani/projects/neuroscience_control"
ATLAS = "/home/dani/nilearn_data/schaefer_2018/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
TS = f"{ROOT}/data/ts_young/ts_young_TR0.72.mat"
OUT = f"{ROOT}/paper_submission/images/ppt/frames_fmri"
os.makedirs(OUT, exist_ok=True)

TR = 0.72
SUBJECT = 0
# every TR (STRIDE=1) and played back at 8 fps -> 5.8 s of brain time per
# second of wall clock, i.e. ~4x slower than the first version.
T0, NFRAMES, STRIDE = 200, 80, 1
WIN = 140                                  # time points shown in the matrix

# ---------------------------------------------------------------- data
atlas_img = nib.load(ATLAS)
labels = atlas_img.get_fdata().astype(int)
n_rois = int(labels.max())

ts = sio.loadmat(TS)["timeseries_all"][:, :, SUBJECT]      # (100, 1200)
ts = (ts - ts.mean(axis=1, keepdims=True)) / ts.std(axis=1, keepdims=True)

# paper preprocessing: bandpass to the slow hemodynamic band, then re-standardise
b, a = butter(2, [0.008, 0.08], btype="band", fs=1.0 / TR)
ts = filtfilt(b, a, ts, axis=1)
ts = (ts - ts.mean(axis=1, keepdims=True)) / ts.std(axis=1, keepdims=True)

mask = labels > 0
idx = labels[mask] - 1
vmax = 2.2


def volume_at(t):
    vol = np.zeros(labels.shape, dtype=np.float32)
    vol[mask] = ts[idx, t]
    return image.new_img_like(atlas_img, vol)


for k in range(NFRAMES):
    t = T0 + k * STRIDE
    fig = plt.figure(figsize=(5.9, 4.95), dpi=112)
    # Two panels only: the matrix inherits the height (and the gap) that the
    # former trace panel used, so the glass brain keeps its previous size and
    # the overall figure height is unchanged.
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[1.05, 1.94],
        hspace=0.182, left=0.115, right=0.975, top=0.925, bottom=0.095,
    )

    # ---- top: glass brain at time t
    ax_b = fig.add_subplot(gs[0])
    plotting.plot_glass_brain(
        volume_at(t), display_mode="lzr", colorbar=False, axes=ax_b,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, plot_abs=False, threshold=0.35,
        annotate=False, alpha=0.55,
    )
    ax_b.set_title(r"$n=100$ regions,  $t=%5.1f$ s" % (t * TR),
                   fontsize=15, pad=2)

    # ---- bottom: n x T matrix with a scan line
    ax_m = fig.add_subplot(gs[1])
    lo = max(0, t - WIN)
    seg = ts[:, lo:lo + WIN]
    ax_m.imshow(seg, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                extent=[lo * TR, (lo + WIN) * TR, n_rois, 0],
                interpolation="nearest")
    ax_m.axvline(t * TR, color="k", lw=1.8)
    ax_m.set_ylabel("region  $i$", fontsize=13)
    ax_m.set_xlabel("time  [s]", fontsize=13)
    ax_m.tick_params(labelsize=11)

    fig.savefig(f"{OUT}/f-{k:03d}.png", dpi=112, facecolor="white")
    plt.close(fig)
    if k % 20 == 0:
        print("frame", k, flush=True)

print("done ->", OUT)
