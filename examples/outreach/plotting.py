"""Figure helpers for the notebook ``hybrid_brain_models.ipynb``.

Only presentation lives here, so that the notebook itself shows the science.
Every function takes plain tensors/arrays and draws one figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

# Palette (validated for colour-vision deficiencies; see README)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#9a9893", "#fcfcfb"
FC_CMAP = LinearSegmentedColormap.from_list("fc", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a09f", "#c42f2f"])
SC_CMAP = LinearSegmentedColormap.from_list("sc", [SURFACE, "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 110, "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "lines.linewidth": 1.6, "mathtext.fontset": "cm", "legend.frameon": False,
})


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _hemisphere_ticks(ax, n):
    half = n // 2
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([half / 2 - 0.5, half + half / 2 - 0.5])
        axis.set_ticklabels(["left", "right"])
    ax.tick_params(length=0)
    for v in (half - 0.5,):
        ax.axhline(v, color="white", lw=1.5)
        ax.axvline(v, color="white", lw=1.5)


def connectome(pos, C):
    """Toy brain seen from above (left) and its connectivity matrix (right)."""
    pos, C = _np(pos), _np(C)
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
    for side in (-1, 1):
        ax.add_patch(Ellipse((side * 0.53, 0), 0.92, 2.75, fc="#f3f2ef", ec=MUTED, lw=1))
    w = C / C.max()
    for i, j in zip(*np.triu_indices(len(C), 1)):
        if w[i, j] > 0.05:
            ax.plot(*pos[[i, j]].T, color=INK_2, lw=3 * w[i, j], alpha=0.25 + 0.6 * w[i, j], zorder=1)
    ax.scatter(*pos.T, s=110, color=SERIES[0], edgecolor="white", linewidth=2, zorder=2)
    ax.text(0, 1.48, "front", ha="center", color=INK_2)
    ax.text(-0.53, -1.62, "left", ha="center", color=INK_2)
    ax.text(0.53, -1.62, "right", ha="center", color=INK_2)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.75, 1.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Toy brain seen from above (20 regions)\nline width = connection strength")
    im = axm.imshow(C, cmap=SC_CMAP, vmin=0)
    _hemisphere_ticks(axm, len(C))
    axm.set_title("Structural connectome $C_{ij}$")
    fig.colorbar(im, ax=axm, shrink=0.8, label="strength (a.u.)")
    fig.tight_layout()
    plt.show()


def regimes(tr, trajectories):
    """One Stuart-Landau oscillator in three regimes: complex plane (top) and Re z(t) (bottom)."""
    names = {-1: "$a<0$: noisy fluctuations", 0: "$a=0$: critical point", 1: "$a>0$: self-sustained rhythm"}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), gridspec_kw={"height_ratios": [1.25, 1]})
    lim = max(np.abs(_np(z)).max() for z in trajectories.values()) * 1.1
    for k, (a, z) in enumerate(trajectories.items()):
        z = _np(z); t = np.arange(len(z)) * tr
        top, bottom = axes[0, k], axes[1, k]
        top.plot(z.real, z.imag, color=SERIES[k], lw=0.9)
        top.plot(0, 0, "+", color=INK, ms=9)
        top.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel=r"$\mathrm{Re}\,z$",
                title=f"{names[int(np.sign(a))]}\n($a={a:+.2f}$)")
        bottom.plot(t, z.real, color=SERIES[k], lw=1.2)
        bottom.set(ylim=(-lim, lim), xlabel="time [s]")
    axes[0, 0].set_ylabel(r"$\mathrm{Im}\,z$")
    axes[1, 0].set_ylabel(r"signal $\mathrm{Re}\,z(t)$")
    fig.tight_layout()
    plt.show()


def scan(tr, bold, regions=(0, 4, 13)):
    """One simulated 'scan': all regions as a heat map, plus a few individual traces."""
    bold = _np(bold); t = np.arange(bold.shape[-1]) * tr
    x = (bold - bold.mean(-1, keepdims=True)) / bold.std(-1, keepdims=True)
    fig, (ax, axt) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    im = ax.imshow(x, aspect="auto", cmap=FC_CMAP, vmin=-2.5, vmax=2.5, extent=[0, t[-1], len(x) - 0.5, -0.5])
    ax.set(ylabel="region $i$", title="Simulated fMRI scan of the virtual brain (standardised signal)")
    fig.colorbar(im, ax=[ax], pad=0.01, label="signal (std. units)")
    for k, i in enumerate(regions):
        axt.plot(t, x[i] + 4.5 * (len(regions) - 1 - k), color=SERIES[k], lw=1.1)
        axt.text(t[-1] * 1.005, 4.5 * (len(regions) - 1 - k), f"region {i}", color=INK_2, va="center")
    axt.set(yticks=[], xlabel="time [s]")
    axt.spines["left"].set_visible(False)
    plt.show()


def analytic(tr, z_hat, z_true, window=250):
    """The analytic signal estimated from BOLD (Hilbert transform) versus the hidden truth."""
    z_hat, z_true = _np(z_hat)[:window], _np(z_true)[:window]
    z_true = z_true / np.abs(z_true).mean() * np.abs(z_hat).mean()   # match scales for display
    t = np.arange(window) * tr
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.plot(t, z_hat.real, color=INK, lw=1.4, label=r"$x(t)$: filtered BOLD (measured)")
    ax.plot(t, z_hat.imag, color=SERIES[0], lw=1.4, label=r"$\mathcal{H}[x](t)$: Hilbert transform")
    ax.plot(t, z_true.imag, color=SERIES[1], lw=1.0, alpha=0.8, label=r"hidden truth $\mathrm{Im}\,z$ (slow part)")
    ax.set(xlabel="time [s]", title="One region: from a real signal to a complex one")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9.5, ncol=3, handlelength=1.5)
    axc.plot(z_hat.real, z_hat.imag, color=SERIES[0], lw=0.9)
    axc.plot(z_hat.real[-1], z_hat.imag[-1], "o", color=SERIES[0], mec="white", mew=2, ms=9)
    axc.axhline(0, color=MUTED, lw=0.6); axc.axvline(0, color=MUTED, lw=0.6)
    axc.set(aspect="equal", xlabel=r"$\mathrm{Re}\,z$", ylabel=r"$\mathrm{Im}\,z$",
            title=r"$z(t)=x(t)+i\,\mathcal{H}[x](t)$")
    fig.tight_layout()
    plt.show()


def matrices(mats, titles, kinds="fc"):
    """A row of connectivity matrices sharing one colour scale per kind ('fc' or 'sc')."""
    mats = [_np(m) for m in mats]
    kinds = [kinds] * len(mats) if isinstance(kinds, str) else list(kinds)
    off = ~np.eye(len(mats[0]), dtype=bool)
    v = max(np.abs(m[off]).max() for m, k in zip(mats, kinds) if k == "fc") if "fc" in kinds else 1
    fig, axes = plt.subplots(1, len(mats), figsize=(3.9 * len(mats), 3.9), squeeze=False)
    for ax, m, title, kind in zip(axes[0], mats, titles, kinds):
        m = m.copy(); np.fill_diagonal(m, np.nan)          # the diagonal (self-correlation) is not informative
        im = ax.imshow(m, cmap=FC_CMAP if kind == "fc" else SC_CMAP,
                       vmin=-v if kind == "fc" else 0, vmax=v if kind == "fc" else None)
        _hemisphere_ticks(ax, len(m))
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.78, label="correlation" if kind == "fc" else "strength (a.u.)")
    fig.tight_layout()
    plt.show()


def scores(results):
    """Bar charts: L2 error (lower is better) and FC similarity (higher is better) per candidate."""
    names = list(results)
    l2 = [float(results[n][0]) for n in names]
    fc = [float(results[n][1]) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, vals, title, best in zip(
        axes, (l2, fc),
        (r"Pointwise $L^2$ error  ($\downarrow$ lower is better)", r"FC similarity  ($\uparrow$ higher is better)"),
        (int(np.argmin(l2)), int(np.argmax(fc))),
    ):
        bars = ax.bar(names, vals, color=SERIES[: len(names)], width=0.6, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", color=INK)
        ax.get_xticklabels()[best].set_fontweight("bold")
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel(r"in units of $\mathbb{E}\,|z|^2$")
    axes[1].set_ylabel("Pearson correlation")
    fig.tight_layout()
    plt.show()
