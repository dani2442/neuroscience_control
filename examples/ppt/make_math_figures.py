"""Mathematical figures for the presentation (Hopf normal form, analytic signal,
grid-search cost, KS non-differentiability)."""

import os
import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

ROOT = "/home/dani/projects/neuroscience_control"
OUT = f"{ROOT}/paper_submission/images/ppt"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, RED, GREEN, GREY = "#2B6CB0", "#E4572E", "#2E9E6B", "#8A8A8A"


def hopf_flow(z, a, w=1.0):
    return (a + 1j * w - np.abs(z) ** 2) * z


def integrate(z0, a, w=1.0, sigma=0.0, T=60.0, dt=0.005):
    n = int(T / dt)
    z = np.empty(n, dtype=complex)
    z[0] = z0
    for k in range(n - 1):
        dw = (rng.normal() + 1j * rng.normal()) * np.sqrt(dt / 2)
        z[k + 1] = z[k] + hopf_flow(z[k], a, w) * dt + sigma * dw
    return np.arange(n) * dt, z


# =====================================================================
# 1. Supercritical Hopf bifurcation: phase portraits
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
for ax, a, ttl in zip(
    axes, [-0.6, 0.0, 0.6],
    [r"$a<0$: stable focus", r"$a=0$: bifurcation", r"$a>0$: limit cycle"],
):
    lim = 1.35
    X, Y = np.meshgrid(np.linspace(-lim, lim, 24), np.linspace(-lim, lim, 24))
    Z = X + 1j * Y
    F = hopf_flow(Z, a)
    ax.streamplot(X, Y, F.real, F.imag, color=GREY, density=0.85,
                  linewidth=0.7, arrowsize=0.8)
    for z0 in [0.06 + 0.03j, 1.28 + 0.15j, -0.9 - 0.7j]:
        _, z = integrate(z0, a, sigma=0.0, T=25.0)
        ax.plot(z.real, z.imag, color=BLUE, lw=1.7)
    if a > 0:
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.sqrt(a) * np.cos(th), np.sqrt(a) * np.sin(th),
                color=RED, lw=3.0, zorder=5)
    ax.plot(0, 0, "o", ms=8, mfc="white" if a > 0 else RED, mec=RED, mew=2, zorder=6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(ttl); ax.set_xlabel(r"$\Re z$")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
axes[0].set_ylabel(r"$\Im z$")

fig.tight_layout()
fig.savefig(f"{OUT}/hopf_bifurcation.pdf"); plt.close(fig)
print("hopf_bifurcation.pdf")

# =====================================================================
# 2. The two regimes with noise -- why the brain sits below the bifurcation
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(12.5, 5.0),
                         gridspec_kw={"width_ratios": [1, 1.9]})
for row, (a, lbl, col, sg) in enumerate([(-0.05, r"$a=-0.05<0$", BLUE, 0.13),
                                         (0.35, r"$a=+0.35>0$", GREEN, 0.10)]):
    t, z = integrate(0.1 + 0.0j, a, w=2 * np.pi * 0.05, sigma=sg, T=260.0, dt=0.02)
    axes[row, 0].plot(z.real, z.imag, color=col, lw=0.7, alpha=0.85)
    axes[row, 0].plot(0, 0, "k+", ms=8)
    axes[row, 0].set_aspect("equal"); axes[row, 0].set_xlim(-1.1, 1.1)
    axes[row, 0].set_ylim(-1.1, 1.1)
    axes[row, 0].set_ylabel(r"$\Im z$"); axes[row, 0].set_title(lbl, color=col)
    axes[row, 1].plot(t, z.real, color=col, lw=1.2)
    axes[row, 1].set_xlim(0, t[-1]); axes[row, 1].set_ylabel(r"$\Re z$")
    axes[row, 1].set_ylim(-1.25, 1.25)
axes[0, 1].set_title("noise-sustained damped oscillation  (resting state)", color=BLUE)
axes[1, 1].set_title("self-sustained limit-cycle oscillation", color=GREEN)
axes[1, 0].set_xlabel(r"$\Re z$"); axes[1, 1].set_xlabel("time  [s]")
fig.tight_layout()
fig.savefig(f"{OUT}/hopf_regimes.pdf"); plt.close(fig)
print("hopf_regimes.pdf")

# =====================================================================
# 3. Why the complex plane: the analytic signal of a real BOLD trace
# =====================================================================
TR = 0.72
ts = sio.loadmat(f"{ROOT}/data/ts_young/ts_young_TR0.72.mat")["timeseries_all"][:, :, 0]
b, a_ = butter(2, [0.008, 0.08], btype="band", fs=1 / TR)
x = filtfilt(b, a_, (ts[7] - ts[7].mean()) / ts[7].std())
x = x[150:400]
t = np.arange(x.size) * TR
zc = hilbert(x)
amp, pha = np.abs(zc), np.unwrap(np.angle(zc))

fig = plt.figure(figsize=(13.5, 4.0))
gs = fig.add_gridspec(2, 3, width_ratios=[1.85, 1.0, 1.0], hspace=0.45, wspace=0.30)

ax = fig.add_subplot(gs[:, 0])
ax.plot(t, x, color="k", lw=1.5, label=r"$x(t)=\Re z(t)$  (filtered BOLD)")
ax.plot(t, amp, color=RED, lw=1.6, label=r"$|z(t)|$  envelope")
ax.plot(t, -amp, color=RED, lw=1.6)
ax.plot(t, zc.imag, color=BLUE, lw=1.0, alpha=0.75, label=r"$\Im z(t)=\mathcal{H}[x](t)$")
ax.set_xlim(t[0], t[-1]); ax.set_xlabel("time  [s]")
ax.set_ylim(-2.1, 2.9)
ax.legend(loc="upper center", ncol=3, fontsize=9.0, frameon=False,
          columnspacing=1.1, handlelength=1.6)
ax.set_title("Hilbert transform: one real trace $\\rightarrow$ one analytic signal")

ax = fig.add_subplot(gs[:, 1])
ax.scatter(zc.real, zc.imag, c=t, cmap="viridis", s=7, zorder=3)
ax.plot(zc.real, zc.imag, color="k", lw=0.45, alpha=0.35)
ax.set_aspect("equal"); ax.set_xlabel(r"$\Re z$"); ax.set_ylabel(r"$\Im z$")
ax.set_title(r"$z(t)\in\mathbb{C}$")
ax.plot(0, 0, "k+", ms=8)

ax = fig.add_subplot(gs[0, 2])
ax.plot(t, amp, color=RED, lw=1.4); ax.set_xlim(t[0], t[-1])
ax.set_title(r"amplitude  $a_i(t)=|z_i|$", fontsize=12)
ax.set_xticklabels([])

ax = fig.add_subplot(gs[1, 2])
ax.plot(t, np.angle(zc), color=BLUE, lw=1.0)
ax.set_xlim(t[0], t[-1]); ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
ax.set_title(r"phase  $\phi_i(t)=\angle z_i$", fontsize=12)
ax.set_xlabel("time  [s]")
fig.tight_layout()
fig.savefig(f"{OUT}/analytic_signal.pdf"); plt.close(fig)
print("analytic_signal.pdf")

# =====================================================================
# 4. Cost of grid search vs gradient descent
# =====================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.4))
p = np.arange(1, 17)
SIM = 2.32                                   # seconds per Hopf simulation (paper)
for K, ls in zip([5, 10], ["-", "--"]):
    ax.semilogy(p, K ** p.astype(float), ls, color=RED, lw=2,
                label=rf"grid search, $K={K}$ values per parameter:  $K^{{p}}$")
ax.semilogy(p, np.full_like(p, 20 * 8, dtype=float), color=BLUE, lw=2.5,
            label=r"gradient descent:  $O(\mathrm{epochs})$, independent of $p$")

ax.scatter([2], [40], color="k", zorder=6, s=55)
ax.annotate("Coupled Hopf\n$p=2$:  40 sims  $=93$ s", (2, 40),
            textcoords="offset points", xytext=(16, 14), fontsize=10)
ax.scatter([10], [5.0 ** 10], color=RED, zorder=6, s=55, marker="D")
ax.annotate(r"$p=10$:  $5^{10}\approx10^{7}$ sims  $\approx 9$ months",
            (10, 5.0 ** 10), textcoords="offset points", xytext=(-232, 26),
            fontsize=10, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.set_xlabel(r"number of fitted parameters  $p$")
ax.set_ylabel("model simulations required")
ax.set_xlim(0.5, 16.5); ax.set_ylim(1e1, 1e16)
ax.legend(fontsize=9.5, loc="upper left")
ax.set_title("Grid search does not scale")
ax.text(0.985, 0.055,
        "Hybrid Hopf: $2\,215$ parameters\nNeural SDE: $220\,176$ parameters",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
        color=GREEN,
        bbox=dict(boxstyle="round,pad=0.4", fc="#EAF6F0", ec=GREEN, lw=1.0))
fig.tight_layout()
fig.savefig(f"{OUT}/grid_cost.pdf"); plt.close(fig)
print("grid_cost.pdf")

# =====================================================================
# 5. KS distance is piecewise constant -> no useful gradient
# =====================================================================
u = rng.normal(0.15, 0.9, 120)
v = rng.normal(-0.35, 1.15, 120)
grid = np.linspace(-4, 4, 800)
Fu = (u[None, :] <= grid[:, None]).mean(1)
Fv = (v[None, :] <= grid[:, None]).mean(1)
k = np.argmax(np.abs(Fu - Fv))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axes[0]
ax.step(grid, Fv, where="post", color="k", lw=1.8, label=r"empirical  $\hat F_{z}$")
ax.step(grid, Fu, where="post", color=RED, lw=1.8, label=r"simulated  $\hat F_{\hat z}$")
ax.annotate("", xy=(grid[k], Fu[k]), xytext=(grid[k], Fv[k]),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=2.2))
ax.text(grid[k] + 0.15, (Fu[k] + Fv[k]) / 2, r"$D_{\mathrm{KS}}$", color=BLUE, fontsize=14)
ax.set_xlabel("FCD entry value"); ax.set_ylabel("CDF")
ax.set_title(r"$D_{\mathrm{KS}}=\sup_x|\hat F_{\hat z}(x)-\hat F_z(x)|$")
ax.legend(fontsize=10, loc="upper left")

ax = axes[1]
eps = np.linspace(-0.9, 0.9, 600)
ks = np.array([np.max(np.abs((u[None, :] + e <= grid[:, None]).mean(1) - Fv))
               for e in eps])
mse = np.array([np.mean((np.sort(u) + e - np.sort(v)) ** 2) for e in eps])
ax.plot(eps, ks, color=RED, lw=2, label=r"$D_{\mathrm{KS}}$  (piecewise constant)")
ax.plot(eps, mse / mse.max() * ks.max(), color=BLUE, lw=2,
        label=r"MSE surrogate  (smooth, rescaled)")
ax.set_xlabel(r"shift of the simulated sample  $\varepsilon$")
ax.set_ylabel("loss")
ax.set_title(r"gradient of $D_{\mathrm{KS}}$ is $0$ a.e.")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/ks_surrogate.pdf"); plt.close(fig)
print("ks_surrogate.pdf")
