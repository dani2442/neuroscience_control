#!/usr/bin/env python3
"""Panels (d) and (e) of Figure 4, in the style of the composite figure.

(e) Personalization: for each of the ten personalized target subjects, the
    static FC correlation of the target with its own model ("Own") and the mean
    over the nine models fitted to other subjects ("Others"), connected per
    target. Source: ``results/personalization_summary.json`` written by
    ``examples/aggregate_personalization.py`` (seed 42, scratch arm), i.e. the
    numbers of Supplementary Table S3.

(d) Data efficiency with a common test set: composite test error on the same 15
    held-out subjects after training on 10 vs. all 65 training-pool subjects.
    Source: ``results/runs/ts_young_<model>_fixtest_tr<K>_seed<s>.json`` written
    by ``logs/fig4d/launch_fig4d.sh``.

Usage::

    python examples/plot_fig4_panels.py --panel e
    python examples/plot_fig4_panels.py --panel d --seeds 42-51
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from aggregate_seeds import _parse_seeds

# Same model order and colours as the composite Figure 4.
MODELS = [
    ("hopf", "Hopf", "#DD8452"),
    ("hybrid_hopf", "Hybrid Hopf", "#55A868"),
    ("nsde", "Neural SDE", "#4C72B0"),
]
OUT_DIR = Path("paper_submission/images/panels")
FIGSIZE = (3.6, 2.4)


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300, transparent=True)
    print(f"Wrote {OUT_DIR}/{name}.{{pdf,png,svg}}")


def panel_e(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())["models"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    dx = 0.2
    for i, (key, _label, color) in enumerate(MODELS):
        rec = summary[key]["fc_correlation"]
        rows = sorted(rec["per_target"], key=lambda r: r["target"])
        others = np.array([r["others_mean"] for r in rows])
        own = np.array([r["own"] for r in rows])
        for o, w in zip(others, own):
            ax.plot([i - dx, i + dx], [o, w], color="#9a9a9a", lw=0.7, zorder=2)
        # Others: filled, Own: open -- the solid/hollow encoding of panel (d).
        ax.scatter(np.full(len(rows), i - dx), others, s=16, color=color,
                   edgecolor=color, linewidth=1.0, zorder=3)
        ax.scatter(np.full(len(rows), i + dx), own, s=16, facecolor="white",
                   edgecolor=color, linewidth=1.2, zorder=3)
        ax.plot([i - dx, i + dx], [others.mean(), own.mean()], color=color, lw=2.2,
                solid_capstyle="round", zorder=4)
        ctrl = rec["controlled"]
        top = max(others.max(), own.max())
        delta = f"{ctrl['mean']:+.3f}".replace("-", "\u2212")
        ax.text(i, top + 0.03, f"$\\Delta$={delta}\n{_stars(ctrl['wilcoxon_p'])}",
                ha="center", va="bottom", fontsize=8.5, linespacing=1.1)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], fontsize=10)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0.3, 1.1)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("FC correlation $\\uparrow$", fontsize=10)
    _style(ax)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#888888", markersize=4.5,
                   label="Others"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="#888888", markeredgewidth=1.2, markersize=4.5,
                   label="Own"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5,
              handletextpad=0.2, borderaxespad=0.1)
    _save(fig, "fig4e_personalization")


def _composite(metrics: dict) -> float:
    return float(np.mean([1 - metrics["fc_correlation"], 1 - metrics["phase_fc_correlation"],
                          metrics["fcd_ks"], metrics["phfcd_ks"]]))


def load_fixtest(runs_dir: Path, seeds: set[int], sizes=(10, 65)) -> tuple[dict, list[int]]:
    """{model: {K: {seed: composite}}} restricted to seeds complete for every cell."""
    data = {key: {k: {} for k in sizes} for key, _, _ in MODELS}
    for key, _, _ in MODELS:
        for k in sizes:
            for s in sorted(seeds):
                fp = runs_dir / f"ts_young_{key}_fixtest_tr{k}_seed{s}.json"
                if fp.exists():
                    data[key][k][s] = _composite(json.loads(fp.read_text())["metrics"]["test_inter"])
    common = sorted(set.intersection(*(set(d) for m in data.values() for d in m.values())))
    return data, common


def panel_d(runs_dir: Path, seeds: set[int]) -> None:
    sizes = (10, 65)
    data, common = load_fixtest(runs_dir, seeds, sizes)
    print(f"complete seeds ({len(common)}): {common}")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.36
    tops = []
    for i, (key, label, color) in enumerate(MODELS):
        small = np.array([data[key][sizes[0]][s] for s in common])
        large = np.array([data[key][sizes[1]][s] for s in common])
        p = wilcoxon(small, large).pvalue
        print(f"  {label:12s} K={sizes[0]}: {small.mean():.3f} ± {small.std(ddof=1):.3f}   "
              f"K={sizes[1]}: {large.mean():.3f} ± {large.std(ddof=1):.3f}   p={p:.4f}")
        ax.bar(i - w / 2, small.mean(), w, yerr=small.std(ddof=1), facecolor=color,
               edgecolor=color, linewidth=1.2, capsize=3,
               error_kw=dict(lw=0.8, capthick=0.8), zorder=3, alpha=0.95)
        ax.bar(i + w / 2, large.mean(), w, yerr=large.std(ddof=1), facecolor="white",
               edgecolor=color, linewidth=1.4, hatch="///", capsize=3,
               error_kw=dict(lw=0.8, capthick=0.8, ecolor=color), zorder=3)
        top = max(small.mean() + small.std(ddof=1), large.mean() + large.std(ddof=1))
        tops.append(top)
        if p < 0.05:
            ax.text(i, top, _stars(p), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[1] for m in MODELS], fontsize=10)
    ax.set_ylim(0, max(tops) * 1.45)
    ax.set_ylabel("Composite test error $\\downarrow$", fontsize=10)
    _style(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#888888", edgecolor="#888888", alpha=0.95),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#888888", hatch="///",
                      linewidth=1.4),
    ]
    ax.legend(handles, [f"{k} subjects" for k in sizes], loc="upper right", frameon=False,
              fontsize=9, title="Training set", title_fontsize=9, borderaxespad=0.1)
    _save(fig, "fig4d_data_efficiency")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", choices=["d", "e"], required=True)
    ap.add_argument("--summary", default="results/personalization_summary.json")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--seeds", default="42-51")
    args = ap.parse_args()
    if args.panel == "e":
        panel_e(Path(args.summary))
    else:
        panel_d(Path(args.runs_dir), _parse_seeds(args.seeds))


if __name__ == "__main__":
    main()
