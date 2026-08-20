#!/usr/bin/env python3
"""Quantify how far the learned coupling departs from its connectome initialization.

Reviewer point: the Hybrid Hopf coupling is *initialized* from a group-averaged
connectome but is then learned without symmetry, positivity, or sign constraints,
so calling the result a "learned structural connectome" is not defensible. It is
an **effective** coupling with a connectome-derived initialization. This script
supplies the numbers that make that statement quantitative rather than rhetorical:

* ``delta`` -- relative Frobenius distance between the learned coupling and its
  own rank-r initialization, plus the correlation and Spearman edge-rank
  correlation between them. This says *how far* learning moved the matrix.
* ``structure`` -- the properties the initialization had and the learned matrix
  need not keep: symmetry (``|C - C^T|`` relative to ``|C|``), the fraction of
  negative entries, and the fraction of the total weight carried by the top
  decile of edges. A structural connectome is symmetric and non-negative; if the
  learned matrix is neither, the terminology has to change.
* ``modules`` -- whether the learned coupling still respects the community
  partition of the initialization, measured by the ratio of mean within-module
  to mean between-module weight. This is the "preservation of important
  modules" check.
* ``stability`` -- across seeds, the mean pairwise correlation between learned
  couplings (computed by the aggregator, not here). A coupling that is not
  reproducible across seeds cannot support an interpretability claim.

Because ``C = L R^T`` is a rank-r factorization, all comparisons are made against
the *rank-r truncation* of the initialization (the model's actual starting
point), not the full connectome, which would inflate the apparent movement.

Usage::

    python examples/coupling_diagnostics.py \
        --checkpoints "checkpoints/ts_young_hybrid_hopf_n94_seed*.pt" \
        --out results/coupling_diagnostics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_model_from_checkpoint
from src.models.hybrid_hopf_model import _svd_init


def _offdiag(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return m[mask]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def _spectral_partition(mat: np.ndarray, k: int = 2) -> np.ndarray:
    """Community labels of a symmetric affinity, via a normalized Laplacian map."""
    from sklearn.cluster import SpectralClustering

    aff = np.abs((mat + mat.T) / 2.0)
    np.fill_diagonal(aff, aff.max() if aff.size else 1.0)
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=0
    )
    return sc.fit_predict(aff)


def _module_contrast(mat: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-module weight divided by mean between-module weight."""
    same = labels[:, None] == labels[None, :]
    n = mat.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    within = mat[same & offdiag]
    between = mat[(~same) & offdiag]
    if within.size == 0 or between.size == 0:
        return float("nan")
    denom = np.abs(between).mean()
    return float(np.abs(within).mean() / denom) if denom else float("nan")


def analyze(checkpoint: Path) -> dict | None:
    model, model_class, _ = load_model_from_checkpoint(str(checkpoint), device="cpu")

    L = getattr(model, "coupling_L", None)
    R = getattr(model, "coupling_R", None)
    full = getattr(model, "coupling_full", None)
    sc = getattr(model, "structural_connectivity", None)
    if sc is None:
        return None

    sc_np = sc.detach().cpu().numpy().astype(float)

    if L is not None and R is not None:
        rank = L.shape[1]
        C = (L.detach().cpu() @ R.detach().cpu().T).numpy().astype(float)
        L0, R0 = _svd_init(sc.detach().cpu(), rank)
        C0 = (L0 @ R0.T).numpy().astype(float)
    elif full is not None:
        rank = full.shape[0]
        C = full.detach().cpu().numpy().astype(float)
        C0 = sc_np.copy()
    else:
        return None

    c_off, c0_off = _offdiag(C), _offdiag(C0)
    sc_off = _offdiag(sc_np)

    # How far learning moved the matrix, relative to where it started.
    denom = np.linalg.norm(C0)
    rel_frob = float(np.linalg.norm(C - C0) / denom) if denom else float("nan")

    # Structural properties the initialization had; the learned matrix need not.
    asym = float(np.linalg.norm(C - C.T) / (np.linalg.norm(C) + 1e-12))
    neg_frac = float((c_off < 0).mean())
    absw = np.abs(c_off)
    order = np.sort(absw)[::-1]
    top_decile = float(order[: max(1, len(order) // 10)].sum() / (absw.sum() + 1e-12))

    labels = _spectral_partition(sc_np, k=2)

    return {
        "checkpoint": str(checkpoint),
        "model_class": model_class,
        "rank": int(rank),
        "delta": {
            "rel_frobenius": rel_frob,
            "pearson_vs_init": _pearson(c_off, c0_off),
            "spearman_vs_init": _spearman(c_off, c0_off),
            "pearson_vs_full_connectome": _pearson(c_off, sc_off),
            "spearman_vs_full_connectome": _spearman(c_off, sc_off),
        },
        "structure": {
            "asymmetry": asym,
            "asymmetry_init": float(
                np.linalg.norm(C0 - C0.T) / (np.linalg.norm(C0) + 1e-12)
            ),
            "negative_fraction": neg_frac,
            "negative_fraction_init": float((c0_off < 0).mean()),
            "top_decile_weight_share": top_decile,
            "top_decile_weight_share_init": float(
                np.sort(np.abs(c0_off))[::-1][: max(1, len(c0_off) // 10)].sum()
                / (np.abs(c0_off).sum() + 1e-12)
            ),
        },
        "modules": {
            "within_between_ratio": _module_contrast(C, labels),
            "within_between_ratio_init": _module_contrast(C0, labels),
            "within_between_ratio_connectome": _module_contrast(sc_np, labels),
        },
        "_coupling_flat": c_off.tolist(),
    }


def build_tex(records: list[dict], stability: dict) -> str:
    """Emit a booktabs tabular summarizing the coupling diagnostics."""
    def agg(path: tuple[str, ...]) -> tuple[float, float]:
        vals = []
        for r in records:
            x = r
            for k in path:
                x = x[k]
            vals.append(float(x))
        if not vals:
            return float("nan"), 0.0
        return float(np.mean(vals)), (
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        )

    # (label, learned path, initialization path or None)
    ROWS_SPEC = [
        (r"Relative Frobenius distance $\lVert C-C_0\rVert/\lVert C_0\rVert$",
         ("delta", "rel_frobenius"), None),
        (r"Edge correlation with initialization $C_0$",
         ("delta", "pearson_vs_init"), None),
        (r"Edge rank correlation with $C_0$ (Spearman)",
         ("delta", "spearman_vs_init"), None),
        (r"Edge rank correlation with full connectome",
         ("delta", "spearman_vs_full_connectome"), None),
        (r"Asymmetry $\lVert C-C^{\top}\rVert/\lVert C\rVert$",
         ("structure", "asymmetry"), ("structure", "asymmetry_init")),
        (r"Fraction of negative edges",
         ("structure", "negative_fraction"), ("structure", "negative_fraction_init")),
        (r"Weight share of the strongest decile",
         ("structure", "top_decile_weight_share"),
         ("structure", "top_decile_weight_share_init")),
        (r"Within-/between-module weight ratio ($K=2$)",
         ("modules", "within_between_ratio"), ("modules", "within_between_ratio_init")),
    ]

    lines = [
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Diagnostic & Learned $C$ & Initialization $C_0$ \\",
        r"\midrule",
    ]
    for label, learned_path, init_path in ROWS_SPEC:
        m, sd = agg(learned_path)
        learned = f"{m:.3f} {{\\scriptsize $\\pm$ {sd:.3f}}}"
        if init_path is None:
            init = "---"
        else:
            im, _ = agg(init_path)
            init = f"{im:.3f}"
        lines.append(f"{label} & {learned} & {init} \\\\")

    lines += [
        r"\midrule",
        f"Across-seed reproducibility (mean pairwise $r$) & "
        f"{stability['mean_pairwise_pearson']:.3f} "
        f"{{\\scriptsize $\\pm$ {stability['std_pairwise_pearson']:.3f}}} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                    help="Glob of checkpoints (quote it).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex-out", default=None,
                    help="Write the summary tabular to this .tex path.")
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints}")

    records = []
    for p in paths:
        try:
            rec = analyze(p)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  skip {p.name}: {exc}")
            continue
        if rec is None:
            print(f"  skip {p.name}: no coupling matrix")
            continue
        records.append(rec)
        d = rec["delta"]
        s = rec["structure"]
        print(f"{p.name}: rel_frob={d['rel_frobenius']:.3f} "
              f"r_init={d['pearson_vs_init']:.3f} "
              f"rho_init={d['spearman_vs_init']:.3f} "
              f"asym={s['asymmetry']:.3f} neg={s['negative_fraction']:.3f}")

    # Across-seed stability of the learned coupling.
    flats = [np.asarray(r.pop("_coupling_flat"), dtype=float) for r in records]
    pairwise = [
        _pearson(flats[i], flats[j])
        for i in range(len(flats))
        for j in range(i + 1, len(flats))
    ]
    stability = {
        "n_checkpoints": len(records),
        "mean_pairwise_pearson": float(np.mean(pairwise)) if pairwise else float("nan"),
        "std_pairwise_pearson": float(np.std(pairwise, ddof=1)) if len(pairwise) > 1 else 0.0,
        "min_pairwise_pearson": float(np.min(pairwise)) if pairwise else float("nan"),
    }
    print(f"\nAcross-seed coupling stability: "
          f"mean pairwise r = {stability['mean_pairwise_pearson']:.3f} "
          f"± {stability['std_pairwise_pearson']:.3f} "
          f"(min {stability['min_pairwise_pearson']:.3f}, n={len(pairwise)} pairs)")

    if args.tex_out:
        tex = build_tex(records, stability)
        tp = Path(args.tex_out)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(tex, encoding="utf-8")
        print(f"Wrote {tp}")

    payload = {"records": records, "stability": stability}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
