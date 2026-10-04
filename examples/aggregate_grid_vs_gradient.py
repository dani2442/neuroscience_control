#!/usr/bin/env python3
"""Build Supplementary Table S2: Coupled Hopf fitted by grid search vs. gradient.

Reads the per-seed result files of the two fitting procedures,

* grid search: ``results/runs/<ds>_hopf_grid_seed<s>.json``
* gradient:    ``results/runs/<ds>_hopf_n94_seed<s>.json`` (the Coupled Hopf runs of Table 1)

restricts both arms to the seeds they share (a balanced, paired design), and
reports mean ± standard deviation across those seeds. The better mean per
column is bold; a superscript marks a two-sided exact Wilcoxon signed-rank test
paired on the seed (``*`` p<0.05, ``**`` p<0.01, ``***`` p<0.001).

Usage::

    python examples/aggregate_grid_vs_gradient.py --seeds 42-51 \
        --out paper_submission/sections/grid_vs_gradient_table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from aggregate_seeds import _parse_seeds

# (json_key, header, higher_is_better)
COLUMNS = [
    ("fc_correlation",       r"FC corr $\uparrow$",           True),
    ("phase_fc_correlation", r"phFC corr $\uparrow$",         True),
    ("fcd_ks",               r"FCD KS $\downarrow$",          False),
    ("phfcd_ks",             r"phFCD KS $\downarrow$",        False),
    ("metastability_diff",   r"Meta $|\Delta|$ $\downarrow$", False),
    ("temporal_correlation", r"TS corr $\uparrow$",           True),
]

# (run family, row label)
ARMS = [("hopf_grid", "Grid search"), ("hopf_n94", "Gradient")]


def _load(runs_dir: Path, ds: str, family: str, seeds: set[int]) -> dict[int, dict]:
    out = {}
    for s in sorted(seeds):
        fp = runs_dir / f"{ds}_{family}_seed{s}.json"
        if fp.exists():
            out[s] = json.loads(fp.read_text())["metrics"]["test_inter"]
    return out


def _sig(p: float) -> str:
    if p < 0.001:
        return "^{***}"
    if p < 0.01:
        return "^{**}"
    if p < 0.05:
        return "^{*}"
    return ""


def _fmt(mean: float, sd: float, bold: bool, sig: str) -> str:
    sup = f"${sig}$" if sig else ""
    body = f"{mean:.3f}{sup} {{\\scriptsize $\\pm$ {sd:.3f}}}"
    return f"\\bf{{{body}}}" if bold else body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--out", default=None, help="Output .tex path (default: print only).")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    runs = {fam: _load(Path(args.runs_dir), args.ds, fam, seeds) for fam, _ in ARMS}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    for fam, label in ARMS:
        dropped = sorted(set(runs[fam]) - set(common))
        print(f"  {label:12s} seeds={sorted(runs[fam])}"
              + (f"  (dropped, not in both arms: {dropped})" if dropped else ""))
    print(f"  paired on {len(common)} seeds: {common}")

    cells: dict[str, list[str]] = {fam: [] for fam, _ in ARMS}
    (fa, _), (fb, _) = ARMS
    for key, _hdr, higher in COLUMNS:
        va = np.array([runs[fa][s][key] for s in common])
        vb = np.array([runs[fb][s][key] for s in common])
        p = wilcoxon(va, vb).pvalue
        better = fa if (va.mean() > vb.mean()) == higher else fb
        for fam, v in ((fa, va), (fb, vb)):
            cells[fam].append(_fmt(v.mean(), v.std(ddof=1), fam == better,
                                   _sig(p) if fam == better else ""))
        print(f"  {key:22s} p={p:.4f}")

    header = " & ".join(["Method"] + [h for _k, h, _b in COLUMNS]) + r" \\"
    lines = [f"\\begin{{tabular}}{{l{' c' * len(COLUMNS)}}}", "\\toprule", header, "\\midrule"]
    for fam, label in ARMS:
        lines.append("    " + " & ".join([label] + cells[fam]) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).write_text(table)
        print(f"Wrote {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
