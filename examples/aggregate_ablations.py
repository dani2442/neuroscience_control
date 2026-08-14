#!/usr/bin/env python3
"""Aggregate ablation-experiment JSONs into an appendix LaTeX table.

Reads per-seed result files written by ``train_models.py backprop`` with
``--run-suffix <ablkey>_seed<N>`` (i.e. ``results/runs/<ds>_<model>_<ablkey>_seed<N>.json``),
averages the requested metrics across seeds, and emits a booktabs ``tabular``
matching the style of ``paper/sections/<ds>_model_table.tex``.

Usage:
    python examples/aggregate_ablations.py --ds ts_young \
        --out paper/sections/ts_young_ablation_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Metric columns: (json_key, header, higher_is_better)
COLUMNS = [
    ("fc_correlation",         r"FC corr $\uparrow$",   True),
    ("fc_mse",                 r"FC MSE $\downarrow$",  False),
    ("phase_fc_correlation",   r"phFC corr $\uparrow$", True),
    ("fcd_ks",                 r"FCD KS $\downarrow$",  False),
    ("phfcd_ks",               r"phFCD KS $\downarrow$", False),
    ("metastability_diff",     r"Meta $|\Delta|$ $\downarrow$", False),
    ("temporal_correlation",   r"TS corr $\uparrow$",   True),
    ("power_spectrum_distance", r"TS PSD $\downarrow$", False),
    ("autocorr_distance",      r"Autocorr $\downarrow$", False),
]

# Row groups: (ablkey, model_key, row_label). Order defines table rows.
# ``sep`` inserts a \midrule before the row.
ROWS = [
    ("baseline",       "hybrid_hopf", r"Hybrid Hopf (reference)",              False),
    ("pmatch",         "nsde",        r"Neural SDE (param-matched)",           True),
    ("scshuffle",      "hybrid_hopf", r"\quad + shuffled connectome",          True),
    ("scrandom",       "hybrid_hopf", r"\quad + random connectome",            False),
    ("fixedcoupling",  "hybrid_hopf", r"\quad + fixed connectivity",           False),
    ("nolocal",        "hybrid_hopf", r"\quad $-$ Hopf local term",            False),
    ("nofdm",          "hybrid_hopf", r"\quad $-$ FDM loss",                   True),
    ("nodyn",          "hybrid_hopf", r"\quad $-$ dyn.\ connectivity losses",  False),
]


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def _fmt(mean: float, std: float, best: bool) -> str:
    if math.isnan(mean):
        return "---"
    body = f"{mean:.3f} {{\\scriptsize $\\pm$ {std:.3f}}}"
    return f"\\bf{{{body}}}" if best else body


def collect(ds: str, runs_dir: Path) -> dict:
    """Return {ablkey: {metric: [per-seed values]}} for present result files."""
    data: dict[str, dict[str, list[float]]] = {}
    for ablkey, model_key, _label, _sep in ROWS:
        files = sorted(runs_dir.glob(f"{ds}_{model_key}_{ablkey}_seed*.json"))
        if not files:
            continue
        metrics: dict[str, list[float]] = {k: [] for k, _, _ in COLUMNS}
        for fp in files:
            with fp.open() as fh:
                payload = json.load(fh)
            inter = payload["metrics"]["test_inter"]
            for key, _, _ in COLUMNS:
                if key in inter:
                    metrics[key].append(float(inter[key]))
        data[ablkey] = {"metrics": metrics, "n_seeds": len(files)}
    return data


def build_table(ds: str, data: dict) -> str:
    # Aggregate mean/std per (ablkey, metric)
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for ablkey, entry in data.items():
        agg[ablkey] = {}
        for key, _, _ in COLUMNS:
            vals = entry["metrics"][key]
            agg[ablkey][key] = _mean_std(vals) if vals else (float("nan"), 0.0)

    # No per-column bolding: in an ablation table the "best" value can be
    # a degenerate artifact (e.g. a collapsed model scores low on a distance
    # metric), which is misleading. The reference row is the anchor instead.
    best: dict[str, str] = {}

    col_spec = "l" + " c" * len(COLUMNS)
    header = " & ".join(["Model"] + [h for _, h, _ in COLUMNS]) + r" \\"
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule", header, "\\midrule"]

    for ablkey, model_key, label, sep in ROWS:
        if ablkey not in agg:
            continue
        if sep:
            lines.append("\\midrule")
        cells = [label]
        for key, _, _ in COLUMNS:
            mean, std = agg[ablkey][key]
            cells.append(_fmt(mean, std, best.get(key) == ablkey))
        lines.append(" & ".join(cells) + r" \\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate ablation JSONs into a LaTeX table.")
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--out", default=None, help="Output .tex path (default: print only).")
    args = ap.parse_args()

    data = collect(args.ds, Path(args.runs_dir))
    if not data:
        raise SystemExit(f"No ablation result files found under {args.runs_dir} for ds={args.ds}")

    # Report coverage
    print("Ablation coverage:")
    for ablkey, _model, label, _sep in ROWS:
        n = data.get(ablkey, {}).get("n_seeds", 0)
        print(f"  {ablkey:15s} ({label.strip()}): {n} seed(s)")

    table = build_table(args.ds, data)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table)
        print(f"\nWrote table to {out}")
    else:
        print("\n" + table)


if __name__ == "__main__":
    main()
