#!/usr/bin/env python3
"""Build the model-comparison tables with the *seed* as the unit of analysis.

Reviewer point 3. The previous tables reported ``mean ± std across held-out test
batches of a single fit``. That spread excludes fitting variability entirely, so
it cannot support a claim that one architecture beats another -- two models whose
intervals do not overlap under that protocol may still swap ranks on a refit.

This script instead reads the per-seed result files written by the N=94 sweep
(``results/runs/ts_young_<model>_n94_seed<seed>.json``), where each file is one
independent refit, and reports:

* mean ± standard deviation **across seeds** (n reported per row);
* a **paired Wilcoxon signed-rank test** against a reference architecture,
  pairing on the seed, with **Holm** correction across the compared
  architectures within each metric;
* a **predefined composite error** -- the same aggregate already used for the
  robustness analysis: the mean of {FC corr, phFC corr, FCD KS, phFCD KS} with
  upward metrics flipped to ``1 - v``, lower is better. Declaring it up front is
  what replaces "best overall balanced performance" read off by eye across
  columns.

Usage::

    python examples/aggregate_seeds.py --ds ts_young \
        --out paper_submission/sections/ts_young_model_table.tex --main
    python examples/aggregate_seeds.py --ds ts_young \
        --out paper_submission/sections/ts_young_full_model_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# (json_key, header, higher_is_better)
ALL_COLUMNS = [
    ("fc_correlation",          r"FC corr $\uparrow$",           True),
    ("fc_mse",                  r"FC MSE $\downarrow$",          False),
    ("phase_fc_correlation",    r"phFC corr $\uparrow$",         True),
    ("fcd_ks",                  r"FCD KS $\downarrow$",          False),
    ("phfcd_ks",                r"phFCD KS $\downarrow$",        False),
    ("metastability_diff",      r"Meta $|\Delta|$ $\downarrow$", False),
    ("temporal_correlation",    r"TS corr $\uparrow$",           True),
    ("autocorr_distance",       r"Autocorr $\downarrow$",        False),
    ("fc_ari_k2",               r"FC ARI $\uparrow$",            True),
]

# Dropped from the supplementary full table (the ARI is only reported at the
# main-table resolution, K=2).
FULL_EXCLUDE = {"fc_ari_k2"}

MAIN_COLUMNS = [
    "fc_correlation",
    "phase_fc_correlation",
    "fcd_ks",
    "phfcd_ks",
    "metastability_diff",
    "fc_ari_k2",
]

# Metrics entering the predefined composite error (lower is better after
# flipping upward metrics). Fixed in advance; do not tune to the result.
COMPOSITE_KEYS = [
    ("fc_correlation", True),
    ("phase_fc_correlation", True),
    ("fcd_ks", False),
    ("phfcd_ks", False),
]

# (model_key, row label). Order defines table rows.
ROWS = [
    ("hopf",          "Coupled Hopf"),
    ("nsde",          "Neural SDE"),
    ("hybrid_hopf",   "Hybrid Hopf"),
    ("gnn_hopf",      "GNN-Hopf"),
    ("hybrid_neural", "Hopf+Neural"),
]

MAIN_ROWS = ["hopf", "nsde", "hybrid_hopf"]

REFERENCE = "hybrid_hopf"


def restrict_to_common(
    data: dict[str, dict[int, dict[str, float]]]
) -> tuple[dict[str, dict[int, dict[str, float]]], list[int]]:
    """Keep only seeds for which *every* architecture produced a model.

    A balanced (complete-case) design is what makes the seed a legitimate pairing
    unit: if each row is summarized over a different seed set, the columns are no
    longer comparable and paired tests silently change their n between metrics.
    Seeds are dropped wholesale rather than per-architecture so that no model is
    credited for surviving a seed on which its competitors crashed -- excluding a
    hard seed only for the models that failed it would flatter exactly those
    models that are least numerically robust.
    """
    if not data:
        return data, []
    common = set.intersection(*(set(v) for v in data.values()))
    dropped = sorted({s for v in data.values() for s in v} - common)
    restricted = {
        mk: {s: rec for s, rec in v.items() if s in common} for mk, v in data.items()
    }
    return restricted, dropped


def collect(ds: str, runs_dir: Path, suffix: str) -> dict[str, dict[int, dict[str, float]]]:
    """Return {model_key: {seed: {metric: value}}}."""
    out: dict[str, dict[int, dict[str, float]]] = {}
    for model_key, _label in ROWS:
        per_seed: dict[int, dict[str, float]] = {}
        for fp in sorted(runs_dir.glob(f"{ds}_{model_key}_{suffix}_seed*.json")):
            seed = int(fp.stem.split("seed")[-1])
            payload = json.loads(fp.read_text(encoding="utf-8"))
            inter = payload.get("metrics", {}).get("test_inter")
            if not inter:
                continue
            per_seed[seed] = {
                k: float(v) for k, v in inter.items() if not k.endswith("_std")
            }
        if per_seed:
            out[model_key] = per_seed
    return out


def merge_ari(
    data: dict[str, dict[int, dict[str, float]]], ari_json: Path | None
) -> None:
    """Fold ``compute_ari.py`` output into the per-seed metric records in place.

    ARI requires re-simulating from a checkpoint, so it is produced by a separate
    pass rather than at training time; merging keeps it on the same seed-level
    footing as every other column.
    """
    if ari_json is None or not ari_json.exists():
        return
    records = json.loads(ari_json.read_text(encoding="utf-8"))
    merged = 0
    for rec in records:
        mk, seed = rec.get("model"), rec.get("seed")
        if mk in data and seed in data[mk]:
            for key, val in rec.items():
                if key.startswith("fc_ari_"):
                    data[mk][seed][key] = float(val)
            merged += 1
    print(f"Merged ARI for {merged} (model, seed) records from {ari_json}")


def composite_error(metrics: dict[str, float]) -> float:
    """Predefined aggregate: mean of flipped core metrics, lower is better."""
    vals = []
    for key, higher_better in COMPOSITE_KEYS:
        if key not in metrics:
            return float("nan")
        v = metrics[key]
        vals.append(1.0 - v if higher_better else v)
    return float(np.mean(vals))


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down correction over a dict of p-values."""
    items = [(k, p) for k, p in pvals.items() if not math.isnan(p)]
    if not items:
        return dict.fromkeys(pvals, float("nan"))
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = (m - i) * p
        running = max(running, val)  # enforce monotonicity
        adjusted[k] = min(1.0, running)
    for k in pvals:
        adjusted.setdefault(k, float("nan"))
    return adjusted


def paired_tests(
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    reference: str,
    models: list[str],
) -> dict[str, float]:
    """Holm-corrected paired Wilcoxon p-values vs *reference*, paired on seed."""
    from scipy.stats import wilcoxon

    raw: dict[str, float] = {}
    if reference not in data:
        return {}
    ref_seeds = data[reference]
    for mk in models:
        if mk == reference or mk not in data:
            continue
        common = sorted(set(ref_seeds) & set(data[mk]))
        a = [ref_seeds[s].get(metric, float("nan")) for s in common]
        b = [data[mk][s].get(metric, float("nan")) for s in common]
        pairs = [
            (x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))
        ]
        if len(pairs) < 5 or all(abs(x - y) < 1e-12 for x, y in pairs):
            raw[mk] = float("nan")
            continue
        try:
            _, p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs],
                            zero_method="wilcox", alternative="two-sided")
            raw[mk] = float(p)
        except ValueError:
            raw[mk] = float("nan")
    return _holm(raw)


def _sig(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return r"$^{***}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def _fmt(mean: float, std: float, *, bold: bool, sig: str = "") -> str:
    if math.isnan(mean):
        return "---"
    # Three decimals throughout: the smallest column (Meta |Delta|) is ~0.01, so
    # a uniform fixed-point format stays readable and avoids mixed notation.
    cell = f"{mean:.3f}{sig} {{\\scriptsize $\\pm$ {std:.3f}}}"
    return f"\\bf{{{cell}}}" if bold else cell


def build_table(
    data: dict[str, dict[int, dict[str, float]]],
    *,
    main: bool,
) -> tuple[str, dict]:
    rows = [(k, lbl) for k, lbl in ROWS if k in data and (not main or k in MAIN_ROWS)]
    keep = MAIN_COLUMNS if main else [c[0] for c in ALL_COLUMNS
                                     if c[0] not in FULL_EXCLUDE]
    cols = [c for c in ALL_COLUMNS if c[0] in keep]
    models = [k for k, _ in rows]

    # Aggregate mean/std across seeds, plus the composite per seed.
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    comp: dict[str, list[float]] = {}
    for mk in models:
        agg[mk] = {}
        seeds = sorted(data[mk])
        for key, _hdr, _hb in cols:
            vals = [data[mk][s][key] for s in seeds if key in data[mk][s]]
            if vals:
                agg[mk][key] = (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                )
            else:
                agg[mk][key] = (float("nan"), 0.0)
        comp[mk] = [composite_error(data[mk][s]) for s in seeds]

    # Significance vs the reference architecture, per metric, Holm-corrected.
    pvals: dict[str, dict[str, float]] = {
        key: paired_tests(data, key, REFERENCE, models) for key, _h, _b in cols
    }
    # Composite is tested on its own, also Holm-corrected across architectures.
    comp_data = {
        mk: {s: {"composite": composite_error(data[mk][s])} for s in sorted(data[mk])}
        for mk in models
    }
    comp_p = paired_tests(comp_data, "composite", REFERENCE, models)

    # Best per column by mean.
    best: dict[str, str] = {}
    for key, _hdr, higher_better in cols:
        scored = [(mk, agg[mk][key][0]) for mk in models if not math.isnan(agg[mk][key][0])]
        if scored:
            best[key] = max(scored, key=lambda kv: kv[1])[0] if higher_better \
                else min(scored, key=lambda kv: kv[1])[0]
    comp_means = {mk: float(np.nanmean(comp[mk])) for mk in models}
    best_comp = min(comp_means, key=lambda k: comp_means[k]) if comp_means else None

    n_col = " c" * (len(cols) + 1)
    header = " & ".join(
        ["Model"] + [h for _k, h, _b in cols] + [r"Composite $\downarrow$"]
    ) + r" \\"
    lines = [f"\\begin{{tabular}}{{l{n_col}}}", "\\toprule", header, "\\midrule"]

    for mk, label in rows:
        n_seeds = len(data[mk])
        cells = [f"{label} ({n_seeds})"]
        for key, _hdr, _hb in cols:
            mean, std = agg[mk][key]
            cells.append(_fmt(mean, std, bold=best.get(key) == mk,
                              sig=_sig(pvals[key].get(mk, float("nan")))))
        cm = comp_means[mk]
        cs = float(np.nanstd(comp[mk], ddof=1)) if len(comp[mk]) > 1 else 0.0
        cells.append(_fmt(cm, cs, bold=best_comp == mk,
                          sig=_sig(comp_p.get(mk, float("nan")))))
        lines.append(" & ".join(cells) + r" \\")

    lines += ["\\bottomrule", "\\end{tabular}"]

    # Seeds present for some architecture but absent for this one. Absence is
    # reported rather than silently averaged over, because a seed can be missing
    # for two very different reasons -- the run has not been executed yet, or the
    # refit diverged and produced no usable model. The second case matters for
    # interpretation: summarizing over only the runs that completed flatters an
    # architecture that sometimes blows up. Which case applies has to be read off
    # the job logs; this function only records that the cell is empty.
    all_seeds = sorted({s for mk in models for s in data[mk]})
    missing = {
        mk: [s for s in all_seeds if s not in data[mk]] for mk in models
    }

    summary = {
        "n_seeds": {mk: len(data[mk]) for mk in models},
        "seeds_attempted": all_seeds,
        "seeds_missing": {mk: v for mk, v in missing.items() if v},
        "reference": REFERENCE,
        "composite_mean": comp_means,
        "composite_p_holm": comp_p,
        "p_holm": {k: pvals[k] for k in pvals},
    }
    return "\n".join(lines) + "\n", summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--suffix", default="n94",
                    help="Run-suffix family to aggregate (default: n94).")
    ap.add_argument("--main", action="store_true",
                    help="Emit the 3-architecture main-text table instead of all five.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--common-seeds", action="store_true",
                    help="Restrict every architecture to the seeds all of them "
                         "completed (balanced complete-case design). Use for the "
                         "final tables; omit while runs are still in flight.")
    ap.add_argument("--ari-json", default="results/ari_per_seed.json",
                    help="Per-seed ARI values from examples/compute_ari.py "
                         "(silently skipped when absent).")
    args = ap.parse_args()

    data = collect(args.ds, Path(args.runs_dir), args.suffix)
    merge_ari(data, Path(args.ari_json) if args.ari_json else None)
    if not data:
        raise SystemExit(f"No runs found under {args.runs_dir} for ds={args.ds}")

    all_seeds = sorted({s for mk in data for s in data[mk]})
    print("Seed coverage:")
    for mk, label in ROWS:
        if mk in data:
            seeds = sorted(data[mk])
            gap = [s for s in all_seeds if s not in seeds]
            note = f"  ABSENT {gap} (not run, or refit diverged -- check logs)" if gap else ""
            print(f"  {label:15s} n={len(seeds):2d}  seeds={seeds}{note}")
        else:
            print(f"  {label:15s} NO RUNS")

    if args.common_seeds:
        data, dropped = restrict_to_common(data)
        if dropped:
            print(f"\nRestricted to common seeds; dropped {dropped} "
                  f"(not completed by every architecture).")
        remaining = sorted(next(iter(data.values()))) if data else []
        print(f"Balanced design over {len(remaining)} seeds: {remaining}")

    table, summary = build_table(data, main=args.main)
    print("\n" + table)
    print("Composite error (lower is better):")
    for mk, v in sorted(summary["composite_mean"].items(), key=lambda kv: kv[1]):
        p = summary["composite_p_holm"].get(mk, float("nan"))
        ptxt = "reference" if mk == REFERENCE else f"Holm p={p:.4g}"
        print(f"  {mk:15s} {v:.4f}   {ptxt}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(f"\nWrote {out}")
    if args.summary_out:
        so = Path(args.summary_out)
        so.parent.mkdir(parents=True, exist_ok=True)
        so.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"Wrote {so}")


if __name__ == "__main__":
    main()
