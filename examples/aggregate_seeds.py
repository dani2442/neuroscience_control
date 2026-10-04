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


def collect(
    ds: str,
    runs_dir: Path,
    suffix: str,
    seeds: set[int] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Return {model_key: {seed: {metric: value}}}.

    ``seeds`` restricts the scan to an explicit seed set. The run directory can
    hold more than one sweep, and a bare ``seed*`` glob would silently pool them
    into a single analysis, so the caller states which sweep a table is built
    from rather than inheriting whatever happens to be on disk.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for model_key, _label in ROWS:
        per_seed: dict[int, dict[str, float]] = {}
        for fp in sorted(runs_dir.glob(f"{ds}_{model_key}_{suffix}_seed*.json")):
            seed = int(fp.stem.split("seed")[-1])
            if seeds is not None and seed not in seeds:
                continue
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
    return {mk: rec["p_holm"] for mk, rec in
            paired_tests_detail(data, metric, reference, models).items()}


def paired_tests_detail(
    data: dict[str, dict[int, dict[str, float]]],
    metric: str,
    reference: str,
    models: list[str],
) -> dict[str, dict[str, float]]:
    """Per-comparison test record vs *reference*, paired on seed.

    Returns ``{model: {n, statistic, p_raw, p_holm, family_size}}``. The Holm
    family is the set of architectures compared against the reference *within
    this metric* -- not the set of metrics in the table. Each column therefore
    answers "which architectures differ from the reference on this descriptor",
    which is the question the table is read for; metrics are separate questions
    and are not pooled into one family. ``family_size`` is recorded per record so
    the correction is auditable from the emitted table alone.

    The test is the exact (permutation) two-sided Wilcoxon signed-rank test.
    ``method="exact"`` is pinned rather than left to scipy's ``"auto"`` so the
    reported p-values cannot silently switch to the normal approximation if the
    seed count later grows past scipy's auto threshold; with n=9 and no zero
    differences or ties the two agree exactly, so pinning changes no published
    value. Note the resolution floor this imposes: with n nonzero pairs the
    smallest attainable two-sided p is 2/2**n (0.0039 at n=9), so a Holm family
    of size m cannot report below m * 2/2**n.
    """
    from scipy.stats import wilcoxon

    raw: dict[str, float] = {}
    detail: dict[str, dict[str, float]] = {}
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
        nonzero = [(x, y) for x, y in pairs if abs(x - y) >= 1e-12]
        if len(pairs) < 5 or not nonzero:
            raw[mk] = float("nan")
            detail[mk] = {"n": len(nonzero), "statistic": float("nan"),
                          "p_raw": float("nan")}
            continue
        try:
            stat, p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs],
                               zero_method="wilcox", alternative="two-sided",
                               method="exact")
            raw[mk] = float(p)
            detail[mk] = {"n": len(nonzero), "statistic": float(stat),
                          "p_raw": float(p)}
        except ValueError:
            raw[mk] = float("nan")
            detail[mk] = {"n": len(nonzero), "statistic": float("nan"),
                          "p_raw": float("nan")}
    adjusted = _holm(raw)
    family_size = sum(1 for v in raw.values() if not math.isnan(v))
    for mk, rec in detail.items():
        rec["p_holm"] = adjusted.get(mk, float("nan"))
        rec["family_size"] = float(family_size)
    return detail


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

    # Per-row seed counts are only informative when they differ; in a balanced
    # (complete-case) table the count is stated once in the caption instead.
    balanced = len({len(data[mk]) for mk, _ in rows}) == 1
    for mk, label in rows:
        cells = [label if balanced else f"{label} ({len(data[mk])})"]
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


def _p(v: float) -> str:
    """Enough digits to show the 2/2**n resolution floor of the exact test."""
    return "---" if math.isnan(v) else f"{v:.4f}"


def build_pvalue_table(
    data: dict[str, dict[int, dict[str, float]]],
    *,
    main: bool,
) -> str:
    """Every annotation in the corresponding table, with its raw and adjusted p.

    Reviewer point: the stars have to be reproducible from the reported
    procedure. One row per (metric, comparison) carries n, the signed-rank
    statistic, the exact raw p, the Holm family size, the adjusted p, and the
    resulting annotation -- so each superscript can be checked end to end.
    """
    rows = [(k, lbl) for k, lbl in ROWS if k in data and (not main or k in MAIN_ROWS)]
    keep = MAIN_COLUMNS if main else [c[0] for c in ALL_COLUMNS
                                      if c[0] not in FULL_EXCLUDE]
    cols = [c for c in ALL_COLUMNS if c[0] in keep]
    models = [k for k, _ in rows]
    labels = dict(ROWS)

    comp_data = {
        mk: {s: {"composite": composite_error(data[mk][s])} for s in sorted(data[mk])}
        for mk in models
    }
    blocks = [(hdr, paired_tests_detail(data, key, REFERENCE, models))
              for key, hdr, _hb in cols]
    blocks.append((r"Composite $\downarrow$",
                   paired_tests_detail(comp_data, "composite", REFERENCE, models)))

    lines = [r"\begin{tabular}{l l c c c c c c}", r"\toprule",
             r"Metric & Comparison & $n$ & $W$ & raw $p$ & $m$ "
             r"& Holm $p$ & Annot. \\",
             r"\midrule"]
    for hdr, recs in blocks:
        first = True
        for mk in models:
            if mk == REFERENCE or mk not in recs:
                continue
            r = recs[mk]
            nn = int(r["n"])
            stat = "---" if math.isnan(r["statistic"]) else f"{r['statistic']:.0f}"
            fam = "---" if math.isnan(r["family_size"]) else f"{int(r['family_size'])}"
            annot = _sig(r["p_holm"]).replace("$^{", "").replace("}$", "") or "n.s."
            lines.append(
                f"{hdr if first else ''} & {labels[mk]} vs.\\ {labels[REFERENCE]} "
                f"& {nn} & {stat} & {_p(r['p_raw'])} & {fam} "
                f"& {_p(r['p_holm'])} & {annot} \\\\"
            )
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def build_per_seed_table(data: dict[str, dict[int, dict[str, float]]], *,
                         main: bool) -> str:
    """Per-seed composite error: the paired input behind the composite tests."""
    rows = [(k, lbl) for k, lbl in ROWS if k in data and (not main or k in MAIN_ROWS)]
    models = [k for k, _ in rows]
    seeds = sorted(set.intersection(*(set(data[mk]) for mk in models))) if models else []

    lines = [r"\begin{tabular}{l" + " c" * len(seeds) + " c}", r"\toprule",
             " & ".join(["Model"] + [str(s) for s in seeds] + [r"Mean $\pm$ SD"])
             + r" \\", r"\midrule"]
    for mk, label in rows:
        vals = [composite_error(data[mk][s]) for s in seeds]
        cells = [f"{v:.3f}" for v in vals]
        mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        lines.append(" & ".join([label] + cells + [f"{mean:.3f} $\\pm$ {sd:.3f}"])
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _parse_seeds(spec: str | None) -> set[int] | None:
    """Parse '1-10' / '1,2,3' / '1-3,7' into a seed set; None means every seed."""
    if not spec:
        return None
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--runs-dir", default="results/runs")
    ap.add_argument("--seeds", default=None,
                    help="Explicit seed set for this table, e.g. '1-10' or "
                         "'1,2,3'. Required whenever results/runs holds more "
                         "than one sweep, so a table names the runs behind it.")
    ap.add_argument("--suffix", default="n94",
                    help="Run-suffix family to aggregate (default: n94).")
    ap.add_argument("--main", action="store_true",
                    help="Emit the 3-architecture main-text table instead of all five.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--pvalues-out", default=None,
                    help="Emit the raw/Holm-adjusted p-value table backing every "
                         "superscript in the corresponding results table.")
    ap.add_argument("--per-seed-out", default=None,
                    help="Emit the per-seed composite errors (the paired input "
                         "to the composite tests).")
    ap.add_argument("--common-seeds", action="store_true",
                    help="Restrict every architecture to the seeds all of them "
                         "completed (balanced complete-case design). Use for the "
                         "final tables; omit while runs are still in flight.")
    ap.add_argument("--ari-json", default="results/ari_per_seed.json",
                    help="Per-seed ARI values from examples/compute_ari.py "
                         "(silently skipped when absent).")
    args = ap.parse_args()

    data = collect(args.ds, Path(args.runs_dir), args.suffix, _parse_seeds(args.seeds))
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
    for path, builder in ((args.pvalues_out, build_pvalue_table),
                          (args.per_seed_out, build_per_seed_table)):
        if not path:
            continue
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(builder(data, main=args.main), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
