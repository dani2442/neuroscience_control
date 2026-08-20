#!/usr/bin/env python3
"""Aggregate the generative-evaluation JSONs into variance, coverage and
personalization summaries (and LaTeX tables).

Consumes ``results/generative/ts_young_<model>_n94_seed<seed>.json`` written by
``examples/evaluate_generative.py`` and produces three blocks.

**Variance decomposition** (reviewer point 4). Each metric is observed on a
``seed x subject x realization`` grid, so the single ``±`` previously reported
was a mixture of three distinct sources. We report the observed standard
deviation of the seed-level means (fitting variability), of the subject-level
means (between-subject heterogeneity), and the within-(seed, subject) spread
across stochastic rollouts (simulation noise). These are descriptive variance
components of the corresponding group means -- not a fitted random-effects
model -- which is what the table caption should say.

**Coverage** (reviewer point 4). For each summary statistic we report the
fraction of held-out subjects whose empirical value falls inside the model's
central 90% predictive interval, and the mean percentile of the empirical value
within the simulated draws. A calibrated generative model covers ~0.90 with a
mean percentile near 0.5; systematic deviation means the model reproduces the
point estimate but not the dispersion.

**Personalization** (reviewer point 2). The subject is the unit. For each
subject we average ``delta_i = within_i - between_i`` over seeds, then run a
two-sided Wilcoxon signed-rank test against zero and a percentile bootstrap CI
over subjects. Top-1 identification accuracy is reported per seed against the
``1/n`` chance level.

Crucially, ``delta > 0`` on its own is **not** evidence of a personalized model.
Every simulation is seeded with the subject's own initial condition, which is a
subject-specific input, so a purely group-level model can score a large positive
delta simply by propagating that state. The Coupled Hopf demonstrates exactly
this: it shares a, kappa, G, sigma and a group-average connectome across the
whole cohort, yet still reaches delta ~ +0.18 and ~42% top-1 identification
against ~1.5% chance -- more than any learned-coupling architecture achieves. It
is therefore the *reference* for initial-condition carryover, and ``delta -
delta_IC`` is reported alongside the raw delta. Every learned model scores a
negative increment, i.e. it preserves *less* subject-specific structure than a
model that cannot be personalized at all. That is evidence against a
personalization claim for these architectures, not weak evidence for one, and it
is why the manuscript no longer claims personalized digital-twin capability.

Usage::

    python examples/aggregate_generative.py --ds ts_young \
        --out-dir paper_submission/sections
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = [
    ("hopf", "Coupled Hopf"),
    ("nsde", "Neural SDE"),
    ("hybrid_hopf", "Hybrid Hopf"),
    ("gnn_hopf", "GNN-Hopf"),
    ("hybrid_neural", "Hopf+Neural"),
]

# NOTE ON THE ESTIMAND. These are *single-trajectory, single-subject* metrics:
# one simulated rollout scored against one subject's window. Table 1 instead
# reports batch Fisher-averaged FC across subjects, which is a far easier target
# (group-average structure averages out both subject idiosyncrasy and simulation
# noise). The two are not comparable in level -- e.g. the Coupled Hopf scores
# ~0.21 here and ~0.91 there -- so the labels below say so explicitly and the
# caption must repeat it. What this table is for is the *decomposition* of
# variability, not the absolute level.
METRICS = [
    ("fc_correlation", r"FC corr (single traj.)"),
    ("phase_fc_correlation", r"phFC corr (single traj.)"),
    ("phfcd_ks", r"phFCD KS (single traj.)"),
    ("metastability_diff", r"Meta $|\Delta|$ (single traj.)"),
]

COVERAGE_STATS = [
    ("fc_strength", "FC strength"),
    ("fc_dispersion", "FC dispersion"),
    ("metastability", "Metastability"),
    ("phfcd_median", "phFCD median"),
]


def load(ds: str, gen_dir: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for model_key, _label in MODELS:
        for fp in sorted(gen_dir.glob(f"{ds}_{model_key}_n94_seed*.json")):
            seed = int(fp.stem.split("seed")[-1])
            out[model_key][seed] = json.loads(fp.read_text(encoding="utf-8"))
    return {k: v for k, v in out.items() if v}


def variance_components(runs: dict[int, dict], metric: str) -> dict[str, float]:
    """Descriptive SDs of seed-, subject-, and realization-level variation."""
    # grid[seed][subject] = list of per-realization values
    grid: dict[int, dict[int, list[float]]] = {}
    for seed, payload in runs.items():
        grid[seed] = {}
        for rec in payload["per_subject"]:
            vals = [
                r[metric] for r in rec["realizations"]
                if metric in r and np.isfinite(r[metric])
            ]
            if vals:
                grid[seed][rec["subject"]] = vals

    seeds = sorted(grid)
    subjects = sorted({s for g in grid.values() for s in g})
    if not seeds or not subjects:
        return {}

    # Within-(seed, subject) spread across stochastic rollouts.
    within = [
        float(np.std(v, ddof=1))
        for g in grid.values() for v in g.values() if len(v) > 1
    ]
    # Seed-level means (averaging over subjects and realizations).
    seed_means = [
        float(np.mean([np.mean(v) for v in grid[s].values()])) for s in seeds if grid[s]
    ]
    # Subject-level means (averaging over seeds and realizations).
    subj_means = []
    for sub in subjects:
        vals = [np.mean(grid[s][sub]) for s in seeds if sub in grid[s]]
        if vals:
            subj_means.append(float(np.mean(vals)))

    return {
        "grand_mean": float(np.mean(seed_means)) if seed_means else float("nan"),
        "sd_seed": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
        "sd_subject": float(np.std(subj_means, ddof=1)) if len(subj_means) > 1 else 0.0,
        "sd_realization": float(np.mean(within)) if within else 0.0,
        "n_seeds": len(seeds),
        "n_subjects": len(subjects),
    }


def coverage_summary(runs: dict[int, dict], stat: str) -> dict[str, float]:
    inside, pcts = [], []
    for payload in runs.values():
        for rec in payload["per_subject"]:
            cov = rec.get("coverage", {}).get(stat)
            if not cov:
                continue
            inside.append(bool(cov["inside_90"]))
            pcts.append(float(cov["percentile"]))
    if not inside:
        return {}
    return {
        "coverage_90": float(np.mean(inside)),
        "mean_percentile": float(np.mean(pcts)),
        "n": len(inside),
    }


def personalization_summary(runs: dict[int, dict]) -> dict[str, float]:
    from scipy.stats import wilcoxon

    # subject -> per-seed deltas (subject is the replication unit)
    by_subject: dict[int, list[float]] = defaultdict(list)
    within_by_subject: dict[int, list[float]] = defaultdict(list)
    between_by_subject: dict[int, list[float]] = defaultdict(list)
    ident_per_seed: list[float] = []
    chance = float("nan")

    for payload in runs.values():
        ident_per_seed.append(float(payload["identification_accuracy"]))
        chance = float(payload["identification_chance"])
        for rec in payload["personalization"]:
            if np.isfinite(rec["delta"]):
                by_subject[rec["subject"]].append(float(rec["delta"]))
                within_by_subject[rec["subject"]].append(float(rec["within"]))
                between_by_subject[rec["subject"]].append(float(rec["between"]))

    deltas = np.array([np.mean(v) for v in by_subject.values()], dtype=float)
    withins = np.array([np.mean(v) for v in within_by_subject.values()], dtype=float)
    betweens = np.array([np.mean(v) for v in between_by_subject.values()], dtype=float)
    if deltas.size < 3:
        return {}

    try:
        _, p = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
        p = float(p)
    except ValueError:
        p = float("nan")

    rng = np.random.default_rng(0)
    boot = rng.choice(deltas, size=(10000, deltas.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "n_subjects": int(deltas.size),
        "within_mean": float(withins.mean()),
        "between_mean": float(betweens.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_median": float(np.median(deltas)),
        "delta_ci_lo": float(lo),
        "delta_ci_hi": float(hi),
        "wilcoxon_p": p,
        "frac_positive": float((deltas > 0).mean()),
        "identification_mean": float(np.mean(ident_per_seed)),
        "identification_sd": float(np.std(ident_per_seed, ddof=1))
        if len(ident_per_seed) > 1 else 0.0,
        "identification_chance": chance,
        "n_seeds": len(ident_per_seed),
    }


def _p_str(p: float) -> str:
    if math.isnan(p):
        return "---"
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.3g}"


def build_variance_table(data: dict[str, dict[int, dict]]) -> str:
    lines = [
        r"\begin{tabular}{l l c c c c}",
        r"\toprule",
        r"Model & Metric & Mean & SD$_\mathrm{seed}$ & SD$_\mathrm{subject}$ "
        r"& SD$_\mathrm{sim}$ \\",
        r"\midrule",
    ]
    for i, (mk, label) in enumerate(MODELS):
        if mk not in data:
            continue
        if i:
            lines.append(r"\midrule")
        for j, (key, mlabel) in enumerate(METRICS):
            vc = variance_components(data[mk], key)
            if not vc:
                continue
            name = label if j == 0 else ""
            lines.append(
                f"{name} & {mlabel} & {vc['grand_mean']:.3f} & "
                f"{vc['sd_seed']:.3f} & {vc['sd_subject']:.3f} & "
                f"{vc['sd_realization']:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_coverage_table(data: dict[str, dict[int, dict]]) -> str:
    header = " & ".join(["Model"] + [lbl for _k, lbl in COVERAGE_STATS])
    lines = [
        r"\begin{tabular}{l" + " c" * len(COVERAGE_STATS) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for mk, label in MODELS:
        if mk not in data:
            continue
        cells = [label]
        for key, _lbl in COVERAGE_STATS:
            cs = coverage_summary(data[mk], key)
            cells.append(
                f"{cs['coverage_90']:.2f} ({cs['mean_percentile']:.2f})" if cs else "---"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# The Coupled Hopf carries no subject-specific parameters: a, kappa, G and sigma
# are shared across the cohort and the connectome is a group average. Any
# within-minus-between advantage it shows therefore cannot be personalization of
# the fitted model -- it can only be carryover from the subject's own initial
# condition, which is the one subject-specific input every simulation receives.
#
# Empirically it does not merely set a floor, it sets the *ceiling*: the group
# model preserves more subject-specific structure than any learned-coupling
# architecture, and the ordering tracks inverse model flexibility. We therefore
# report `delta - delta_IC` as a signed increment against this reference. A
# negative increment -- which is what every learned model actually produces --
# means the architecture destroys subject-specific information relative to a
# model that cannot be personalized at all, and is evidence *against* a
# personalization claim rather than weak evidence for one.
IC_REFERENCE = "hopf"


def build_personalization_table(data: dict[str, dict[int, dict]]) -> tuple[str, dict]:
    lines = [
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Model & Within & Between & $\Delta$ [95\% CI] & $p$ & "
        r"$\Delta-\Delta_{\mathrm{IC}}$ & Identification \\",
        r"\midrule",
    ]
    summary: dict[str, dict] = {}
    ref = personalization_summary(data[IC_REFERENCE]) if IC_REFERENCE in data else None
    ref_delta = ref["delta_mean"] if ref else float("nan")

    for mk, label in MODELS:
        if mk not in data:
            continue
        ps = personalization_summary(data[mk])
        if not ps:
            continue
        ps["delta_vs_ic_reference"] = (
            float("nan") if math.isnan(ref_delta) else ps["delta_mean"] - ref_delta
        )
        summary[mk] = ps
        if mk == IC_REFERENCE:
            incr = r"--- (reference)"
        elif math.isnan(ps["delta_vs_ic_reference"]):
            incr = "---"
        else:
            incr = f"${ps['delta_vs_ic_reference']:+.3f}$"
        lines.append(
            f"{label} & {ps['within_mean']:.3f} & {ps['between_mean']:.3f} & "
            f"${ps['delta_mean']:+.3f}$ [${ps['delta_ci_lo']:+.3f}$, "
            f"${ps['delta_ci_hi']:+.3f}$] & {_p_str(ps['wilcoxon_p'])} & {incr} & "
            f"{ps['identification_mean']:.2f} {{\\scriptsize $\\pm$ "
            f"{ps['identification_sd']:.2f}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--gen-dir", default="results/generative")
    ap.add_argument("--out-dir", default=None,
                    help="Directory to write the three .tex tabulars into.")
    ap.add_argument("--summary-out", default="results/generative_summary.json")
    args = ap.parse_args()

    data = load(args.ds, Path(args.gen_dir))
    if not data:
        raise SystemExit(f"No generative results under {args.gen_dir}")

    print("Coverage of generative runs:")
    for mk, label in MODELS:
        if mk in data:
            print(f"  {label:15s} seeds={sorted(data[mk])}")

    var_tab = build_variance_table(data)
    cov_tab = build_coverage_table(data)
    pers_tab, pers_summary = build_personalization_table(data)

    print("\n--- variance decomposition ---\n" + var_tab)
    print("--- coverage (fraction inside 90% PI, mean percentile) ---\n" + cov_tab)
    print("--- personalization ---\n" + pers_tab)

    print("Personalization detail:")
    for mk, ps in pers_summary.items():
        print(f"  {mk:15s} delta={ps['delta_mean']:+.4f} "
              f"[{ps['delta_ci_lo']:+.4f},{ps['delta_ci_hi']:+.4f}] "
              f"p={ps['wilcoxon_p']:.3g} pos={ps['frac_positive']:.2f} "
              f"ident={ps['identification_mean']:.3f}"
              f"(chance {ps['identification_chance']:.3f}) "
              f"n_subj={ps['n_subjects']}")

    if args.out_dir:
        od = Path(args.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "ts_young_variance_table.tex").write_text(var_tab, encoding="utf-8")
        (od / "ts_young_coverage_table.tex").write_text(cov_tab, encoding="utf-8")
        (od / "ts_young_personalization_table.tex").write_text(pers_tab, encoding="utf-8")
        print(f"\nWrote three tabulars to {od}")

    if args.summary_out:
        so = Path(args.summary_out)
        so.parent.mkdir(parents=True, exist_ok=True)
        so.write_text(json.dumps(
            {
                "personalization": pers_summary,
                "variance": {
                    mk: {k: variance_components(data[mk], k) for k, _ in METRICS}
                    for mk in data
                },
                "coverage": {
                    mk: {k: coverage_summary(data[mk], k) for k, _ in COVERAGE_STATS}
                    for mk in data
                },
            }, indent=1), encoding="utf-8")
        print(f"Wrote {so}")


if __name__ == "__main__":
    main()
