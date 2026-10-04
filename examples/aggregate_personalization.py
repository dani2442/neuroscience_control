#!/usr/bin/env python3
"""Aggregate the N=1 personalization runs into within- vs between-subject summaries.

Consumes ``results/personalization/<ds>_<model>_sub<idx>_seed<seed>.json`` written
by ``examples/train_personalized.py`` and reports two contrasts per model and
metric. The replication unit is always the *subject*, never the test batch.

**Pooled contrast.** For each personalised fit, ``delta = intra - inter`` on the
two pooled test loaders (sign-flipped for lower-is-better metrics, so a positive
delta always means "better on its own subject"). Wilcoxon signed-rank against
zero, plus a percentile bootstrap CI over fits.

**Difficulty-controlled contrast.** The pooled delta confounds personalization
with subject difficulty: a target subject with unusually model-friendly dynamics
yields a positive delta no matter what the fit learned. Using the per-subject
scores, we instead build the fit x target matrix and, for each target subject t
that was itself personalised, compare

    own       = M[t][t]                       (t's own model on t)
    others_on = mean_{f != t} M[f][t]         (other subjects' models on t)

Because the target is held fixed, its difficulty cancels; only the identity of
the *fitting* subject varies. This is the contrast that actually tests
personalization, and the one to quote. The naive row-wise version (own vs. the
same model on other targets) is reported alongside it so the gap between the two
is visible.

Usage::

    python examples/aggregate_personalization.py --ds ts_young
"""

from __future__ import annotations

import argparse
import json
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

# Metrics where a larger value is a better fit. Everything else in
# EVAL_METRIC_KEYS is a distance/error and improves as it shrinks.
HIGHER_BETTER = {"fc_correlation", "phase_fc_correlation", "temporal_correlation"}

METRICS = [
    "fc_correlation",
    "phase_fc_correlation",
    "phfcd_ks",
    "fcd_ks",
    "metastability_diff",
]


def _oriented(metric: str, value: float) -> float:
    """Return *value* sign-flipped so that larger is always better."""
    return value if metric in HIGHER_BETTER else -value


def load(ds: str, res_dir: Path, tag: str = "", seed: int = 42) -> dict[str, dict[int, dict]]:
    """model_key -> {subject index of the fit -> record}, for one arm and seed.

    The filename glob alone cannot separate arms: ``..._seed42.json`` and
    ``..._seed42_ft.json`` both match ``_sub*_seed*.json``, so globbing without
    a filter would silently average a scratch fit and a fine-tuned one into the
    same contrast. Records are therefore filtered on the ``tag`` field they
    carry. Records written before tagging existed have no field and read as the
    untagged (scratch) arm.

    Records are also filtered on ``seed``: each subject must map to exactly one
    fit, and two seeds of the same subject would otherwise overwrite each other
    in glob order.
    """
    tag = (tag or "").strip().strip("_")
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for model_key, _label in MODELS:
        for fp in sorted(res_dir.glob(f"{ds}_{model_key}_sub*_seed*.json")):
            rec = json.loads(fp.read_text(encoding="utf-8"))
            if (rec.get("tag") or "") != tag or int(rec["seed"]) != seed:
                continue
            subj = int(rec["subject"])
            if subj in out[model_key]:
                raise SystemExit(f"Two {model_key} records for subject {subj} "
                                 f"(tag={tag!r}, seed={seed}): {fp.name}")
            out[model_key][subj] = rec
    return {k: v for k, v in out.items() if v}


def _arm_label(runs: dict[int, dict]) -> str:
    """One-line description of how this arm's fits were initialised."""
    arms = {r.get("arm", "scratch") for r in runs.values()}
    if arms == {"scratch"}:
        return "from scratch"
    if arms == {"base"}:
        bases = {(r.get("finetune") or {}).get("init_checkpoint") for r in runs.values()}
        name = Path(next(iter(bases))).name if len(bases) == 1 and None not in bases else "?"
        return f"unadapted base {name} (no training)"
    ft = [r.get("finetune") or {} for r in runs.values() if r.get("arm") == "finetune"]
    if not ft:
        return f"arms={sorted(arms)}"
    bases = {f.get("init_checkpoint") for f in ft}
    lrs = {f.get("lr") for f in ft}
    frozen = {tuple(f.get("freeze_patterns") or ()) for f in ft}
    bits = [f"fine-tuned from {Path(next(iter(bases))).name}" if len(bases) == 1
            else f"fine-tuned from {len(bases)} different bases (!)"]
    if len(lrs) == 1:
        bits.append(f"lr={next(iter(lrs))}")
    if frozen != {()}:
        bits.append(f"frozen={sorted(frozen)}")
    if len(arms) > 1:
        bits.append(f"MIXED ARMS: {sorted(arms)} (!)")
    return ", ".join(bits)


def _stats(deltas: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Wilcoxon signed-rank vs zero plus a percentile bootstrap CI of the mean."""
    from scipy.stats import wilcoxon

    deltas = np.asarray([d for d in deltas if np.isfinite(d)], dtype=float)
    if deltas.size == 0:
        return {}
    if deltas.size >= 3:
        try:
            _, p = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
            p = float(p)
        except ValueError:
            p = float("nan")
        rng = np.random.default_rng(seed)
        boot = rng.choice(deltas, size=(10000, deltas.size), replace=True).mean(axis=1)
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    else:
        p, lo, hi = float("nan"), float("nan"), float("nan")
    return {
        "n": int(deltas.size),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "ci_lo": lo,
        "ci_hi": hi,
        "wilcoxon_p": p,
        "frac_positive": float((deltas > 0).mean()),
    }


def pooled_contrast(runs: dict[int, dict], metric: str) -> dict[str, float]:
    """intra - inter on the two pooled test loaders, one delta per fit."""
    intra, inter = [], []
    for rec in runs.values():
        m = rec["metrics"]
        if metric in m["test_intra"] and metric in m["test_inter"]:
            intra.append(float(m["test_intra"][metric]))
            inter.append(float(m["test_inter"][metric]))
    if not intra:
        return {}
    intra_a, inter_a = np.array(intra), np.array(inter)
    deltas = np.array([_oriented(metric, a) - _oriented(metric, b)
                       for a, b in zip(intra_a, inter_a)])
    out = _stats(deltas)
    out.update({"intra_mean": float(intra_a.mean()), "inter_mean": float(inter_a.mean())})
    return out


def matrix_contrasts(runs: dict[int, dict], metric: str) -> dict[str, dict]:
    """Column-wise (difficulty-controlled) and row-wise (naive) contrasts.

    ``M[f][t]`` is the score of the fit personalised on subject *f*, evaluated on
    subject *t*'s held-out segment.
    """
    fits = sorted(runs)
    M: dict[int, dict[int, float]] = {}
    for f in fits:
        per = runs[f].get("per_subject") or {}
        # A subject flagged as non-finite had a diverged rollout, which poisons
        # the whole batch. Some metrics (notably the KS distances) still return a
        # finite number from a NaN simulation, so those values are meaningless
        # rather than merely noisy — drop every metric for such a subject, not
        # just the ones that happen to read NaN.
        bad = {int(j) for j in runs[f].get("nonfinite_subjects", [])}
        M[f] = {int(t): float(v[metric]) for t, v in per.items()
                if metric in v and int(t) not in bad and np.isfinite(v[metric])}
    if not any(M.values()):
        return {}

    # Column-wise: hold the target fixed, vary which subject's model scores it.
    col_deltas, col_rows = [], []
    for t in fits:
        own = M.get(t, {}).get(t, float("nan"))
        if not np.isfinite(own):
            continue
        others = [M[f][t] for f in fits
                  if f != t and np.isfinite(M[f].get(t, float("nan")))]
        if not others:
            continue
        others_mean = float(np.mean(others))
        col_deltas.append(_oriented(metric, own) - _oriented(metric, others_mean))
        col_rows.append({"target": t, "own": own, "others_mean": others_mean,
                         "n_other_models": len(others)})

    # Row-wise: hold the model fixed, vary the target (confounded by difficulty).
    row_deltas, row_rows = [], []
    for f in fits:
        own = M.get(f, {}).get(f, float("nan"))
        if not np.isfinite(own):
            continue
        others = [v for t, v in M[f].items() if t != f and np.isfinite(v)]
        if not others:
            continue
        others_mean = float(np.mean(others))
        row_deltas.append(_oriented(metric, own) - _oriented(metric, others_mean))
        row_rows.append({"fit": f, "own": own, "others_mean": others_mean,
                         "n_other_targets": len(others)})

    excluded = sorted({int(j) for f in fits
                       for j in runs[f].get("nonfinite_subjects", [])})
    return {
        "controlled": {**_stats(np.array(col_deltas)), "per_target": col_rows},
        "naive": {**_stats(np.array(row_deltas)), "per_fit": row_rows},
        "excluded_subjects": excluded,
    }


def _p_str(p: float) -> str:
    if not np.isfinite(p):
        return "  ---"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def _fmt(d: dict, key: str, width: int = 8) -> str:
    v = d.get(key)
    return f"{v:>{width}.4f}" if isinstance(v, float) and np.isfinite(v) else " " * (width - 3) + "---"


def report(data: dict[str, dict[int, dict]]) -> dict:
    summary: dict[str, dict] = {}
    labels = dict(MODELS)

    for metric in METRICS:
        arrow = "higher better" if metric in HIGHER_BETTER else "lower better"
        print(f"\n{'=' * 96}")
        print(f"  {metric}  ({arrow})")
        print(f"{'=' * 96}")
        print(f"{'model':<14}{'n':>3} | {'intra':>8}{'inter':>9} | "
              f"{'Δ pooled':>9}{'p':>8} | {'Δ ctrl':>9}{'95% CI':>19}{'p':>8}{'>0':>6}")
        print("-" * 96)

        for model_key, _ in MODELS:
            runs = data.get(model_key)
            if not runs:
                continue
            pooled = pooled_contrast(runs, metric)
            mats = matrix_contrasts(runs, metric)
            ctrl = mats.get("controlled", {})
            summary.setdefault(model_key, {})[metric] = {
                "pooled": pooled,
                "controlled": {k: v for k, v in ctrl.items() if k != "per_target"},
                "naive": {k: v for k, v in mats.get("naive", {}).items() if k != "per_fit"},
                "per_target": ctrl.get("per_target", []),
                "excluded_subjects": mats.get("excluded_subjects", []),
            }
            ci = (f"[{ctrl['ci_lo']:+.4f},{ctrl['ci_hi']:+.4f}]"
                  if np.isfinite(ctrl.get("ci_lo", np.nan)) else "---")
            print(f"{labels[model_key]:<14}{pooled.get('n', 0):>3} | "
                  f"{_fmt(pooled, 'intra_mean')}{_fmt(pooled, 'inter_mean', 9)} | "
                  f"{_fmt(pooled, 'mean', 9)}{_p_str(pooled.get('wilcoxon_p', float('nan'))):>8} | "
                  f"{_fmt(ctrl, 'mean', 9)}{ci:>19}"
                  f"{_p_str(ctrl.get('wilcoxon_p', float('nan'))):>8}"
                  f"{ctrl.get('frac_positive', float('nan')):>6.2f}")

    print(f"\n{'=' * 96}")
    print("Δ pooled = intra - inter on the pooled loaders (confounded by subject difficulty).")
    print("Δ ctrl   = own model vs other subjects' models on the SAME target subject.")
    print("           Target held fixed, so difficulty cancels. This is the personalization test.")
    print("All deltas are oriented so that positive = better on the subject it was fitted to.")

    excluded = sorted({j for per_metric in summary.values()
                       for blk in per_metric.values()
                       for j in blk.get("excluded_subjects", [])})
    if excluded:
        print(f"\nExcluded {len(excluded)} subject(s) with non-finite scores "
              f"(diverged rollouts): {excluded}")
    return summary


def compare_arms(a: dict[str, dict[int, dict]], b: dict[str, dict[int, dict]],
                 label_a: str, label_b: str,
                 metrics: list[str] | None = None) -> None:
    """Print Own / Others / delta for two arms next to each other.

    The comparison the paper needs: fine-tuning is expected to raise *both* Own
    and Others, so a rise in Own means nothing on its own. Only the two deltas
    side by side say whether the extra fidelity cost any subject specificity.
    """
    metrics = metrics or ["fc_correlation", "phase_fc_correlation"]
    print(f"\n{'=' * 104}")
    print(f"  ARM COMPARISON   A = {label_a}   B = {label_b}")
    print(f"{'=' * 104}")
    for metric in metrics:
        print(f"\n{metric}  (higher better)" if metric in HIGHER_BETTER
              else f"\n{metric}  (lower better)")
        print(f"{'model':<14}{'arm':<6} | {'Own':>8}{'Others':>9}{'Δ ctrl':>10}"
              f"{'95% CI':>19}{'p':>8}{'>0':>6}")
        print("-" * 104)
        for model_key, label in MODELS:
            for data, arm_tag in ((a, "A"), (b, "B")):
                runs = data.get(model_key)
                if not runs:
                    continue
                ctrl = matrix_contrasts(runs, metric).get("controlled", {})
                rows = ctrl.get("per_target", [])
                own = np.mean([r["own"] for r in rows]) if rows else float("nan")
                oth = np.mean([r["others_mean"] for r in rows]) if rows else float("nan")
                ci = (f"[{ctrl['ci_lo']:+.4f},{ctrl['ci_hi']:+.4f}]"
                      if np.isfinite(ctrl.get("ci_lo", np.nan)) else "---")
                print(f"{label:<14}{arm_tag:<6} | {own:>8.4f}{oth:>9.4f}"
                      f"{_fmt(ctrl, 'mean', 10)}{ci:>19}"
                      f"{_p_str(ctrl.get('wilcoxon_p', float('nan'))):>8}"
                      f"{ctrl.get('frac_positive', float('nan')):>6.2f}")
    print(f"\n{'=' * 104}")
    print("Own rising from A to B is expected and is NOT evidence of personalization:")
    print("a base model pretrained on other subjects already reconstructs an unseen")
    print("subject well. Read the Δ ctrl column — that is the quantity under test.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", default="ts_young")
    ap.add_argument("--res-dir", default="results/personalization")
    ap.add_argument("--tag", default="",
                    help="Arm to report, matching --tag passed to train_personalized.py "
                         "(default: the untagged scratch arm).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Fit seed to report (default: 42, the seed used in the paper).")
    ap.add_argument("--compare-tag", default=None,
                    help="Second arm to print side by side with --tag, e.g. 'ft_lr1e-4'.")
    ap.add_argument("--summary-out", default=None,
                    help="Default: results/personalization_summary[_<tag>].json")
    args = ap.parse_args()

    res_dir = Path(args.res_dir)
    tag = (args.tag or "").strip().strip("_")
    data = load(args.ds, res_dir, tag, seed=args.seed)
    if not data:
        raise SystemExit(
            f"No personalization records found under {res_dir}/ for ds={args.ds} "
            f"and tag={tag!r}. Tags present: "
            f"{sorted({(json.loads(fp.read_text()).get('tag') or '') for fp in res_dir.glob(f'{args.ds}_*_sub*_seed*.json')})}"
        )

    for model_key, runs in sorted(data.items()):
        print(f"loaded {model_key}: {len(runs)} fits (subjects {sorted(runs)})  "
              f"[{_arm_label(runs)}]")

    summary = report(data)

    if args.compare_tag is not None:
        other_tag = args.compare_tag.strip().strip("_")
        other = load(args.ds, res_dir, other_tag, seed=args.seed)
        if not other:
            raise SystemExit(f"No records for --compare-tag {other_tag!r} under {res_dir}/")
        for model_key, runs in sorted(other.items()):
            print(f"loaded {model_key}: {len(runs)} fits (subjects {sorted(runs)})  "
                  f"[{_arm_label(runs)}]")
        compare_arms(data, other,
                     tag or "scratch (untagged)", other_tag)

    out = Path(args.summary_out) if args.summary_out else Path(
        f"results/personalization_summary{'_' + tag if tag else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"dataset_type": args.ds, "tag": tag, "models": summary},
                  fh, indent=2, sort_keys=True)
    print(f"\nSummary written to: {out}")


if __name__ == "__main__":
    main()
