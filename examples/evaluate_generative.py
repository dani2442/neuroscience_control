#!/usr/bin/env python3
"""Generative (posterior-predictive) evaluation of a fitted whole-brain model.

This script implements the evaluation protocol added during revision. It is a
pure *evaluation-time* analysis: it loads an already-fitted checkpoint and never
retrains, so it can be run over the existing seed sweep.

It produces three blocks, each answering a specific reviewer request.

``realizations``
    Because these are stochastic generative models, a single rollout is one
    draw, not "the" model prediction. For every held-out subject we draw
    ``--n-realizations`` independent trajectories under fresh noise and keep the
    per-realization metric values. Combined across seeds this separates the
    three variance sources that were previously conflated in a single ``±``:
    subjects, fitted models (seeds), and stochastic simulations.

``coverage``
    A posterior-predictive check. For a set of scalar summary statistics we
    compare the empirical value against the model-generated distribution for the
    same subject and report (i) the empirical percentile within that
    distribution and (ii) whether it falls inside the central 90% interval.
    A calibrated generative model should cover ~90% of subjects and produce
    percentiles that are roughly uniform, not piled up at 0 or 1.

``personalization``
    A per-subject, properly paired version of the within-vs-between analysis.
    For each subject *i* we simulate from that subject's own initial condition
    and correlate the simulated FC against (a) that same subject's held-out
    second-half empirical FC (``within_i``) and (b) every other subject's
    empirical FC (``between_i``, averaged). ``delta_i = within_i - between_i``
    is then a paired per-subject quantity that supports a signed-rank test and a
    bootstrap CI. We additionally report top-1 subject identification accuracy,
    which is the stronger claim: can the simulated FC pick its own subject out
    of the cohort?

Usage::

    python examples/evaluate_generative.py \
        --checkpoint checkpoints/ts_young_hybrid_hopf_n94_seed42.pt \
        --data-path data/ts_young/ts_young_TR0.72.mat \
        --out results/generative/ts_young_hybrid_hopf_n94_seed42.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import compute_split_indices, load_dataset
from src.metrics import (
    compute_static_fc,
    ks_distance_2samp,
    metastability_value,
    phase_coherence_fc,
)
from src.metrics.dynamics_metrics import phfcd_distribution
from src.models import load_model_from_checkpoint
from src.training import HopfConfig
from src.training.evaluation import rollout_model
from src.utils import resolve_device, seed_all

# Scalar statistics used for the posterior-predictive coverage check. Each maps
# a (n_rois, T) complex trajectory to a scalar. These are deliberately *summary*
# statistics of the trajectory rather than discrepancies against the target, so
# that empirical and simulated values live on the same scale and can be compared
# directly.
def _mean_fc_strength(ts: torch.Tensor) -> float:
    fc = compute_static_fc(ts)[0]
    n = fc.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(fc[iu[0], iu[1]].mean())


def _fc_dispersion(ts: torch.Tensor) -> float:
    fc = compute_static_fc(ts)[0]
    n = fc.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(fc[iu[0], iu[1]].std())


def _metastability(ts: torch.Tensor) -> float:
    return float(metastability_value(ts.unsqueeze(0)))


def _phfcd_median(ts: torch.Tensor) -> float:
    phases = torch.angle(ts).transpose(0, 1)  # (T, N)
    dist = phfcd_distribution(phases)
    if dist.numel() == 0:
        return float("nan")
    return float(dist.median())


COVERAGE_STATS = {
    "fc_strength": _mean_fc_strength,
    "fc_dispersion": _fc_dispersion,
    "metastability": _metastability,
    "phfcd_median": _phfcd_median,
}


def _fc_corr(fc_a: torch.Tensor, fc_b: torch.Tensor) -> float:
    """Pearson correlation between the upper triangles of two FC matrices."""
    n = fc_a.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=fc_a.device)
    a = fc_a[iu[0], iu[1]]
    b = fc_b[iu[0], iu[1]]
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a @ b) / denom)


def _phase_fc(ts: torch.Tensor) -> torch.Tensor:
    return phase_coherence_fc(ts.unsqueeze(0))[0]


def _simulate(model, initial_state: torch.Tensor, n_steps: int, cfg) -> torch.Tensor:
    """One stochastic rollout, returned as (n_rois, T)."""
    with torch.no_grad():
        out = rollout_model(
            model,
            initial_state.unsqueeze(0),
            n_steps,
            cfg.tr,
            sde_type=cfg.sde_type,
            method=cfg.sde_method,
            dt_min=cfg.dt_min,
            denoise_f_lo=cfg.denoise_f_lo,
            denoise_f_hi=cfg.denoise_f_hi,
        )
    return out[0]


def _metrics_against_target(sim: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Discrepancy metrics for one simulated trajectory vs one empirical window."""
    fc_sim = compute_static_fc(sim)[0]
    fc_emp = compute_static_fc(target)[0]
    pfc_sim = _phase_fc(sim)
    pfc_emp = _phase_fc(target)

    ph_sim = torch.angle(sim).transpose(0, 1)
    ph_emp = torch.angle(target).transpose(0, 1)
    d_sim = phfcd_distribution(ph_sim)
    d_emp = phfcd_distribution(ph_emp)
    phfcd_ks = (
        float(ks_distance_2samp(d_sim, d_emp))
        if d_sim.numel() and d_emp.numel()
        else float("nan")
    )

    return {
        "fc_correlation": _fc_corr(fc_sim, fc_emp),
        "phase_fc_correlation": _fc_corr(pfc_sim, pfc_emp),
        "phfcd_ks": phfcd_ks,
        "metastability_diff": abs(_metastability(sim) - _metastability(target)),
    }


def evaluate(
    checkpoint: Path,
    dataset,
    cfg,
    *,
    split_seed: int,
    n_realizations: int,
    window: int,
    device: str,
    eval_seed: int,
    max_eval_subjects: int | None = None,
) -> dict:
    model, model_class, _ = load_model_from_checkpoint(str(checkpoint), device=device)
    model.eval()

    train_idx, _val_idx, test_idx = compute_split_indices(
        dataset, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=split_seed
    )

    if max_eval_subjects is not None:
        test_idx = test_idx[:max_eval_subjects]
        train_idx = train_idx[:max_eval_subjects]

    ts = dataset.timeseries  # (n_subjects, n_rois, T) complex
    n_timepoints = ts.shape[2]
    half = n_timepoints // 2
    window = min(window, half)

    # ---------------------------------------------------------------- block 1+2
    # Repeated realizations on held-out (inter-subject) test data, plus the
    # posterior-predictive coverage check on the same draws.
    seed_all(eval_seed)
    per_subject: list[dict] = []
    for si in test_idx.tolist():
        target = ts[si, :, :window]
        initial_state = target[:, 0]

        realizations: list[dict[str, float]] = []
        stats_sim: dict[str, list[float]] = {k: [] for k in COVERAGE_STATS}
        for _ in range(n_realizations):
            sim = _simulate(model, initial_state, window, cfg)
            realizations.append(_metrics_against_target(sim, target))
            for name, fn in COVERAGE_STATS.items():
                stats_sim[name].append(fn(sim))

        stats_emp = {name: fn(target) for name, fn in COVERAGE_STATS.items()}
        coverage: dict[str, dict[str, float]] = {}
        for name, emp in stats_emp.items():
            draws = np.asarray(stats_sim[name], dtype=float)
            draws = draws[np.isfinite(draws)]
            if draws.size == 0 or not np.isfinite(emp):
                continue
            # Percentile of the empirical value within the simulated draws, and
            # whether it lies inside the central 90% predictive interval.
            pct = float((draws < emp).mean())
            lo, hi = np.percentile(draws, [5.0, 95.0])
            coverage[name] = {
                "empirical": float(emp),
                "sim_mean": float(draws.mean()),
                "sim_std": float(draws.std(ddof=1)) if draws.size > 1 else 0.0,
                "percentile": pct,
                "inside_90": bool(lo <= emp <= hi),
            }

        per_subject.append(
            {
                "subject": int(si),
                "realizations": realizations,
                "coverage": coverage,
            }
        )

    # ------------------------------------------------------------------ block 3
    # Personalization: paired per-subject within-vs-between FC reconstruction and
    # top-1 subject identification. Simulations start from each subject's own
    # second-half initial condition (seen subject, unseen time window) so that
    # `within` and `between` differ only in *which* subject we score against.
    seed_all(eval_seed + 1)
    pers_idx = train_idx.tolist()
    emp_fc_second: dict[int, torch.Tensor] = {}
    for si in pers_idx:
        emp_fc_second[si] = compute_static_fc(ts[si, :, half:half + window])[0]

    personalization: list[dict] = []
    n_ident_hits = 0
    for si in pers_idx:
        initial_state = ts[si, :, half]
        # Average the simulated FC over realizations so the personalization
        # estimate is not dominated by a single noise draw.
        fc_acc = None
        for _ in range(n_realizations):
            sim = _simulate(model, initial_state, window, cfg)
            fc = compute_static_fc(sim)[0]
            fc_acc = fc if fc_acc is None else fc_acc + fc
        fc_sim = fc_acc / n_realizations

        within = _fc_corr(fc_sim, emp_fc_second[si])
        others = [_fc_corr(fc_sim, emp_fc_second[sj]) for sj in pers_idx if sj != si]
        others_arr = np.asarray(others, dtype=float)
        between = float(np.nanmean(others_arr))
        # Top-1 identification: does the subject's own empirical FC win?
        hit = bool(np.all(within > others_arr[np.isfinite(others_arr)]))
        n_ident_hits += int(hit)

        personalization.append(
            {
                "subject": int(si),
                "within": within,
                "between": between,
                "delta": within - between,
                "identified": hit,
            }
        )

    return {
        "checkpoint": str(checkpoint),
        "model_class": model_class,
        "split_seed": split_seed,
        "n_realizations": n_realizations,
        "window": window,
        "n_test_subjects": len(per_subject),
        "n_personalization_subjects": len(personalization),
        "identification_accuracy": n_ident_hits / max(1, len(personalization)),
        "identification_chance": 1.0 / max(1, len(personalization)),
        "per_subject": per_subject,
        "personalization": personalization,
    }


def _infer_seed(checkpoint: Path, override: int | None) -> int:
    if override is not None:
        return override
    m = re.search(r"seed(\d+)", checkpoint.stem)
    if m:
        return int(m.group(1))
    return 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", default="data/ts_young/ts_young_TR0.72.mat")
    ap.add_argument("--dataset-type", default="ts_young")
    ap.add_argument("--n-realizations", type=int, default=20,
                    help="Independent stochastic rollouts per subject.")
    ap.add_argument("--window", type=int, default=100,
                    help="Simulated/evaluated window length in time points.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Split seed; inferred from the checkpoint name when omitted.")
    ap.add_argument("--eval-seed", type=int, default=12345,
                    help="Seed for the evaluation noise draws (kept fixed across models).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-eval-subjects", type=int, default=None,
                    help="Limit subjects per block (smoke testing only).")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = resolve_device(args.device)
    split_seed = _infer_seed(ckpt, args.seed)

    cfg = HopfConfig()
    cfg.dataset_type = args.dataset_type
    cfg.data_path = args.data_path
    dataset = load_dataset(cfg, device)

    result = evaluate(
        ckpt,
        dataset,
        cfg,
        split_seed=split_seed,
        n_realizations=args.n_realizations,
        window=args.window,
        device=device,
        eval_seed=args.eval_seed,
        max_eval_subjects=args.max_eval_subjects,
    )

    deltas = [p["delta"] for p in result["personalization"]]
    print(f"{ckpt.name}: split_seed={split_seed} "
          f"test_subjects={result['n_test_subjects']} "
          f"realizations={result['n_realizations']}")
    print(f"  personalization delta = {np.mean(deltas):+.4f} "
          f"(median {np.median(deltas):+.4f}, n={len(deltas)})")
    print(f"  identification top-1  = {result['identification_accuracy']:.3f} "
          f"(chance {result['identification_chance']:.3f})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
