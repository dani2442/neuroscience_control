#!/usr/bin/env python3
"""Compute the FC adjusted Rand index (ARI) per fitted checkpoint.

The modularity claim in the Results is quantified by the ARI between the
community partition of the empirical FC and that of each model's simulated FC.
Previously this number existed only inside an analysis notebook, so it could not
be reported per seed alongside the other metrics. This script recomputes it from
checkpoints as part of the reproducible pipeline, writing one value per
(model, seed) so that ARI can carry the same seed-level error bars and paired
tests as every other column.

Protocol (matched to the main text): spectral clustering with ``K`` clusters on
the affinity ``A = (1 + FC) / 2`` using a precomputed affinity and k-means label
assignment. The empirical partition is derived from the group-averaged FC of the
held-out test subjects; the simulated partition from the FC of trajectories
generated from those subjects' initial conditions, averaged over
``--n-realizations`` independent draws so the partition is not driven by a single
noise draw.

Usage::

    python examples/compute_ari.py \
        --checkpoints "checkpoints/ts_young_*_n94_seed*.pt" \
        --out results/ari_per_seed.json
"""

from __future__ import annotations

import argparse
import glob
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
from src.metrics import compute_static_fc, fisher_batch_average
from src.models import load_model_from_checkpoint
from src.training import HopfConfig
from src.training.evaluation import rollout_model
from src.utils import resolve_device, seed_all


def _partition(fc: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    aff = (1.0 + fc) / 2.0
    np.fill_diagonal(aff, 1.0)
    aff = np.clip(aff, 0.0, None)
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=0
    )
    return sc.fit_predict(aff)


def _model_key(stem: str) -> str:
    m = re.match(r"ts_young_(.+?)_n94_seed\d+$", stem)
    return m.group(1) if m else stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True, help="Glob (quote it).")
    ap.add_argument("--data-path", default="data/ts_young/ts_young_TR0.72.mat")
    ap.add_argument("--dataset-type", default="ts_young")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n-realizations", type=int, default=10)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sklearn.metrics import adjusted_rand_score

    paths = sorted(Path(p) for p in glob.glob(args.checkpoints))
    if not paths:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints}")

    device = resolve_device(args.device)
    cfg = HopfConfig()
    cfg.dataset_type = args.dataset_type
    cfg.data_path = args.data_path
    dataset = load_dataset(cfg, device)
    ts = dataset.timeseries

    records: list[dict] = []
    emp_partitions: dict[tuple[int, int], np.ndarray] = {}

    for ckpt in paths:
        split_seed = int(re.search(r"seed(\d+)", ckpt.stem).group(1))
        try:
            model, _cls, _ = load_model_from_checkpoint(str(ckpt), device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {ckpt.name}: {exc}")
            continue
        model.eval()

        _tr, _va, test_idx = compute_split_indices(
            dataset, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=split_seed
        )
        window = min(args.window, ts.shape[2])
        target = ts[test_idx, :, :window]

        # Empirical partition for this split (cached per split seed and K).
        emp_fc = fisher_batch_average(compute_static_fc(target)).cpu().numpy()

        seed_all(args.eval_seed)
        initial_state = target[:, :, 0]
        fc_acc = None
        with torch.no_grad():
            for _ in range(args.n_realizations):
                sim = rollout_model(
                    model, initial_state, window, cfg.tr,
                    sde_type=cfg.sde_type, method=cfg.sde_method, dt_min=cfg.dt_min,
                    denoise_f_lo=cfg.denoise_f_lo, denoise_f_hi=cfg.denoise_f_hi,
                )
                fc = fisher_batch_average(compute_static_fc(sim))
                fc_acc = fc if fc_acc is None else fc_acc + fc
        sim_fc = (fc_acc / args.n_realizations).cpu().numpy()

        rec = {
            "checkpoint": str(ckpt),
            "model": _model_key(ckpt.stem),
            "seed": split_seed,
        }
        for k in args.k:
            key = (split_seed, k)
            if key not in emp_partitions:
                emp_partitions[key] = _partition(emp_fc, k)
            rec[f"fc_ari_k{k}"] = float(
                adjusted_rand_score(emp_partitions[key], _partition(sim_fc, k))
            )
        records.append(rec)
        detail = "  ".join(f"ARI(K={k})={rec[f'fc_ari_k{k}']:.3f}" for k in args.k)
        print(f"{ckpt.name}: {detail}")

    # Summary per model.
    print("\nPer-model summary (mean +- sd across seeds):")
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    for mk, rs in sorted(by_model.items()):
        parts = []
        for k in args.k:
            v = np.array([r[f"fc_ari_k{k}"] for r in rs], dtype=float)
            sd = v.std(ddof=1) if v.size > 1 else 0.0
            parts.append(f"K={k}: {v.mean():.3f} +- {sd:.3f}")
        print(f"  {mk:15s} n={len(rs):2d}  " + "   ".join(parts))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
