#!/usr/bin/env python3
"""Personalization experiment — fit one model per *single* subject (N = 1).

Where the main suite fits one group model to N = 94 subjects, this script fits
an independent model to a single subject and asks how far that personalised fit
transfers:

* ``test_intra`` — **within-subject**: the same subject, held-out second half of
  the scan. Unseen timepoints, seen individual.
* ``test_inter`` — **between-subject**: every *other* subject, the *same*
  held-out second half. Unseen individual.

Both test splits cover the identical temporal window, so the intra/inter gap
isolates subject identity rather than a time-of-scan effect.

Temporal layout (T = n_timepoints, half = T // 2)::

    [0 .................... half-n_val)  train   (target subject only)
    [half-n_val ................. half)  val     (target subject, early stopping)
    [half .......................... T)  test    (intra: target; inter: others)

Only the training segment of the target subject shapes the fit: the FC target,
the Hopf ``omega`` initialisation and the reported reference statistics are all
recomputed from it, so no held-out timepoint leaks into training. The one
group-level prior that remains is the structural connectivity (``--sc-source
group``, default): the connectome prior of the cohort-level fit with split seed
``--prior-split-seed`` (42), i.e. the mean FC over the first halves of that
split's training subjects. The target must not be one of those subjects, so no
data of the personalized subject enters the prior; the script refuses such a
target and lists the eligible ones. ``--sc-source subject`` makes the fit fully
individual.

**Why per-subject scores are also recorded.** A raw intra-minus-inter gap
confounds personalization with subject difficulty: if the target subject simply
has more model-friendly dynamics, the gap is positive whatever the fit learned.
Alongside the two pooled splits this script therefore scores the fitted model on
*every* subject's held-out half separately (``per_subject``), with an identical
window budget for each. Pooling those records across the personalized fits gives
a model x target-subject matrix whose diagonal (own model) can be compared to
its own column (other subjects' models on the same target), which holds the
target fixed and cancels difficulty. See ``examples/aggregate_personalization.py``.

Note that the initial-condition carryover documented in
``examples/aggregate_generative.py`` does *not* apply in the same way here: every
rollout is seeded from the first timepoint of the window it is scored against,
so both the intra and the inter arm get their own subject's initial state and
the advantage cancels between them.

**Two arms.** By default the fit starts from scratch (per-subject ``omega``
initialisation, no pretrained weights). With ``--init-checkpoint`` it instead
fine-tunes a base model pretrained on a pool of *other* subjects, which is the
clinically realistic protocol and removes the data-quantity handicap of the
scratch arm: a scratch fit sees one subject's ~450 training timepoints, whereas
the group models see tens of thousands.

The base model must be pretrained with ``--exclude-subjects`` covering every
subject that will later be personalised. Otherwise the base already encodes each
target, every fine-tuned model inherits that information, and the own-vs-other
contrast is suppressed before the experiment starts.

Expect fine-tuning to raise the *absolute* scores on the target and to shrink
``delta``: all fine-tuned models share one base, so the other subjects' models
also reconstruct the target well. Absolute fidelity and subject specificity are
distinct quantities here — the Coupled Hopf arm reaches the highest own-subject
FC correlation of any architecture with no personalization effect at all. The
point of sweeping ``--lr``, ``--n-epochs`` and ``--freeze`` is to trace that
fidelity-versus-specificity trade-off rather than to maximise either alone.

Usage::

    # scratch arm (as before)
    python examples/train_personalized.py --model hopf --subject 0 \
        --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat \
        --no-wandb --skip-figures

    # fine-tuning arm
    python examples/train_personalized.py --model hybrid_hopf --subject 0 \
        --init-checkpoint checkpoints/ts_young_hybrid_hopf_basepool.pt \
        --lr 1e-4 --tag ft_lr1e-4 \
        --dataset-type ts_young --data-path data/ts_young/ts_young_TR0.72.mat \
        --no-wandb --skip-figures

Writes ``results/personalization/<ds>_<model>_sub<idx>_seed<seed>[_<tag>].json``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.cli_args import add_dataset_args
from examples.train_models import (
    _CONFIG_CLS,
    _MODEL_TITLES,
    _apply_common_cfg_overrides,
    _apply_dataset_cfg_overrides,
    _apply_dataset_time_defaults,
)
from src.dataset import (
    compute_omega_from_timeseries,
    load_dataset,
    resolve_split,
    training_segment_stop,
    training_view,
)
from src.dataset.preprocessing import RandomWindowDataset
from src.metrics import compute_static_fc
from src.models import build_model
from src.training import Trainer
from src.utils import (
    EVAL_METRIC_KEYS,
    as_numeric_metrics,
    ensure_proxy_env,
    evaluate_model_loader_metrics,
    format_metrics_mean_std,
    print_section,
    resolve_device,
    save_checkpoint,
    seed_all,
)


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def _build_splits(dataset, subject: int, window_size: int, val_frac: float):
    """Return the temporal slices and subject index sets for one target subject.

    Raises ``SystemExit`` when the requested layout cannot fit *window_size*.
    """
    n_subjects = int(dataset.n_subjects)
    if not 0 <= subject < n_subjects:
        raise SystemExit(f"--subject must be in [0, {n_subjects - 1}], got {subject}")
    if n_subjects < 2:
        raise SystemExit(
            f"Between-subject testing needs >= 2 loaded subjects, got {n_subjects}. "
            "Do not restrict --max-subjects to 1: N=1 refers to the *training* set, "
            "the remaining subjects form the between-subject test set."
        )

    T = int(dataset.n_timepoints)
    half = T // 2
    n_val = max(window_size, int(round(val_frac * half)))
    train_end = half - n_val
    if train_end < window_size:
        raise SystemExit(
            f"Training segment would be {train_end} timepoints, shorter than "
            f"window_size={window_size}. Lower --val-frac or --window-size."
        )

    others = [i for i in range(n_subjects) if i != subject]
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, half),
        "test": slice(half, T),
        "others": others,
        "n_timepoints": T,
    }


def _group_prior(dataset, cfg, subject: int, split_seed: int):
    """Connectome prior shared by every personalized fit.

    The training view of the cohort-level split with *split_seed*: mean FC over
    the training segments (first halves) of that split's training subjects,
    i.e. the same prior as the cohort-level fit with that seed. Raises if
    *subject* is one of those training subjects.
    """
    train_idx, _, _ = resolve_split(dataset, cfg.train_ratio, cfg.val_ratio, split_seed)
    if subject in set(int(i) for i in train_idx):
        eligible = sorted(set(range(int(dataset.n_subjects))) - set(int(i) for i in train_idx))
        raise SystemExit(
            f"Subject {subject} is one of the {len(train_idx)} training subjects whose data "
            f"build the group connectome prior (split seed {split_seed}), so its own data "
            f"would enter the prior. Personalize a subject outside that pool: {eligible}"
        )
    t_stop = training_segment_stop(dataset.n_timepoints, cfg.use_full_timeseries)
    return training_view(dataset, train_idx, t_stop=t_stop, held_out_idx=[subject])


def _loader(ts, fc, ctrl, window_size, n_windows, batch_size, device, shuffle):
    ds = RandomWindowDataset(ts, fc, window_size, n_windows, device, control=ctrl)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def _per_subject_metrics(model, dataset, test_slice, cfg, window_size,
                         n_windows, device) -> dict[str, dict[str, float]]:
    """Score *model* on every subject's held-out segment, one subject at a time.

    Every subject — the personalised target included — gets the same window
    budget, so the target's entry is directly comparable to the others'.
    """
    ctrl = getattr(dataset, "control", None)
    out: dict[str, dict[str, float]] = {}
    for j in range(int(dataset.n_subjects)):
        ts_j = dataset.timeseries[j:j + 1, :, test_slice]
        loader = _loader(
            ts_j, compute_static_fc(ts_j),
            ctrl[j:j + 1] if ctrl is not None else None,
            window_size, n_windows, cfg.batch_size, device, False,
        )
        metrics = evaluate_model_loader_metrics(model, loader, cfg, n_steps=window_size)
        out[str(j)] = {k: float(v) for k, v in as_numeric_metrics(metrics).items()}
    return out


def _batch_coverage(per_batch: dict[str, list[float]], expected: int) -> dict[str, object]:
    """How many batches actually reached each metric, out of *expected*.

    ``MetricAccumulator.update`` silently drops NaN batches, so a metric mean can
    rest on fewer batches than the loader produced with nothing in the output to
    say so. Recording the contributing count makes a diverged rollout visible
    instead of quietly shrinking the sample.
    """
    counts = {k: len(v) for k, v in per_batch.items() if k in EVAL_METRIC_KEYS}
    short = {k: c for k, c in counts.items() if c < expected}
    return {"expected_batches": int(expected), "counts": counts, "short": short}


def _nonfinite_subjects(per_subject: dict[str, dict[str, float]]) -> list[int]:
    """Subject indices whose scores contain a NaN/inf (diverged rollout)."""
    return sorted(
        int(j) for j, m in per_subject.items()
        if any(not math.isfinite(v) for v in m.values())
    )


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------

class _SkipTraining(Exception):
    """Control-flow marker for --no-train, so the Trainer teardown still runs."""


def _apply_freeze(model, patterns: list[str] | None) -> dict[str, list[str]]:
    """Freeze every parameter whose name contains one of *patterns*.

    Substring matching rather than fixed group names, so the same flag works
    across architectures: ``--freeze key_net output_net`` adapts only the
    mechanistic Hopf parameters of the hybrid model, ``--freeze omega a g
    kappa`` does the opposite.
    """
    named = list(model.named_parameters())
    if not patterns:
        return {"frozen": [], "trainable": [n for n, _ in named]}

    frozen = []
    for name, param in named:
        if any(pat in name for pat in patterns):
            param.requires_grad_(False)
            frozen.append(name)
    trainable = [n for n, q in model.named_parameters() if q.requires_grad]
    if not frozen:
        raise SystemExit(
            f"--freeze {patterns} matched no parameter. Available names: "
            f"{[n for n, _ in named]}"
        )
    if not trainable:
        raise SystemExit("--freeze matched every parameter; nothing left to adapt.")
    print(f"  froze {len(frozen)}/{len(named)} parameter tensors: {frozen}")
    print(f"  adapting: {trainable}")
    return {"frozen": frozen, "trainable": trainable}


def _reinit_omega(model, ts_train, dataset, cfg) -> bool:
    """Reset the loaded model's intrinsic frequencies to the target subject's.

    Copies in place: ``omega`` is handed to the SDE function by reference at
    construction, so rebinding the attribute would leave the integrator reading
    the old tensor.
    """
    if not hasattr(model, "omega"):
        print("  --reinit-omega ignored: this architecture has no omega parameter.")
        return False
    omega = compute_omega_from_timeseries(
        ts_train, dt=dataset.dt, f_lo=cfg.f_lo, f_hi=cfg.f_hi, method="peak",
    ).to(model.omega.device, model.omega.dtype)
    with torch.no_grad():
        model.omega.copy_(omega)
    print(f"  omega re-initialised from the target subject's training segment: "
          f"[{omega.min() / 6.2832:.4f}, {omega.max() / 6.2832:.4f}] Hz")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, object]:
    title = _MODEL_TITLES.get(args.model, args.model)
    print_section(f"PERSONALIZATION (N=1) — {title.upper()} — SUBJECT {args.subject}")

    cfg = _CONFIG_CLS[args.model]()
    _apply_common_cfg_overrides(cfg, args)
    _apply_dataset_cfg_overrides(cfg, args)
    _apply_dataset_time_defaults(cfg, args)
    if args.window_size is not None:
        cfg.window_size = args.window_size
    if args.lr is not None:
        cfg.lr = args.lr
    ds_tag = cfg.dataset_type
    tag = (args.tag or "").strip().strip("_")
    tag_suffix = f"_{tag}" if tag else ""
    if args.init_checkpoint is not None and args.sc_source == "subject":
        raise SystemExit(
            "--init-checkpoint with --sc-source subject is contradictory: loading the "
            "checkpoint overwrites the structural_connectivity buffer with the base "
            "model's group prior, so the subject-specific SC would be silently "
            "discarded. Use --sc-source group for fine-tuned runs."
        )
    cfg.experiment_name = (
        f"personalized_{args.model}_{ds_tag}_sub{args.subject}_seed{cfg.seed}{tag_suffix}"
    )
    cfg.run_name = f"{cfg.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_proxy_env()
    print(f"Config: {dataclasses.asdict(cfg)}")
    device = resolve_device(cfg.device)
    seed_all(cfg.seed)

    print_section("STEP 1: Loading Data and Building N=1 Splits")
    dataset = load_dataset(cfg, device)
    window_size = min(cfg.window_size, dataset.n_timepoints // 2)
    splits = _build_splits(dataset, args.subject, window_size, args.val_frac)

    s = args.subject
    ctrl = getattr(dataset, "control", None)
    target_ctrl = ctrl[s:s + 1] if ctrl is not None else None
    other_ctrl = ctrl[splits["others"]] if ctrl is not None else None

    ts_train = dataset.timeseries[s:s + 1, :, splits["train"]]
    ts_val = dataset.timeseries[s:s + 1, :, splits["val"]]
    ts_intra = dataset.timeseries[s:s + 1, :, splits["test"]]
    ts_inter = dataset.timeseries[splits["others"], :, splits["test"]]

    # FC targets are recomputed per segment so the training target never sees
    # held-out timepoints. (Test-loader FC is unused by the metric path, but is
    # kept segment-consistent for clarity.)
    fc_train = compute_static_fc(ts_train)
    fc_val = compute_static_fc(ts_val)
    fc_intra = compute_static_fc(ts_intra)
    fc_inter = compute_static_fc(ts_inter)

    n_test_windows = args.n_test_windows
    train_loader = _loader(ts_train, fc_train, target_ctrl, window_size,
                           cfg.n_windows_per_epoch, cfg.batch_size, device, True)
    val_loader = _loader(ts_val, fc_val, target_ctrl, window_size,
                         max(cfg.batch_size, cfg.n_windows_per_epoch // 4),
                         cfg.batch_size, device, False)
    intra_loader = _loader(ts_intra, fc_intra, target_ctrl, window_size,
                           n_test_windows, cfg.batch_size, device, False)
    inter_loader = _loader(ts_inter, fc_inter, other_ctrl, window_size,
                           n_test_windows, cfg.batch_size, device, False)

    print(f"  subject={s} of {dataset.n_subjects}  window_size={window_size}")
    print(f"  train  : subject {s}, t[{splits['train'].start}:{splits['train'].stop}]"
          f"  ({ts_train.shape[2]} tp, {len(train_loader)} batches)")
    print(f"  val    : subject {s}, t[{splits['val'].start}:{splits['val'].stop}]"
          f"  ({ts_val.shape[2]} tp, {len(val_loader)} batches)")
    print(f"  intra  : subject {s}, t[{splits['test'].start}:{splits['test'].stop}]"
          f"  ({ts_intra.shape[2]} tp, {len(intra_loader)} batches)  [within-subject]")
    print(f"  inter  : {len(splits['others'])} other subjects, same t-range"
          f"  ({len(inter_loader)} batches)  [between-subject]")

    print_section("STEP 2: Training on the Single Target Subject")
    group_prior = None
    if args.init_checkpoint is not None:
        # Placeholder only: loading the base checkpoint below overwrites it with
        # the base model's own connectome prior.
        sc = fc_train[0]
    elif args.sc_source == "group":
        prior_view = _group_prior(dataset, cfg, s, args.prior_split_seed)
        sc = prior_view.fc_mean
        group_prior = {"split_seed": int(args.prior_split_seed), **prior_view.group_prior_info}
        print(f"  structural connectivity: group connectome prior of the split-seed "
              f"{args.prior_split_seed} cohort fit ({prior_view.n_subjects} training subjects, "
              f"timepoints [0, {prior_view.n_timepoints}))")
    else:
        sc = fc_train[0]
        print("  structural connectivity: target-subject training-segment FC (fully individual)")

    # Restrict the view handed to build_model so omega is initialised from the
    # target subject's training segment alone.
    train_view = copy.copy(dataset)
    train_view.timeseries = ts_train
    model = build_model(args.model, train_view, cfg, device,
                        structural_connectivity=sc,
                        n_control_dims=dataset.n_control_dims)

    # Fine-tuning arm: start from a base model pretrained on a disjoint pool of
    # subjects instead of from the per-subject initialisation above. ``load``
    # replaces the whole state dict, so the target-subject omega init and the SC
    # buffer are both inherited from the base unless --reinit-omega asks for the
    # subject's own frequencies back.
    base_meta: dict = {}
    if args.init_checkpoint is not None:
        ckpt = Path(args.init_checkpoint)
        if not ckpt.is_file():
            raise SystemExit(f"--init-checkpoint not found: {ckpt}")
        base_meta = model.load(str(ckpt)) or {}
        print(f"  initialised from base checkpoint: {ckpt}")
        if base_meta:
            print(f"  base metadata: {base_meta}")
    else:
        print("  initialised from scratch (per-subject omega, no pretrained weights)")

    omega_reinit = False
    if args.reinit_omega:
        if args.init_checkpoint is None:
            print("  --reinit-omega is a no-op without --init-checkpoint "
                  "(omega is already the target subject's).")
        else:
            omega_reinit = _reinit_omega(model, ts_train, dataset, cfg)

    freeze_info = _apply_freeze(model, args.freeze)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  adapting {n_trainable}/{n_total} parameters  (lr={cfg.lr}, "
          f"n_epochs={cfg.n_epochs})")


    # Only build an explicit optimizer when freezing, so the un-frozen path stays
    # byte-identical to the Trainer default it used before.
    optimizer = None
    if args.freeze:
        optimizer = torch.optim.Adam(
            [q for q in model.parameters() if q.requires_grad], lr=cfg.lr,
        )

    trainer = None
    if args.no_train:
        print("  --no-train: skipping adaptation, scoring the loaded model as-is.")
    try:
        if args.no_train:
            raise _SkipTraining
        trainer = Trainer(
            model=model, optimizer=optimizer, lr=cfg.lr, device=device,
            checkpoint_dir=cfg.checkpoint_dir, experiment_name=cfg.experiment_name,
            cfg=cfg, use_wandb=cfg.use_wandb,
        )
        trainer.train(
            train_loader=train_loader, val_loader=val_loader,
            n_epochs=cfg.n_epochs, n_steps=window_size, dt=cfg.tr,
            early_stopping_patience=cfg.early_stopping_patience, verbose=True,
        )
    except _SkipTraining:
        pass
    finally:
        if trainer is not None:
            trainer.finish()

    print_section("STEP 3: Within- vs Between-Subject Evaluation")
    train_metrics = evaluate_model_loader_metrics(model, train_loader, cfg, n_steps=window_size)
    val_metrics = evaluate_model_loader_metrics(model, val_loader, cfg, n_steps=window_size)
    intra_metrics, intra_batches = evaluate_model_loader_metrics(
        model, intra_loader, cfg, n_steps=window_size,
        return_std=True, return_values=True)
    inter_metrics, inter_batches = evaluate_model_loader_metrics(
        model, inter_loader, cfg, n_steps=window_size,
        return_std=True, return_values=True)
    coverage = {
        "test_intra": _batch_coverage(intra_batches, len(intra_loader)),
        "test_inter": _batch_coverage(inter_batches, len(inter_loader)),
    }
    for split, cov in coverage.items():
        if cov["short"]:
            print(f"  WARNING: {split} lost batches to non-finite rollouts "
                  f"(of {cov['expected_batches']}): {cov['short']}")

    print(f"  scoring all {dataset.n_subjects} subjects individually "
          f"({args.n_subject_windows} windows each) ...")
    per_subject = _per_subject_metrics(
        model, dataset, splits["test"], cfg, window_size,
        args.n_subject_windows, device)

    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = save_checkpoint(
            model,
            checkpoint_name=f"{ds_tag}_{args.model}_personalized_sub{s}_seed{cfg.seed}{tag_suffix}.pt",
            artifact_name=f"{ds_tag}_{args.model}_personalized_sub{s}_seed{cfg.seed}{tag_suffix}",
            checkpoint_dir=cfg.checkpoint_dir,
            use_wandb=cfg.use_wandb,
        )

    record = {
        "experiment": "personalization_n1",
        "dataset_type": ds_tag,
        "model": args.model,
        "subject": s,
        "seed": int(cfg.seed),
        "n_train_subjects": 1,
        "n_inter_subjects": len(splits["others"]),
        "sc_source": args.sc_source,
        # Training subjects and segment behind the group prior (None if unused).
        # Older JSONs lack this field: their prior was the all-94-subject mean FC.
        "group_prior": group_prior,
        # FCD/phFCD training loss; older JSONs lack this field (entrywise MSE).
        "dynamics_loss": "sorted_entry_mse",
        "arm": ("base" if args.no_train
                else "finetune" if args.init_checkpoint is not None
                else "scratch"),
        "trained": not args.no_train,
        "tag": tag,
        "finetune": {
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "base_metadata": base_meta or None,
            "lr": float(cfg.lr),
            "freeze_patterns": list(args.freeze) if args.freeze else [],
            "frozen_params": freeze_info["frozen"],
            "n_trainable_params": int(n_trainable),
            "n_total_params": int(n_total),
            "reinit_omega": bool(omega_reinit),
        },
        "window_size": int(window_size),
        "n_epochs": int(cfg.n_epochs),
        "n_test_windows": int(n_test_windows),
        "n_subject_windows": int(args.n_subject_windows),
        "splits": {
            "train": [splits["train"].start, splits["train"].stop],
            "val": [splits["val"].start, splits["val"].stop],
            "test": [splits["test"].start, splits["test"].stop],
            "n_timepoints": splits["n_timepoints"],
        },
        "metrics": {
            "train": {k: float(v) for k, v in as_numeric_metrics(train_metrics).items()},
            "val": {k: float(v) for k, v in as_numeric_metrics(val_metrics).items()},
            "test_intra": {k: float(v) for k, v in as_numeric_metrics(intra_metrics).items()},
            "test_inter": {k: float(v) for k, v in as_numeric_metrics(inter_metrics).items()},
        },
        "per_subject": per_subject,
        "nonfinite_subjects": _nonfinite_subjects(per_subject),
        "batch_coverage": coverage,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds_tag}_{args.model}_sub{s}_seed{cfg.seed}{tag_suffix}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)

    bad = _nonfinite_subjects(per_subject)
    if bad:
        print(f"  WARNING: non-finite metrics for {len(bad)} of {dataset.n_subjects} subjects "
              f"{bad} — diverged rollout or degenerate window. Recorded as NaN and excluded "
              "from the means below; the aggregator excludes them too and reports the count.")

    own = per_subject[str(s)].get("fc_correlation", float("nan"))
    others_fc = [v["fc_correlation"] for k, v in per_subject.items()
                 if k != str(s) and math.isfinite(v.get("fc_correlation", float("nan")))]
    others_mean = sum(others_fc) / len(others_fc) if others_fc else float("nan")

    arm = "FINE-TUNED" if args.init_checkpoint is not None else "FROM SCRATCH"
    print(f"\n{'=' * 60}")
    print(f"  PERSONALIZED {title.upper()} — SUBJECT {s} — {arm} — COMPLETED")
    print(f"{'=' * 60}")
    print(f"Train      : {format_metrics_mean_std(train_metrics)}")
    print(f"Val        : {format_metrics_mean_std(val_metrics)}")
    print(f"Test intra : {format_metrics_mean_std(intra_metrics)}   [within-subject]")
    print(f"Test inter : {format_metrics_mean_std(inter_metrics)}   [between-subject]")
    print(f"Per-subject fc_correlation: own={own:.4f}  others_mean={others_mean:.4f}  "
          f"delta={own - others_mean:+.4f}")
    print(f"Results saved to: {out_path}")
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit one model per single subject (N=1) and test within/between subject.",
    )
    p.add_argument("--model", type=str, default="nsde", choices=list(_CONFIG_CLS))
    p.add_argument("--subject", type=int, required=True,
                   help="Index of the target subject to personalise on.")
    p.add_argument("--val-frac", type=float, default=0.25,
                   help="Fraction of the first half reserved for validation (default 0.25).")
    p.add_argument("--n-test-windows", type=int, default=1024,
                   help="Windows drawn for each pooled intra/inter test split (default 1024).")
    p.add_argument("--n-subject-windows", type=int, default=128,
                   help="Windows drawn per subject for the per-subject matrix (default 128).")
    p.add_argument("--sc-source", type=str, default="group", choices=["group", "subject"],
                   help="Structural connectivity prior: the group prior of the cohort-level "
                        "fit (default, see --prior-split-seed) or the target subject's "
                        "training-segment FC.")
    p.add_argument("--prior-split-seed", type=int, default=42,
                   help="Split seed of the cohort-level fit whose connectome prior is used "
                        "with --sc-source group (default 42). The target must not be one of "
                        "that split's training subjects.")
    p.add_argument("--init-checkpoint", type=str, default=None,
                   help="Pretrained base checkpoint to fine-tune from. The base MUST have "
                        "been trained with --exclude-subjects covering every subject that "
                        "will be personalised, otherwise it already encodes the targets and "
                        "the own-vs-other contrast is suppressed by construction.")
    p.add_argument("--lr", type=float, default=None,
                   help="Override cfg.lr. Lower values (e.g. 1e-4) keep a fine-tuned model "
                        "closer to its base; this is the main adaptation-strength knob.")
    p.add_argument("--freeze", nargs="+", default=None, metavar="SUBSTR",
                   help="Freeze parameters whose name contains any of these substrings, e.g. "
                        "'--freeze key_net output_net' to adapt only the mechanistic Hopf "
                        "parameters of the hybrid model.")
    p.add_argument("--no-train", action="store_true",
                   help="Skip adaptation and score the loaded checkpoint as-is. With "
                        "--init-checkpoint this measures the unadapted base on every "
                        "subject, the reference row the fine-tuned Own column needs: one "
                        "run scores all subjects, so a single job covers every target.")
    p.add_argument("--reinit-omega", action="store_true",
                   help="After loading the base checkpoint, reset omega to the target "
                        "subject's training-segment frequencies (warm-start the physics).")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for the result/checkpoint filenames and the aggregator's "
                        "arm filter (e.g. 'ft_lr1e-4'). Keeps arms side by side.")
    p.add_argument("--window-size", type=int, default=None, help="Override cfg.window_size.")
    p.add_argument("--out-dir", type=str, default="results/personalization",
                   help="Directory for the per-run JSON record.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Skip writing the fitted model checkpoint.")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--device", type=str, default=None, help="Device (auto, cuda, cpu)")
    p.add_argument("--skip-figures", action="store_true",
                   help="Accepted for parity with train_models.py; no figures are made")
    p.add_argument("--n-epochs", type=int, default=None, help="Override number of training epochs")
    p.add_argument("--group-size", type=int, default=None,
                   help="Samples per group for metric evaluation (0 = full batch)")
    p.add_argument("--seed", type=int, default=None, help="Override cfg.seed (default 42)")
    p.add_argument("--tr", type=float, default=None, help="Override repetition time in seconds.")
    add_dataset_args(p)
    return p


def main(argv: list[str] | None = None) -> dict[str, object]:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
