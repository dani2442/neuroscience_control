#!/usr/bin/env python3
"""In-silico control experiment for the LSD dataset.

This script is a command-line conversion of `examples/in_silico_control.ipynb`.
It optimises constant control inputs that steer trained models toward the
empirical Placebo, LSD+Ketanserin, and LSD functional-connectivity regimes,
evaluates fidelity, analyses endpoint-control similarity/transfer, and saves
figures plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import torch
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_lsd_data
from src.dataset.data_loader import compute_fc_from_timeseries
from src.models import load_model_from_checkpoint

plt.style.use(["science", "no-latex"])

TARGET_CONDITIONS = ["Placebo", "LSD+Ket", "LSD"]
CONTRAST_CONDITIONS = ("Placebo", "LSD")
MODEL_CHECKPOINTS = {
    "Hopf": "lsd_hopf.pt",
    "GNN Hopf": "lsd_gnn_hopf.pt",
    "Hybrid Hopf": "lsd_hybrid_hopf.pt",
    "Hybrid+Neural": "lsd_hybrid_neural.pt",
    "Neural SDE": "lsd_nsde.pt",
}
COND_SEED_OFFSETS = {"Placebo": 11, "LSD+Ket": 23, "LSD": 29}
COND_DISPLAY = {"Placebo": "Placebo", "LSD+Ket": "LSD+Ket.", "LSD": "LSD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-type", default="lsd")
    parser.add_argument("--lsd-data-dir", default="data/lsd")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-path", default="results/lsd_in_silico_control.json")
    parser.add_argument("--out-dir-pdf", default="paper/images/lsd")
    parser.add_argument("--out-dir-png", default="paper/images_png/lsd")
    parser.add_argument("--out-dir-svg", default="paper/images_svg/lsd")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dt", type=float, default=0.72)
    parser.add_argument("--dt-min", type=float, default=0.05)
    parser.add_argument("--n-sim-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-opt-steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--reg-l2", type=float, default=1e-3)
    parser.add_argument("--fc-loss-weight", type=float, default=1.0)
    parser.add_argument("--n-eval-batches", type=int, default=8)
    parser.add_argument("--n-eval-steps", type=int, default=240)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Skip cross-architecture transfer matrices to shorten the run.",
    )
    return parser.parse_args()


def repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but no CUDA device is available.")
    return device_arg


def _get_n_control_dims(model: torch.nn.Module) -> int:
    n = getattr(model, "n_control_dims", None)
    if n is None:
        n = getattr(getattr(model, "sde_func", None), "n_control_dims", None)
    return int(n or 0)


def fcd_windowed(ts: torch.Tensor, window: int = 30, step: int = 3) -> torch.Tensor:
    """Compute sliding-window FCD upper-triangle entries for a batch."""
    ts_real = ts.real if torch.is_complex(ts) else ts
    _, n_rois, n_timepoints = ts_real.shape
    idx = torch.triu_indices(n_rois, n_rois, offset=1)
    windows = []
    for start in range(0, n_timepoints - window + 1, step):
        window_ts = ts_real[:, :, start : start + window]
        window_ts = window_ts - window_ts.mean(dim=2, keepdim=True)
        window_ts = window_ts / (window_ts.std(dim=2, keepdim=True) + 1e-8)
        fc_window = torch.bmm(window_ts, window_ts.transpose(1, 2)) / (window - 1)
        windows.append(fc_window[:, idx[0], idx[1]])

    features = torch.stack(windows, dim=0).permute(1, 0, 2)
    features = features - features.mean(dim=2, keepdim=True)
    features = features / (features.norm(dim=2, keepdim=True) + 1e-8)
    fcd = torch.bmm(features, features.transpose(1, 2))
    n_windows = fcd.shape[1]
    tri = torch.triu_indices(n_windows, n_windows, offset=1)
    return fcd[:, tri[0], tri[1]]


def _simulate(
    model: torch.nn.Module,
    u: torch.Tensor,
    z0: torch.Tensor,
    n_steps: int,
    dt: float,
    dt_min: float,
) -> torch.Tensor:
    """Run a rollout with the batch of controls u shaped (B, n_control_dims)."""
    return model.forward(z0, n_steps=n_steps, dt=dt, dt_min=dt_min, control=u)


def _fc_loss(sim_ts: torch.Tensor, target_fc: torch.Tensor) -> torch.Tensor:
    fc = torch.stack([compute_fc_from_timeseries(s.unsqueeze(0))[0] for s in sim_ts])
    diff = fc - target_fc.unsqueeze(0).to(fc.device)
    mask = 1 - torch.eye(fc.shape[-1], device=fc.device)
    return ((diff * mask) ** 2).sum(dim=(1, 2)).mean()


def optimise_control(
    model: torch.nn.Module,
    target_fc: torch.Tensor,
    n_control_dims: int,
    n_rois: int,
    device: str,
    dt: float,
    dt_min: float,
    fc_loss_weight: float,
    n_opt_steps: int,
    n_sim_steps: int,
    batch_size: int,
    lr: float,
    reg_l2: float,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(seed)
    u = torch.nn.Parameter(torch.zeros(1, n_control_dims, device=device))
    opt = torch.optim.Adam([u], lr=lr)
    target = target_fc.to(device)
    history: list[float] = []

    for _ in range(n_opt_steps):
        z0 = torch.randn(batch_size, n_rois, dtype=torch.complex64, device=device) * 0.1
        u_batch = u.expand(batch_size, -1).contiguous()
        sim = _simulate(model, u_batch, z0, n_sim_steps, dt=dt, dt_min=dt_min)
        fc_err = _fc_loss(sim, target)
        loss = fc_loss_weight * fc_err + reg_l2 * (u**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(fc_err.detach().cpu()))

    return u.detach().cpu().squeeze(0).numpy(), history


def _upper_off_diag(fc: torch.Tensor) -> torch.Tensor:
    idx = torch.triu_indices(fc.shape[-1], fc.shape[-1], offset=1)
    return fc[..., idx[0], idx[1]]


@torch.no_grad()
def simulate_with_control(
    model: torch.nn.Module,
    u_vec: np.ndarray,
    n_rois: int,
    device: str,
    dt: float,
    dt_min: float,
    n_batches: int,
    batch_size: int,
    n_steps: int,
    seed: int,
) -> torch.Tensor:
    """Run independent rollouts and return (n_batches * batch_size, n_rois, n_steps)."""
    torch.manual_seed(seed)
    n_control_dims = u_vec.shape[0]
    u = torch.as_tensor(u_vec, device=device, dtype=torch.float32).view(1, n_control_dims)
    all_ts = []
    for _ in range(n_batches):
        z0 = torch.randn(batch_size, n_rois, dtype=torch.complex64, device=device) * 0.1
        u_batch = u.expand(batch_size, -1).contiguous()
        ts = model.forward(z0, n_steps=n_steps, dt=dt, dt_min=dt_min, control=u_batch)
        all_ts.append(ts.detach().cpu())
    return torch.cat(all_ts, dim=0)


def fidelity_metrics(
    sim_ts: torch.Tensor,
    target_fc: torch.Tensor,
    empirical_fcd: torch.Tensor,
) -> dict[str, float | torch.Tensor]:
    """Return fidelity metrics for a batch of simulated trajectories."""
    fcs = compute_fc_from_timeseries(sim_ts)
    off = _upper_off_diag(fcs)
    tgt_off = _upper_off_diag(target_fc)

    a = off - off.mean(dim=1, keepdim=True)
    b = tgt_off - tgt_off.mean()
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm() + 1e-8
    fc_corr = num / den
    fc_mse = ((off - tgt_off) ** 2).mean(dim=1)
    sim_fcd = fcd_windowed(sim_ts)
    ks_stat, ks_p = ks_2samp(sim_fcd.flatten().numpy(), empirical_fcd.flatten().numpy())

    return {
        "fc_corr_mean": float(fc_corr.mean()),
        "fc_corr_std": float(fc_corr.std(unbiased=False)),
        "fc_mse_mean": float(fc_mse.mean()),
        "fc_mse_std": float(fc_mse.std(unbiased=False)),
        "fcd_ks": float(ks_stat),
        "fcd_ks_pvalue": float(ks_p),
        "sim_fcd": sim_fcd,
    }


def pairwise_ks(samples_by_model: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = list(samples_by_model.keys())
    pvals = np.ones((len(labels), len(labels)))
    ks_stats = np.zeros((len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            stat, pval = ks_2samp(samples_by_model[a], samples_by_model[b])
            ks_stats[i, j] = stat
            pvals[i, j] = pval
    return labels, ks_stats, pvals


def save_figure(fig: plt.Figure, name: str, out_dirs: dict[str, Path]) -> None:
    for ext, out_dir in out_dirs.items():
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_dir = repo_path(args.checkpoint_dir)
    lsd_data_dir = repo_path(args.lsd_data_dir)
    results_path = repo_path(args.results_path)
    out_dirs = {
        "pdf": repo_path(args.out_dir_pdf),
        "png": repo_path(args.out_dir_png),
        "svg": repo_path(args.out_dir_svg),
    }
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Simulation: dt={args.dt}, n_steps={args.n_sim_steps}, "
        f"batch={args.batch_size}"
    )
    print(
        f"Optimisation: n_steps={args.n_opt_steps}, lr={args.lr}, reg_l2={args.reg_l2}"
    )

    ts_np, ctrl_np, patient_ids, cond_names = load_lsd_data(str(lsd_data_dir))
    _ = patient_ids, cond_names
    ts_t = torch.from_numpy(ts_np).float()
    _, n_rois, _ = ts_t.shape

    condition_of_subject = []
    for row in ctrl_np:
        if tuple(row) == (0.0, 0.0):
            condition_of_subject.append("Placebo")
        elif tuple(row) == (1.0, 0.0):
            condition_of_subject.append("LSD")
        else:
            condition_of_subject.append("LSD+Ket")
    condition_of_subject = np.array(condition_of_subject)

    empirical_fc_all = compute_fc_from_timeseries(ts_t)
    empirical_fc_by_cond = {}
    empirical_fc_mean = {}
    for cond in TARGET_CONDITIONS:
        mask = condition_of_subject == cond
        fcs = empirical_fc_all[mask]
        empirical_fc_by_cond[cond] = fcs
        empirical_fc_mean[cond] = fcs.mean(dim=0)
        off_diag_mean = (empirical_fc_mean[cond] - torch.eye(n_rois)).abs().mean().item()
        print(f"{cond:8s}: {mask.sum():3d} subjects, FC mean |off-diag| = {off_diag_mean:.3f}")

    empirical_fcd_by_cond = {
        cond: fcd_windowed(ts_t[condition_of_subject == cond]) for cond in TARGET_CONDITIONS
    }
    for cond in TARGET_CONDITIONS:
        print(f"{cond:8s}: empirical FCD samples = {empirical_fcd_by_cond[cond].numel()}")

    models: dict[str, torch.nn.Module] = {}
    for label, ckpt_name in MODEL_CHECKPOINTS.items():
        ckpt_path = checkpoint_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"  [skip] missing checkpoint {ckpt_path}")
            continue
        model, model_class, _ = load_model_from_checkpoint(str(ckpt_path), device=device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models[label] = model
        print(
            f"  loaded {label:14s} ({model_class}, n_rois={model.n_rois}, "
            f"n_control_dims={_get_n_control_dims(model)})"
        )

    if not models:
        raise RuntimeError(f"No checkpoints were loaded from {checkpoint_dir}")

    n_control_dims_by_model = {label: _get_n_control_dims(model) for label, model in models.items()}

    print("Optimising controls...")
    start = time.time()
    optimised: dict[str, dict[str, dict[str, np.ndarray | list[float]]]] = {}
    for label, model in models.items():
        optimised[label] = {}
        n_control_dims = n_control_dims_by_model[label]
        for cond in TARGET_CONDITIONS:
            t0 = time.time()
            seed = args.random_seed + COND_SEED_OFFSETS[cond]
            u_star, hist = optimise_control(
                model=model,
                target_fc=empirical_fc_mean[cond],
                n_control_dims=n_control_dims,
                n_rois=model.n_rois,
                device=device,
                dt=args.dt,
                dt_min=args.dt_min,
                fc_loss_weight=args.fc_loss_weight,
                n_opt_steps=args.n_opt_steps,
                n_sim_steps=args.n_sim_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                reg_l2=args.reg_l2,
                seed=seed,
            )
            optimised[label][cond] = {"u": u_star, "loss_history": hist}
            print(
                f"  {label:14s} | target={cond:8s} | u*={np.round(u_star, 3).tolist()} "
                f"| fc_err_final={hist[-1]:.4f} | {time.time() - t0:.1f}s"
            )
    print(f"Total optimisation time: {time.time() - start:.1f}s")

    labels = list(models.keys())
    n_labels = len(labels)
    ncols = min(3, n_labels)
    nrows = int(np.ceil(n_labels / ncols))
    colors = {"Placebo": "#4C72B0", "LSD+Ket": "#55A868", "LSD": "#DD8452"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, label in zip(axes, labels):
        for cond in TARGET_CONDITIONS:
            hist = optimised[label][cond]["loss_history"]
            ax.plot(hist, color=colors[cond], label=cond, linewidth=1.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_ylabel(r"off-diag FC error $\|\cdot\|_F^2$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)
    for ax in axes[len(labels) :]:
        ax.set_visible(False)
    fig.suptitle("Control optimisation loss curves", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "control_loss_curves", out_dirs)

    print("Evaluating optimised controls on fresh rollouts...")
    fidelity: dict[str, dict[str, dict[str, float | torch.Tensor]]] = {}
    for label, model in models.items():
        fidelity[label] = {}
        for cond in TARGET_CONDITIONS:
            u_star = optimised[label][cond]["u"]
            sim_ts = simulate_with_control(
                model=model,
                u_vec=u_star,
                n_rois=model.n_rois,
                device=device,
                dt=args.dt,
                dt_min=args.dt_min,
                n_batches=args.n_eval_batches,
                batch_size=args.batch_size,
                n_steps=args.n_eval_steps,
                seed=123 + COND_SEED_OFFSETS[cond],
            )
            metrics = fidelity_metrics(sim_ts, empirical_fc_mean[cond], empirical_fcd_by_cond[cond])
            fidelity[label][cond] = metrics
            print(
                f"  {label:14s} | target={cond:8s} | "
                f"FC corr={metrics['fc_corr_mean']:.3f}+/-{metrics['fc_corr_std']:.3f} | "
                f"FC MSE={metrics['fc_mse_mean']:.4f} | FCD KS={metrics['fcd_ks']:.3f}"
            )

    metric_cfg = [
        ("fc_corr_mean", "fc_corr_std", "FC correlation ↑"),
        ("fc_mse_mean", "fc_mse_std", "FC MSE (off-diagonal) ↓"),
        ("fcd_ks", None, "FCD KS vs. empirical ↓"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(labels))
    bar_width = 0.24
    for ax, (metric_mean, metric_std, title) in zip(axes, metric_cfg):
        for i, cond in enumerate(TARGET_CONDITIONS):
            means = [fidelity[label][cond][metric_mean] for label in labels]
            stds = [fidelity[label][cond][metric_std] if metric_std else 0 for label in labels]
            ax.bar(
                x + (i - (len(TARGET_CONDITIONS) - 1) / 2) * bar_width,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                color=colors[cond],
                label=COND_DISPLAY[cond],
                alpha=0.9,
                edgecolor="white",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("Controlled-simulation fidelity vs. target FC regimes", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "control_fidelity_bars", out_dirs)

    m_max = max(n_control_dims_by_model.values())
    rows_list = []
    for label in labels:
        u_placebo = optimised[label][CONTRAST_CONDITIONS[0]]["u"]
        u_lsd = optimised[label][CONTRAST_CONDITIONS[1]]["u"]
        delta_u = u_lsd - u_placebo

        def pad(vec: np.ndarray) -> np.ndarray:
            out = np.full(m_max, np.nan)
            out[: vec.shape[0]] = vec
            return out

        rows_list.append(
            {
                "Model": label,
                "n_ctrl": n_control_dims_by_model[label],
                **{
                    f"u_{cond}[{i}]": float(v)
                    for cond in TARGET_CONDITIONS
                    for i, v in enumerate(pad(optimised[label][cond]["u"]))
                },
                **{f"delta_u[{i}]": float(v) for i, v in enumerate(pad(delta_u))},
                "||delta_u||": float(np.linalg.norm(delta_u)),
            }
        )
    df_u = pd.DataFrame(rows_list).set_index("Model")
    print("\nOptimised controls:")
    print(df_u.round(3).to_string())

    fig, axes = plt.subplots(1, m_max, figsize=(5 * m_max, 4), squeeze=False)
    for dim in range(m_max):
        ax = axes[0, dim]
        vals_by_cond = {cond: [] for cond in TARGET_CONDITIONS}
        for label in labels:
            for cond in TARGET_CONDITIONS:
                u_cond = optimised[label][cond]["u"]
                vals_by_cond[cond].append(u_cond[dim] if dim < u_cond.shape[0] else np.nan)
        x = np.arange(len(labels))
        for idx, cond in enumerate(TARGET_CONDITIONS):
            ax.bar(
                x + (idx - (len(TARGET_CONDITIONS) - 1) / 2) * bar_width,
                vals_by_cond[cond],
                bar_width,
                color=colors[cond],
                label=COND_DISPLAY[cond],
                edgecolor="white",
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"$u^\\star$ component {dim}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle("Optimised control inputs $u^\\star$ per architecture and target", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "control_u_star", out_dirs)

    labels_by_m = {
        m_dim: [label for label in labels if n_control_dims_by_model[label] == m_dim]
        for m_dim in set(n_control_dims_by_model.values())
    }

    def pairwise_cosine(vecs: list[np.ndarray]) -> np.ndarray:
        arr = np.stack(vecs, axis=0)
        normalized = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return normalized @ normalized.T

    for m_dim, m_labels in labels_by_m.items():
        if len(m_labels) < 2:
            continue
        deltas = [
            optimised[label][CONTRAST_CONDITIONS[1]]["u"]
            - optimised[label][CONTRAST_CONDITIONS[0]]["u"]
            for label in m_labels
        ]
        if m_dim == 1:
            arr = np.array(deltas).squeeze()
            sim = np.sign(arr[:, None] * arr[None, :])
            title = rf"sign agreement of $\Delta u$ ({CONTRAST_CONDITIONS[1]} - {CONTRAST_CONDITIONS[0]})"
        else:
            sim = pairwise_cosine(deltas)
            title = (
                rf"cosine similarity of $\Delta u$ "
                rf"({CONTRAST_CONDITIONS[1]} - {CONTRAST_CONDITIONS[0]}, m={m_dim})"
            )

        fig, ax = plt.subplots(figsize=(1.5 + 0.6 * len(m_labels), 1.5 + 0.6 * len(m_labels)))
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(m_labels)))
        ax.set_yticks(range(len(m_labels)))
        ax.set_xticklabels(m_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(m_labels, fontsize=8)
        for i in range(len(m_labels)):
            for j in range(len(m_labels)):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
        ax.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        save_figure(fig, f"control_delta_u_similarity_m{m_dim}", out_dirs)

    target_cond = CONTRAST_CONDITIONS[1]
    fc_samples = {}
    fcd_samples = {}
    for label, model in models.items():
        sim_ts = simulate_with_control(
            model=model,
            u_vec=optimised[label][target_cond]["u"],
            n_rois=model.n_rois,
            device=device,
            dt=args.dt,
            dt_min=args.dt_min,
            n_batches=args.n_eval_batches,
            batch_size=args.batch_size,
            n_steps=args.n_eval_steps,
            seed=777,
        )
        fcs = compute_fc_from_timeseries(sim_ts)
        fc_samples[label] = _upper_off_diag(fcs).flatten().numpy()
        fcd_samples[label] = fcd_windowed(sim_ts).flatten().numpy()
        print(f"  {label}: {fc_samples[label].size} FC edges, {fcd_samples[label].size} FCD samples")

    lbls_fc, fc_ks, fc_p = pairwise_ks(fc_samples)
    lbls_fcd, fcd_ks, fcd_p = pairwise_ks(fcd_samples)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plots = [
        (fc_ks, fc_p, "FC edges — KS statistic (p-value annotated)", lbls_fc),
        (fcd_ks, fcd_p, "FCD — KS statistic (p-value annotated)", lbls_fcd),
    ]
    for ax, (ks_mat, p_mat, title, plot_labels) in zip(axes, plots):
        im = ax.imshow(ks_mat, cmap="viridis")
        ax.set_xticks(range(len(plot_labels)))
        ax.set_yticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(plot_labels, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(len(plot_labels)):
            for j in range(len(plot_labels)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="white")
                else:
                    color = "white" if ks_mat[i, j] > ks_mat.max() * 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"ks={ks_mat[i, j]:.2f}\np={p_mat[i, j]:.0e}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=color,
                    )
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(
        "Pairwise statistical (in)distinguishability between architectures under LSD control",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "control_pairwise_ks", out_dirs)

    alpha = 0.05
    indist_pairs = []
    for i, a in enumerate(lbls_fc):
        for j, b in enumerate(lbls_fc):
            if j <= i:
                continue
            if fc_p[i, j] >= alpha and fcd_p[i, j] >= alpha:
                indist_pairs.append((a, b, float(fc_p[i, j]), float(fcd_p[i, j])))
    print(f"\nStatistically indistinguishable pairs (p>={alpha:.2f} on BOTH FC and FCD):")
    if not indist_pairs:
        print("  (none)")
    for a, b, p_fc, p_fcd in indist_pairs:
        print(f"  {a} ↔ {b}: p_FC={p_fc:.3f}, p_FCD={p_fcd:.3f}")

    transfer = {cond: {} for cond in TARGET_CONDITIONS}
    if args.skip_transfer:
        print("Skipping transfer matrices (--skip-transfer).")
    else:
        for cond in TARGET_CONDITIONS:
            for m_dim, m_labels in labels_by_m.items():
                if len(m_labels) < 2:
                    continue
                mat = np.full((len(m_labels), len(m_labels)), np.nan)
                for i, src in enumerate(m_labels):
                    u_src = optimised[src][cond]["u"]
                    for j, tgt in enumerate(m_labels):
                        sim_ts = simulate_with_control(
                            model=models[tgt],
                            u_vec=u_src,
                            n_rois=models[tgt].n_rois,
                            device=device,
                            dt=args.dt,
                            dt_min=args.dt_min,
                            n_batches=args.n_eval_batches,
                            batch_size=args.batch_size,
                            n_steps=args.n_eval_steps,
                            seed=999 + i * 100 + j,
                        )
                        metrics = fidelity_metrics(
                            sim_ts=sim_ts,
                            target_fc=empirical_fc_mean[cond],
                            empirical_fcd=empirical_fcd_by_cond[cond],
                        )
                        mat[i, j] = metrics["fc_corr_mean"]
                transfer[cond][m_dim] = {"labels": m_labels, "fc_corr": mat}
                print(f"Transfer matrix ({cond}, m={m_dim}):")
                print(pd.DataFrame(mat, index=m_labels, columns=m_labels).round(3).to_string())
                print()

        m_dims_plotted = [m_dim for m_dim, m_labels in labels_by_m.items() if len(m_labels) >= 2]
        fig, axes = plt.subplots(
            len(m_dims_plotted),
            len(TARGET_CONDITIONS),
            figsize=(5 * len(TARGET_CONDITIONS), 4.2 * len(m_dims_plotted)),
            squeeze=False,
        )
        for row_idx, m_dim in enumerate(m_dims_plotted):
            m_labels = labels_by_m[m_dim]
            for col_idx, cond in enumerate(TARGET_CONDITIONS):
                ax = axes[row_idx, col_idx]
                mat = transfer[cond][m_dim]["fc_corr"]
                im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.5, vmax=1.0)
                ax.set_xticks(range(len(m_labels)))
                ax.set_yticks(range(len(m_labels)))
                ax.set_xticklabels(m_labels, rotation=30, ha="right", fontsize=8)
                ax.set_yticklabels(m_labels, fontsize=8)
                ax.set_xlabel("target architecture", fontsize=9)
                ax.set_ylabel("source architecture (u from)", fontsize=9)
                ax.set_title(f"{cond} target | m={m_dim}", fontsize=10, fontweight="bold")
                for i in range(len(m_labels)):
                    for j in range(len(m_labels)):
                        value = mat[i, j]
                        if not np.isnan(value):
                            ax.text(
                                j,
                                i,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color="black" if value > 0.4 else "white",
                                fontweight="bold" if i == j else "normal",
                            )
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(
            "Cross-architecture transfer of optimised controls (FC correlation to target)",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()
        save_figure(fig, "control_transfer_matrix", out_dirs)

        if indist_pairs:
            rows = []
            for cond in TARGET_CONDITIONS:
                for a, b, p_fc, p_fcd in indist_pairs:
                    m_dim = n_control_dims_by_model[a]
                    if n_control_dims_by_model[b] != m_dim:
                        continue
                    m_labels = transfer[cond][m_dim]["labels"]
                    i_a = m_labels.index(a)
                    i_b = m_labels.index(b)
                    mat = transfer[cond][m_dim]["fc_corr"]
                    rows.append(
                        {
                            "target_cond": cond,
                            "pair": f"{a} ↔ {b}",
                            "matched_AA": mat[i_a, i_a],
                            "matched_BB": mat[i_b, i_b],
                            "transfer_A→B": mat[i_a, i_b],
                            "transfer_B→A": mat[i_b, i_a],
                            "drop_A→B": mat[i_a, i_a] - mat[i_a, i_b],
                            "drop_B→A": mat[i_b, i_b] - mat[i_b, i_a],
                            "p_FC": p_fc,
                            "p_FCD": p_fcd,
                        }
                    )
            df_trans = pd.DataFrame(rows)
            print("\nTransfer performance on statistically indistinguishable pairs:")
            print(df_trans.round(3).to_string(index=False))
        else:
            print("No statistically indistinguishable pairs to highlight.")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_type": args.dataset_type,
        "target_conditions": TARGET_CONDITIONS,
        "n_control_dims": n_control_dims_by_model,
        "optimised": {
            label: {
                cond: {"u": values["u"].tolist(), "loss_history": values["loss_history"]}
                for cond, values in per_cond.items()
            }
            for label, per_cond in optimised.items()
        },
        "fidelity": {
            label: {
                cond: {k: v for k, v in metrics.items() if k != "sim_fcd"}
                for cond, metrics in per_cond.items()
            }
            for label, per_cond in fidelity.items()
        },
        "transfer": {
            cond: {
                str(m_dim): {
                    "labels": values["labels"],
                    "fc_corr_matrix": values["fc_corr"].tolist(),
                }
                for m_dim, values in by_m.items()
            }
            for cond, by_m in transfer.items()
        },
        "indistinguishable_pairs": [
            {"a": a, "b": b, "p_FC": p_fc, "p_FCD": p_fcd}
            for a, b, p_fc, p_fcd in indist_pairs
        ],
        "config": {
            "contrast_conditions": list(CONTRAST_CONDITIONS),
            "dt": args.dt,
            "dt_min": args.dt_min,
            "n_sim_steps": args.n_sim_steps,
            "batch_size": args.batch_size,
            "n_opt_steps": args.n_opt_steps,
            "lr": args.lr,
            "reg_l2": args.reg_l2,
            "n_eval_batches": args.n_eval_batches,
            "n_eval_steps": args.n_eval_steps,
        },
    }
    with results_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
