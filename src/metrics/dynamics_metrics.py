"""Dynamics metrics: FCD and metastability.

All inputs are **complex** analytic-signal tensors with layout
``(batch, n_rois, n_timepoints)`` (or ``(n_rois, n_timepoints)`` which is
auto-batched).  The data has already been bandpass-filtered and converted to
a complex analytic signal at dataset-load time, so **no further signal
preprocessing** is applied here.

Time averages use ``.mean()`` over the time dimension — the standard
discrete approximation to :math:`\\frac{1}{T}\\int_0^T f(t)\\,dt` for
uniformly sampled data.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ._utils import (
    align_batch_and_time,
    ensure_batch,
    fisher_batch_average,
    to_real,
    upper_tri_vec,
    zscore,
)


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

def fc_matrix(x: torch.Tensor) -> torch.Tensor:
    """Pearson FC from a z-scored real time series: ``FC = X^T X / (T-1)``.

    Args:
        x: ``(T, N)`` real tensor.

    Returns:
        ``(N, N)`` FC matrix.
    """
    xz = zscore(x, dim=0)
    T = xz.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 timepoints to compute FC.")
    return (xz.T @ xz) / (T - 1)


def _windowed_fc_vectors(
    x: torch.Tensor, win_len: int, win_step: int,
) -> Optional[torch.Tensor]:
    """Upper-triangle FC vectors for sliding windows.

    Args:
        x: ``(T, N)`` real tensor.

    Returns:
        ``(W, M)`` z-scored FC vectors, or ``None`` if no valid windows.
    """
    T = x.shape[0]
    vecs: list[torch.Tensor] = []
    for start in range(0, T - win_len + 1, win_step):
        w = x[start : start + win_len]
        vecs.append(upper_tri_vec(fc_matrix(w), k=1))

    if not vecs:
        return None

    V = torch.stack(vecs, dim=0)  # (W, M)
    if V.shape[1] <= 1:
        return None
    return zscore(V, dim=1)


def fcd_matrix(
    x: torch.Tensor, win_len: int, win_step: int,
) -> Optional[torch.Tensor]:
    """FCD matrix: correlation among windowed FC vectors.

    Args:
        x: ``(T, N)`` real tensor.
    """
    V = _windowed_fc_vectors(x, win_len, win_step)
    if V is None:
        return None
    n_features = V.shape[1]
    return (V @ V.T) / max(n_features - 1, 1)


def fcd_distribution(
    x: torch.Tensor, win_len: int, win_step: int,
) -> torch.Tensor:
    """Upper-triangle distribution of the FCD matrix.

    Args:
        x: ``(T, N)`` real tensor.
    """
    FCD = fcd_matrix(x, win_len, win_step)
    if FCD is None:
        return torch.empty(0, device=x.device, dtype=x.dtype)
    return upper_tri_vec(FCD, k=1)


# ---------------------------------------------------------------------------
# Grand-average phase-coherence FC (Eq. 11 in Deco et al. 2019)
# ---------------------------------------------------------------------------

def phase_coherence_fc(ts: torch.Tensor) -> torch.Tensor:
    r"""Grand-average phase-coherence FC matrix.

    Implements Eq. 11 of Deco et al. (2019):

    .. math::

        \mathrm{FC}_{ij} = \left\langle
            \cos\!\bigl(\phi_j(t) - \phi_i(t)\bigr)
        \right\rangle_t

    where :math:`\phi_i(t)` is the instantaneous phase extracted via
    ``torch.angle(z)`` from the complex analytic signal.

    Args:
        ts: ``(batch, n_rois, T)`` **complex** analytic-signal tensor.
            If ``(n_rois, T)`` it is auto-batched.

    Returns:
        ``(batch, n_rois, n_rois)`` phase-coherence FC matrices.  Each
        entry is in :math:`[-1, 1]` with diagonal identically 1.
    """
    ts = ensure_batch(ts)
    if not torch.is_complex(ts):
        raise TypeError(
            "phase_coherence_fc expects complex input; "
            "got real tensor.  Ensure data/model output is complex."
        )
    phases = torch.angle(ts)
    diff = phases.unsqueeze(2) - phases.unsqueeze(1)
    return torch.cos(diff).mean(dim=-1)  # (B, N, N)




# ---------------------------------------------------------------------------
# Phase coherence & phase-based FCD (phFCD)
# ---------------------------------------------------------------------------

def phase_coherence_matrix(phases: torch.Tensor) -> torch.Tensor:
    r"""Phase coherence matrices across all time points.

    For each pair of regions :math:`(n, m)` at time :math:`t`:

    .. math::

        P_{nm}(t) = \cos\!\bigl(\phi_n(t) - \phi_m(t)\bigr)

    Args:
        phases: ``(T, N)`` phase angles in radians.

    Returns:
        ``(T, N, N)`` phase coherence matrices.
    """
    diff = phases.unsqueeze(2) - phases.unsqueeze(1)
    return torch.cos(diff)


def phfcd_matrix(phases: torch.Tensor) -> Optional[torch.Tensor]:
    r"""Phase FCD (phFCD) similarity matrix.

    Implements the time-varying FC assessment described in Deco et al. (2019):

    1. Compute phase coherence :math:`P_{nm}(t)` for every time point.
    2. Vectorise the upper-triangular entries:
       :math:`\mathbf{p}(t) = \operatorname{vec}_{\triangle}(P(t)) \in \mathbb{R}^M`.
    3. Build a cosine-similarity matrix across time:

       .. math::

           \mathrm{phFCD}_{ij}
           = \frac{\mathbf{p}(t_i)^\top \mathbf{p}(t_j)}
                  {\|\mathbf{p}(t_i)\|_2\,\|\mathbf{p}(t_j)\|_2}

    Args:
        phases: ``(T, N)`` phase angles in radians.

    Returns:
        ``(T, T)`` phFCD matrix, or ``None`` if fewer than 2 ROIs.
    """
    T, N = phases.shape
    if N < 2:
        return None

    P = phase_coherence_matrix(phases)

    idx = torch.triu_indices(N, N, offset=1, device=phases.device)
    vecs = P[:, idx[0], idx[1]]  # (T, M)

    norms = torch.linalg.norm(vecs, dim=1, keepdim=True).clamp(min=1e-12)
    vecs_normed = vecs / norms
    return vecs_normed @ vecs_normed.T


def phfcd_distribution(phases: torch.Tensor) -> torch.Tensor:
    """Upper-triangular distribution of the phFCD matrix.

    Args:
        phases: ``(T, N)`` phase angles in radians.

    Returns:
        1-D tensor of phFCD similarity values.
    """
    fcd = phfcd_matrix(phases)
    if fcd is None:
        return torch.empty(0, device=phases.device, dtype=phases.dtype)
    return upper_tri_vec(fcd, k=1)


# ---------------------------------------------------------------------------
# Symmetrized KL divergence (Eq. 6 in Deco et al. 2019)
# ---------------------------------------------------------------------------

def symmetric_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Symmetrized Kullback-Leibler divergence between two distributions.

    .. math::

        D_{\mathrm{KL}}^{\mathrm{sym}}(P, Q)
        = \frac{1}{2}\sum_i P(i)\ln\frac{P(i)}{Q(i)}
        + \frac{1}{2}\sum_i Q(i)\ln\frac{Q(i)}{P(i)}

    Args:
        p: 1-D probability distribution (sums to 1).
        q: 1-D probability distribution (same length as *p*).
        eps: Small constant to avoid log(0).

    Returns:
        Scalar tensor — symmetric KL divergence.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = (p * torch.log(p / q)).sum()
    kl_qp = (q * torch.log(q / p)).sum()
    return 0.5 * (kl_pq + kl_qp)


# ---------------------------------------------------------------------------
# Markov entropy rate (Eqs. 8-9 in Deco et al. 2019)
# ---------------------------------------------------------------------------

def markov_entropy_rate(T_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Entropy rate of a discrete-time Markov chain.

    Given a transition probability matrix :math:`\mathbf{T}` with stationary
    distribution :math:`\pi` (left eigenvector for eigenvalue 1):

    .. math::

        S(\mathbf{T})
        = -\sum_i \pi(i)\sum_j T_{ij}\log T_{ij}

    Args:
        T_mat: ``(k, k)`` row-stochastic transition probability matrix.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor — entropy rate in nats.
    """
    T_mat = T_mat.clamp(min=eps)
    eigenvalues, eigenvectors = torch.linalg.eig(T_mat.T.to(torch.float64))
    idx = torch.argmin(torch.abs(eigenvalues - 1.0))
    pi = eigenvectors[:, idx].real
    pi = pi.abs()
    pi = pi / pi.sum()
    pi = pi.to(T_mat.dtype)
    row_entropies = -(T_mat * torch.log(T_mat)).sum(dim=1)  # (k,)
    return (pi * row_entropies).sum()


def tpm_entropy_distance(
    tpm_pred: torch.Tensor,
    tpm_target: torch.Tensor,
) -> torch.Tensor:
    r"""Absolute difference in Markov entropy rates between two TPMs.

    .. math::

        D_{\mathrm{ME}} = \bigl|S(\mathbf{T}_{\text{pred}})
                          - S(\mathbf{T}_{\text{target}})\bigr|

    Args:
        tpm_pred: ``(k, k)`` row-stochastic TPM.
        tpm_target: ``(k, k)`` row-stochastic TPM.

    Returns:
        Scalar tensor — absolute entropy-rate difference.
    """
    return torch.abs(markov_entropy_rate(tpm_pred) - markov_entropy_rate(tpm_target))


# ---------------------------------------------------------------------------
# KS distance
# ---------------------------------------------------------------------------

def ks_distance_2samp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two-sample Kolmogorov-Smirnov statistic (torch-only, exact CDF)."""
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    x = torch.sort(x.flatten()).values
    y = torch.sort(y.flatten()).values
    n, m = x.numel(), y.numel()
    z = torch.sort(torch.cat([x, y])).values
    cdf_x = torch.searchsorted(x, z, right=True).to(z.dtype) / n
    cdf_y = torch.searchsorted(y, z, right=True).to(z.dtype) / m
    return torch.max(torch.abs(cdf_x - cdf_y))


# ---------------------------------------------------------------------------
# Metastability
# ---------------------------------------------------------------------------

def kuramoto_metastability(phases: torch.Tensor) -> torch.Tensor:
    r"""Metastability from phase time series.

    .. math::

        R(t) = \left|\frac{1}{N}\sum_i e^{i\phi_i(t)}\right|,\quad
        \text{Meta} = \operatorname{std}_t R(t)

    Args:
        phases: ``(T, N)`` in radians.
    """
    order = torch.abs(torch.mean(torch.exp(1j * phases), dim=1))
    return order.std()


def metastability_value(ts: torch.Tensor) -> torch.Tensor:
    r"""Batch-mean metastability (differentiable).

    Phases are extracted directly via ``torch.angle(z)`` from the complex
    analytic signal -- no Hilbert transform or bandpass is needed.

    Args:
        ts: ``(batch, n_rois, T)`` **complex** tensor.

    Returns:
        Scalar tensor -- mean metastability across the batch.
    """
    ts = ensure_batch(ts)
    if not torch.is_complex(ts):
        raise TypeError(
            "metastability_value expects complex input; "
            "got real tensor.  Ensure data/model output is complex."
        )
    phases = torch.angle(ts).permute(0, 2, 1)
    vals = torch.stack([kuramoto_metastability(phases[b]) for b in range(phases.shape[0])])
    return vals.mean()


# ---------------------------------------------------------------------------
# nn.Module metric/loss classes
# ---------------------------------------------------------------------------

class FCD(nn.Module):
    """FCD (Functional Connectivity Dynamics) MSE loss + KS evaluation metric.

    ``forward()`` returns the differentiable FCD MSE loss (surrogate for KS).
    ``evaluate()`` returns ``{"fcd_mse": float, "fcd_ks": float}``.

    Args:
        tr: Repetition time in seconds.
        fcd_win_sec: FCD window length in seconds.
        fcd_step_sec: FCD step size in seconds.
    """

    def __init__(self, tr: float = 0.72, fcd_win_sec: float = 30.0, fcd_step_sec: float = 2.0):
        super().__init__()
        self.tr = tr
        self.fcd_win_sec = fcd_win_sec
        self.fcd_step_sec = fcd_step_sec

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        pred, target, batch, _ = align_batch_and_time(ts_pred, ts_target)
        pred, target = to_real(pred), to_real(target)
        win_len = int(round(self.fcd_win_sec / self.tr))
        win_step = int(round(self.fcd_step_sec / self.tr))
        if win_len < 10 or win_step <= 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        losses: list[torch.Tensor] = []
        for b in range(batch):
            pred_fcd = fcd_matrix(pred[b].T, win_len, win_step)
            targ_fcd = fcd_matrix(target[b].T, win_len, win_step)
            if pred_fcd is None or targ_fcd is None:
                continue
            n = min(pred_fcd.shape[0], targ_fcd.shape[0])
            if n <= 1:
                continue
            losses.append(((pred_fcd[:n, :n] - targ_fcd[:n, :n]) ** 2).mean())
        if not losses:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        ts_pred_b = ensure_batch(ts_pred)
        ts_target_b = ensure_batch(ts_target)
        B = min(ts_pred_b.shape[0], ts_target_b.shape[0])
        T = min(ts_pred_b.shape[2], ts_target_b.shape[2])
        pred_real = to_real(ts_pred_b[:B, :, :T])
        targ_real = to_real(ts_target_b[:B, :, :T])
        win_len = int(round(self.fcd_win_sec / self.tr))
        win_step = int(round(self.fcd_step_sec / self.tr))
        fcd_ks = float("nan")
        if win_len >= 10 and (T - win_len) > 10 and win_step > 0:
            pred_dists: List[torch.Tensor] = []
            targ_dists: List[torch.Tensor] = []
            valid = True
            for b in range(B):
                pd = fcd_distribution(pred_real[b].T, win_len, win_step)
                td = fcd_distribution(targ_real[b].T, win_len, win_step)
                if pd.numel() == 0 or td.numel() == 0:
                    valid = False
                    break
                pred_dists.append(pd)
                targ_dists.append(td)
            if valid and pred_dists:
                fcd_ks = float(ks_distance_2samp(torch.cat(pred_dists), torch.cat(targ_dists)).item())
        return {
            "fcd_mse": self(ts_pred, ts_target).item(),
            "fcd_ks": fcd_ks,
        }


class PhFCD(nn.Module):
    """Phase FCD (phFCD) MSE loss + KS evaluation metric.

    ``forward()`` returns the differentiable phFCD MSE loss (surrogate for KS).
    ``evaluate()`` returns ``{"phfcd_mse": float, "phfcd_ks": float}``.
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        pred = ensure_batch(ts_pred)
        target = ensure_batch(ts_target)
        B = min(pred.shape[0], target.shape[0])
        T = min(pred.shape[2], target.shape[2])
        pred, target = pred[:B, :, :T], target[:B, :, :T]
        if not torch.is_complex(pred) or not torch.is_complex(target):
            return torch.zeros((), device=pred.device, dtype=pred.real.dtype)
        losses: list[torch.Tensor] = []
        for b in range(B):
            pred_phases = torch.angle(pred[b]).T
            targ_phases = torch.angle(target[b]).T
            pred_phfcd = phfcd_matrix(pred_phases)
            targ_phfcd = phfcd_matrix(targ_phases)
            if pred_phfcd is None or targ_phfcd is None:
                continue
            n = min(pred_phfcd.shape[0], targ_phfcd.shape[0])
            if n <= 1:
                continue
            losses.append(((pred_phfcd[:n, :n] - targ_phfcd[:n, :n]) ** 2).mean())
        if not losses:
            return torch.zeros((), device=pred.device, dtype=pred.real.dtype)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        pred = ensure_batch(ts_pred)
        target = ensure_batch(ts_target)
        B = min(pred.shape[0], target.shape[0])
        T = min(pred.shape[2], target.shape[2])
        pred, target = pred[:B, :, :T], target[:B, :, :T]
        phfcd_ks = float("nan")
        if torch.is_complex(pred) and torch.is_complex(target):
            pred_ph_dists: List[torch.Tensor] = []
            targ_ph_dists: List[torch.Tensor] = []
            valid = True
            for b in range(B):
                pd_ph = phfcd_distribution(torch.angle(pred[b]).T)
                td_ph = phfcd_distribution(torch.angle(target[b]).T)
                if pd_ph.numel() == 0 or td_ph.numel() == 0:
                    valid = False
                    break
                pred_ph_dists.append(pd_ph)
                targ_ph_dists.append(td_ph)
            if valid and pred_ph_dists:
                phfcd_ks = float(ks_distance_2samp(
                    torch.cat(pred_ph_dists), torch.cat(targ_ph_dists)
                ).item())
        return {
            "phfcd_mse": self(ts_pred, ts_target).item(),
            "phfcd_ks": phfcd_ks,
        }


class Metastability(nn.Module):
    """L1 difference between predicted and target Kuramoto metastability.

    Metric and loss are the same value.
    ``evaluate()`` returns ``{"metastability_diff": float}``.
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        return torch.abs(metastability_value(ts_pred) - metastability_value(ts_target))

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"metastability_diff": self(ts_pred, ts_target).item()}


class PhaseFC(nn.Module):
    """1 − Pearson correlation between phase-coherence FC matrices.

    ``forward()`` returns the loss (lower is better).
    ``evaluate()`` returns ``{"phase_fc_correlation": float}`` (higher is better).
    """

    def forward(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> torch.Tensor:
        fc_pred = fisher_batch_average(phase_coherence_fc(ts_pred))
        fc_target = fisher_batch_average(phase_coherence_fc(ts_target))
        v_pred = upper_tri_vec(fc_pred, k=1)
        v_target = upper_tri_vec(fc_target, k=1)
        if v_pred.numel() < 2:
            return torch.tensor(float("nan"), device=ts_pred.device, dtype=ts_pred.real.dtype)
        p = v_pred - v_pred.mean()
        t = v_target - v_target.mean()
        num = (p * t).sum()
        den = torch.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-8
        return 1.0 - num / den

    @torch.no_grad()
    def evaluate(self, ts_pred: torch.Tensor, ts_target: torch.Tensor) -> dict:
        return {"phase_fc_correlation": 1.0 - self(ts_pred, ts_target).item()}
