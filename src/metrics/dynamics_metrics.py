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

from typing import Dict, List, Optional

import torch

from ._utils import ensure_batch, to_real, upper_tri_vec, zscore


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
    # Z-score each window vector across features so
    # V V^T / (M-1) yields Pearson correlation between window vectors.
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
    # (T, N, 1) - (T, 1, N) -> (T, N, N)
    diff = phases.unsqueeze(2) - phases.unsqueeze(1)
    return torch.cos(diff)


def phfcd_matrix(phases: torch.Tensor) -> Optional[torch.Tensor]:
    r"""Phase FCD (phFCD) similarity matrix.

    Implements the time-varying FC assessment described in Deco et al. (2019):

    1. Compute phase coherence :math:`P_{nm}(t)` for every time point.
    2. Vectorise the upper-triangular entries :
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

    # Phase coherence at each time: (T, N, N)
    P = phase_coherence_matrix(phases)

    # Upper-triangle of each P(t): (T, M)  where M = N*(N-1)/2
    idx = torch.triu_indices(N, N, offset=1, device=phases.device)
    vecs = P[:, idx[0], idx[1]]  # (T, M)

    # Cosine similarity
    norms = torch.linalg.norm(vecs, dim=1, keepdim=True).clamp(min=1e-12)
    vecs_normed = vecs / norms
    return vecs_normed @ vecs_normed.T


def phfcd_distribution(phases: torch.Tensor) -> torch.Tensor:
    """Upper-triangular distribution of the phFCD matrix.

    This is the tv-FC summary distribution used by the paper.

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

    The time-domain standard deviation of the Kuramoto order parameter is
    the discrete approximation to
    :math:`\sqrt{\frac{1}{T}\int_0^T (R(t) - \bar R)^2\,dt}`.
    The result is averaged over the batch.

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
    # phases: (B, N, T) -> (B, T, N) for kuramoto_metastability
    phases = torch.angle(ts).permute(0, 2, 1)
    vals = torch.stack([kuramoto_metastability(phases[b]) for b in range(phases.shape[0])])
    return vals.mean()


# ---------------------------------------------------------------------------
# Differentiable losses
# ---------------------------------------------------------------------------

def metastability_l1_loss(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> torch.Tensor:
    """L1 loss between predicted and target metastability."""
    return torch.abs(metastability_value(ts_pred) - metastability_value(ts_target))


def fcd_mse_loss(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    fcd_win_sec: float = 60.0,
    fcd_step_sec: float = 2.0,
) -> torch.Tensor:
    """Differentiable FCD surrogate: MSE between FCD matrices.

    Operates on the **real part** of the complex analytic signal.  No
    additional preprocessing (detrend / bandpass / zscore) is applied
    because the data was already preprocessed at load time.

    Returns ``0`` when the window configuration cannot produce a valid
    FCD matrix (e.g. too-short series), so training proceeds without NaNs.
    """
    pred, target, batch, _ = _align_real(ts_pred, ts_target)

    win_len = int(round(fcd_win_sec / tr))
    win_step = int(round(fcd_step_sec / tr))

    if win_len < 10 or win_step <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    losses: list[torch.Tensor] = []
    for b in range(batch):
        # (N, T) -> (T, N) for windowed-FC helpers
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


def phfcd_mse_loss(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
) -> torch.Tensor:
    """Differentiable phFCD surrogate: MSE between phFCD matrices.

    Uses the **complex** analytic signal to extract instantaneous phases
    via ``torch.angle(z)``, then computes the phFCD matrix for each batch
    element and returns the mean MSE.

    Returns ``0`` when the input is real-valued or has fewer than 2 ROIs.
    """
    pred = ensure_batch(ts_pred)
    target = ensure_batch(ts_target)
    B = min(pred.shape[0], target.shape[0])
    T = min(pred.shape[2], target.shape[2])
    pred = pred[:B, :, :T]
    target = target[:B, :, :T]

    if not torch.is_complex(pred) or not torch.is_complex(target):
        return torch.zeros((), device=pred.device, dtype=pred.real.dtype)

    losses: list[torch.Tensor] = []
    for b in range(B):
        pred_phases = torch.angle(pred[b]).T   # (T, N)
        targ_phases = torch.angle(target[b]).T  # (T, N)
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


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_dynamics_fit_metrics(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    fcd_win_sec: float = 60.0,
    fcd_step_sec: float = 2.0,
    compute_fcd: bool = True,
    compute_metastability: bool = True,
) -> Dict[str, float]:
    """Compute FCD-KS and metastability-diff evaluation metrics.

    All time-domain averages use ``.mean()`` -- the discrete approximation
    to :math:`\\frac{1}{T}\\int_0^T f(t)\\,dt` for uniform sampling -- then
    averaged over the batch and ROIs.

    Args:
        ts_pred: ``(batch, n_rois, T)`` complex.
        ts_target: ``(batch, n_rois, T)`` complex.
        tr: Repetition time (seconds), used only for FCD window sizing.
        fcd_win_sec: FCD window length in seconds.
        fcd_step_sec: FCD window step in seconds.
        compute_fcd: Whether to compute ``fcd_ks``.
        compute_metastability: Whether to compute ``metastability_diff``.

    Returns:
        ``{"fcd_ks": float, "metastability_diff": float}``

    Notes:
        ``fcd_ks`` is reported as ``NaN`` when FCD windowing is not feasible.
    """
    ts_pred = ensure_batch(ts_pred)
    ts_target = ensure_batch(ts_target)
    B = min(ts_pred.shape[0], ts_target.shape[0])
    T = min(ts_pred.shape[2], ts_target.shape[2])
    ts_pred = ts_pred[:B, :, :T]
    ts_target = ts_target[:B, :, :T]

    # Real parts for FCD
    pred_real = to_real(ts_pred)
    targ_real = to_real(ts_target)

    win_len = int(round(fcd_win_sec / tr))
    win_step = int(round(fcd_step_sec / tr))

    # ---- FCD KS (windowed-correlation based) ----
    fcd_ks = float("nan")
    if compute_fcd and win_len >= 10 and (T - win_len) > 10 and win_step > 0:
        pred_dists: List[torch.Tensor] = []
        targ_dists: List[torch.Tensor] = []
        for b in range(B):
            pd = fcd_distribution(pred_real[b].T, win_len, win_step)
            td = fcd_distribution(targ_real[b].T, win_len, win_step)
            if pd.numel() == 0 or td.numel() == 0:
                pred_dists.clear()
                targ_dists.clear()
                break
            pred_dists.append(pd)
            targ_dists.append(td)
        if pred_dists and targ_dists:
            fcd_ks = float(
                ks_distance_2samp(
                    torch.cat(pred_dists), torch.cat(targ_dists),
                ).item()
            )

    # ---- phFCD KS (phase-based FCD, the paper's main fitting metric) ----
    phfcd_ks_val = float("nan")
    if compute_fcd and torch.is_complex(ts_pred) and torch.is_complex(ts_target):
        pred_ph_dists: List[torch.Tensor] = []
        targ_ph_dists: List[torch.Tensor] = []
        for b in range(B):
            pred_phases = torch.angle(ts_pred[b]).T   # (T, N)
            targ_phases = torch.angle(ts_target[b]).T  # (T, N)
            pd_ph = phfcd_distribution(pred_phases)
            td_ph = phfcd_distribution(targ_phases)
            if pd_ph.numel() == 0 or td_ph.numel() == 0:
                pred_ph_dists.clear()
                targ_ph_dists.clear()
                break
            pred_ph_dists.append(pd_ph)
            targ_ph_dists.append(td_ph)
        if pred_ph_dists and targ_ph_dists:
            phfcd_ks_val = float(
                ks_distance_2samp(
                    torch.cat(pred_ph_dists), torch.cat(targ_ph_dists),
                ).item()
            )

    # ---- Metastability ----
    meta_diff = float("nan")
    if compute_metastability:
        meta_pred = metastability_value(ts_pred)
        meta_targ = metastability_value(ts_target)
        meta_diff = float(torch.abs(meta_pred - meta_targ).item())

    return {
        "fcd_ks": fcd_ks,
        "phfcd_ks": phfcd_ks_val,
        "metastability_diff": meta_diff,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _align_real(
    pred: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Ensure batch dim, convert to real, align batch & time."""
    pred = to_real(ensure_batch(pred))
    target = to_real(ensure_batch(target))
    B = min(pred.shape[0], target.shape[0])
    T = min(pred.shape[2], target.shape[2])
    return pred[:B, :, :T], target[:B, :, :T], B, T
