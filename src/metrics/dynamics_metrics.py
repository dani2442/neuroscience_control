"""Dynamics metrics and differentiable dynamics losses."""

from typing import Dict, Optional, List

import torch


def detrend_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Remove per-channel linear trend.

    Args:
        x: (T, N)
    """
    if x.ndim != 2:
        raise ValueError("x must be (T, N)")
    T, _ = x.shape
    if T < 2:
        return x

    device, dtype = x.device, x.dtype

    # Closed-form per-channel linear regression avoids backend lstsq issues.
    t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
    t_centered = t - t.mean(dim=0, keepdim=True)  # (T, 1)
    x_mean = x.mean(dim=0, keepdim=True)  # (1, N)

    denom = (t_centered * t_centered).sum(dim=0, keepdim=True).clamp_min(1e-12)  # (1, 1)
    slope = (t_centered * (x - x_mean)).sum(dim=0, keepdim=True) / denom  # (1, N)
    intercept = x_mean - slope * t.mean(dim=0, keepdim=True)  # (1, N)
    trend = t * slope + intercept  # (T, N)
    return x - trend


def zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, keepdim=True) + eps
    return (x - mu) / sd


def fft_bandpass(x: torch.Tensor, tr: float, f_lo: float, f_hi: float) -> torch.Tensor:
    """
    FFT brick-wall bandpass.

    Args:
        x: (T, N), real
    """
    if x.ndim != 2:
        raise ValueError("x must be (T, N)")
    T = x.shape[0]
    device = x.device
    dtype = x.dtype

    X = torch.fft.rfft(x, dim=0)  # (F, N), complex
    freqs = torch.fft.rfftfreq(T, d=tr).to(device=device, dtype=dtype)  # (F,)

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X_filtered = torch.where(mask.unsqueeze(1), X, torch.zeros_like(X))
    x_filt = torch.fft.irfft(X_filtered, n=T, dim=0)
    return x_filt


def analytic_signal(x: torch.Tensor) -> torch.Tensor:
    """
    Hilbert transform via FFT to get analytic signal.

    Args:
        x: (T, N), real

    Returns:
        (T, N), complex
    """
    if x.ndim != 2:
        raise ValueError("x must be (T, N)")
    T = x.shape[0]
    X = torch.fft.fft(x, dim=0)  # (T, N), complex

    h = torch.zeros(T, device=x.device, dtype=X.dtype)
    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1:T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(T + 1) // 2] = 2.0

    Z = torch.fft.ifft(X * h.unsqueeze(1), dim=0)
    return Z


def fc_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Pearson FC using z-scored time series: FC = X^T X / (T-1).

    Args:
        x: (T, N)

    Returns:
        (N, N)
    """
    xz = zscore(x, dim=0)
    T = xz.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 timepoints to compute FC.")
    return (xz.transpose(0, 1) @ xz) / (T - 1)


def upper_tri_vec(M: torch.Tensor, k: int = 1) -> torch.Tensor:
    n = M.shape[0]
    if n < 2:
        return torch.empty(0, device=M.device, dtype=M.dtype)
    iu = torch.triu_indices(n, n, offset=k, device=M.device)
    return M[iu[0], iu[1]]


def _windowed_fc_vectors(x: torch.Tensor, win_len: int, win_step: int) -> Optional[torch.Tensor]:
    """
    Compute upper-triangle FC vectors for sliding windows.

    Args:
        x: (T, N)
    """
    T, _ = x.shape
    vecs = []
    for start in range(0, T - win_len + 1, win_step):
        w = x[start:start + win_len]
        FCw = fc_matrix(w)
        vecs.append(upper_tri_vec(FCw, k=1))

    if not vecs:
        return None

    V = torch.stack(vecs, dim=0)  # (W, M)
    if V.shape[1] <= 1:
        return None
    # Z-score each window vector across features (dim=1) so that
    # V @ V^T / (M-1) yields Pearson correlation *between* window vectors,
    # matching the FCD definition: corr(w(τ1), w(τ2)).
    return zscore(V, dim=1)


def fcd_matrix(x: torch.Tensor, win_len: int, win_step: int) -> Optional[torch.Tensor]:
    """
    Windowed FC vectors -> FCD matrix (corr among vectors).

    Args:
        x: (T, N)
    """
    V = _windowed_fc_vectors(x, win_len, win_step)
    if V is None:
        return None

    n_features = V.shape[1]
    denom = max(n_features - 1, 1)
    return (V @ V.transpose(0, 1)) / denom


def fcd_distribution(x: torch.Tensor, win_len: int, win_step: int) -> torch.Tensor:
    """
    Windowed FC vectors -> FCD matrix (corr among vectors) -> upper-triangle distribution.

    Args:
        x: (T, N)
    """
    FCD = fcd_matrix(x, win_len, win_step)
    if FCD is None:
        return torch.empty(0, device=x.device, dtype=x.dtype)
    return upper_tri_vec(FCD, k=1)


def ks_distance_2samp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Two-sample KS statistic (torch-only, exact for empirical CDF).

    Args:
        x, y: 1D
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    x = torch.sort(x.flatten()).values
    y = torch.sort(y.flatten()).values
    n = x.numel()
    m = y.numel()

    z = torch.sort(torch.cat([x, y], dim=0)).values
    cdf_x = torch.searchsorted(x, z, right=True).to(dtype=z.dtype) / n
    cdf_y = torch.searchsorted(y, z, right=True).to(dtype=z.dtype) / m
    return torch.max(torch.abs(cdf_x - cdf_y))


def kuramoto_metastability(phases: torch.Tensor) -> torch.Tensor:
    """
    phases: (T, N) radians
    metastability = std_t |mean_i exp(i*phi_i(t))|
    """
    order = torch.abs(torch.mean(torch.exp(1j * phases), dim=1))
    return order.std()


def _preprocess_timeseries(x: torch.Tensor, tr: float, f_lo: float, f_hi: float) -> torch.Tensor:
    x = detrend_linear(x)
    x = fft_bandpass(x, tr, f_lo, f_hi)
    x = zscore(x, dim=0)
    return x


def _to_real(ts: torch.Tensor) -> torch.Tensor:
    """Extract real part if complex, else pass through."""
    return ts.real if torch.is_complex(ts) else ts


def _ensure_batch(ts: torch.Tensor) -> torch.Tensor:
    ts = _to_real(ts)
    if ts.ndim == 2:
        return ts.unsqueeze(0)
    if ts.ndim != 3:
        raise ValueError("Timeseries must be (batch, n_rois, n_timepoints) or (n_rois, n_timepoints)")
    return ts


def metastability_value(
    ts: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07
) -> torch.Tensor:
    """
    Compute batch-mean metastability (differentiable).

    Args:
        ts: (batch, n_rois, n_timepoints) or (n_rois, n_timepoints)
    """
    ts = _ensure_batch(ts)
    vals = []
    for b in range(ts.shape[0]):
        x = _preprocess_timeseries(ts[b].transpose(0, 1), tr, f_lo, f_hi)
        phases = torch.angle(analytic_signal(x))
        vals.append(kuramoto_metastability(phases))
    return torch.stack(vals).mean()


def metastability_l1_loss(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07
) -> torch.Tensor:
    """
    Differentiable L1 loss on metastability.
    """
    meta_pred = metastability_value(ts_pred, tr=tr, f_lo=f_lo, f_hi=f_hi)
    meta_targ = metastability_value(ts_target, tr=tr, f_lo=f_lo, f_hi=f_hi)
    return torch.abs(meta_pred - meta_targ)


def fcd_mse_loss(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    fcd_win_sec: float = 60.0,
    fcd_step_sec: float = 2.0
) -> torch.Tensor:
    """
    Differentiable training surrogate for FCD using MSE between FCD matrices.

    Notes:
        - This is intentionally different from the evaluation metric ``fcd_ks``,
          which compares FCD value distributions via KS distance.
        - If the chosen window configuration cannot produce a valid FCD matrix
          (e.g., too-short series, invalid step), this returns ``0`` so training
          can proceed without NaNs.
    """
    ts_pred = _ensure_batch(ts_pred)
    ts_target = _ensure_batch(ts_target)

    batch = min(ts_pred.shape[0], ts_target.shape[0])
    if batch == 0:
        raise ValueError("Timeseries batch dimension must be > 0.")

    ts_pred = ts_pred[:batch]
    ts_target = ts_target[:batch]

    win_len = int(round(fcd_win_sec / tr))
    win_step = int(round(fcd_step_sec / tr))

    device = ts_pred.device
    dtype = ts_pred.dtype

    if win_len < 10 or win_step <= 0:
        return torch.zeros((), device=device, dtype=dtype)

    losses = []
    for b in range(batch):
        pred = _preprocess_timeseries(ts_pred[b].transpose(0, 1), tr, f_lo, f_hi)
        targ = _preprocess_timeseries(ts_target[b].transpose(0, 1), tr, f_lo, f_hi)

        pred_fcd = fcd_matrix(pred, win_len, win_step)
        targ_fcd = fcd_matrix(targ, win_len, win_step)
        if pred_fcd is None or targ_fcd is None:
            continue

        n = min(pred_fcd.shape[0], targ_fcd.shape[0])
        if n <= 1:
            continue
        losses.append(((pred_fcd[:n, :n] - targ_fcd[:n, :n]) ** 2).mean())

    if not losses:
        return torch.zeros((), device=device, dtype=dtype)

    return torch.stack(losses).mean()


def compute_dynamics_fit_metrics(
    ts_pred: torch.Tensor,
    ts_target: torch.Tensor,
    tr: float = 0.72,
    f_lo: float = 0.04,
    f_hi: float = 0.07,
    fcd_win_sec: float = 60.0,
    fcd_step_sec: float = 2.0,
    compute_fcd: bool = True,
    compute_metastability: bool = True,
) -> Dict[str, float]:
    """
    Compute FCD and metastability metrics.

    Args:
        ts_pred: (batch, n_rois, n_timepoints) or (n_rois, n_timepoints)
        ts_target: (batch, n_rois, n_timepoints) or (n_rois, n_timepoints)

    Returns:
        {"fcd_ks": float, "metastability_diff": float}

    Notes:
        - ``fcd_ks`` is an evaluation metric (KS distance on FCD distributions),
          not the differentiable training loss.
        - ``fcd_ks`` is reported as ``NaN`` when FCD windowing is not feasible for
          the provided series length / window parameters.
    """
    ts_pred = _ensure_batch(ts_pred)
    ts_target = _ensure_batch(ts_target)

    batch = min(ts_pred.shape[0], ts_target.shape[0])
    if batch == 0:
        raise ValueError("Timeseries batch dimension must be > 0.")

    ts_pred = ts_pred[:batch]
    ts_target = ts_target[:batch]

    # Align time dimensions to the shorter of the two to avoid length bias.
    n_timepoints = min(ts_pred.shape[2], ts_target.shape[2])
    ts_pred = ts_pred[:, :, :n_timepoints]
    ts_target = ts_target[:, :, :n_timepoints]

    win_len = int(round(fcd_win_sec / tr))
    win_step = int(round(fcd_step_sec / tr))

    fcd_ks = float("nan")
    if compute_fcd:
        # Keep NaN when FCD cannot be computed with this window config so callers
        # can distinguish "disabled/unavailable" from a valid numeric distance.
        if win_len >= 10 and (n_timepoints - win_len) > 10 and win_step > 0:
            pred_dists: List[torch.Tensor] = []
            targ_dists: List[torch.Tensor] = []
            for b in range(batch):
                pred = _preprocess_timeseries(ts_pred[b].transpose(0, 1), tr, f_lo, f_hi)
                targ = _preprocess_timeseries(ts_target[b].transpose(0, 1), tr, f_lo, f_hi)
                pred_dist = fcd_distribution(pred, win_len, win_step)
                targ_dist = fcd_distribution(targ, win_len, win_step)
                if pred_dist.numel() == 0 or targ_dist.numel() == 0:
                    pred_dists = []
                    targ_dists = []
                    break
                pred_dists.append(pred_dist)
                targ_dists.append(targ_dist)

            if pred_dists and targ_dists:
                pred_cat = torch.cat(pred_dists, dim=0)
                targ_cat = torch.cat(targ_dists, dim=0)
                fcd_ks = float(ks_distance_2samp(pred_cat, targ_cat).item())

    meta_diff = float("nan")
    if compute_metastability:
        meta_pred_vals = []
        meta_targ_vals = []
        for b in range(batch):
            pred = _preprocess_timeseries(ts_pred[b].transpose(0, 1), tr, f_lo, f_hi)
            targ = _preprocess_timeseries(ts_target[b].transpose(0, 1), tr, f_lo, f_hi)
            ph_pred = torch.angle(analytic_signal(pred))
            ph_targ = torch.angle(analytic_signal(targ))
            meta_pred_vals.append(kuramoto_metastability(ph_pred))
            meta_targ_vals.append(kuramoto_metastability(ph_targ))

        meta_pred = torch.stack(meta_pred_vals).mean()
        meta_targ = torch.stack(meta_targ_vals).mean()
        meta_diff = float(torch.abs(meta_pred - meta_targ).item())

    return {
        "fcd_ks": fcd_ks,
        "metastability_diff": meta_diff,
    }
