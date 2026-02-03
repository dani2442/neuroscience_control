"""Dynamics metrics for timeseries comparisons (FCD, metastability)."""

from typing import Dict, Optional, Tuple, List

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
    device, dtype = x.device, x.dtype

    t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
    A = torch.cat([t, torch.ones_like(t)], dim=1)  # (T, 2)
    beta = torch.linalg.lstsq(A, x).solution  # (2, N)
    trend = A @ beta
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


def fcd_distribution(x: torch.Tensor, win_len: int, win_step: int) -> torch.Tensor:
    """
    Windowed FC vectors -> FCD matrix (corr among vectors) -> upper-triangle distribution.

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
        return torch.empty(0, device=x.device, dtype=x.dtype)

    V = torch.stack(vecs, dim=0)  # (W, M)
    if V.shape[1] <= 1:
        return torch.empty(0, device=x.device, dtype=x.dtype)

    V = zscore(V, dim=0)
    Mfeat = V.shape[1]
    if Mfeat <= 1:
        return torch.empty(0, device=x.device, dtype=x.dtype)
    FCD = (V @ V.transpose(0, 1)) / (Mfeat - 1)
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


def _ensure_batch(ts: torch.Tensor) -> torch.Tensor:
    if ts.ndim == 2:
        return ts.unsqueeze(0)
    if ts.ndim != 3:
        raise ValueError("Timeseries must be (batch, n_rois, n_timepoints) or (n_rois, n_timepoints)")
    return ts


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
    """
    ts_pred = _ensure_batch(ts_pred)
    ts_target = _ensure_batch(ts_target)

    batch = min(ts_pred.shape[0], ts_target.shape[0])
    if batch == 0:
        raise ValueError("Timeseries batch dimension must be > 0.")

    ts_pred = ts_pred[:batch]
    ts_target = ts_target[:batch]

    n_timepoints = ts_pred.shape[2]
    win_len = int(round(fcd_win_sec / tr))
    win_step = int(round(fcd_step_sec / tr))

    fcd_ks = float("nan")
    if compute_fcd:
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
