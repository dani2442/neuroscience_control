"""Shared utilities for metrics computation.

All metrics operate on **complex** analytic-signal tensors with layout
``(batch, n_rois, n_timepoints)``.  Helpers here handle the common
operations: extracting the real part, ensuring a batch dimension, z-scoring,
and extracting the upper triangle of a matrix.
"""

import torch


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def to_real(ts: torch.Tensor) -> torch.Tensor:
    """Return the real part of a complex tensor, or pass through if already real."""
    return ts.real if torch.is_complex(ts) else ts


def ensure_batch(ts: torch.Tensor) -> torch.Tensor:
    """Add a leading batch dimension when the input is ``(n_rois, T)``.

    Returns the tensor unchanged if it already has 3 dimensions.

    Raises:
        ValueError: If the tensor has fewer than 2 or more than 3 dimensions.
    """
    if ts.ndim == 2:
        return ts.unsqueeze(0)
    if ts.ndim != 3:
        raise ValueError(
            "Timeseries must be (batch, n_rois, n_timepoints) or "
            f"(n_rois, n_timepoints), got {ts.ndim}D"
        )
    return ts


def align_batch_and_time(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Ensure both tensors are batched and trimmed to the shorter time/batch.

    Returns:
        pred, target  — trimmed tensors with shape ``(B, N, T)``
        B             — common batch size
        T             — common time length
    """
    pred = ensure_batch(pred)
    target = ensure_batch(target)
    B = min(pred.shape[0], target.shape[0])
    T = min(pred.shape[2], target.shape[2])
    return pred[:B, :, :T], target[:B, :, :T], B, T


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def zscore(x: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    """Z-score normalise along *dim*."""
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, keepdim=True) + eps
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def upper_tri_vec(M: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Extract the upper-triangular elements of a 2-D matrix as a flat vector.

    Args:
        M: ``(n, n)`` matrix.
        k: Diagonal offset (default ``1`` excludes the main diagonal).
    """
    n = M.shape[-1]
    if n < 2:
        return torch.empty(0, device=M.device, dtype=M.dtype)
    idx = torch.triu_indices(n, n, offset=k, device=M.device)
    return M[idx[0], idx[1]]
