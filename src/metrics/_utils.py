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


def fisher_batch_average(
    matrices: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Average correlation-like matrices across batch in Fisher-z space.

    The off-diagonal entries are clipped to ``(-1 + eps, 1 - eps)``,
    transformed with ``atanh``, averaged across the leading batch
    dimension, and mapped back with ``tanh``. The diagonal is restored
    from the arithmetic batch mean so identity diagonals remain exact.

    Args:
        matrices: ``(batch, n, n)`` or ``(n, n)`` tensor.
        eps: Numerical guard for ``atanh`` near ``±1``.

    Returns:
        ``(n, n)`` tensor.
    """
    matrices = to_real(matrices)
    if matrices.ndim == 2:
        return matrices
    if matrices.ndim != 3:
        raise ValueError(
            "Expected correlation matrices with shape (batch, n, n) or (n, n), "
            f"got {matrices.shape}."
        )

    clipped = matrices.clamp(min=-1.0 + eps, max=1.0 - eps)
    fisher_mean = torch.atanh(clipped).mean(dim=0)
    averaged = torch.tanh(fisher_mean)

    diag = matrices.diagonal(dim1=-2, dim2=-1).mean(dim=0)
    diag_matrix = torch.diag_embed(diag)
    off_diag_mask = (~torch.eye(
        averaged.shape[-1], dtype=torch.bool, device=averaged.device
    )).to(averaged.dtype)
    return averaged * off_diag_mask + diag_matrix


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def upper_tri_vec(M: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Extract the upper-triangular elements as a flat vector.

    Handles both 2-D ``(N, N)`` and batched 3-D ``(B, N, N)`` inputs.

    Args:
        M: ``(N, N)`` or ``(B, N, N)`` matrix.
        k: Diagonal offset (default ``1`` excludes the main diagonal).

    Returns:
        ``(M_elems,)`` for 2-D input or ``(B, M_elems)`` for 3-D input.
    """
    n = M.shape[-1]
    if n < 2:
        return torch.empty(0, device=M.device, dtype=M.dtype)
    idx = torch.triu_indices(n, n, offset=k, device=M.device)
    if M.dim() == 2:
        return M[idx[0], idx[1]]
    return M[:, idx[0], idx[1]]  # batched (B, N, N) -> (B, M_elems)
