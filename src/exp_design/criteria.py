"""Alphabetic optimality criteria for persistent-excitation Gramians."""

from __future__ import annotations

from typing import Iterable

import torch

CRITERIA = ("e", "a", "d", "t")

_ALIASES = {
    "e": "e",
    "e-optimal": "e",
    "e_optimal": "e",
    "a": "a",
    "a-optimal": "a",
    "a_optimal": "a",
    "d": "d",
    "d-optimal": "d",
    "d_optimal": "d",
    "t": "t",
    "t-optimal": "t",
    "t_optimal": "t",
}


def canonicalize_criterion(name: str) -> str:
    """Normalize criterion aliases to a short canonical name."""
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unsupported criterion: {name}")
    return _ALIASES[key]


def ensure_supported(criteria: Iterable[str]) -> tuple[str, ...]:
    """Validate and canonicalize a criterion list."""
    return tuple(canonicalize_criterion(name) for name in criteria)


def criterion_objective(
    gramian: torch.Tensor,
    criterion: str,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Return the differentiable objective to maximize."""
    criterion = canonicalize_criterion(criterion)
    gramian = 0.5 * (gramian + gramian.T)
    eye = torch.eye(gramian.shape[0], dtype=gramian.dtype, device=gramian.device)
    regularized = gramian + jitter * eye

    if criterion == "e":
        return torch.linalg.eigvalsh(regularized).min()
    if criterion == "a":
        return -torch.trace(torch.linalg.inv(regularized))
    if criterion == "d":
        sign, logabsdet = torch.linalg.slogdet(regularized)
        return torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -torch.inf))
    if criterion == "t":
        return torch.trace(gramian)
    raise AssertionError("Unreachable.")


def criterion_metrics(gramian: torch.Tensor, jitter: float = 1e-6) -> dict[str, float]:
    """Return standard Gramian summary metrics."""
    gramian = 0.5 * (gramian + gramian.T)
    eye = torch.eye(gramian.shape[0], dtype=gramian.dtype, device=gramian.device)
    regularized = gramian + jitter * eye
    eigvals = torch.linalg.eigvalsh(gramian)
    sign, logabsdet = torch.linalg.slogdet(regularized)
    log_det = logabsdet if bool((sign > 0).item()) else torch.full_like(logabsdet, float("-inf"))

    return {
        "lambda_min": float(eigvals.min().item()),
        "lambda_max": float(eigvals.max().item()),
        "trace": float(torch.trace(gramian).item()),
        "log_det": float(log_det.item()),
        "trace_inv": float(torch.trace(torch.linalg.inv(regularized)).item()),
    }
