"""Optimization and SDP relaxation for PE experiment design."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .basis import Basis
from .criteria import CRITERIA, canonicalize_criterion, criterion_metrics, criterion_objective


@dataclass(slots=True)
class DesignResult:
    """Container for one optimized waveform."""

    criterion: str
    method: str
    coefficients: np.ndarray
    gramian: np.ndarray
    eigenvalues: np.ndarray
    waveform: np.ndarray
    score: float
    metrics: dict[str, float]
    solver_status: str | None = None
    upper_bound: float | None = None


@dataclass
class PersistentExcitationDesign:
    """Finite-dimensional PE design problem from the stage-1 appendix."""

    basis: Basis
    pe_order: int
    sobolev_order: int
    signal_dim: int = 1
    derivative_bound: float = 1.0
    n_grid: int = 2001
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    grid_np: np.ndarray = field(init=False, repr=False)
    q_np: dict[tuple[int, int], np.ndarray] = field(init=False, repr=False)
    q_torch: dict[tuple[int, int], torch.Tensor] = field(init=False, repr=False)
    basis_tables_np: dict[int, np.ndarray] = field(init=False, repr=False)
    constraint_weights_np: np.ndarray = field(init=False, repr=False)
    sobolev_r_np: np.ndarray = field(init=False, repr=False)
    sobolev_r_torch: torch.Tensor = field(init=False, repr=False)
    r_np: np.ndarray = field(init=False, repr=False)
    r_torch: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.signal_dim <= 0:
            raise ValueError("signal_dim must be positive.")
        if self.pe_order <= 0:
            raise ValueError("pe_order must be positive.")
        if self.sobolev_order < self.pe_order - 1:
            raise ValueError("sobolev_order must satisfy sobolev_order >= pe_order - 1.")
        if self.derivative_bound <= 0.0:
            raise ValueError("derivative_bound must be positive.")
        if self.n_grid < 3:
            raise ValueError("n_grid must be at least 3.")

        self.grid_np = np.linspace(0.0, self.basis.horizon, self.n_grid, dtype=np.float64)
        weights = self._trapezoidal_weights(self.grid_np)

        self.basis_tables_np = {
            order: self.basis.evaluate_derivative(order=order, grid=self.grid_np)
            for order in range(self.sobolev_order + 1)
        }

        self.q_np = {}
        for i in range(self.sobolev_order + 1):
            for j in range(self.sobolev_order + 1):
                left = self.basis_tables_np[i] * weights[None, :]
                self.q_np[(i, j)] = left @ self.basis_tables_np[j].T

        self.constraint_weights_np = self._constraint_weights()
        self.sobolev_r_np = sum(self.q_np[(k, k)] for k in range(self.sobolev_order + 1))
        self.sobolev_r_np = 0.5 * (self.sobolev_r_np + self.sobolev_r_np.T)
        self.r_np = sum(
            self.constraint_weights_np[k] * self.q_np[(k, k)] for k in range(self.sobolev_order + 1)
        )
        self.r_np = 0.5 * (self.r_np + self.r_np.T)

        self.q_torch = {
            key: torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for key, value in self.q_np.items()
        }
        self.sobolev_r_torch = torch.as_tensor(self.sobolev_r_np, dtype=self.dtype, device=self.device)
        self.r_torch = torch.as_tensor(self.r_np, dtype=self.dtype, device=self.device)

    @staticmethod
    def _trapezoidal_weights(grid: np.ndarray) -> np.ndarray:
        dt = grid[1] - grid[0]
        weights = np.full_like(grid, dt)
        weights[0] = 0.5 * dt
        weights[-1] = 0.5 * dt
        return weights

    @property
    def n_basis(self) -> int:
        return self.basis.n_basis

    @property
    def n_coefficients(self) -> int:
        return self.signal_dim * self.n_basis

    def _constraint_weights(self) -> np.ndarray:
        """Return the diagonal Sobolev weights used in the quadratic budget."""
        weights = np.ones(self.sobolev_order + 1, dtype=np.float64)
        if self.sobolev_order >= 1:
            weights[1] = 1.0 / (self.derivative_bound ** 2)
        return weights

    def _reshape_coefficients_numpy(self, coefficients: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coeffs.size != self.n_coefficients:
            raise ValueError(
                f"Expected {self.n_coefficients} coefficients for signal_dim={self.signal_dim} "
                f"and n_basis={self.n_basis}, got {coeffs.size}."
            )
        return coeffs.reshape(self.signal_dim, self.n_basis)

    def _reshape_coefficients_torch(self, coefficients: torch.Tensor) -> torch.Tensor:
        coeffs = coefficients.reshape(-1)
        if coeffs.numel() != self.n_coefficients:
            raise ValueError(
                f"Expected {self.n_coefficients} coefficients for signal_dim={self.signal_dim} "
                f"and n_basis={self.n_basis}, got {coeffs.numel()}."
            )
        return coeffs.reshape(self.signal_dim, self.n_basis)

    @staticmethod
    def _orientation_sign(values: np.ndarray, eps: float = 1e-12) -> float:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return 1.0
        index = int(np.argmax(np.abs(flat)))
        if abs(float(flat[index])) <= eps:
            return 1.0
        return 1.0 if float(flat[index]) >= 0.0 else -1.0

    def canonicalize_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        """Fix the global sign ambiguity by orienting the dominant sample positively."""
        coeffs = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        waveform = self.waveform(coeffs)
        return self._orientation_sign(waveform) * coeffs

    def normalize_coefficients(self, coefficients: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Project coefficients to the Sobolev unit sphere."""
        coeffs = self._reshape_coefficients_torch(coefficients)
        norm_sq = torch.einsum("ai,ij,aj->", coeffs, self.r_torch, coeffs).clamp_min(eps)
        return (coeffs / torch.sqrt(norm_sq)).reshape(-1)

    def constraint_value(self, coefficients: np.ndarray) -> float:
        """Return the weighted quadratic budget value."""
        coeffs = self._reshape_coefficients_numpy(coefficients)
        return float(np.einsum("ai,ij,aj->", coeffs, self.r_np, coeffs))

    def sobolev_value(self, coefficients: np.ndarray) -> float:
        """Return the standard unweighted ``H^s`` quadratic value."""
        coeffs = self._reshape_coefficients_numpy(coefficients)
        return float(np.einsum("ai,ij,aj->", coeffs, self.sobolev_r_np, coeffs))

    def sobolev_norm(self, coefficients: np.ndarray) -> float:
        """Return the standard unweighted ``H^s`` norm."""
        return float(np.sqrt(max(self.sobolev_value(coefficients), 0.0)))

    def gramian(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Return the lifted PE Gramian for the coefficient vector."""
        coeffs = self._reshape_coefficients_torch(coefficients)
        rows: list[torch.Tensor] = []
        for i in range(self.pe_order):
            entries = [
                torch.einsum("ai,ij,aj->", coeffs, self.q_torch[(i, j)], coeffs)
                for j in range(self.pe_order)
            ]
            rows.append(torch.stack(entries))
        gramian = torch.stack(rows)
        return 0.5 * (gramian + gramian.T)

    def gramian_numpy(self, coefficients: np.ndarray) -> np.ndarray:
        """NumPy wrapper for the PE Gramian."""
        coeffs = self._reshape_coefficients_numpy(coefficients)
        gramian = np.empty((self.pe_order, self.pe_order), dtype=np.float64)
        for i in range(self.pe_order):
            for j in range(self.pe_order):
                gramian[i, j] = np.einsum("ai,ij,aj->", coeffs, self.q_np[(i, j)], coeffs)
        return 0.5 * (gramian + gramian.T)

    def waveform(self, coefficients: np.ndarray, derivative_order: int = 0) -> np.ndarray:
        """Evaluate the waveform or one of its derivatives on the grid."""
        coeffs = self._reshape_coefficients_numpy(coefficients)
        values = coeffs @ self.basis_tables_np[derivative_order]
        if self.signal_dim == 1:
            return values.reshape(-1)
        return values

    def optimize_backprop(
        self,
        criterion: str,
        *,
        steps: int = 800,
        lr: float = 5e-2,
        restarts: int = 8,
        jitter: float = 1e-6,
        seed: int = 0,
        init: np.ndarray | None = None,
    ) -> DesignResult:
        """Optimize basis coefficients with torch autodiff."""
        criterion = canonicalize_criterion(criterion)
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if restarts <= 0:
            raise ValueError("restarts must be positive.")

        best_score = float("-inf")
        best_coeffs: torch.Tensor | None = None

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        initializations: list[torch.Tensor] = []
        if init is not None:
            initializations.append(torch.as_tensor(init, dtype=self.dtype, device=self.device))
        while len(initializations) < restarts:
            initializations.append(
                torch.randn(
                    self.n_coefficients,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        for theta_init in initializations:
            theta = torch.nn.Parameter(theta_init.clone())
            optimizer = torch.optim.Adam([theta], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()
                coeffs = self.normalize_coefficients(theta)
                gramian = self.gramian(coeffs)
                objective = criterion_objective(gramian, criterion=criterion, jitter=jitter)
                (-objective).backward()
                optimizer.step()

            with torch.no_grad():
                coeffs = self.normalize_coefficients(theta)
                gramian = self.gramian(coeffs)
                score = float(criterion_objective(gramian, criterion=criterion, jitter=jitter).item())
                if score > best_score:
                    best_score = score
                    best_coeffs = coeffs.detach().clone()

        if best_coeffs is None:
            raise RuntimeError("Backprop optimization did not produce a solution.")

        return self._build_result(
            coefficients=best_coeffs.cpu().numpy(),
            criterion=criterion,
            method="backprop",
            score=best_score,
        )

    def solve_sdp_relaxation(
        self,
        criterion: str,
        *,
        jitter: float = 1e-6,
        solver: str | None = None,
        verbose: bool = False,
        refine_steps: int = 0,
        refine_lr: float = 5e-2,
        solver_options: dict[str, Any] | None = None,
    ) -> DesignResult:
        """Solve the lifted SDP, round it, and optionally refine with backprop."""
        criterion = canonicalize_criterion(criterion)
        solver_options = {} if solver_options is None else dict(solver_options)

        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError(
                "cvxpy is required for the SDP relaxation. Install it with "
                "`uv sync --extra exp_design`."
            ) from exc

        total_coeffs = self.n_coefficients
        block_r = np.kron(np.eye(self.signal_dim, dtype=np.float64), self.r_np)
        block_q = {
            key: np.kron(np.eye(self.signal_dim, dtype=np.float64), value)
            for key, value in self.q_np.items()
        }

        X = cp.Variable((total_coeffs, total_coeffs), PSD=True)
        rows = []
        for i in range(self.pe_order):
            rows.append(cp.hstack([cp.trace(block_q[(i, j)] @ X) for j in range(self.pe_order)]))
        gramian = cp.vstack(rows)
        gramian = 0.5 * (gramian + gramian.T)
        eye = np.eye(self.pe_order)

        constraints = [cp.trace(block_r @ X) <= 1.0]
        if criterion == "e":
            gamma = cp.Variable()
            constraints.append(gramian - gamma * eye >> 0)
            objective = cp.Maximize(gamma)
        elif criterion == "a":
            objective = cp.Maximize(-cp.tr_inv(gramian + jitter * eye))
        elif criterion == "d":
            objective = cp.Maximize(cp.log_det(gramian + jitter * eye))
        elif criterion == "t":
            objective = cp.Maximize(cp.trace(gramian))
        else:
            raise AssertionError("Unreachable.")

        problem = cp.Problem(objective, constraints)
        solver_candidates = [solver] if solver is not None else self._default_solvers(criterion)
        last_error: Exception | None = None
        last_status: str | None = None

        for solver_name in solver_candidates:
            options = dict(solver_options)
            if solver_name == "SCS":
                options.setdefault("eps", 1e-6)
                options.setdefault("max_iters", 10_000)

            try:
                with warnings.catch_warnings():
                    if not verbose:
                        warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
                    problem.solve(solver=solver_name, verbose=verbose, **options)
            except Exception as exc:
                last_error = exc
                continue

            last_status = problem.status
            if problem.status in {"optimal", "optimal_inaccurate"} and np.isfinite(problem.value):
                break
        else:
            if last_error is not None:
                raise RuntimeError(
                    f"SDP solve failed for criterion {criterion} with solvers {solver_candidates}."
                ) from last_error
            raise RuntimeError(f"SDP solve failed with status {last_status}.")

        x_value = np.asarray(X.value, dtype=np.float64)
        x_value = 0.5 * (x_value + x_value.T)
        eigenvalues, eigenvectors = np.linalg.eigh(x_value)
        top_index = int(np.argmax(eigenvalues))
        rounded = np.sqrt(max(eigenvalues[top_index], 0.0)) * eigenvectors[:, top_index]

        norm_sq = self.constraint_value(rounded)
        if norm_sq <= 0.0:
            raise RuntimeError("Rounded SDP solution has zero Sobolev norm.")
        rounded = rounded / np.sqrt(norm_sq)

        if refine_steps > 0:
            refined = self.optimize_backprop(
                criterion=criterion,
                steps=refine_steps,
                lr=refine_lr,
                restarts=1,
                init=rounded,
            )
            refined.method = "sdp+backprop"
            refined.solver_status = problem.status
            refined.upper_bound = float(problem.value)
            return refined

        return self._build_result(
            coefficients=rounded,
            criterion=criterion,
            method="sdp",
            score=float(problem.value),
            solver_status=problem.status,
            upper_bound=float(problem.value),
        )

    @staticmethod
    def _default_solvers(criterion: str) -> list[str]:
        """Return a stable default solver order for each criterion."""
        if criterion in {"e", "t"}:
            return ["CLARABEL", "SCS"]
        return ["SCS", "CLARABEL"]

    def compare_criteria(
        self,
        criteria: tuple[str, ...] = CRITERIA,
        methods: tuple[str, ...] = ("backprop", "sdp"),
        *,
        backprop_kwargs: dict[str, Any] | None = None,
        sdp_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, DesignResult]]:
        """Run the requested criteria and methods."""
        backprop_kwargs = {} if backprop_kwargs is None else dict(backprop_kwargs)
        sdp_kwargs = {} if sdp_kwargs is None else dict(sdp_kwargs)

        results: dict[str, dict[str, DesignResult]] = {}
        for criterion in criteria:
            key = canonicalize_criterion(criterion)
            method_results: dict[str, DesignResult] = {}
            if "backprop" in methods:
                result = self.optimize_backprop(key, **backprop_kwargs)
                method_results[result.method] = result
            if "sdp" in methods:
                result = self.solve_sdp_relaxation(key, **sdp_kwargs)
                method_results[result.method] = result
            results[key] = method_results
        return results

    def _build_result(
        self,
        *,
        coefficients: np.ndarray,
        criterion: str,
        method: str,
        score: float,
        solver_status: str | None = None,
        upper_bound: float | None = None,
    ) -> DesignResult:
        coeffs = self.canonicalize_coefficients(coefficients)
        gramian = self.gramian_numpy(coeffs)
        torch_gramian = torch.as_tensor(gramian, dtype=self.dtype)
        return DesignResult(
            criterion=criterion,
            method=method,
            coefficients=coeffs,
            gramian=gramian,
            eigenvalues=np.linalg.eigvalsh(gramian),
            waveform=self.waveform(coeffs),
            score=float(score),
            metrics=criterion_metrics(torch_gramian),
            solver_status=solver_status,
            upper_bound=upper_bound,
        )
