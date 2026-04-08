"""Basis functions for smooth persistent-excitation design."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi

import numpy as np
from numpy.polynomial.legendre import Legendre


class Basis(ABC):
    """Abstract basis on ``[0, T]`` used to parameterize the waveform."""

    def __init__(self, n_basis: int, horizon: float):
        if n_basis <= 0:
            raise ValueError("n_basis must be positive.")
        if horizon <= 0.0:
            raise ValueError("horizon must be positive.")
        self.n_basis = int(n_basis)
        self.horizon = float(horizon)

    @property
    def name(self) -> str:
        """Human-readable basis name."""
        return self.__class__.__name__

    def evaluate(self, grid: np.ndarray) -> np.ndarray:
        """Return basis values on the time grid."""
        return self.evaluate_derivative(order=0, grid=grid)

    @abstractmethod
    def evaluate_derivative(self, order: int, grid: np.ndarray) -> np.ndarray:
        """Return the derivative table with shape ``(n_basis, len(grid))``."""


class FourierBasis(Basis):
    """L2-normalized trigonometric basis with one constant term."""

    def evaluate_derivative(self, order: int, grid: np.ndarray) -> np.ndarray:
        if order < 0:
            raise ValueError("order must be non-negative.")

        grid = np.asarray(grid, dtype=np.float64)
        values = np.zeros((self.n_basis, grid.size), dtype=np.float64)
        values[0] = 0.0 if order > 0 else np.full_like(grid, 1.0 / np.sqrt(self.horizon))

        harmonic = 1
        column = 1
        norm = np.sqrt(2.0 / self.horizon)
        while column < self.n_basis:
            omega = 2.0 * pi * harmonic / self.horizon
            if column < self.n_basis:
                values[column] = norm * (omega ** order) * np.sin(grid * omega + order * pi / 2.0)
                column += 1
            if column < self.n_basis:
                values[column] = norm * (omega ** order) * np.cos(grid * omega + order * pi / 2.0)
                column += 1
            harmonic += 1
        return values


class LegendreBasis(Basis):
    """Shifted Legendre basis, orthonormal in ``L2(0, T)``."""

    def evaluate_derivative(self, order: int, grid: np.ndarray) -> np.ndarray:
        if order < 0:
            raise ValueError("order must be non-negative.")

        grid = np.asarray(grid, dtype=np.float64)
        scaled_grid = (2.0 * grid / self.horizon) - 1.0
        values = np.zeros((self.n_basis, grid.size), dtype=np.float64)

        for index in range(self.n_basis):
            polynomial = Legendre.basis(index).deriv(order)
            normalization = np.sqrt((2 * index + 1) / self.horizon)
            chain_rule = (2.0 / self.horizon) ** order
            values[index] = normalization * chain_rule * polynomial(scaled_grid)

        return values
