"""Base model class for neuroscience models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pathlib import Path


class BaseNeuroscienceModel(nn.Module, ABC):
    """Abstract base class for neuroscience simulation models.

    All models take a **complex** initial state and produce complex timeseries.
    FC is computed from the real part of the output.
    """

    CHECKPOINT_VERSION = 3
    REQUIRED_CHECKPOINT_KEYS = (
        "checkpoint_version",
        "model_class",
        "model_config",
        "model_state_dict",
    )

    def __init__(self, n_rois: int, device: str = "cpu"):
        super().__init__()
        self.n_rois = n_rois
        self.device = device

    @abstractmethod
    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        dt: float = 0.01,
    ) -> torch.Tensor:
        """Generate complex timeseries from complex initial conditions.

        Args:
            initial_state: Complex tensor of shape (batch, n_rois).
            n_steps: Number of time steps to simulate.
            dt: Time step size.

        Returns:
            Complex timeseries of shape (batch, n_rois, n_steps).
        """
        pass

    @abstractmethod
    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of model parameters."""
        pass

    def get_model_config(self) -> Dict[str, Any]:
        """Return constructor config for checkpoint reconstruction."""
        return {"n_rois": int(self.n_rois)}

    def compute_fc(self, timeseries: torch.Tensor) -> torch.Tensor:
        """Compute FC (Pearson correlation) from the real part of timeseries.

        Args:
            timeseries: (batch, n_rois, n_timepoints), real or complex.

        Returns:
            FC matrices of shape (batch, n_rois, n_rois).
        """
        ts = timeseries.real if torch.is_complex(timeseries) else timeseries
        ts_centered = ts - ts.mean(dim=2, keepdim=True)
        std = ts_centered.std(dim=2, keepdim=True) + 1e-8
        ts_norm = ts_centered / std
        return torch.bmm(ts_norm, ts_norm.transpose(1, 2)) / ts.shape[2]

    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            'checkpoint_version': self.CHECKPOINT_VERSION,
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_config': self.get_model_config(),
        }
        if metadata is not None:
            checkpoint['metadata'] = metadata
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)

    def load(self, filepath: str, strict: bool = True) -> Dict[str, Any]:
        """Load model checkpoint. Returns metadata dict."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a dict: {filepath}")

        missing = [k for k in self.REQUIRED_CHECKPOINT_KEYS if k not in checkpoint]
        if missing:
            raise ValueError(f"Invalid checkpoint (missing: {', '.join(missing)}): {filepath}")

        if checkpoint['model_class'] != self.__class__.__name__:
            raise ValueError(
                f"Model class mismatch: expected {self.__class__.__name__}, "
                f"got {checkpoint['model_class']}"
            )

        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        return checkpoint.get('metadata', {})
