"""Base model class for neuroscience models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pathlib import Path


class BaseNeuroscienceModel(nn.Module, ABC):
    """
    Abstract base class for neuroscience simulation models.
    
    All models should inherit from this class and implement:
    - forward(): Generate timeseries given initial conditions
    - compute_fc(): Compute functional connectivity from generated timeseries
    """
    
    CHECKPOINT_VERSION = 3
    REQUIRED_CHECKPOINT_KEYS = (
        "checkpoint_version",
        "model_class",
        "model_config",
        "model_state_dict",
    )

    def __init__(self, n_rois: int, device: str = "cpu"):
        """
        Initialize base model.
        
        Args:
            n_rois: Number of brain regions (ROIs)
            device: Device to run model on
        """
        super().__init__()
        self.n_rois = n_rois
        self.device = device
    
    @abstractmethod
    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Generate timeseries from initial conditions.
        
        Args:
            initial_state: Initial state of shape (batch, n_rois, state_dim)
            n_steps: Number of time steps to simulate
            dt: Time step size
            
        Returns:
            Simulated timeseries of shape (batch, n_rois, n_steps)
        """
        pass
    
    @abstractmethod
    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of model parameters."""
        pass

    def get_model_config(self) -> Dict[str, Any]:
        """
        Return model constructor config required for checkpoint reconstruction.

        Subclasses should override this when they require extra constructor args.
        """
        return {"n_rois": int(self.n_rois)}
    
    def compute_fc(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Compute functional connectivity from timeseries.
        
        Args:
            timeseries: Tensor of shape (batch, n_rois, n_timepoints)
            
        Returns:
            FC matrices of shape (batch, n_rois, n_rois)
        """
        # Center the timeseries
        ts_centered = timeseries - timeseries.mean(dim=2, keepdim=True)
        
        # Compute correlation
        std = ts_centered.std(dim=2, keepdim=True) + 1e-8
        ts_normalized = ts_centered / std
        
        # Batch matrix multiplication for correlation
        n_timepoints = timeseries.shape[2]
        fc = torch.bmm(ts_normalized, ts_normalized.transpose(1, 2)) / n_timepoints
        
        return fc
    
    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            metadata: Optional metadata to save with checkpoint
        """
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
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
            strict: Whether to enforce an exact parameter/key match
            
        Returns:
            Metadata from checkpoint if available
        """
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a dict: {filepath}")

        missing = [k for k in self.REQUIRED_CHECKPOINT_KEYS if k not in checkpoint]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Legacy or invalid checkpoint (missing: {missing_str}): {filepath}"
            )

        model_class = checkpoint['model_class']
        if model_class != self.__class__.__name__:
            raise ValueError(
                f"Checkpoint model class mismatch: expected {self.__class__.__name__}, "
                f"got {model_class}"
            )

        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        return checkpoint.get('metadata', {})
