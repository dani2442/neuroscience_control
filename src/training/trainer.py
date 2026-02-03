"""Trainer class for backpropagation-based training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
from pathlib import Path
import time

from ..models.base_model import BaseNeuroscienceModel
from ..metrics import fc_correlation, fc_mse, compute_all_fc_metrics
from ..metrics.metrics_store import MetricsStore


class Trainer:
    """
    Trainer for neuroscience models using backpropagation.
    
    Supports:
    - Mini-batch training with windowed data
    - Multiple loss functions
    - Early stopping
    - Checkpointing
    - Fine-tuning from pretrained models
    """
    
    def __init__(
        self,
        model: BaseNeuroscienceModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        loss_fn: str = "mse",
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "experiment"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optional optimizer (Adam by default)
            lr: Learning rate
            loss_fn: Loss function ("mse", "correlation", or "combined")
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            experiment_name: Name for experiment
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.loss_fn_name = loss_fn
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        # Set up loss function
        self.loss_fn = self._get_loss_fn(loss_fn)
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Metrics storage
        self.metrics_store = MetricsStore(
            experiment_name=experiment_name,
            save_dir=f"results/metrics/{experiment_name}"
        )
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _get_loss_fn(self, loss_fn: str) -> Callable:
        """Get loss function by name."""
        if loss_fn == "mse":
            return lambda pred, target: fc_mse(pred, target)
        elif loss_fn == "correlation":
            return lambda pred, target: 1 - fc_correlation(pred, target)
        elif loss_fn == "combined":
            return lambda pred, target: (
                fc_mse(pred, target) + 0.5 * (1 - fc_correlation(pred, target))
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            n_steps: Simulation steps
            dt: Time step
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_fc_corr = 0
        n_batches = 0
        
        for batch in train_loader:
            windows, fc_targets, _ = batch
            windows = windows.to(self.device)
            fc_targets = fc_targets.to(self.device)
            
            batch_size = windows.shape[0]
            n_timepoints = windows.shape[2]
            
            self.optimizer.zero_grad()
            
            # Simulate from model
            # Use first timepoint as initial condition for some models
            simulated = self.model.forward(
                initial_state=None,
                n_steps=n_timepoints,
                dt=dt,
                batch_size=batch_size
            )
            
            # Compute FC from simulated
            fc_pred = self.model.compute_fc(simulated)
            
            # Compute loss
            loss = self.loss_fn(fc_pred, fc_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                fc_corr = fc_correlation(fc_pred, fc_targets)
            
            total_loss += loss.item()
            total_fc_corr += fc_corr.item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'fc_correlation': total_fc_corr / n_batches
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            n_steps: Simulation steps
            dt: Time step
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_fc_corr = 0
        n_batches = 0
        
        for batch in val_loader:
            windows, fc_targets, _ = batch
            windows = windows.to(self.device)
            fc_targets = fc_targets.to(self.device)
            
            batch_size = windows.shape[0]
            n_timepoints = windows.shape[2]
            
            # Simulate
            simulated = self.model.forward(
                initial_state=None,
                n_steps=n_timepoints,
                dt=dt,
                batch_size=batch_size
            )
            
            # Compute FC
            fc_pred = self.model.compute_fc(simulated)
            
            # Compute metrics
            loss = self.loss_fn(fc_pred, fc_targets)
            fc_corr = fc_correlation(fc_pred, fc_targets)
            
            total_loss += loss.item()
            total_fc_corr += fc_corr.item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'fc_correlation': total_fc_corr / n_batches
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        n_steps: int = 100,
        dt: float = 0.01,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        verbose: bool = True
    ) -> MetricsStore:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            n_steps: Simulation steps per sample
            dt: Time step
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            verbose: Print progress
            
        Returns:
            MetricsStore with training history
        """
        # Store hyperparameters
        self.metrics_store.set_hyperparameters({
            'lr': self.lr,
            'loss_fn': self.loss_fn_name,
            'n_epochs': n_epochs,
            'n_steps': n_steps,
            'dt': dt,
            'model_class': self.model.__class__.__name__
        })
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, n_steps, dt)
            self.metrics_store.log_train(epoch, train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader, n_steps, dt)
            self.metrics_store.log_val(epoch, val_metrics)
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(f"best_{self.experiment_name}.pt")
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val FC Corr: {val_metrics['fc_correlation']:.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Save final metrics
        self.metrics_store.save()
        
        return self.metrics_store
    
    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            n_steps: Simulation steps
            dt: Time step
            
        Returns:
            Test metrics
        """
        metrics = self.validate(test_loader, n_steps, dt)
        self.metrics_store.log_test(metrics)
        return metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename
        self.model.save(
            str(filepath),
            metadata={
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch
            }
        )
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename
        metadata = self.model.load(str(filepath))
        
        if 'optimizer_state_dict' in metadata:
            self.optimizer.load_state_dict(metadata['optimizer_state_dict'])
        
        self.best_val_loss = metadata.get('best_val_loss', float('inf'))
        self.best_epoch = metadata.get('best_epoch', 0)
