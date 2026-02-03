"""Trainer class for backpropagation-based training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any, Union
from pathlib import Path
import time
import dataclasses
import os
import wandb

from ..models.base_model import BaseNeuroscienceModel
from ..metrics import fc_correlation, fc_mse, compute_all_fc_metrics
from ..metrics.metrics_store import MetricsStore
from .config import TrainingConfig


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
        experiment_name: str = "experiment",
        cfg: Optional[TrainingConfig] = None,
        use_wandb: bool = True
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
            cfg: Optional TrainingConfig for wandb logging
            use_wandb: Whether to use wandb logging
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.loss_fn_name = loss_fn
        self.cfg = cfg
        self.use_wandb = use_wandb
        
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
        
        # Initialize wandb if config provided
        self.wandb_run = None
        if self.use_wandb and cfg is not None:
            self._init_wandb(cfg)
    
    def _init_wandb(self, cfg: TrainingConfig):
        """Initialize wandb with proxy settings."""
                
        # Set proxy environment variables
        os.environ["HTTP_PROXY"] = "http://proxy.nhr.fau.de:80"
        os.environ["HTTPS_PROXY"] = "http://proxy.nhr.fau.de:80"
        
        # Configure wandb settings with proxy
        settings = wandb.Settings(
            _service_transport="http",
        )
        
        # Get run name from config or generate
        run_name = getattr(cfg, 'run_name', None) or self.experiment_name
        
        self.wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=getattr(cfg, 'wandb_entity', None),
            name=run_name,
            config=dataclasses.asdict(cfg),
            settings=settings
        )
        
        # Watch model for gradient logging
        wandb.watch(self.model, log="all", log_freq=100)
        
        print(f"Wandb initialized: {cfg.wandb_project}/{run_name}")
    
    def _log_wandb(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to wandb."""
        if self.wandb_run is not None:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict["epoch"] = step
            wandb.log(log_dict, step=step)
    
    def _log_wandb_artifact(self, filepath: str, name: str, artifact_type: str = "model"):
        """Log artifact to wandb."""
        if self.wandb_run is not None:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
    
    def _log_wandb_figure(self, fig, name: str, step: Optional[int] = None):
        """Log matplotlib figure to wandb."""
        if self.wandb_run is not None:
            wandb.log({name: wandb.Image(fig)}, step=step)
    
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
            
            # Log to wandb
            self._log_wandb(train_metrics, epoch, prefix="train")
            self._log_wandb(val_metrics, epoch, prefix="val")
            self._log_wandb({"best_val_loss": self.best_val_loss}, epoch, prefix="best")
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    checkpoint_path = f"best_{self.experiment_name}.pt"
                    self.save_checkpoint(checkpoint_path)
                    # Log best model as artifact to wandb
                    self._log_wandb_artifact(
                        str(self.checkpoint_dir / checkpoint_path),
                        name=f"best_model_{self.experiment_name}",
                        artifact_type="model"
                    )
                    # Log best metrics
                    self._log_wandb({
                        "best_epoch": epoch,
                        "best_val_loss": self.best_val_loss,
                        "best_val_fc_correlation": val_metrics['fc_correlation']
                    }, epoch, prefix="best")
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
                # Log early stopping to wandb
                self._log_wandb({"early_stopped": True, "final_epoch": epoch}, epoch)
                break
        
        # Save final metrics
        self.metrics_store.save()
        
        # Log final summary to wandb
        if self.wandb_run is not None:
            wandb.summary["best_epoch"] = self.best_epoch
            wandb.summary["best_val_loss"] = self.best_val_loss
            wandb.summary["total_epochs"] = epoch + 1
        
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
        
        # Log test metrics to wandb
        self._log_wandb(metrics, step=0, prefix="test")
        if self.wandb_run is not None:
            for k, v in metrics.items():
                wandb.summary[f"test_{k}"] = v
        
        return metrics
    
    def finish(self):
        """Finish training and cleanup wandb."""
        if self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
    
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
