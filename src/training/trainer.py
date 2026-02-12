"""Trainer class for backpropagation-based training."""

import dataclasses
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..models.base_model import BaseNeuroscienceModel
from ..metrics import (
    compute_all_fc_metrics,
    compute_all_timeseries_metrics,
    compute_dynamics_fit_metrics,
)
from ..metrics.metrics_store import MetricsStore
from .config import TrainingConfig
from .losses import CompositeLoss, build_loss

FC_METRICS = ("loss", "fc_correlation", "fc_mse", "fc_upper_corr")
TS_METRICS = ("power_spectrum_distance", "temporal_correlation", "autocorr_distance")
DYN_METRICS = ("fcd_ks", "metastability_diff")
LOSS_COMPONENT_METRICS = (
    "loss_fc_mse", "loss_fc_corr",
    "loss_l2",
    "loss_hilbert_amp", "loss_hilbert_omega",
    "loss_fcd", "loss_metastability",
)
ALL_METRICS = FC_METRICS + TS_METRICS + DYN_METRICS + LOSS_COMPONENT_METRICS
WANDB_EPOCH_METRICS = ALL_METRICS + ("metrics_sampled_batches",)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def average(self, key: str) -> float:
        count = self.counts.get(key, 0)
        if count == 0:
            return float("nan")
        return self.sums[key] / count


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
            loss_fn: Loss function ("mse", "correlation", "combined", "fc_fcd_meta")
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
        self.metrics_sample_batches = cfg.metrics_sample_batches if cfg is not None else 1

        # Build composite loss from config weights
        dyn_kwargs = self._dynamics_kwargs_from_cfg(cfg)
        weight_overrides = self._weight_overrides_from_cfg(cfg)
        self.loss_obj = build_loss(
            name=loss_fn,
            weight_overrides=weight_overrides,
            dyn_kwargs=dyn_kwargs,
        )
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        # (validation happens inside build_loss)
        
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

        # Ensure all epoch metrics are tracked against epoch
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("best/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        
        # Watch model for gradient logging
        wandb.watch(self.model, log="all", log_freq=100)
        
        print(f"Wandb initialized: {cfg.wandb_project}/{run_name}")
    
    def _log_wandb(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to wandb."""
        if self.wandb_run is not None:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict["epoch"] = step
            wandb.log(log_dict, step=step)

    def _normalize_epoch_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Ensure a consistent set of epoch metrics for wandb logging."""
        normalized = {key: metrics.get(key, float("nan")) for key in WANDB_EPOCH_METRICS}
        for key, value in metrics.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    def _log_wandb_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        step: int
    ) -> None:
        """Log train/val metrics together for a single epoch."""
        if self.wandb_run is None:
            return
        train_metrics = self._normalize_epoch_metrics(train_metrics)
        val_metrics = self._normalize_epoch_metrics(val_metrics)
        log_dict = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "epoch": step
        }
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
    
    # -- static helpers for building CompositeLoss from config -----------

    @staticmethod
    def _dynamics_kwargs_from_cfg(cfg: Optional[TrainingConfig]) -> Dict[str, float]:
        """Extract dynamics parameters forwarded to loss terms."""
        if cfg is None:
            return {"tr": 0.72, "f_lo": 0.04, "f_hi": 0.07,
                    "fcd_win_sec": 60.0, "fcd_step_sec": 2.0}
        return {
            "tr": cfg.tr, "f_lo": cfg.f_lo, "f_hi": cfg.f_hi,
            "fcd_win_sec": cfg.fcd_win_sec, "fcd_step_sec": cfg.fcd_step_sec,
        }

    @staticmethod
    def _weight_overrides_from_cfg(cfg: Optional[TrainingConfig]) -> Dict[str, float]:
        """Collect per-term weight overrides from config."""
        if cfg is None:
            return {}
        overrides: Dict[str, float] = {}
        _MAP = {
            "loss_weight_fc": "fc_corr",
            "loss_weight_fc_mse": "fc_mse",
            "loss_weight_l2": "l2",
            "loss_weight_hilbert_amp": "hilbert_amp",
            "loss_weight_hilbert_omega": "hilbert_omega",
            "loss_weight_fcd": "fcd",
            "loss_weight_metastability": "metastability",
        }
        for attr, term in _MAP.items():
            val = getattr(cfg, attr, None)
            if val is not None:
                overrides[term] = val
        return overrides

    def _dynamics_kwargs(self) -> Dict[str, float]:
        """Convenience accessor used by metrics computation."""
        kw = self._dynamics_kwargs_from_cfg(self.cfg)
        if self.cfg is not None:
            kw["compute_fcd"] = self.cfg.compute_fcd_metrics
            kw["compute_metastability"] = self.cfg.compute_metastability_metrics
        else:
            kw["compute_fcd"] = True
            kw["compute_metastability"] = True
        return kw

    def _compute_loss(
        self,
        fc_pred: torch.Tensor,
        fc_targets: torch.Tensor,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Delegate to the :class:`CompositeLoss` object."""
        return self.loss_obj(fc_pred, fc_targets, ts_pred, ts_target)

    def _should_compute_expensive(self, batch_idx: int) -> bool:
        limit = self.metrics_sample_batches
        if limit is None:
            return True
        if limit <= 0:
            return False
        return batch_idx < limit

    def _compute_batch_metrics(
        self,
        fc_pred: torch.Tensor,
        fc_targets: torch.Tensor,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        loss: torch.Tensor,
        loss_components: Dict[str, torch.Tensor],
        compute_expensive: bool
    ) -> Dict[str, float]:
        metrics = compute_all_fc_metrics(fc_pred, fc_targets)
        metrics["loss"] = float(loss.item())
        for name, value in loss_components.items():
            metrics[name] = float(value.item())

        if compute_expensive:
            metrics.update(compute_all_timeseries_metrics(ts_pred, ts_target))
            dyn_metrics = compute_dynamics_fit_metrics(
                ts_pred,
                ts_target,
                **self._dynamics_kwargs()
            )
            metrics.update(dyn_metrics)

        return metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        n_steps: int,
        dt: float,
        epoch: int,
        n_epochs: int,
        train: bool,
        verbose: bool
    ) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        accumulator = _MetricAccumulator()
        sampled_batches = 0

        iterable = loader
        if verbose:
            phase = "train" if train else "val"
            iterable = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{n_epochs} [{phase}]",
                leave=False,
                dynamic_ncols=True
            )

        for batch_idx, batch in enumerate(iterable):
            windows, fc_targets, _ = batch
            windows = windows.to(self.device)
            fc_targets = fc_targets.to(self.device)

            batch_size = windows.shape[0]
            n_timepoints = windows.shape[2]
            n_sim_steps = min(n_steps, n_timepoints)
            if n_sim_steps <= 0:
                raise ValueError(f"n_steps must be >= 1, got {n_steps}")

            # Keep simulation horizon, loss window, and initial condition aligned.
            target_window = windows[:, :, :n_sim_steps]
            initial_state = target_window[:, :, 0]  # (batch, n_rois)

            if train:
                self.optimizer.zero_grad()

            if train:
                simulated = self.model.forward(
                    initial_state=initial_state,
                    n_steps=n_sim_steps,
                    dt=dt,
                    batch_size=batch_size
                )
                fc_pred = self.model.compute_fc(simulated)
                loss, loss_components = self._compute_loss(
                    fc_pred,
                    fc_targets,
                    simulated,
                    target_window
                )
            else:
                with torch.no_grad():
                    simulated = self.model.forward(
                        initial_state=initial_state,
                        n_steps=n_sim_steps,
                        dt=dt,
                        batch_size=batch_size
                    )
                    fc_pred = self.model.compute_fc(simulated)
                    loss, loss_components = self._compute_loss(
                        fc_pred,
                        fc_targets,
                        simulated,
                        target_window
                    )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            compute_expensive = self._should_compute_expensive(batch_idx)
            if compute_expensive:
                sampled_batches += 1

            with torch.no_grad():
                metrics = self._compute_batch_metrics(
                    fc_pred.detach(),
                    fc_targets,
                    simulated.detach(),
                    target_window,
                    loss,
                    loss_components,
                    compute_expensive
                )
            accumulator.update(metrics)

            if verbose:
                postfix = {
                    "loss": f"{accumulator.average('loss'):.4f}",
                    "fc_corr": f"{accumulator.average('fc_correlation'):.4f}"
                }
                iterable.set_postfix(postfix)

        metrics = {name: accumulator.average(name) for name in ALL_METRICS}
        if self.metrics_sample_batches is None:
            metrics["metrics_sampled_batches"] = len(loader)
        else:
            metrics["metrics_sampled_batches"] = sampled_batches

        return metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.01,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Train a single epoch (compatibility helper for fine-tuning)."""
        return self._run_epoch(
            loader=train_loader,
            n_steps=n_steps,
            dt=dt,
            epoch=0,
            n_epochs=1,
            train=True,
            verbose=verbose
        )

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.01,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Evaluate on validation data."""
        return self._run_epoch(
            loader=val_loader,
            n_steps=n_steps,
            dt=dt,
            epoch=0,
            n_epochs=1,
            train=False,
            verbose=verbose
        )
    
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
            'loss_weights': self.loss_obj.weights,
            'n_epochs': n_epochs,
            'n_steps': n_steps,
            'dt': dt,
            'model_class': self.model.__class__.__name__
        })
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self._run_epoch(
                loader=train_loader,
                n_steps=n_steps,
                dt=dt,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True,
                verbose=verbose
            )
            self.metrics_store.log_train(epoch, train_metrics)
            
            # Validate
            val_metrics = self._run_epoch(
                loader=val_loader,
                n_steps=n_steps,
                dt=dt,
                epoch=epoch,
                n_epochs=n_epochs,
                train=False,
                verbose=verbose
            )
            self.metrics_store.log_val(epoch, val_metrics)
            
            # Log to wandb
            self._log_wandb_epoch(train_metrics, val_metrics, epoch)
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
        # Keep test independent from validate() to avoid API compatibility issues.
        if hasattr(self, "validate") and callable(getattr(self, "validate")):
            metrics = self.validate(test_loader, n_steps, dt)
        else:
            metrics = self._run_epoch(
                loader=test_loader,
                n_steps=n_steps,
                dt=dt,
                epoch=0,
                n_epochs=1,
                train=False,
                verbose=False
            )
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
