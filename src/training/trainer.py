"""Trainer class for backpropagation-based training."""

import dataclasses
import math
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

FC_METRICS = ("loss", "fc_correlation", "fc_mse")
TS_METRICS = ("power_spectrum_distance", "temporal_correlation", "autocorr_distance")
DYN_METRICS = ("fcd_ks", "phfcd_ks", "phase_fc_correlation", "metastability_diff")
LOSS_COMPONENT_METRICS = (
    "loss_fc_mse", "loss_fc_correlation",
    "loss_l2",
    "loss_amplitude", "loss_omega",
    "loss_fcd", "loss_phfcd", "loss_phase_fc_correlation", "loss_metastability",
)
ALL_METRICS = FC_METRICS + TS_METRICS + DYN_METRICS + LOSS_COMPONENT_METRICS
WANDB_EPOCH_METRICS = ALL_METRICS


class _MetricAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.sum_squares: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            numeric = float(value)
            self.sums[key] = self.sums.get(key, 0.0) + numeric
            self.sum_squares[key] = self.sum_squares.get(key, 0.0) + (numeric * numeric)
            self.counts[key] = self.counts.get(key, 0) + 1

    def average(self, key: str) -> float:
        count = self.counts.get(key, 0)
        if count == 0:
            return float("nan")
        return self.sums[key] / count

    def std(self, key: str) -> float:
        count = self.counts.get(key, 0)
        if count == 0:
            return float("nan")
        mean = self.sums[key] / count
        mean_square = self.sum_squares[key] / count
        variance = max(0.0, mean_square - (mean * mean))
        return math.sqrt(variance)


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
        use_wandb: bool = True,
        extra_dyn_kwargs: Optional[Dict[str, object]] = None,
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
        self.sde_type = cfg.sde_type if cfg is not None else "stratonovich"
        self.sde_method = cfg.sde_method if cfg is not None else "reversible_heun"
        self.dt_min = cfg.dt_min if cfg is not None else 0.04
        self.use_adjoint = cfg.use_adjoint if cfg is not None else True
        if cfg is not None and cfg.adjoint_method is not None:
            self.adjoint_method = cfg.adjoint_method
        else:
            self.adjoint_method = "adjoint_reversible_heun" if self.sde_method == "reversible_heun" else self.sde_method

        # Build composite loss from config weights.
        dyn_kwargs = self._dynamics_kwargs_from_cfg(cfg)
        if extra_dyn_kwargs:
            dyn_kwargs.update(extra_dyn_kwargs)
        weight_overrides = self._weight_overrides_from_cfg(cfg)
        self.loss_obj = build_loss(
            name=loss_fn,
            weight_overrides=weight_overrides,
            dyn_kwargs=dyn_kwargs,
        )
        self.active_loss_component_metrics = tuple(
            f"loss_{name}" for name in self.loss_obj.weights.keys()
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
        # os.environ["HTTP_PROXY"] = "http://proxy.nhr.fau.de:80"
        # os.environ["HTTPS_PROXY"] = "http://proxy.nhr.fau.de:80"
        
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
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("best/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("params/*", step_metric="epoch")
        
        # Watch model for gradient / parameter histogram logging.
        # wandb.watch does not support complex-valued parameters — it casts
        # tensors to FloatTensor, silently discarding imaginary parts and
        # emitting a noisy UserWarning.  Skip watching when the model
        # contains complex parameters; scalar metrics are still logged
        # normally via _log_wandb.
        has_complex = any(p.is_complex() for p in self.model.parameters())
        if has_complex:
            print("  (skipping wandb.watch – model has complex parameters)")
        else:
            wandb.watch(self.model, log="all", log_freq=100)
        
        print(f"Wandb initialized: {cfg.wandb_project}/{run_name}")
    
    def _log_wandb(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to wandb."""
        if self.wandb_run is not None:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            log_dict["epoch"] = step
            wandb.log(log_dict)

    def _normalize_epoch_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Ensure a consistent set of epoch metrics for wandb logging."""
        expected_keys = tuple(
            dict.fromkeys(WANDB_EPOCH_METRICS + self.active_loss_component_metrics)
        )
        normalized = {key: metrics.get(key, float("nan")) for key in expected_keys}
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
            **{f"validation/{k}": v for k, v in val_metrics.items()},
            "epoch": step
        }
        # Log Hopf learnable parameters (a, G, kappa) per epoch when available.
        hopf_params = self._get_hopf_params()
        if hopf_params:
            log_dict.update({f"params/{k}": v for k, v in hopf_params.items()})
        wandb.log(log_dict)
    
    def _get_hopf_params(self) -> Dict[str, float]:
        """Extract current Hopf learnable parameters from the model (if any)."""
        params: Dict[str, float] = {}
        model = self.model
        if hasattr(model, "a"):
            a = model.a
            params["a"] = float(a.item() if a.dim() == 0 else a.mean().item())
        if hasattr(model, "g"):
            g = model.g
            params["G"] = float(g.item() if g.dim() == 0 else g.mean().item())
        if hasattr(model, "kappa"):
            kappa = model.kappa
            if isinstance(kappa, torch.Tensor):
                params["kappa"] = float(kappa.item() if kappa.dim() == 0 else kappa.mean().item())
            else:
                params["kappa"] = float(kappa)
        return params

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
            return {"tr": 0.72, "fcd_win_sec": 60.0, "fcd_step_sec": 2.0}
        return {
            "tr": cfg.tr,
            "fcd_win_sec": cfg.fcd_win_sec, "fcd_step_sec": cfg.fcd_step_sec,
        }

    @staticmethod
    def _weight_overrides_from_cfg(cfg: Optional[TrainingConfig]) -> Dict[str, float]:
        """Collect per-term weight overrides from config."""
        if cfg is None:
            return {}
        overrides: Dict[str, float] = {}
        _MAP = {
            "loss_weight_fc": "fc_correlation",
            "loss_weight_fc_mse": "fc_mse",
            "loss_weight_l2": "l2",
            "loss_weight_amplitude": "amplitude",
            "loss_weight_omega": "omega",
            "loss_weight_fcd": "fcd",
            "loss_weight_phfcd": "phfcd",
            "loss_weight_phase_fc_correlation": "phase_fc_correlation",
            "loss_weight_metastability": "metastability",
        }
        for attr, term in _MAP.items():
            val = getattr(cfg, attr, None)
            if val is not None:
                overrides[term] = val
        return overrides

    def _dynamics_kwargs(self) -> Dict[str, float]:
        """Convenience accessor used by metrics computation."""
        return self._dynamics_kwargs_from_cfg(self.cfg)

    def _compute_loss(
        self,
        fc_pred: torch.Tensor,
        fc_targets: torch.Tensor,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Delegate to the :class:`CompositeLoss` object."""
        return self.loss_obj(fc_pred, fc_targets, ts_pred, ts_target)

    def _compute_batch_metrics(
        self,
        fc_pred: torch.Tensor,
        fc_targets: torch.Tensor,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        loss: torch.Tensor,
        loss_components: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        metrics = compute_all_fc_metrics(fc_pred, fc_targets)
        metrics["loss"] = float(loss.item())
        for name, value in loss_components.items():
            metrics[name] = float(value.item())

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
        iterable = loader
        if verbose:
            phase = "train" if train else "val"
            iterable = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{n_epochs} [{phase}]",
                leave=False,
                dynamic_ncols=True
            )

        for batch in iterable:
            windows, fc_targets, _, control = batch
            windows = windows.to(self.device)
            fc_targets = fc_targets.to(self.device)
            control = control.to(self.device)
            # Treat empty control (n_control_dims==0) as None
            ctrl_arg = control if control.shape[-1] > 0 else None

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
                    sde_type=self.sde_type,
                    method=self.sde_method,
                    dt_min=self.dt_min,
                    use_adjoint=self.use_adjoint,
                    adjoint_method=self.adjoint_method,
                    control=ctrl_arg,
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
                        sde_type=self.sde_type,
                        method=self.sde_method,
                        dt_min=self.dt_min,
                        use_adjoint=self.use_adjoint,
                        adjoint_method=self.adjoint_method,
                        control=ctrl_arg,
                    )
                    fc_pred = self.model.compute_fc(simulated)
                    loss, loss_components = self._compute_loss(
                        fc_pred,
                        fc_targets,
                        simulated,
                        target_window
                    )

            if not torch.isfinite(simulated.real).all():
                raise RuntimeError(
                    "Non-finite simulated trajectory detected. "
                    "Try reducing --noise-sigma and/or --dt-min, or switch solver settings."
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. "
                    "Try reducing --noise-sigma and/or --dt-min, or switch solver settings."
                )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            with torch.no_grad():
                metrics = self._compute_batch_metrics(
                    fc_pred.detach(),
                    fc_targets,
                    simulated.detach(),
                    target_window,
                    loss,
                    loss_components,
                )
            accumulator.update(metrics)

            if verbose:
                postfix = {
                    "loss": f"{accumulator.average('loss'):.4f}",
                    "fc_correlation": f"{accumulator.average('fc_correlation'):.4f}"
                }
                iterable.set_postfix(postfix)

        metric_names = sorted(set(ALL_METRICS) | set(accumulator.sums.keys()))
        metrics = {name: accumulator.average(name) for name in metric_names}
        return metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.72,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Train a single epoch."""
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
        dt: float = 0.72,
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
        dt: float = 0.72,
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
            'sde_type': self.sde_type,
            'sde_method': self.sde_method,
            'dt_min': self.dt_min,
            'use_adjoint': self.use_adjoint,
            'adjoint_method': self.adjoint_method,
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
                      f"Val fc_correlation: {val_metrics['fc_correlation']:.4f} | "
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
        dt: float = 0.72
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
        accumulator = _MetricAccumulator()
        for batch in test_loader:
            windows, fc_targets, _, control = batch
            windows = windows.to(self.device)
            fc_targets = fc_targets.to(self.device)
            control = control.to(self.device)
            ctrl_arg = control if control.shape[-1] > 0 else None

            n_timepoints = windows.shape[2]
            n_sim_steps = min(n_steps, n_timepoints)
            if n_sim_steps <= 0:
                raise ValueError(f"n_steps must be >= 1, got {n_steps}")

            target_window = windows[:, :, :n_sim_steps]
            initial_state = target_window[:, :, 0]
            simulated = self.model.forward(
                initial_state=initial_state,
                n_steps=n_sim_steps,
                dt=dt,
                sde_type=self.sde_type,
                method=self.sde_method,
                dt_min=self.dt_min,
                use_adjoint=self.use_adjoint,
                adjoint_method=self.adjoint_method,
                control=ctrl_arg,
            )
            fc_pred = self.model.compute_fc(simulated)
            loss, loss_components = self._compute_loss(
                fc_pred,
                fc_targets,
                simulated,
                target_window,
            )
            batch_metrics = self._compute_batch_metrics(
                fc_pred,
                fc_targets,
                simulated,
                target_window,
                loss,
                loss_components,
            )
            accumulator.update(batch_metrics)

        metric_names = sorted(set(ALL_METRICS) | set(accumulator.sums.keys()))
        metrics = {name: accumulator.average(name) for name in metric_names}
        metrics.update({f"{name}_std": accumulator.std(name) for name in metric_names})
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
    
