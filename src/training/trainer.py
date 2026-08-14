"""Trainer class for backpropagation-based training."""

import dataclasses
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..models.base_model import BaseNeuroscienceModel
from ..metrics.metrics_store import MetricsStore
from .config import TrainingConfig
from .evaluation import (
    EVAL_METRIC_KEYS,
    MetricAccumulator,
    accumulate_timeseries_metrics,
    build_eval_metrics,
    evaluate_timeseries_metrics,
    rollout_model,
)
from .losses import CompositeLoss


class Trainer:
    """
    Trainer for neuroscience models using backpropagation.

    Supports:
    - Mini-batch training with windowed data
    - Composite loss built from config loss_weights
    - Early stopping and checkpointing
    - Fine-tuning from pretrained models
    """

    def __init__(
        self,
        model: BaseNeuroscienceModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "experiment",
        cfg: Optional[TrainingConfig] = None,
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.sde_type = cfg.sde_type if cfg is not None else "stratonovich"
        self.sde_method = cfg.sde_method if cfg is not None else "reversible_heun"
        self.dt_min = cfg.dt_min if cfg is not None else 0.04
        self.use_adjoint = cfg.use_adjoint if cfg is not None else True
        if cfg is not None and cfg.adjoint_method is not None:
            self.adjoint_method = cfg.adjoint_method
        else:
            self.adjoint_method = (
                "adjoint_reversible_heun"
                if self.sde_method == "reversible_heun"
                else self.sde_method
            )

        # Dynamics parameters for FCD windowing
        tr = cfg.tr if cfg is not None else 0.72
        fcd_win_sec = cfg.fcd_win_sec if cfg is not None else 30.0
        fcd_step_sec = cfg.fcd_step_sec if cfg is not None else 2.0

        # Build composite loss from config loss_weights
        loss_weights = cfg.loss_weights if cfg is not None else {"fc_mse": 1.0}
        self.loss_fn = CompositeLoss(
            weights=loss_weights,
            tr=tr,
            fcd_win_sec=fcd_win_sec,
            fcd_step_sec=fcd_step_sec,
            fdm_n_pairs=getattr(cfg, "fdm_n_pairs", 32),
            fdm_max_lag=getattr(cfg, "fdm_max_lag", 50),
            fdm_sigma=getattr(cfg, "fdm_sigma", 1.0),
        )

        # Evaluation metric modules (always compute all, for logging)
        self.eval_metrics: List[torch.nn.Module] = build_eval_metrics(
            tr=tr,
            fcd_win_sec=fcd_win_sec,
            fcd_step_sec=fcd_step_sec,
        )

        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        # Process-unique name for the temporary best-model checkpoint so that
        # concurrent runs sharing an experiment_name (e.g. seed/ablation sweeps
        # launched in parallel) never read or overwrite each other's file.
        self._best_ckpt_name = f"best_{experiment_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}.pt"

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
        settings = wandb.Settings(
            _service_transport="http",
        )
        run_name = getattr(cfg, 'run_name', None) or self.experiment_name
        self.wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=getattr(cfg, 'wandb_entity', None),
            name=run_name,
            config=dataclasses.asdict(cfg),
            settings=settings
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("best/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("params/*", step_metric="epoch")

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
        all_keys = ("loss",) + EVAL_METRIC_KEYS + tuple(self.loss_fn.component_names)
        normalized = {key: metrics.get(key, float("nan")) for key in dict.fromkeys(all_keys)}
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

    def _compute_batch_metrics(
        self,
        ts_pred: torch.Tensor,
        ts_target: torch.Tensor,
        loss: torch.Tensor,
        loss_components: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        group_size = getattr(self.cfg, "group_size", 0) if self.cfg is not None else 0
        metrics = evaluate_timeseries_metrics(
            ts_pred, ts_target, self.eval_metrics, group_size=group_size,
        )
        metrics["loss"] = float(loss.item())
        for name, value in loss_components.items():
            metrics[name] = float(value.item())
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

        accumulator = MetricAccumulator()
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
            windows, _fc, _, control = batch
            windows = windows.to(self.device)
            control = control.to(self.device)
            ctrl_arg = control if control.shape[-1] > 0 else None

            n_timepoints = windows.shape[2]
            n_sim_steps = min(n_steps, n_timepoints)
            if n_sim_steps <= 0:
                raise ValueError(f"n_steps must be >= 1, got {n_steps}")

            target_window = windows[:, :, :n_sim_steps]
            initial_state = target_window[:, :, 0]

            if train:
                self.optimizer.zero_grad()

            denoise_f_lo = None
            denoise_f_hi = None
            if self.cfg is not None and (not train or self.cfg.denoise_predictions):
                denoise_f_lo = self.cfg.denoise_f_lo
                denoise_f_hi = self.cfg.denoise_f_hi

            if train:
                simulated = rollout_model(
                    self.model,
                    initial_state,
                    n_sim_steps,
                    dt,
                    control=ctrl_arg,
                    sde_type=self.sde_type,
                    method=self.sde_method,
                    dt_min=self.dt_min,
                    use_adjoint=self.use_adjoint,
                    adjoint_method=self.adjoint_method,
                    denoise_f_lo=denoise_f_lo,
                    denoise_f_hi=denoise_f_hi,
                )
                loss, loss_components = self.loss_fn(simulated, target_window)
            else:
                with torch.no_grad():
                    simulated = rollout_model(
                        self.model,
                        initial_state,
                        n_sim_steps,
                        dt,
                        control=ctrl_arg,
                        sde_type=self.sde_type,
                        method=self.sde_method,
                        dt_min=self.dt_min,
                        use_adjoint=self.use_adjoint,
                        adjoint_method=self.adjoint_method,
                        denoise_f_lo=denoise_f_lo,
                        denoise_f_hi=denoise_f_hi,
                    )
                    loss, loss_components = self.loss_fn(simulated, target_window)

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

        all_keys = sorted(set(("loss",) + EVAL_METRIC_KEYS + tuple(self.loss_fn.component_names))
                          | set(accumulator.sums.keys()))
        return {name: accumulator.average(name) for name in all_keys}

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
        """Full training loop."""
        self.metrics_store.set_hyperparameters({
            'lr': self.lr,
            'loss_weights': self.loss_fn.weights,
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

            self._log_wandb_epoch(train_metrics, val_metrics, epoch)
            self._log_wandb({"best_val_loss": self.best_val_loss}, epoch, prefix="best")

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                patience_counter = 0

                if save_best:
                    checkpoint_path = self._best_ckpt_name
                    self.save_checkpoint(checkpoint_path)
                    self._log_wandb_artifact(
                        str(self.checkpoint_dir / checkpoint_path),
                        name=f"best_model_{self.experiment_name}",
                        artifact_type="model"
                    )
                    self._log_wandb({
                        "best_epoch": epoch,
                        "best_val_loss": self.best_val_loss,
                        "best_val_fc_correlation": val_metrics['fc_correlation']
                    }, epoch, prefix="best")
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val fc_correlation: {val_metrics['fc_correlation']:.4f} | "
                      f"Time: {elapsed:.1f}s")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                self._log_wandb({"early_stopped": True, "final_epoch": epoch}, epoch)
                break

        self.metrics_store.save()

        if self.wandb_run is not None:
            wandb.summary["best_epoch"] = self.best_epoch
            wandb.summary["best_val_loss"] = self.best_val_loss
            wandb.summary["total_epochs"] = epoch + 1

        # Restore best weights and remove the temporary checkpoint file
        if save_best:
            best_ckpt = self.checkpoint_dir / self._best_ckpt_name
            if best_ckpt.exists():
                self.model.load(str(best_ckpt))
                best_ckpt.unlink()

        return self.metrics_store

    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
        n_steps: int = 100,
        dt: float = 0.72,
        label: str = "test",
    ) -> Dict[str, float]:
        """Evaluate on test set.

        Metrics are computed at the batch level and aggregated across
        repeated stochastic rollouts when configured.
        """
        num_simulations = max(1, getattr(self.cfg, "n_simulations", 1))
        simulation_accumulator = MetricAccumulator()

        for _ in range(num_simulations):
            batch_accumulator = MetricAccumulator()
            for batch in test_loader:
                windows, _fc, _, control = batch
                windows = windows.to(self.device)
                control = control.to(self.device)
                ctrl_arg = control if control.shape[-1] > 0 else None

                n_timepoints = windows.shape[2]
                n_sim_steps = min(n_steps, n_timepoints)
                if n_sim_steps <= 0:
                    raise ValueError(f"n_steps must be >= 1, got {n_steps}")

                target_window = windows[:, :, :n_sim_steps]
                initial_state = target_window[:, :, 0]
                simulated = rollout_model(
                    self.model,
                    initial_state,
                    n_sim_steps,
                    dt,
                    control=ctrl_arg,
                    sde_type=self.sde_type,
                    method=self.sde_method,
                    dt_min=self.dt_min,
                    use_adjoint=self.use_adjoint,
                    adjoint_method=self.adjoint_method,
                    denoise_f_lo=self.cfg.denoise_f_lo if self.cfg is not None else None,
                    denoise_f_hi=self.cfg.denoise_f_hi if self.cfg is not None else None,
                )

                group_size = getattr(self.cfg, "group_size", 0) if self.cfg is not None else 0
                accumulate_timeseries_metrics(
                    batch_accumulator,
                    simulated,
                    target_window,
                    self.eval_metrics,
                    group_size=group_size,
                )

                loss, loss_components = self.loss_fn(simulated, target_window)
                batch_loss_metrics: Dict[str, float] = {"loss": float(loss.item())}
                for name, value in loss_components.items():
                    batch_loss_metrics[name] = float(value.item())
                batch_accumulator.update(batch_loss_metrics)

            simulation_accumulator.update(batch_accumulator.summary())

        all_keys = sorted(
            set(("loss",) + EVAL_METRIC_KEYS + tuple(self.loss_fn.component_names))
            | set(simulation_accumulator.sums.keys())
        )
        metrics = {name: simulation_accumulator.average(name) for name in all_keys}
        metrics.update({f"{name}_std": simulation_accumulator.std(name) for name in all_keys})
        self.metrics_store.log_test(metrics, label=label)

        self._log_wandb(metrics, step=0, prefix=label)
        if self.wandb_run is not None:
            for k, v in metrics.items():
                wandb.summary[f"{label}_{k}"] = v

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
