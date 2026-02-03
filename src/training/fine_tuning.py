"""Fine-tuning capabilities for pre-trained models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any
from pathlib import Path

from ..models.base_model import BaseNeuroscienceModel
from ..metrics import fc_correlation, fc_mse
from ..metrics.metrics_store import MetricsStore
from .trainer import Trainer


class FineTuner:
    """
    Fine-tuner for pre-trained neuroscience models.
    
    Supports:
    - Loading from checkpoints
    - Freezing/unfreezing layers
    - Learning rate warmup
    - Discriminative learning rates
    """
    
    def __init__(
        self,
        model: BaseNeuroscienceModel,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model: Model to fine-tune
            checkpoint_path: Path to checkpoint to load
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)
        
        self.frozen_params: List[str] = []
    
    def load_pretrained(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load pretrained weights.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Metadata from checkpoint
        """
        print(f"Loading pretrained weights from {checkpoint_path}")
        metadata = self.model.load(checkpoint_path)
        return metadata
    
    def freeze_parameters(self, param_names: Optional[List[str]] = None):
        """
        Freeze specified parameters.
        
        Args:
            param_names: List of parameter names to freeze. If None, freeze all.
        """
        if param_names is None:
            param_names = [name for name, _ in self.model.named_parameters()]
        
        for name, param in self.model.named_parameters():
            if name in param_names:
                param.requires_grad = False
                self.frozen_params.append(name)
        
        print(f"Froze {len(self.frozen_params)} parameters")
    
    def unfreeze_parameters(self, param_names: Optional[List[str]] = None):
        """
        Unfreeze specified parameters.
        
        Args:
            param_names: List of parameter names to unfreeze. If None, unfreeze all.
        """
        if param_names is None:
            param_names = list(self.frozen_params)
        
        for name, param in self.model.named_parameters():
            if name in param_names:
                param.requires_grad = True
                if name in self.frozen_params:
                    self.frozen_params.remove(name)
        
        print(f"Unfroze parameters. {len(self.frozen_params)} still frozen")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_parameter_groups(
        self,
        base_lr: float = 1e-4,
        layer_decay: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups with discriminative learning rates.
        
        Args:
            base_lr: Base learning rate
            layer_decay: Decay factor for earlier layers
            
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        
        # Group parameters by their depth/position
        named_params = list(self.model.named_parameters())
        n_params = len(named_params)
        
        for i, (name, param) in enumerate(named_params):
            if param.requires_grad:
                # Earlier parameters get lower learning rate
                depth_factor = layer_decay ** (n_params - i - 1)
                lr = base_lr * depth_factor
                
                param_groups.append({
                    'params': [param],
                    'lr': lr,
                    'name': name
                })
        
        return param_groups
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        lr: float = 1e-4,
        warmup_epochs: int = 5,
        freeze_layers: Optional[List[str]] = None,
        use_discriminative_lr: bool = False,
        layer_decay: float = 0.9,
        n_steps: int = 100,
        dt: float = 0.01,
        experiment_name: str = "finetune"
    ) -> MetricsStore:
        """
        Fine-tune the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            lr: Learning rate
            warmup_epochs: Epochs for learning rate warmup
            freeze_layers: Layers to keep frozen
            use_discriminative_lr: Use different LRs for different layers
            layer_decay: Decay factor for discriminative LR
            n_steps: Simulation steps
            dt: Time step
            experiment_name: Name for experiment
            
        Returns:
            MetricsStore with training history
        """
        # Freeze layers if specified
        if freeze_layers is not None:
            self.freeze_parameters(freeze_layers)
        
        # Set up optimizer
        if use_discriminative_lr:
            param_groups = self.get_parameter_groups(lr, layer_decay)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
        else:
            trainable_params = self.get_trainable_parameters()
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            device=self.device,
            experiment_name=experiment_name
        )
        
        # Training loop with warmup
        metrics_store = MetricsStore(
            experiment_name=experiment_name,
            save_dir=f"results/metrics/{experiment_name}"
        )
        
        metrics_store.set_hyperparameters({
            'lr': lr,
            'warmup_epochs': warmup_epochs,
            'use_discriminative_lr': use_discriminative_lr,
            'layer_decay': layer_decay,
            'frozen_layers': freeze_layers,
            'n_trainable_params': len(self.get_trainable_parameters())
        })
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = trainer.train_epoch(train_loader, n_steps, dt)
            metrics_store.log_train(epoch, train_metrics)
            
            # Validate
            val_metrics = trainer.validate(val_loader, n_steps, dt)
            metrics_store.log_val(epoch, val_metrics)
            
            # Step scheduler
            scheduler.step()
            
            # Track best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                trainer.save_checkpoint(f"best_{experiment_name}.pt")
            
            if epoch % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"LR: {current_lr:.2e}")
        
        metrics_store.save()
        return metrics_store
    
    def gradual_unfreeze(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs_per_stage: int = 10,
        initial_lr: float = 1e-4,
        lr_decay: float = 0.5,
        n_steps: int = 100,
        dt: float = 0.01
    ) -> MetricsStore:
        """
        Gradually unfreeze layers during training.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs_per_stage: Epochs before unfreezing next layer
            initial_lr: Initial learning rate
            lr_decay: LR decay per stage
            n_steps: Simulation steps
            dt: Time step
            
        Returns:
            MetricsStore with training history
        """
        # Get all parameter names
        param_names = [name for name, _ in self.model.named_parameters()]
        
        # Start with all frozen
        self.freeze_parameters()
        
        metrics_store = MetricsStore(
            experiment_name="gradual_unfreeze",
            save_dir="results/metrics/gradual_unfreeze"
        )
        
        epoch_counter = 0
        current_lr = initial_lr
        
        # Gradually unfreeze from last to first
        for i, param_name in enumerate(reversed(param_names)):
            self.unfreeze_parameters([param_name])
            
            # Create optimizer with current trainable params
            trainable = self.get_trainable_parameters()
            optimizer = torch.optim.Adam(trainable, lr=current_lr)
            
            trainer = Trainer(
                model=self.model,
                optimizer=optimizer,
                device=self.device,
                experiment_name=f"gradual_stage_{i}"
            )
            
            print(f"\nStage {i}: Unfroze {param_name}, LR={current_lr:.2e}")
            
            for epoch in range(epochs_per_stage):
                train_metrics = trainer.train_epoch(train_loader, n_steps, dt)
                val_metrics = trainer.validate(val_loader, n_steps, dt)
                
                metrics_store.log_train(epoch_counter, train_metrics)
                metrics_store.log_val(epoch_counter, val_metrics)
                
                epoch_counter += 1
            
            # Decay learning rate
            current_lr *= lr_decay
        
        metrics_store.save()
        return metrics_store
