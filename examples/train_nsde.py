#!/usr/bin/env python3
"""
Training script for Neural SDE Model.

This script trains the Neural SDE model via backpropagation
and logs results to Weights & Biases.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import dataclasses
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set up proxy before importing wandb
os.environ["HTTP_PROXY"] = "http://proxy.nhr.fau.de:80"
os.environ["HTTPS_PROXY"] = "http://proxy.nhr.fau.de:80"

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

# Import project modules
from src.dataset import NeuroscienceDataset, create_data_loaders
from src.models import NeuralSDE
from src.metrics import compute_all_fc_metrics
from src.training import Trainer, FineTuner, NeuralSDEConfig
from src.utils import (
    plot_fc_comparison,
    plot_timeseries,
    plot_training_curves,
    FIGURES_DIR
)


def setup_device(device: str = "auto") -> str:
    """Set up computation device."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU")
    return device


def load_data(cfg: NeuralSDEConfig, device: str):
    """Load and prepare dataset."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading and Processing Data")
    print('='*60)
    
    dataset = NeuroscienceDataset(
        filepath=cfg.data_path,
        normalize=True,
        device=device
    )
    
    print(f"Loaded dataset:")
    print(f"  - Number of subjects: {dataset.n_subjects}")
    print(f"  - Number of ROIs: {dataset.n_rois}")
    print(f"  - Number of timepoints: {dataset.n_timepoints}")
    print(f"  - FC matrix shape: {dataset.fc_mean.shape}")
    
    return dataset


def create_loaders(dataset, cfg: NeuralSDEConfig, device: str):
    """Create data loaders."""
    window_size = min(cfg.window_size, dataset.n_timepoints // 4)
    print(f"Window size: {window_size}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        stride=cfg.stride or window_size // 2,
        batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        device=device
    )
    
    print(f"Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, window_size


def train_neural_sde(dataset, train_loader, val_loader, test_loader, 
                     window_size: int, cfg: NeuralSDEConfig, device: str):
    """Train Neural SDE model using backpropagation."""
    print(f"\n{'='*60}")
    print("STEP 2: Training Neural SDE Model (Backpropagation)")
    print('='*60)
    
    # Create Neural SDE model
    sde_model = NeuralSDE(
        n_rois=dataset.n_rois,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        device=device
    )
    
    print(f"\nNeural SDE model created:")
    print(f"  - Parameters: {sum(p.numel() for p in sde_model.parameters())}")
    
    # Create trainer with wandb integration
    trainer = Trainer(
        model=sde_model,
        lr=cfg.lr,
        loss_fn=cfg.loss_fn,
        device=device,
        experiment_name=cfg.experiment_name,
        cfg=cfg,
        use_wandb=cfg.use_wandb
    )
    
    # Train
    print(f"\nTraining for {cfg.n_epochs} epochs...")
    
    metrics_store = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.n_epochs,
        n_steps=window_size,
        dt=cfg.dt,
        early_stopping_patience=cfg.early_stopping_patience,
        verbose=True
    )
    
    # Test
    test_metrics = trainer.test(test_loader, n_steps=window_size)
    print(f"\nTest metrics: {test_metrics}")
    
    return sde_model, trainer, metrics_store, test_metrics


def fine_tune_model(sde_model, train_loader, val_loader, 
                    window_size: int, cfg: NeuralSDEConfig, device: str):
    """Fine-tune trained model."""
    if not cfg.fine_tune:
        return sde_model, None
    
    print(f"\n{'='*60}")
    print("STEP 3: Fine-tuning Neural SDE Model")
    print('='*60)
    
    fine_tuner = FineTuner(
        model=sde_model,
        device=device
    )
    
    print("Fine-tuning with lower learning rate...")
    ft_metrics = fine_tuner.fine_tune(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.fine_tune_epochs,
        lr=cfg.fine_tune_lr,
        warmup_epochs=cfg.warmup_epochs,
        n_steps=window_size,
        experiment_name=f"{cfg.experiment_name}_finetuned"
    )
    
    # Log fine-tuning metrics to wandb
    if HAS_WANDB and cfg.use_wandb and wandb.run is not None:
        wandb.log({"fine_tuning/completed": True})
    
    return sde_model, ft_metrics


def save_model_and_figures(sde_model, metrics_store, test_metrics, 
                           dataset, cfg: NeuralSDEConfig, device: str):
    """Save model and generate figures."""
    print(f"\n{'='*60}")
    print("STEP 4: Saving Model and Generating Figures")
    print('='*60)
    
    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)
    
    # Save model checkpoint
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"nsde_best_{cfg.run_name}.pt"
    sde_model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")
    
    # Log model artifact to wandb
    if HAS_WANDB and cfg.use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(f"nsde_model_{cfg.run_name}", type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    
    # Generate figures
    with torch.no_grad():
        sde_ts = sde_model.forward(n_steps=n_timepoints, batch_size=1)
        sde_fc = sde_model.compute_fc(sde_ts)[0]
    
    # Final metrics
    final_metrics = compute_all_fc_metrics(sde_fc.unsqueeze(0), target_fc.unsqueeze(0))
    print(f"Final metrics: {final_metrics}")
    
    # Log final metrics to wandb
    if HAS_WANDB and cfg.use_wandb and wandb.run is not None:
        for k, v in final_metrics.items():
            wandb.summary[f"final_{k}"] = v
    
    # FC comparison
    fig = plot_fc_comparison(
        sde_fc, target_fc,
        title="Neural SDE Model - FC Comparison",
        default_name="nsde_fc_comparison",
        use_pdf=True
    )
    
    if HAS_WANDB and cfg.use_wandb and wandb.run is not None:
        wandb.log({"figures/fc_comparison": wandb.Image(fig)})
    plt.close()
    
    # Timeseries plot
    fig = plot_timeseries(
        sde_ts[0],
        n_rois=5,
        title="Neural SDE - Simulated Timeseries",
        default_name="nsde_timeseries",
        use_pdf=True
    )
    
    if HAS_WANDB and cfg.use_wandb and wandb.run is not None:
        wandb.log({"figures/timeseries": wandb.Image(fig)})
    plt.close()
    
    # Training curves
    fig = plot_training_curves(
        metrics_store,
        default_name="nsde_training_curves",
        use_pdf=True
    )
    
    if HAS_WANDB and cfg.use_wandb and wandb.run is not None:
        wandb.log({"figures/training_curves": wandb.Image(fig)})
    plt.close()
    
    print(f"Figures saved to {FIGURES_DIR}")
    
    return checkpoint_path, final_metrics


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Neural SDE Model")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat",
                        help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control",
                        help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default="neural_sde",
                        help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Enable fine-tuning after training")
    args = parser.parse_args()
    
    print("="*60)
    print("NEURAL SDE MODEL TRAINING")
    print("="*60)
    
    # Create config
    cfg = NeuralSDEConfig(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=args.device,
        seed=args.seed,
        dt=0.1,
        n_epochs=args.n_epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        fine_tune=args.fine_tune
    )
    
    # Setup
    device = setup_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    trainer = None
    
    try:
        # Step 1: Load data
        dataset = load_data(cfg, device)
        
        # Create data loaders
        train_loader, val_loader, test_loader, window_size = create_loaders(dataset, cfg, device)
        
        # Step 2: Train Neural SDE
        sde_model, trainer, metrics_store, test_metrics = train_neural_sde(
            dataset, train_loader, val_loader, test_loader, 
            window_size, cfg, device
        )
        
        # Step 3: Fine-tune (optional)
        sde_model, ft_metrics = fine_tune_model(
            sde_model, train_loader, val_loader,
            window_size, cfg, device
        )
        
        # Step 4: Save model and figures
        checkpoint_path, final_metrics = save_model_and_figures(
            sde_model, metrics_store, test_metrics,
            dataset, cfg, device
        )
        
        print("\n" + "="*60)
        print("NEURAL SDE TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nTest metrics: {test_metrics}")
        print(f"Final metrics: {final_metrics}")
        print(f"Model saved to: {checkpoint_path}")
        print(f"Figures saved to: {FIGURES_DIR}")
        
        return {
            "model": sde_model,
            "metrics": final_metrics,
            "test_metrics": test_metrics,
            "checkpoint": checkpoint_path
        }
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Finish wandb run
        if trainer is not None:
            trainer.finish()
        elif HAS_WANDB and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
