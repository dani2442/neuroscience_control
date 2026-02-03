#!/usr/bin/env python3
"""
Training script for Coupled Hopf Model.

This script trains the Coupled Hopf model via grid search
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

import wandb

# Import project modules
from src.dataset import NeuroscienceDataset
from src.models import CoupledHopfModel
from src.metrics import compute_all_fc_metrics
from src.training import grid_search_hopf, HopfConfig
from src.utils import (
    plot_fc_comparison,
    plot_timeseries,
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


def init_wandb(cfg: HopfConfig) -> None:
    """Initialize wandb with proxy settings."""
    settings = wandb.Settings(_service_transport="http")
    
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config=dataclasses.asdict(cfg),
        settings=settings,
        tags=["hopf", "grid_search"]
    )
    
    print(f"Wandb initialized: {cfg.wandb_project}/{cfg.run_name}")
    return run


def load_data(cfg: HopfConfig, device: str):
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


def train_hopf_grid_search(dataset, cfg: HopfConfig, device: str):
    """Train Hopf model using grid search."""
    print(f"\n{'='*60}")
    print("STEP 2: Training Coupled Hopf Model (Grid Search)")
    print('='*60)
    
    target_fc = dataset.fc_mean
    n_rois = dataset.n_rois
    n_timepoints = min(dataset.n_timepoints, 200)
    
    print(f"Grid search over {len(cfg.g_values) * len(cfg.a_values)} parameter combinations")
    print(f"  - G values: {cfg.g_values}")
    print(f"  - a values: {cfg.a_values}")
    
    best_params, hopf_model = grid_search_hopf(
        target_fc=target_fc,
        n_rois=n_rois,
        g_values=cfg.g_values,
        a_values=cfg.a_values,
        n_timepoints=n_timepoints,
        n_simulations=cfg.n_simulations,
        device=device
    )
    
    # Log grid search results to wandb
    if cfg.use_wandb and wandb.run is not None:
        wandb.log({
            "best_params/G": best_params.get('initial_g', 0),
            "best_params/a": best_params.get('initial_a', 0),
        })
    
    # Evaluate on full timeseries
    print("\nEvaluating best Hopf model...")
    with torch.no_grad():
        hopf_ts = hopf_model.forward(n_steps=n_timepoints, batch_size=10)
        hopf_fc = hopf_model.compute_fc(hopf_ts)
        hopf_fc_mean = hopf_fc.mean(dim=0)
    
    hopf_metrics = compute_all_fc_metrics(hopf_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
    print(f"Hopf model metrics: {hopf_metrics}")
    
    # Log final metrics to wandb
    if cfg.use_wandb and wandb.run is not None:
        for k, v in hopf_metrics.items():
            wandb.log({f"metrics/{k}": v})
        wandb.summary.update(hopf_metrics)
    
    return hopf_model, hopf_metrics, best_params


def save_model_and_figures(hopf_model, hopf_metrics, dataset, cfg: HopfConfig, device: str):
    """Save model and generate figures."""
    print(f"\n{'='*60}")
    print("STEP 3: Saving Model and Generating Figures")
    print('='*60)
    
    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)
    
    # Save model checkpoint
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"hopf_best_{cfg.run_name}.pt"
    hopf_model.save(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")
    
    # Log model artifact to wandb
    if cfg.use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(f"hopf_model_{cfg.run_name}", type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    
    # Generate figures
    with torch.no_grad():
        hopf_ts = hopf_model.forward(n_steps=n_timepoints, batch_size=1)
        hopf_fc = hopf_model.compute_fc(hopf_ts)[0]
    
    # FC comparison
    fig = plot_fc_comparison(
        hopf_fc, target_fc,
        title="Coupled Hopf Model - FC Comparison",
        default_name="hopf_fc_comparison",
        use_pdf=True
    )
    
    # Log figure to wandb
    if cfg.use_wandb and wandb.run is not None:
        wandb.log({"figures/fc_comparison": wandb.Image(fig)})
    plt.close()
    
    # Timeseries plot
    fig = plot_timeseries(
        hopf_ts[0],
        n_rois=5,
        title="Coupled Hopf - Simulated Timeseries",
        default_name="hopf_timeseries",
        use_pdf=True
    )
    
    if cfg.use_wandb and wandb.run is not None:
        wandb.log({"figures/timeseries": wandb.Image(fig)})
    plt.close()
    
    print(f"Figures saved to {FIGURES_DIR}")
    
    return checkpoint_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Coupled Hopf Model")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat", help="Path to data file")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control", help="Wandb project name")
    parser.add_argument("--experiment-name", type=str, default="hopf", help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("="*60)
    print("COUPLED HOPF MODEL TRAINING")
    print("="*60)
    
    # Create config
    cfg = HopfConfig(
        experiment_name=args.experiment_name,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=args.device,
        seed=args.seed,
        dt=0.1
    )
    
    # Setup
    device = setup_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Initialize wandb
    init_wandb(cfg)
    
    # Step 1: Load data
    dataset = load_data(cfg, device)
    
    # Step 2: Train via grid search
    hopf_model, hopf_metrics, best_params = train_hopf_grid_search(dataset, cfg, device)
    
    # Step 3: Save model and figures
    checkpoint_path = save_model_and_figures(hopf_model, hopf_metrics, dataset, cfg, device)
    
    print("\n" + "="*60)
    print("HOPF TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nBest parameters: {best_params}")
    print(f"Final metrics: {hopf_metrics}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Figures saved to: {FIGURES_DIR}")
    
    return {
        "model": hopf_model,
        "metrics": hopf_metrics,
        "params": best_params,
        "checkpoint": checkpoint_path
    }


if __name__ == "__main__":
    main()
