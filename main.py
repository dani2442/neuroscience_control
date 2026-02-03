#!/usr/bin/env python3
"""
Main script for neuroscience control project.

This script demonstrates the complete workflow:
1. Load and process data
2. Train Hopf model via grid search
3. Train Neural SDE model via backpropagation
4. Evaluate both models
5. Compare and visualize results
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import project modules
from src.dataset import NeuroscienceDataset, create_data_loaders
from src.models import CoupledHopfModel, NeuralSDE
from src.metrics import compute_all_fc_metrics, MetricsStore
from src.training import Trainer, grid_search_hopf, FineTuner
from src.utils import (
    plot_fc_comparison,
    plot_training_curves,
    plot_model_comparison,
    plot_timeseries,
    create_comparison_report
)


def setup_device():
    """Set up computation device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def load_data(filepath: str, device: str):
    """Load and prepare dataset."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading and Processing Data")
    print('='*60)
    
    dataset = NeuroscienceDataset(
        filepath=filepath,
        normalize=True,
        device=device
    )
    
    print(f"Loaded dataset:")
    print(f"  - Number of subjects: {dataset.n_subjects}")
    print(f"  - Number of ROIs: {dataset.n_rois}")
    print(f"  - Number of timepoints: {dataset.n_timepoints}")
    print(f"  - FC matrix shape: {dataset.fc_mean.shape}")
    
    return dataset


def train_hopf_grid_search(dataset, device):
    """Train Hopf model using grid search."""
    print(f"\n{'='*60}")
    print("STEP 2: Training Coupled Hopf Model (Grid Search)")
    print('='*60)
    
    target_fc = dataset.fc_mean
    n_rois = dataset.n_rois
    n_timepoints = min(dataset.n_timepoints, 200)  # Use subset for speed
    
    # Grid search parameters
    g_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    a_values = [-0.05, -0.02, -0.01, 0.0, 0.01]
    
    print(f"Grid search over {len(g_values) * len(a_values)} parameter combinations")
    print(f"  - G values: {g_values}")
    print(f"  - a values: {a_values}")
    
    best_params, hopf_model = grid_search_hopf(
        target_fc=target_fc,
        n_rois=n_rois,
        g_values=g_values,
        a_values=a_values,
        n_timepoints=n_timepoints,
        n_simulations=5,
        device=device
    )
    
    # Evaluate on full timeseries
    print("\nEvaluating best Hopf model...")
    with torch.no_grad():
        hopf_ts = hopf_model.forward(n_steps=n_timepoints, batch_size=10)
        hopf_fc = hopf_model.compute_fc(hopf_ts)
        hopf_fc_mean = hopf_fc.mean(dim=0)
    
    hopf_metrics = compute_all_fc_metrics(hopf_fc_mean.unsqueeze(0), target_fc.unsqueeze(0))
    print(f"Hopf model metrics: {hopf_metrics}")
    
    return hopf_model, hopf_metrics


def train_neural_sde(dataset, device):
    """Train Neural SDE model using backpropagation."""
    print(f"\n{'='*60}")
    print("STEP 3: Training Neural SDE Model (Backpropagation)")
    print('='*60)
    
    # Create windowed data loaders
    window_size = min(50, dataset.n_timepoints // 4)
    print(f"Window size: {window_size}")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        stride=window_size // 2,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        device=device
    )
    
    print(f"Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    # Create Neural SDE model
    sde_model = NeuralSDE(
        n_rois=dataset.n_rois,
        hidden_dim=32,
        n_layers=2,
        device=device
    )
    
    print(f"\nNeural SDE model created:")
    print(f"  - Parameters: {sum(p.numel() for p in sde_model.parameters())}")
    
    # Create trainer
    trainer = Trainer(
        model=sde_model,
        lr=1e-3,
        loss_fn="combined",
        device=device,
        experiment_name="neural_sde"
    )
    
    # Train (reduced epochs for demo)
    n_epochs = 50
    print(f"\nTraining for {n_epochs} epochs...")
    
    metrics_store = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_steps=window_size,
        dt=0.01,
        early_stopping_patience=15,
        verbose=True
    )
    
    # Test
    test_metrics = trainer.test(test_loader, n_steps=window_size)
    print(f"\nTest metrics: {test_metrics}")
    
    return sde_model, metrics_store, test_metrics


def fine_tune_model(sde_model, dataset, device):
    """Fine-tune trained model."""
    print(f"\n{'='*60}")
    print("STEP 4: Fine-tuning Neural SDE Model")
    print('='*60)
    
    window_size = min(50, dataset.n_timepoints // 4)
    
    train_loader, val_loader, _ = create_data_loaders(
        dataset=dataset,
        window_size=window_size,
        stride=window_size // 2,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        device=device
    )
    
    # Fine-tune
    fine_tuner = FineTuner(
        model=sde_model,
        device=device
    )
    
    print("Fine-tuning with lower learning rate...")
    ft_metrics = fine_tuner.fine_tune(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=20,
        lr=1e-4,
        warmup_epochs=3,
        n_steps=window_size,
        experiment_name="neural_sde_finetuned"
    )
    
    return sde_model, ft_metrics


def compare_models(hopf_model, sde_model, dataset, hopf_metrics, sde_metrics, device):
    """Compare both models."""
    print(f"\n{'='*60}")
    print("STEP 5: Model Comparison and Visualization")
    print('='*60)
    
    target_fc = dataset.fc_mean
    n_timepoints = min(dataset.n_timepoints, 200)
    
    # Generate predictions from both models
    with torch.no_grad():
        # Hopf
        hopf_ts = hopf_model.forward(n_steps=n_timepoints, batch_size=1)
        hopf_fc = hopf_model.compute_fc(hopf_ts)[0]
        
        # Neural SDE
        sde_ts = sde_model.forward(n_steps=n_timepoints, batch_size=1)
        sde_fc = sde_model.compute_fc(sde_ts)[0]
    
    # Final metrics
    hopf_final = compute_all_fc_metrics(hopf_fc.unsqueeze(0), target_fc.unsqueeze(0))
    sde_final = compute_all_fc_metrics(sde_fc.unsqueeze(0), target_fc.unsqueeze(0))
    
    print("\n" + "-"*40)
    print("Final Results:")
    print("-"*40)
    print(f"Coupled Hopf Model:")
    for k, v in hopf_final.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nNeural SDE Model:")
    for k, v in sde_final.items():
        print(f"  {k}: {v:.4f}")
    
    # Create visualizations
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # FC comparisons
    print("\nGenerating visualizations...")
    
    plot_fc_comparison(
        hopf_fc, target_fc,
        title="Coupled Hopf Model - FC Comparison",
        save_path=str(results_dir / "hopf_fc_comparison.png")
    )
    plt.close()
    
    plot_fc_comparison(
        sde_fc, target_fc,
        title="Neural SDE Model - FC Comparison",
        save_path=str(results_dir / "sde_fc_comparison.png")
    )
    plt.close()
    
    # Model comparison bar plot
    model_results = {
        "Coupled Hopf": hopf_final,
        "Neural SDE": sde_final
    }
    
    plot_model_comparison(
        model_results,
        metric_names=["fc_correlation", "fc_mse"],
        save_path=str(results_dir / "model_comparison.png")
    )
    plt.close()
    
    # Timeseries plots
    plot_timeseries(
        hopf_ts[0],
        n_rois=5,
        title="Coupled Hopf - Simulated Timeseries",
        save_path=str(results_dir / "hopf_timeseries.png")
    )
    plt.close()
    
    plot_timeseries(
        sde_ts[0],
        n_rois=5,
        title="Neural SDE - Simulated Timeseries",
        save_path=str(results_dir / "sde_timeseries.png")
    )
    plt.close()
    
    # Comprehensive comparison report
    models = {
        "Coupled Hopf": hopf_model,
        "Neural SDE": sde_model
    }
    
    create_comparison_report(
        models=models,
        target_fc=target_fc,
        n_timepoints=n_timepoints,
        save_dir=str(results_dir / "comparison")
    )
    plt.close()
    
    print(f"\nAll visualizations saved to {results_dir}")
    
    return model_results


def main():
    """Main execution function."""
    print("="*60)
    print("NEUROSCIENCE CONTROL - Brain Dynamics Simulation")
    print("="*60)
    
    # Setup
    device = setup_device()
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data path
    data_path = "data/ts_young/ts_young_TR0.72.mat"
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"\nData file not found: {data_path}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        n_rois = 68
        n_timepoints = 200
        n_subjects = 20
        
        # Generate synthetic timeseries
        timeseries = np.random.randn(n_rois, n_timepoints, n_subjects) * 0.5
        
        # Add some structure (correlated activity)
        for i in range(n_subjects):
            for j in range(n_rois):
                timeseries[j, :, i] += 0.3 * np.sin(np.linspace(0, 4*np.pi, n_timepoints) + j*0.1)
        
        # Compute FC
        fc_all = np.zeros((n_rois, n_rois, n_subjects))
        for i in range(n_subjects):
            fc_all[:, :, i] = np.corrcoef(timeseries[:, :, i])
        
        fc_mean = fc_all.mean(axis=2)
        
        # Save synthetic data
        from scipy.io import savemat
        Path("data/ts_young").mkdir(parents=True, exist_ok=True)
        savemat(data_path, {
            'timeseries_all': timeseries,
            'FC_all': fc_all,
            'FC_mean': fc_mean
        })
        print(f"Synthetic data saved to {data_path}")
    
    # Execute pipeline
    try:
        # Step 1: Load data
        dataset = load_data(data_path, device)
        
        # Step 2: Train Hopf model via grid search
        hopf_model, hopf_metrics = train_hopf_grid_search(dataset, device)
        
        # Step 3: Train Neural SDE via backpropagation
        sde_model, sde_store, sde_metrics = train_neural_sde(dataset, device)
        
        # Step 4: Fine-tune (optional)
        sde_model, ft_metrics = fine_tune_model(sde_model, dataset, device)
        
        # Step 5: Compare and visualize
        results = compare_models(hopf_model, sde_model, dataset, hopf_metrics, sde_metrics, device)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nResults saved to: results/")
        print("  - figures/: Visualization plots")
        print("  - metrics/: Training metrics")
        print("  - grid_search/: Grid search results")
        
        return results
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
