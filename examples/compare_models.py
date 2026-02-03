#!/usr/bin/env python3
"""
Model Comparison Script.

This script loads trained Hopf and Neural SDE models,
compares their performance, and generates publication-quality figures.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import os
import wandb

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set up proxy before importing wandb
os.environ["HTTP_PROXY"] = "http://proxy.nhr.fau.de:80"
os.environ["HTTPS_PROXY"] = "http://proxy.nhr.fau.de:80"

# Import project modules
from src.dataset import NeuroscienceDataset
from src.models import CoupledHopfModel, NeuralSDE
from src.metrics import compute_all_fc_metrics
from src.utils import (
    plot_fc_comparison,
    plot_model_comparison,
    plot_timeseries,
    create_comparison_report,
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


def init_wandb(project: str, run_name: str, use_wandb: bool = True):
    """Initialize wandb for comparison run."""    
    settings = wandb.Settings(_service_transport="http")
    
    run = wandb.init(
        project=project,
        name=run_name,
        settings=settings,
        tags=["comparison", "evaluation"]
    )
    
    print(f"Wandb initialized: {project}/{run_name}")
    return run


def load_data(data_path: str, device: str):
    """Load dataset."""
    print(f"\n{'='*60}")
    print("Loading Data")
    print('='*60)
    
    dataset = NeuroscienceDataset(
        filepath=data_path,
        normalize=True,
        device=device
    )
    
    print(f"Loaded dataset:")
    print(f"  - Number of subjects: {dataset.n_subjects}")
    print(f"  - Number of ROIs: {dataset.n_rois}")
    print(f"  - Number of timepoints: {dataset.n_timepoints}")
    
    return dataset


def load_models(hopf_checkpoint: str, nsde_checkpoint: str, 
                n_rois: int, device: str):
    """Load trained models from checkpoints."""
    print(f"\n{'='*60}")
    print("Loading Models")
    print('='*60)
    
    models = {}
    
    # Load Hopf model
    if hopf_checkpoint and Path(hopf_checkpoint).exists():
        print(f"Loading Hopf model from {hopf_checkpoint}")
        hopf_model = CoupledHopfModel(n_rois=n_rois, device=device)
        hopf_model.load(hopf_checkpoint)
        models["Coupled Hopf"] = hopf_model
        print("  - Hopf model loaded successfully")
    else:
        print(f"Warning: Hopf checkpoint not found at {hopf_checkpoint}")
    
    # Load Neural SDE model
    if nsde_checkpoint and Path(nsde_checkpoint).exists():
        print(f"Loading Neural SDE model from {nsde_checkpoint}")
        sde_model = NeuralSDE(n_rois=n_rois, device=device)
        sde_model.load(nsde_checkpoint)
        models["Neural SDE"] = sde_model
        print("  - Neural SDE model loaded successfully")
    else:
        print(f"Warning: Neural SDE checkpoint not found at {nsde_checkpoint}")
    
    return models


def evaluate_models(models: dict, target_fc: torch.Tensor, 
                    n_timepoints: int, n_simulations: int = 10):
    """Evaluate all models and compute metrics."""
    print(f"\n{'='*60}")
    print("Evaluating Models")
    print('='*60)
    
    all_results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        all_fc_corrs = []
        all_fc_mses = []
        
        with torch.no_grad():
            for _ in range(n_simulations):
                ts = model.forward(n_steps=n_timepoints, batch_size=1)
                fc_pred = model.compute_fc(ts)
                
                metrics = compute_all_fc_metrics(fc_pred, target_fc.unsqueeze(0))
                all_fc_corrs.append(metrics['fc_correlation'])
                all_fc_mses.append(metrics['fc_mse'])
        
        results = {
            'fc_correlation': np.mean(all_fc_corrs),
            'fc_correlation_std': np.std(all_fc_corrs),
            'fc_mse': np.mean(all_fc_mses),
            'fc_mse_std': np.std(all_fc_mses),
        }
        
        all_results[name] = results
        print(f"  FC Correlation: {results['fc_correlation']:.4f} ± {results['fc_correlation_std']:.4f}")
        print(f"  FC MSE: {results['fc_mse']:.4f} ± {results['fc_mse_std']:.4f}")
    
    return all_results


def generate_comparison_figures(models: dict, target_fc: torch.Tensor,
                                 results: dict, n_timepoints: int,
                                 use_wandb: bool = True):
    """Generate all comparison figures."""
    print(f"\n{'='*60}")
    print("Generating Comparison Figures")
    print('='*60)
    
    figures = {}
    
    # Individual FC comparisons
    for name, model in models.items():
        with torch.no_grad():
            ts = model.forward(n_steps=n_timepoints, batch_size=1)
            fc_pred = model.compute_fc(ts)[0]
        
        # FC comparison
        fig = plot_fc_comparison(
            fc_pred, target_fc,
            title=f"{name} - FC Comparison",
            default_name=f"{name.lower().replace(' ', '_')}_fc_comparison",
            use_pdf=True
        )
        figures[f"{name}_fc"] = fig
        
        if use_wandb and wandb.run is not None:
            wandb.log({f"figures/{name}_fc_comparison": wandb.Image(fig)})
        plt.close()
        
        # Timeseries
        fig = plot_timeseries(
            ts[0],
            n_rois=5,
            title=f"{name} - Simulated Timeseries",
            default_name=f"{name.lower().replace(' ', '_')}_timeseries",
            use_pdf=True
        )
        figures[f"{name}_ts"] = fig
        
        if use_wandb and wandb.run is not None:
            wandb.log({f"figures/{name}_timeseries": wandb.Image(fig)})
        plt.close()
    
    # Model comparison bar plot
    fig = plot_model_comparison(
        results,
        metric_names=["fc_correlation", "fc_mse"],
        default_name="model_comparison_metrics",
        use_pdf=True
    )
    figures["comparison"] = fig
    
    if use_wandb and wandb.run is not None:
        wandb.log({"figures/model_comparison": wandb.Image(fig)})
    plt.close()
    
    # Comprehensive comparison report
    fig = create_comparison_report(
        models=models,
        target_fc=target_fc,
        n_timepoints=n_timepoints,
        save_dir=str(FIGURES_DIR),
        use_pdf=True
    )
    figures["report"] = fig
    
    if use_wandb and wandb.run is not None:
        wandb.log({"figures/comparison_report": wandb.Image(fig)})
    plt.close()
    
    print(f"All figures saved to {FIGURES_DIR}")
    
    return figures


def save_results(results: dict, save_path: str):
    """Save comparison results to JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy values to Python floats
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            k: float(v) for k, v in metrics.items()
        }
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {save_path}")


def print_summary(results: dict):
    """Print comparison summary."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\n" + "-"*40)
    print(f"{'Model':<20} {'FC Corr':<15} {'FC MSE':<15}")
    print("-"*40)
    
    for name, metrics in results.items():
        fc_corr = f"{metrics['fc_correlation']:.4f} ± {metrics['fc_correlation_std']:.4f}"
        fc_mse = f"{metrics['fc_mse']:.4f} ± {metrics['fc_mse_std']:.4f}"
        print(f"{name:<20} {fc_corr:<15} {fc_mse:<15}")
    
    print("-"*40)
    
    # Determine best model
    best_model = max(results.keys(), key=lambda x: results[x]['fc_correlation'])
    print(f"\nBest model (by FC correlation): {best_model}")
    print(f"  FC Correlation: {results[best_model]['fc_correlation']:.4f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat",
                        help="Path to data file")
    parser.add_argument("--hopf-checkpoint", type=str, default="checkpoints/hopf_best.pt",
                        help="Path to Hopf model checkpoint")
    parser.add_argument("--nsde-checkpoint", type=str, default="checkpoints/nsde_best.pt",
                        help="Path to Neural SDE model checkpoint")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control",
                        help="Wandb project name")
    parser.add_argument("--run-name", type=str, default="model_comparison",
                        help="Run name for wandb")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cuda, cpu)")
    parser.add_argument("--n-simulations", type=int, default=10,
                        help="Number of simulations for evaluation")
    parser.add_argument("--output-path", type=str, default="results/comparison_results.json",
                        help="Path to save comparison results")
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Setup
    device = setup_device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)
    
    use_wandb = not args.no_wandb
    
    # Initialize wandb
    init_wandb(args.wandb_project, args.run_name, use_wandb)
    
    try:
        # Load data
        dataset = load_data(args.data_path, device)
        target_fc = dataset.fc_mean
        n_timepoints = min(dataset.n_timepoints, 200)
        
        # Load models
        models = load_models(
            args.hopf_checkpoint,
            args.nsde_checkpoint,
            n_rois=dataset.n_rois,
            device=device
        )
        
        if not models:
            print("Error: No models could be loaded. Please check checkpoint paths.")
            return
        
        # Evaluate models
        results = evaluate_models(
            models, target_fc, n_timepoints, 
            n_simulations=args.n_simulations
        )
        
        # Log results to wandb
        if use_wandb and wandb.run is not None:
            for model_name, metrics in results.items():
                for k, v in metrics.items():
                    wandb.log({f"{model_name}/{k}": v})
            wandb.summary.update({
                "best_model": max(results.keys(), key=lambda x: results[x]['fc_correlation'])
            })
        
        # Generate figures
        figures = generate_comparison_figures(
            models, target_fc, results, n_timepoints,
            use_wandb=use_wandb
        )
        
        # Save results
        save_results(results, args.output_path)
        
        # Print summary
        print_summary(results)
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nResults saved to: {args.output_path}")
        print(f"Figures saved to: {FIGURES_DIR}")
        
        return results
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
