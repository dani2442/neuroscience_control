"""Visualization utilities for neuroscience models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..metrics import fc_correlation, compute_all_fc_metrics
from ..metrics.metrics_store import MetricsStore

# Default figures directory for paper
FIGURES_DIR = Path("paper/images")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_save_path(save_path: Optional[str], default_name: str, use_pdf: bool = True) -> Optional[Path]:
    """
    Get the save path, defaulting to paper/images/ with PDF format.
    
    Args:
        save_path: Explicit save path or None
        default_name: Default filename (without extension)
        use_pdf: Whether to use PDF format
        
    Returns:
        Path object or None
    """
    if save_path is None:
        ext = ".pdf" if use_pdf else ".png"
        return FIGURES_DIR / f"{default_name}{ext}"
    
    path = Path(save_path)
    # Convert to PDF if requested and not already PDF
    if use_pdf and path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    
    return path


def plot_fc_comparison(
    fc_pred: torch.Tensor,
    fc_target: torch.Tensor,
    title: str = "FC Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    use_pdf: bool = True,
    default_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison between predicted and target FC matrices.
    
    Args:
        fc_pred: Predicted FC (n_rois x n_rois)
        fc_target: Target FC (n_rois x n_rois)
        title: Plot title
        save_path: Path to save figure (None uses paper/images/)
        figsize: Figure size
        use_pdf: Whether to save as PDF
        default_name: Default filename if save_path is None
        
    Returns:
        Matplotlib figure
    """
    if fc_pred.dim() > 2:
        fc_pred = fc_pred[0]
    if fc_target.dim() > 2:
        fc_target = fc_target[0]
    
    fc_pred_np = fc_pred.detach().cpu().numpy()
    fc_target_np = fc_target.detach().cpu().numpy()
    fc_diff = fc_pred_np - fc_target_np
    
    # Compute correlation
    corr = fc_correlation(fc_pred.unsqueeze(0), fc_target.unsqueeze(0)).item()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Target FC
    im0 = axes[0].imshow(fc_target_np, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Target FC')
    axes[0].set_xlabel('ROI')
    axes[0].set_ylabel('ROI')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Predicted FC
    im1 = axes[1].imshow(fc_pred_np, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title(f'Predicted FC (r={corr:.3f})')
    axes[1].set_xlabel('ROI')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Difference
    vmax_diff = max(abs(fc_diff.min()), abs(fc_diff.max()))
    im2 = axes[2].imshow(fc_diff, cmap='coolwarm', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title('Difference (Pred - Target)')
    axes[2].set_xlabel('ROI')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Determine save path
    if save_path is not None or default_name is not None:
        final_path = _get_save_path(
            save_path, 
            default_name or "fc_comparison",
            use_pdf=use_pdf
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='pdf' if use_pdf else 'png')
        print(f"Saved figure to {final_path}")
    
    return fig


def plot_training_curves(
    metrics_store: MetricsStore,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    use_pdf: bool = True,
    default_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        metrics_store: MetricsStore with training history
        save_path: Path to save figure
        figsize: Figure size
        use_pdf: Whether to save as PDF
        default_name: Default filename if save_path is None
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    train_loss = metrics_store.get_metric_history('loss', 'train')
    val_loss = metrics_store.get_metric_history('loss', 'val')
    epochs = list(range(len(train_loss)))
    
    axes[0].plot(epochs, train_loss, label='Train', color='blue')
    axes[0].plot(epochs, val_loss, label='Validation', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # FC Correlation curves
    train_corr = metrics_store.get_metric_history('fc_correlation', 'train')
    val_corr = metrics_store.get_metric_history('fc_correlation', 'val')
    
    axes[1].plot(epochs, train_corr, label='Train', color='blue')
    axes[1].plot(epochs, val_corr, label='Validation', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('FC Correlation')
    axes[1].set_title('FC Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Training History: {metrics_store.experiment_name}", fontsize=14)
    plt.tight_layout()
    
    # Determine save path
    if save_path is not None or default_name is not None:
        final_path = _get_save_path(
            save_path, 
            default_name or "training_curves",
            use_pdf=use_pdf
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='pdf' if use_pdf else 'png')
        print(f"Saved figure to {final_path}")
    
    return fig


def plot_model_comparison(
    model_results: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    use_pdf: bool = True,
    default_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of metrics across models.
    
    Args:
        model_results: Dict mapping model name to metrics dict
        metric_names: List of metrics to compare
        save_path: Path to save figure
        figsize: Figure size
        use_pdf: Whether to save as PDF
        default_name: Default filename if save_path is None
        
    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        metric_names = ['fc_correlation', 'fc_mse', 'loss']
    
    model_names = list(model_results.keys())
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(len(model_names))
    width = 0.6
    
    for i, metric in enumerate(metric_names):
        values = [model_results[m].get(metric, 0) for m in model_names]
        
        bars = axes[i].bar(x, values, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)])
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout()
    
    # Determine save path
    if save_path is not None or default_name is not None:
        final_path = _get_save_path(
            save_path, 
            default_name or "model_comparison",
            use_pdf=use_pdf
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='pdf' if use_pdf else 'png')
        print(f"Saved figure to {final_path}")
    
    return fig


def plot_timeseries(
    timeseries: torch.Tensor,
    roi_indices: Optional[List[int]] = None,
    n_rois: int = 5,
    title: str = "Timeseries",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    use_pdf: bool = True,
    default_name: Optional[str] = None,
    show_multiple_paths: bool = True,
    alpha: float = 0.7
) -> plt.Figure:
    """
    Plot timeseries for selected ROIs.
    
    Args:
        timeseries: Tensor of shape (n_rois, n_timepoints) or (batch, n_rois, n_timepoints)
        roi_indices: Specific ROIs to plot
        n_rois: Number of ROIs to plot if indices not specified
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        use_pdf: Whether to save as PDF
        default_name: Default filename if save_path is None
        show_multiple_paths: If True and batch dim exists, show all paths with transparency
        alpha: Transparency for multiple paths
        
    Returns:
        Matplotlib figure
    """
    has_batch = timeseries.dim() == 3
    
    if has_batch and show_multiple_paths:
        # Keep batch dimension for multi-path visualization
        ts_np = timeseries.detach().cpu().numpy()  # (batch, n_rois, n_timepoints)
        n_paths = ts_np.shape[0]
        n_total_rois = ts_np.shape[1]
        n_timepoints = ts_np.shape[2]
    else:
        if has_batch:
            timeseries = timeseries[0]
        ts_np = timeseries.detach().cpu().numpy()
        n_paths = 1
        n_total_rois = ts_np.shape[0]
        n_timepoints = ts_np.shape[1]
        ts_np = ts_np[np.newaxis, ...]  # Add batch dimension for uniform handling
    
    if roi_indices is None:
        # Select evenly spaced ROIs
        roi_indices = np.linspace(0, n_total_rois - 1, n_rois, dtype=int).tolist()
    
    fig, axes = plt.subplots(len(roi_indices), 1, figsize=figsize, sharex=True)
    if len(roi_indices) == 1:
        axes = [axes]
    
    time = np.arange(n_timepoints)
    
    # Color map for different paths
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_paths, 10)))
    
    for i, roi_idx in enumerate(roi_indices):
        for path_idx in range(n_paths):
            path_alpha = alpha if n_paths > 1 else 1.0
            path_label = f'Path {path_idx + 1}' if path_idx == 0 and n_paths > 1 and i == 0 else None
            axes[i].plot(
                time, ts_np[path_idx, roi_idx], 
                linewidth=0.8, 
                alpha=path_alpha,
                color=colors[path_idx % len(colors)],
                label=path_label if i == 0 else None
            )
        axes[i].set_ylabel(f'ROI {roi_idx}')
        axes[i].grid(True, alpha=0.3)
    
    # Add legend if multiple paths
    if n_paths > 1:
        axes[0].legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time')
    plt.suptitle(title + (f' ({n_paths} paths)' if n_paths > 1 else ''), fontsize=14)
    plt.tight_layout()
    
    # Determine save path
    if save_path is not None or default_name is not None:
        final_path = _get_save_path(
            save_path, 
            default_name or "timeseries",
            use_pdf=use_pdf
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='pdf' if use_pdf else 'png')
        print(f"Saved figure to {final_path}")
    
    return fig


def plot_power_spectrum(
    timeseries: torch.Tensor,
    roi_indices: Optional[List[int]] = None,
    sampling_rate: float = 1.0,
    title: str = "Power Spectrum",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    use_pdf: bool = True,
    default_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot power spectrum of timeseries.
    
    Args:
        timeseries: Tensor of shape (n_rois, n_timepoints)
        roi_indices: ROIs to plot
        sampling_rate: Sampling rate in Hz
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        use_pdf: Whether to save as PDF
        default_name: Default filename if save_path is None
        
    Returns:
        Matplotlib figure
    """
    if timeseries.dim() == 3:
        timeseries = timeseries[0]
    
    ts_np = timeseries.detach().cpu().numpy()
    n_rois = ts_np.shape[0]
    n_timepoints = ts_np.shape[1]
    
    if roi_indices is None:
        roi_indices = list(range(min(5, n_rois)))
    
    # Compute FFT
    freqs = np.fft.rfftfreq(n_timepoints, d=1/sampling_rate)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for roi_idx in roi_indices:
        fft = np.fft.rfft(ts_np[roi_idx])
        power = np.abs(fft) ** 2
        power = power / power.sum()  # Normalize
        
        ax.semilogy(freqs, power, label=f'ROI {roi_idx}', alpha=0.7)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Power')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Determine save path
    if save_path is not None or default_name is not None:
        final_path = _get_save_path(
            save_path, 
            default_name or "power_spectrum",
            use_pdf=use_pdf
        )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(final_path, dpi=300, bbox_inches='tight', format='pdf' if use_pdf else 'png')
        print(f"Saved figure to {final_path}")
    
    return fig


def create_comparison_report(
    models: Dict[str, Any],
    target_fc: torch.Tensor,
    n_timepoints: int = 200,
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    use_pdf: bool = True
) -> plt.Figure:
    """
    Create comprehensive comparison report for multiple models.
    
    Args:
        models: Dict mapping model names to model objects
        target_fc: Target functional connectivity
        n_timepoints: Timepoints to simulate
        save_dir: Directory to save results (defaults to paper/images/)
        figsize: Figure size
        use_pdf: Whether to save as PDF
        
    Returns:
        Matplotlib figure
    """
    # Default to paper/images for save directory
    if save_dir is None:
        save_dir = FIGURES_DIR
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_models = len(models)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, n_models + 1, figure=fig)
    
    # First column: Target FC
    ax_target = fig.add_subplot(gs[0, 0])
    target_np = target_fc.detach().cpu().numpy()
    if target_np.ndim > 2:
        target_np = target_np[0]
    im = ax_target.imshow(target_np, cmap='coolwarm', vmin=-1, vmax=1)
    ax_target.set_title('Target FC')
    plt.colorbar(im, ax=ax_target, fraction=0.046)
    
    all_metrics = {}
    
    for i, (name, model) in enumerate(models.items()):
        # Simulate
        with torch.no_grad():
            ts = model.forward(n_steps=n_timepoints, batch_size=1)
            fc_pred = model.compute_fc(ts)
        
        fc_pred_np = fc_pred[0].detach().cpu().numpy()
        ts_np = ts[0].detach().cpu().numpy()
        
        # Plot FC
        ax_fc = fig.add_subplot(gs[0, i + 1])
        im = ax_fc.imshow(fc_pred_np, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Compute metrics
        metrics = compute_all_fc_metrics(fc_pred, target_fc.unsqueeze(0))
        all_metrics[name] = metrics
        
        ax_fc.set_title(f'{name}\n(r={metrics["fc_correlation"]:.3f})')
        plt.colorbar(im, ax=ax_fc, fraction=0.046)
        
        # Plot sample timeseries
        ax_ts = fig.add_subplot(gs[1, i + 1])
        for roi in range(min(5, ts_np.shape[0])):
            ax_ts.plot(ts_np[roi, :100], alpha=0.7, linewidth=0.8)
        ax_ts.set_title(f'{name} - Timeseries')
        ax_ts.set_xlabel('Time')
        
        # Plot FC scatter
        ax_scatter = fig.add_subplot(gs[2, i + 1])
        n_rois = fc_pred_np.shape[0]
        idx = np.triu_indices(n_rois, k=1)
        pred_vals = fc_pred_np[idx]
        target_vals = target_np[idx]
        
        ax_scatter.scatter(target_vals, pred_vals, alpha=0.3, s=10)
        ax_scatter.plot([-1, 1], [-1, 1], 'r--', linewidth=1)
        ax_scatter.set_xlabel('Target FC')
        ax_scatter.set_ylabel('Predicted FC')
        ax_scatter.set_title(f'{name} - FC Scatter')
        ax_scatter.set_xlim(-1, 1)
        ax_scatter.set_ylim(-1, 1)
    
    # Empty subplot for layout
    ax_empty1 = fig.add_subplot(gs[1, 0])
    ax_empty1.axis('off')
    ax_empty1.text(0.5, 0.5, 'Sample\nTimeseries', ha='center', va='center', fontsize=12)
    
    ax_empty2 = fig.add_subplot(gs[2, 0])
    ax_empty2.axis('off')
    ax_empty2.text(0.5, 0.5, 'FC\nScatter', ha='center', va='center', fontsize=12)
    
    plt.suptitle('Model Comparison Report', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure as PDF for paper
    ext = ".pdf" if use_pdf else ".png"
    fig_path = save_dir / f"comparison_report{ext}"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf' if use_pdf else 'png')
    print(f"Saved figure to {fig_path}")
    
    # Save metrics
    import json
    with open(save_dir / "comparison_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Comparison report saved to {save_dir}")
    
    return fig
