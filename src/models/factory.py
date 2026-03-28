"""Model factory: create model instances from config + dataset."""

from __future__ import annotations

import torch

from ..dataset import NeuroscienceDataset, compute_omega_from_timeseries
from ..training.config import HopfConfig, HybridHopfConfig, HybridNeuralConfig, GNNHopfConfig, NeuralSDEConfig, TrainingConfig
from .hopf_model import CoupledHopfModel
from .hybrid_hopf_model import HybridHopfModel
from .hybrid_neural_model import HybridHopfNeuralModel
from .gnn_hopf_model import GNNHopfModel
from .neural_sde import NeuralSDE


def build_model(
    model_name: str,
    dataset: NeuroscienceDataset,
    cfg: TrainingConfig,
    device: str,
    structural_connectivity: torch.Tensor | None = None,
    n_control_dims: int = 0,
) -> CoupledHopfModel | HybridHopfModel | NeuralSDE:
    """Create model instance for *model_name* using *cfg* parameters.

    All hyper-parameters are read from *cfg* so callers don't need to forward
    individual values from argparse.

    Args:
        n_control_dims: Number of control input dimensions (0 = autonomous).
    """
    if model_name == "nsde":
        model = NeuralSDE(
            n_rois=dataset.n_rois,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            device=device,
            n_control_dims=n_control_dims,
        )
        print(f"\nNeural SDE model — params: {sum(p.numel() for p in model.parameters())}")
        return model

    # Hopf-family models share omega computation.
    omega = compute_omega_from_timeseries(
        dataset.timeseries,
        dt=dataset.dt,
        f_lo=cfg.f_lo,
        f_hi=cfg.f_hi,
        method="peak",
    )
    print(
        f"  omega: shape={omega.shape}, "
        f"range=[{omega.min().item() / (2 * 3.14159):.4f}, "
        f"{omega.max().item() / (2 * 3.14159):.4f}] Hz"
    )

    if model_name == "hopf":
        if not isinstance(cfg, (HopfConfig, HybridHopfConfig)):
            raise ValueError(f"Expected HopfConfig or HybridHopfConfig for 'hopf', got {type(cfg).__name__}")
        model = CoupledHopfModel(
            n_rois=dataset.n_rois,
            structural_connectivity=structural_connectivity,
            omega=omega,
            initial_a=cfg.initial_a,
            initial_g=cfg.initial_g,
            initial_kappa=cfg.initial_kappa,
            noise_sigma=cfg.noise_sigma,
            device=device,
            learnable_a=cfg.learnable_a,
            learnable_g=cfg.learnable_g,
            learnable_kappa=cfg.learnable_kappa,
            learnable_omega=cfg.learnable_omega,
            learnable_fc=cfg.learnable_fc,
            n_control_dims=n_control_dims,
        )
        n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nCoupled Hopf model — learnable params: {n_learn}")
        return model

    if model_name == "hybrid_hopf":
        if not isinstance(cfg, HybridHopfConfig):
            raise ValueError(f"Expected HybridHopfConfig for 'hybrid_hopf', got {type(cfg).__name__}")
        model = HybridHopfModel(
            n_rois=dataset.n_rois,
            structural_connectivity=structural_connectivity,
            omega=omega,
            initial_a=cfg.initial_a,
            initial_g=cfg.initial_g,
            initial_kappa=cfg.initial_kappa,
            noise_sigma=cfg.noise_sigma,
            coupling_hidden_dim=cfg.coupling_hidden_dim,
            coupling_n_layers=cfg.coupling_n_layers,
            coupling_rank=cfg.coupling_rank,
            d_key=cfg.d_key,
            device=device,
            learnable_a=cfg.learnable_a,
            learnable_g=cfg.learnable_g,
            learnable_kappa=cfg.learnable_kappa,
            learnable_omega=cfg.learnable_omega,
            n_control_dims=n_control_dims,
        )
        n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank_str = f"rank={cfg.coupling_rank}" if cfg.coupling_rank else "full-rank"
        print(
            f"\nHybrid Hopf model — learnable params: {n_learn}, "
            f"{rank_str}, d_key={cfg.d_key}, "
            f"hidden={cfg.coupling_hidden_dim}, layers={cfg.coupling_n_layers}"
        )
        return model

    if model_name == "gnn_hopf":
        if not isinstance(cfg, (GNNHopfConfig, HybridHopfConfig)):
            raise ValueError(f"Expected GNNHopfConfig or HybridHopfConfig for 'gnn_hopf', got {type(cfg).__name__}")
        model = GNNHopfModel(
            n_rois=dataset.n_rois,
            structural_connectivity=structural_connectivity,
            omega=omega,
            initial_a=cfg.initial_a,
            initial_g=cfg.initial_g,
            initial_kappa=cfg.initial_kappa,
            noise_sigma=cfg.noise_sigma,
            node_hidden_dim=cfg.node_hidden_dim if hasattr(cfg, 'node_hidden_dim') else cfg.coupling_hidden_dim,
            node_n_layers=cfg.node_n_layers if hasattr(cfg, 'node_n_layers') else cfg.coupling_n_layers,
            device=device,
            learnable_a=cfg.learnable_a,
            learnable_g=cfg.learnable_g,
            learnable_kappa=cfg.learnable_kappa,
            learnable_omega=cfg.learnable_omega,
            n_control_dims=n_control_dims,
        )
        n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\nGNN-Hopf model — learnable params: {n_learn}, "
            f"node_hidden_dim={model.node_hidden_dim}, node_n_layers={model.node_n_layers}"
        )
        return model

    if model_name == "hybrid_neural":
        if not isinstance(cfg, HybridNeuralConfig):
            raise ValueError(f"Expected HybridNeuralConfig for 'hybrid_neural', got {type(cfg).__name__}")
        model = HybridHopfNeuralModel(
            n_rois=dataset.n_rois,
            structural_connectivity=structural_connectivity,
            omega=omega,
            initial_a=cfg.initial_a,
            initial_g=cfg.initial_g,
            initial_sigma=cfg.noise_sigma,
            drift_hidden_dim=cfg.drift_hidden_dim,
            drift_n_layers=cfg.drift_n_layers,
            device=device,
            n_control_dims=n_control_dims,
        )
        n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\nHybrid Neural model — learnable params: {n_learn}, "
            f"drift_hidden_dim={cfg.drift_hidden_dim}, drift_n_layers={cfg.drift_n_layers}"
        )
        return model

    raise ValueError(f"Unsupported model: {model_name}")
