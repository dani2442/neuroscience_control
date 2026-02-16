"""Training configuration dataclass."""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import datetime


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    
    # Experiment settings
    experiment_name: str = "experiment"
    run_name: Optional[str] = None  # Auto-generated if None
    seed: int = 42
    
    # WandB settings
    wandb_project: str = "neuroscience-control"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True
    
    # Data settings
    data_path: str = "data/ts_young/ts_young_TR0.72.mat"
    max_subjects: Optional[int] = 5
    window_size: int = 100
    n_windows_per_epoch: int = 256
    batch_size: int = 16
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Fourier denoising
    fourier_denoise: bool = True
    denoise_f_lo: float = 0.008
    denoise_f_hi: float = 0.08

    # Dynamics metrics settings
    tr: float = 0.72
    f_lo: float = 0.04
    f_hi: float = 0.07
    fcd_win_sec: float = 60.0
    fcd_step_sec: float = 2.0
    compute_fcd_metrics: bool = True
    compute_metastability_metrics: bool = True
    metrics_sample_batches: Optional[int] = 1  # Limit expensive metrics per epoch (None = all)
    
    # Model settings
    hidden_dim: int = 32
    n_layers: int = 2
    coupling_strength: float = 0.1
    
    # Training settings
    n_epochs: int = 50
    lr: float = 1e-3
    loss_fn: str = "combined"
    loss_weight_fc: Optional[float] = None
    loss_weight_fc_mse: Optional[float] = None
    loss_weight_l2: Optional[float] = None
    loss_weight_amplitude: Optional[float] = None
    loss_weight_omega: Optional[float] = None
    loss_weight_fcd: Optional[float] = None
    loss_weight_metastability: Optional[float] = None
    early_stopping_patience: int = 15
    n_steps: int = 100
    dt_min: Optional[float] = None  # Minimum time step for adaptive SDE solvers
    sde_method: str = "euler"  # SDE solver method ('euler', 'milstein', 'srk', etc.)
    
    # Fine-tuning settings
    fine_tune: bool = False
    fine_tune_epochs: int = 20
    fine_tune_lr: float = 1e-4
    warmup_epochs: int = 3
    
    # Grid search settings (for Hopf)
    g_values: List[float] = field(default_factory=lambda: [-0.3, 0.3, 0.5, 0.7, 1.0, 1.5])
    a_values: List[float] = field(default_factory=lambda: [-0.02, -0.01, 0.0, 0.01, 0.05,])
    kappa_values: List[float] = field(default_factory=lambda: [0.1])
    n_simulations: int = 5
    
    # Directories
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    figures_dir: str = "paper/images"
    
    # Device
    device: str = "auto"  # "auto", "cuda", or "cpu"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.experiment_name}_{timestamp}"

        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HopfConfig(TrainingConfig):
    """Configuration for Hopf model training."""
    experiment_name: str = "hopf"
    noise_sigma: float = 0.05
    learnable_a: bool = False
    learnable_g: bool = False
    learnable_kappa: bool = False

    # Composite grid-search scoring weights
    weight_fc: float = 1.0
    weight_fcd: float = 0.5
    weight_meta: float = 0.5


@dataclass
class HybridHopfConfig(TrainingConfig):
    """Configuration for HybridHopf model training (learnable coupling network)."""
    experiment_name: str = "hybrid_hopf"
    noise_sigma: float = 0.05
    learnable_a: bool = True
    learnable_g: bool = True
    learnable_kappa: bool = True
    learnable_omega: bool = False
    
    # Coupling network architecture
    coupling_hidden_dim: int = 32
    coupling_n_layers: int = 2


@dataclass
class NeuralSDEConfig(TrainingConfig):
    """Configuration for Neural SDE training."""
    experiment_name: str = "neural_sde"
    use_structural_connectivity: bool = False
