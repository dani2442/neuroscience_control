"""Training configuration dataclasses."""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import datetime


def _default_openneuro_derivative_patterns() -> List[str]:
    """Default subset of files that enables BIDS-derivative ROI extraction."""
    return [
        "derivatives/fmriprep/sub-*/**/*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "derivatives/fmriprep/sub-*/**/*_desc-confounds_timeseries.tsv",
    ]


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
    dataset_type: str = "ts_young"  # "ts_young", "lsd", "nilearn", "openneuro", "datalad", "bids"
    data_path: str = "data/ts_young/ts_young_TR0.72.mat"
    lsd_data_dir: str = "data/lsd"
    nilearn_dataset: str = "development_fmri"
    nilearn_data_dir: str = "data/nilearn"
    nilearn_n_subjects: Optional[int] = None
    nilearn_reduce_confounds: bool = True
    openneuro_dataset: str = "ds000030"
    openneuro_tag: Optional[str] = None
    openneuro_target_dir: str = "data/openneuro"
    openneuro_include: List[str] = field(default_factory=_default_openneuro_derivative_patterns)
    openneuro_exclude: List[str] = field(default_factory=list)
    datalad_source: str = "https://github.com/OpenNeuroDatasets/ds000030.git"
    datalad_dataset_dir: str = "data/datalad/ds000030"
    datalad_get_paths: List[str] = field(default_factory=_default_openneuro_derivative_patterns)
    bids_root: Optional[str] = None
    bids_relative_path: Optional[str] = None
    bids_derivatives_dir: str = "derivatives/fmriprep"
    bids_task: Optional[str] = None
    bids_space: Optional[str] = "MNI152NLin2009cAsym"
    bids_desc: Optional[str] = "preproc"
    bids_subject_ids: List[str] = field(default_factory=list)
    bids_runs_per_subject: int = 1
    atlas_n_rois: int = 100
    atlas_yeo_networks: int = 7
    atlas_resolution_mm: int = 1
    atlas_smoothing_fwhm: Optional[float] = None
    max_subjects: Optional[int] = None  # None means use all subjects
    window_size: int = 50
    n_windows_per_epoch: int = 1024
    batch_size: int = 128
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
    fcd_win_sec: float = 20.0
    fcd_step_sec: float = 2.0
    
    # Model settings
    coupling_strength: float = 0.1
    
    # Training settings
    n_epochs: int = 20
    lr: float = 1e-3
    loss_weights: dict = field(default_factory=lambda: {
        "fc_correlation": 1.0, "fc_mse": 1.0, "l2": 1.0,
        "amplitude": 1.0, "omega": 1.0, "power_spectrum": 1.0,
        "temporal_correlation": 1.0, "autocorrelation": 1.0,
        "fcd": 1.0, "phfcd": 1.0, "phase_fc_correlation": 1.0,
        "metastability": 1.0, "fdm": 0.25,
    })
    early_stopping_patience: int = 15
    n_steps: int = 50

    # FDM loss hyperparameters
    fdm_n_pairs: int = 32
    fdm_max_lag: int = 50
    fdm_sigma: float = 1.0
    dt_min: Optional[float] = 0.05  # Fixed solver sub-step (passed as torchsde `dt`)
    sde_type: str = "ito"  # SDE interpretation required by reversible_heun
    sde_method: str = "euler"  # Stable Stratonovich solver
    use_adjoint: bool = False  # Use torchsde.sdeint_adjoint for backprop memory efficiency
    adjoint_method: Optional[str] = "euler"  # Matching adjoint for reversible_heun
    
    # Grid search settings (for Hopf)
    g_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0, 1.5])
    a_values: List[float] = field(default_factory=lambda: [-0.02, -0.01, 0.0, 0.01])
    kappa_values: List[float] = field(default_factory=lambda: [0.1, 1.0])
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
    noise_sigma: float = 0.1
    initial_a: float = -0.02
    initial_g: float = 0.5
    initial_kappa: float = 1.0
    learnable_a: bool = True
    learnable_g: bool = True
    learnable_kappa: bool = False
    learnable_omega: bool = False
    learnable_fc: bool = False  # Learnable functional connectivity matrix

    # Composite grid-search scoring weights
    weight_fc: float = 1.0
    weight_fcd: float = 0.0
    weight_phfcd: float = 1.0
    weight_meta: float = 1.0


@dataclass
class HybridHopfConfig(TrainingConfig):
    """Configuration for HybridHopf model training (low-rank coupling + node networks)."""
    experiment_name: str = "hybrid_hopf"
    noise_sigma: float = 0.1
    initial_a: float = -0.02
    initial_g: float = 0.5
    initial_kappa: float = 1.0
    learnable_a: bool = True
    learnable_g: bool = True
    learnable_kappa: bool = False
    learnable_omega: bool = False
    learnable_fc: bool = False  # Kept for backward compat with HopfConfig checks
    use_adjoint: bool = True  # Adjoint SDE solver to reduce CUDA memory usage

    # Node-network architecture (key + output MLPs)
    coupling_hidden_dim: int = 16
    coupling_n_layers: int = 1

    # Low-rank coupling: None = full n×n, int = rank-r factorisation
    coupling_rank: Optional[int] = 10
    d_key: int = 4  # dimension of the key embedding


@dataclass
class GNNHopfConfig(TrainingConfig):
    """Configuration for GNN-Hopf model training (node-wise neural coupling)."""
    experiment_name: str = "gnn_hopf"
    noise_sigma: float = 0.1
    initial_a: float = -0.02
    initial_g: float = 0.5
    initial_kappa: float = 1.0
    learnable_a: bool = True
    learnable_g: bool = True
    learnable_kappa: bool = False
    learnable_omega: bool = False

    # Node-wise network architecture
    node_hidden_dim: int = 16
    node_n_layers: int = 1
    use_adjoint: bool = True  # Adjoint SDE solver to reduce CUDA memory usage


@dataclass
class HybridNeuralConfig(TrainingConfig):
    """Configuration for HybridHopfNeuralModel training (Hopf + neural drift correction)."""
    experiment_name: str = "hybrid_neural"
    noise_sigma: float = 0.1
    initial_a: float = -0.02
    initial_g: float = 0.5
    initial_kappa: float = 1.0
    learnable_a: bool = True
    learnable_g: bool = True
    learnable_kappa: bool = False
    learnable_omega: bool = False

    # Neural drift architecture
    drift_hidden_dim: int = 64
    drift_n_layers: int = 2
    use_adjoint: bool = True


@dataclass
class NeuralSDEConfig(TrainingConfig):
    """Configuration for Neural SDE training."""
    experiment_name: str = "neural_sde"
    hidden_dim: int = 128
    n_layers: int = 2
    use_structural_connectivity: bool = False
