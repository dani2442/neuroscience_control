"""
Neural SDE Model for brain dynamics simulation.

This model uses neural networks to parameterize stochastic differential
equations for more flexible and learnable brain dynamics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .base_model import BaseNeuroscienceModel

try:
    import torchsde
    HAS_TORCHSDE = True
except ImportError:
    HAS_TORCHSDE = False


class DriftNetwork(nn.Module):
    """Neural network for the drift term of the SDE."""
    
    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 64,
        n_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        in_dim = n_rois
        
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, n_rois))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionNetwork(nn.Module):
    """Neural network for the diffusion term of the SDE."""
    
    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 32,
        noise_type: str = "diagonal"
    ):
        super().__init__()
        
        self.noise_type = noise_type
        self.n_rois = n_rois
        
        if noise_type == "diagonal":
            # Diagonal noise: each ROI has independent noise
            self.net = nn.Sequential(
                nn.Linear(n_rois, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_rois),
                nn.Softplus()  # Ensure positive diffusion
            )
        else:
            # Scalar noise
            self.sigma = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "diagonal":
            return self.net(x)
        else:
            return self.sigma.abs() * torch.ones_like(x)


class NeuralSDEFunc(nn.Module):
    """
    SDE function for torchsde integration.
    
    Implements: dX = f(X)dt + g(X)dW
    """
    
    noise_type = "diagonal"
    sde_type = "ito"
    
    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        structural_connectivity: Optional[torch.Tensor] = None,
        coupling_strength: float = 0.1
    ):
        super().__init__()
        
        self.n_rois = n_rois
        self.drift_net = DriftNetwork(n_rois, hidden_dim, n_layers)
        self.diffusion_net = DiffusionNetwork(n_rois, hidden_dim // 2)
        
        # Optional structural connectivity coupling
        if structural_connectivity is not None:
            self.register_buffer('sc', structural_connectivity)
            self.coupling = nn.Parameter(torch.tensor(coupling_strength))
        else:
            self.sc = None
            self.coupling = None
    
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift function."""
        drift = self.drift_net(y)
        
        # Add structural coupling if available
        if self.sc is not None:
            coupling = self.coupling * torch.matmul(y, self.sc.T)
            drift = drift + coupling
        
        return drift
    
    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion function."""
        return self.diffusion_net(y)


class NeuralSDE(BaseNeuroscienceModel):
    """
    Neural SDE model for brain dynamics.
    
    Uses neural networks to learn the drift and diffusion terms
    of a stochastic differential equation that governs brain dynamics.
    """
    
    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        structural_connectivity: Optional[torch.Tensor] = None,
        coupling_strength: float = 0.1,
        device: str = "cpu"
    ):
        """
        Initialize Neural SDE model.
        
        Args:
            n_rois: Number of brain regions
            hidden_dim: Hidden dimension for networks
            n_layers: Number of layers in drift network
            structural_connectivity: Optional connectivity matrix
            coupling_strength: Initial coupling strength
            device: Device to run on
        """
        super().__init__(n_rois, device)
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # SDE function
        self.sde_func = NeuralSDEFunc(
            n_rois=n_rois,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            structural_connectivity=structural_connectivity,
            coupling_strength=coupling_strength
        )
        
        # Initial state network
        self.init_net = nn.Sequential(
            nn.Linear(n_rois, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_rois)
        )
        
        self.to(device)
    
    def forward(
        self,
        initial_state: Optional[torch.Tensor] = None,
        n_steps: int = 100,
        dt: float = 0.01,
        batch_size: int = 1,
        method: str = "euler"
    ) -> torch.Tensor:
        """
        Simulate brain dynamics using Neural SDE.
        
        Args:
            initial_state: Initial state (batch, n_rois) or None
            n_steps: Number of time steps
            dt: Time step size
            batch_size: Batch size if initial_state is None
            method: SDE solver method
            
        Returns:
            Simulated timeseries of shape (batch, n_rois, n_steps)
        """
        if initial_state is None:
            # Random initial state
            y0 = 0.1 * torch.randn(batch_size, self.n_rois, device=self.device)
        else:
            y0 = initial_state.to(self.device)
            batch_size = y0.shape[0]
        
        # Transform initial state
        y0 = self.init_net(y0)
        
        # Time points
        ts = torch.linspace(0, n_steps * dt, n_steps, device=self.device)
        
        if HAS_TORCHSDE:
            # Use torchsde for proper SDE integration
            trajectory = torchsde.sdeint(
                self.sde_func,
                y0,
                ts,
                method=method
            )
            # Shape: (n_steps, batch, n_rois) -> (batch, n_rois, n_steps)
            result = trajectory.permute(1, 2, 0)
        else:
            # Fallback to simple Euler-Maruyama
            result = self._euler_maruyama(y0, n_steps, dt)
        
        return result
    
    def _euler_maruyama(
        self,
        y0: torch.Tensor,
        n_steps: int,
        dt: float
    ) -> torch.Tensor:
        """
        Simple Euler-Maruyama integration.
        
        Args:
            y0: Initial state (batch, n_rois)
            n_steps: Number of steps
            dt: Time step
            
        Returns:
            Trajectory of shape (batch, n_rois, n_steps)
        """
        y = y0
        trajectory = [y]
        t = torch.tensor(0.0, device=self.device)
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=self.device))
        
        for _ in range(n_steps - 1):
            drift = self.sde_func.f(t, y)
            diffusion = self.sde_func.g(t, y)
            noise = torch.randn_like(y)
            
            y = y + drift * dt + diffusion * sqrt_dt * noise
            trajectory.append(y)
            t = t + dt
        
        return torch.stack(trajectory, dim=2)
    
    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of key model parameters."""
        params = {}
        
        # Get coupling if available
        if self.sde_func.coupling is not None:
            params['coupling'] = self.sde_func.coupling.detach().clone()
        
        # Get diffusion scale
        if hasattr(self.sde_func.diffusion_net, 'sigma'):
            params['sigma'] = self.sde_func.diffusion_net.sigma.detach().clone()
        
        return params
    
    def generate_bold(
        self,
        n_timepoints: int,
        tr: float = 0.72,
        dt: float = 0.001,
        batch_size: int = 1,
        initial_transient: int = 1000
    ) -> torch.Tensor:
        """
        Generate BOLD-like signal.
        
        Args:
            n_timepoints: Number of BOLD timepoints
            tr: Repetition time in seconds
            dt: Integration time step
            batch_size: Number of simulations
            initial_transient: Steps to discard
            
        Returns:
            BOLD signal of shape (batch, n_rois, n_timepoints)
        """
        # Calculate steps needed
        steps_per_tr = int(tr / dt)
        total_steps = initial_transient + n_timepoints * steps_per_tr
        
        # Simulate full trajectory
        full_traj = self.forward(
            initial_state=None,
            n_steps=total_steps,
            dt=dt,
            batch_size=batch_size
        )
        
        # Remove transient
        traj = full_traj[:, :, initial_transient:]
        
        # Downsample to TR
        bold = traj[:, :, ::steps_per_tr][:, :, :n_timepoints]
        
        return bold
