"""
Coupled Hopf Model for brain dynamics simulation.

The Hopf model describes oscillatory neural dynamics using coupled 
nonlinear oscillators at the bifurcation point.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .base_model import BaseNeuroscienceModel


class CoupledHopfModel(BaseNeuroscienceModel):
    """
    Coupled Hopf oscillator model for brain dynamics.
    
    The model simulates brain activity using coupled nonlinear oscillators
    where each brain region is represented by a Hopf oscillator.
    
    The dynamics are governed by:
        dz/dt = z * (a + i*omega - |z|^2) + G * C @ z + noise
    
    where z is complex, a is the bifurcation parameter, omega is the 
    intrinsic frequency, G is global coupling, and C is the structural
    connectivity matrix.
    """
    
    def __init__(
        self,
        n_rois: int,
        structural_connectivity: Optional[torch.Tensor] = None,
        initial_a: float = -0.02,
        initial_g: float = 0.5,
        omega: Optional[torch.Tensor] = None,
        noise_sigma: float = 0.01,
        device: str = "cpu",
        learnable_a: bool = True,
        learnable_g: bool = True,
        learnable_omega: bool = False
    ):
        """
        Initialize Coupled Hopf Model.
        
        Args:
            n_rois: Number of brain regions
            structural_connectivity: Connectivity matrix (n_rois x n_rois)
            initial_a: Initial bifurcation parameter
            initial_g: Initial global coupling strength
            omega: Intrinsic frequencies for each region
            noise_sigma: Noise standard deviation
            device: Device to run on
            learnable_a: Whether bifurcation parameter is learnable
            learnable_g: Whether global coupling is learnable
            learnable_omega: Whether frequencies are learnable
        """
        super().__init__(n_rois, device)
        
        self.noise_sigma = noise_sigma
        self.learnable_a = learnable_a
        self.learnable_g = learnable_g
        self.learnable_omega = learnable_omega
        
        # Structural connectivity
        if structural_connectivity is None:
            # Use identity as default (no coupling between regions)
            sc = torch.eye(n_rois, device=device)
        else:
            sc = structural_connectivity.to(device)
        
        # Normalize SC
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('structural_connectivity', sc)
        
        # Bifurcation parameter (controls oscillation amplitude)
        if learnable_a:
            self.a = nn.Parameter(torch.full((n_rois,), initial_a, device=device))
        else:
            self.register_buffer('a', torch.full((n_rois,), initial_a, device=device))
        
        # Global coupling strength
        if learnable_g:
            self.g = nn.Parameter(torch.tensor(initial_g, device=device))
        else:
            self.register_buffer('g', torch.tensor(initial_g, device=device))
        
        # Intrinsic frequencies
        if omega is None:
            omega = torch.linspace(0.04, 0.07, n_rois, device=device) * 2 * torch.pi
        else:
            omega = omega.to(device)
        
        if learnable_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer('omega', omega)
        
        self.to(device)
    
    def hopf_dynamics(
        self,
        z: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Compute one step of Hopf dynamics.
        
        Args:
            z: Complex state of shape (batch, n_rois)
            dt: Time step
            
        Returns:
            Updated complex state
        """
        # Compute |z|^2
        z_squared = torch.abs(z) ** 2
        
        # Local dynamics: z * (a + i*omega - |z|^2)
        local = z * (self.a.unsqueeze(0) + 1j * self.omega.unsqueeze(0) - z_squared)
        
        # Coupling: G * C @ z
        # z has shape (batch, n_rois), SC has shape (n_rois, n_rois)
        # We need to compute SC @ z^T for each batch, then transpose back
        # Alternatively: (z @ SC^T) which gives (batch, n_rois)
        # Convert SC to complex for matmul compatibility
        sc_complex = self.structural_connectivity.to(z.dtype)
        coupling = self.g * torch.matmul(z, sc_complex.T)
        
        # Add noise
        noise = self.noise_sigma * torch.randn_like(z.real) + \
                1j * self.noise_sigma * torch.randn_like(z.imag)
        
        # Euler integration
        z_new = z + dt * (local + coupling) + torch.sqrt(torch.tensor(dt, device=z.device)) * noise
        
        return z_new


    
    def forward(
        self,
        initial_state: Optional[torch.Tensor] = None,
        n_steps: int = 100,
        dt: float = 0.01,
        batch_size: int = 1,
        return_complex: bool = False
    ) -> torch.Tensor:
        """
        Simulate brain dynamics.
        
        Args:
            initial_state: Initial complex state (batch, n_rois) or None for random
            n_steps: Number of time steps
            dt: Time step size
            batch_size: Batch size (used if initial_state is None)
            return_complex: Whether to return complex state or just real part
            
        Returns:
            Simulated timeseries of shape (batch, n_rois, n_steps)
        """
        if initial_state is None:
            # Random initial conditions near origin
            z = 0.1 * (torch.randn(batch_size, self.n_rois, device=self.device) + 
                       1j * torch.randn(batch_size, self.n_rois, device=self.device))
        else:
            z = initial_state.to(self.device)
            batch_size = z.shape[0]
        
        # Store trajectory
        trajectory = []
        
        for _ in range(n_steps):
            z = self.hopf_dynamics(z, dt)
            if return_complex:
                trajectory.append(z)
            else:
                trajectory.append(z.real)
        
        # Stack: (n_steps, batch, n_rois) -> (batch, n_rois, n_steps)
        result = torch.stack(trajectory, dim=2)
        
        if not return_complex:
            result = result.real if torch.is_complex(result) else result
        
        return result
    
    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of model parameters."""
        params = {
            'a': self.a.detach().clone(),
            'g': self.g.detach().clone(),
            'omega': self.omega.detach().clone(),
        }
        return params
    
    def set_parameters(
        self,
        a: Optional[torch.Tensor] = None,
        g: Optional[float] = None,
        omega: Optional[torch.Tensor] = None
    ):
        """
        Set model parameters.
        
        Args:
            a: Bifurcation parameters
            g: Global coupling
            omega: Intrinsic frequencies
        """
        with torch.no_grad():
            if a is not None:
                if self.learnable_a:
                    self.a.copy_(a.to(self.device))
                else:
                    self.a = a.to(self.device)
            
            if g is not None:
                if self.learnable_g:
                    self.g.copy_(torch.tensor(g, device=self.device))
                else:
                    self.g = torch.tensor(g, device=self.device)
            
            if omega is not None:
                if self.learnable_omega:
                    self.omega.copy_(omega.to(self.device))
                else:
                    self.omega = omega.to(self.device)
    
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
            batch_size=batch_size,
            return_complex=False
        )
        
        # Remove transient
        traj = full_traj[:, :, initial_transient:]
        
        # Downsample to TR
        bold = traj[:, :, ::steps_per_tr][:, :, :n_timepoints]
        
        return bold
