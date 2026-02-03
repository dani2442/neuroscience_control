"""
Coupled Hopf Model for brain dynamics simulation.

The Hopf model describes oscillatory neural dynamics using coupled 
nonlinear oscillators at the bifurcation point.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .base_model import BaseNeuroscienceModel
import torchsde


class HopfSDEFunc(nn.Module):
    """
    SDE function for Hopf oscillator dynamics compatible with torchsde.
    
    Implements: dz = f(z)dt + g(z)dW
    where f is the Hopf dynamics and g is the diffusion.
    """
    
    noise_type = "diagonal"
    sde_type = "ito"
    
    def __init__(
        self,
        n_rois: int,
        a: torch.Tensor,
        g: torch.Tensor,
        omega: torch.Tensor,
        structural_connectivity: torch.Tensor,
        noise_sigma: float = 0.01
    ):
        """
        Initialize Hopf SDE function.
        
        Args:
            n_rois: Number of brain regions
            a: Bifurcation parameters (n_rois,)
            g: Global coupling strength
            omega: Intrinsic frequencies (n_rois,)
            structural_connectivity: Connectivity matrix (n_rois, n_rois)
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        self.n_rois = n_rois
        self.noise_sigma = noise_sigma
        
        # Store references to parameters (will be updated by parent model)
        self.a = a
        self.global_coupling = g  # Renamed to avoid shadowing the g() method required by torchsde
        self.omega = omega
        self.structural_connectivity = structural_connectivity
    
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift function for Hopf dynamics.
        
        y is stored as (batch, 2*n_rois) where first n_rois are real parts
        and second n_rois are imaginary parts.
        """
        # Split real and imaginary parts
        y_real = y[:, :self.n_rois]
        y_imag = y[:, self.n_rois:]
        
        # Compute |z|^2 = real^2 + imag^2
        z_squared = y_real ** 2 + y_imag ** 2
        
        # Local dynamics: z * (a + i*omega - |z|^2)
        # (x + iy) * (a - |z|^2 + i*omega)
        # = x*(a - |z|^2) - y*omega + i*(y*(a - |z|^2) + x*omega)
        a_minus_z2 = self.a.unsqueeze(0) - z_squared
        
        local_real = y_real * a_minus_z2 - y_imag * self.omega.unsqueeze(0)
        local_imag = y_imag * a_minus_z2 + y_real * self.omega.unsqueeze(0)
        
        # Coupling: G * C @ z (in complex form)
        # For real matrices and complex z: (C @ z) = C @ real + i * C @ imag
        coupling_real = self.global_coupling * torch.matmul(y_real, self.structural_connectivity.T)
        coupling_imag = self.global_coupling * torch.matmul(y_imag, self.structural_connectivity.T)
        
        drift_real = local_real + coupling_real
        drift_imag = local_imag + coupling_imag
        
        return torch.cat([drift_real, drift_imag], dim=1)
    
    def g_func(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function for Hopf dynamics.
        
        Returns diagonal diffusion (same noise for all components).
        """
        return self.noise_sigma * torch.ones_like(y)
    
    # torchsde expects 'g' method, but we've defined g as a parameter
    # So we alias it
    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.g_func(t, y)


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
        
        # Create SDE function for torchsde integration
        self.sde_func = HopfSDEFunc(
            n_rois=n_rois,
            a=self.a,
            g=self.g,
            omega=self.omega,
            structural_connectivity=self.structural_connectivity,
            noise_sigma=noise_sigma
        )
        
        self.to(device)
    
    def _update_sde_func_params(self):
        """Update SDE function parameters from model parameters."""
        self.sde_func.a = self.a
        self.sde_func.global_coupling = self.g
        self.sde_func.omega = self.omega
        self.sde_func.structural_connectivity = self.structural_connectivity
    
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
        dt: float = 0.72,
        batch_size: int = 1,
        return_complex: bool = False,
        method: str = "euler",
        dt_min: Optional[float] = 0.1
    ) -> torch.Tensor:
        """
        Simulate brain dynamics using SDE integration.
        
        Args:
            initial_state: Initial complex state (batch, n_rois) or None for random
            n_steps: Number of time steps
            dt: Time step size
            batch_size: Batch size (used if initial_state is None)
            return_complex: Whether to return complex state or just real part
            method: SDE solver method ('euler', 'milstein', 'srk', etc.)
            dt_min: Minimum time step for adaptive solvers
            
        Returns:
            Simulated timeseries of shape (batch, n_rois, n_steps)
        """
        # Update SDE function parameters
        self._update_sde_func_params()
        
        if initial_state is None:
            # Random initial conditions near origin (real representation)
            y0_real = 0.1 * torch.randn(batch_size, self.n_rois, device=self.device)
            y0_imag = 0.1 * torch.randn(batch_size, self.n_rois, device=self.device)
            y0 = torch.cat([y0_real, y0_imag], dim=1)
        else:
            # Convert complex initial state to real representation
            z = initial_state.to(self.device)
            batch_size = z.shape[0]
            if torch.is_complex(z):
                y0 = torch.cat([z.real, z.imag], dim=1)
            else:
                # Assume it's already in real representation
                y0 = z
        
        # Time points
        ts = torch.linspace(0, (n_steps - 1) * dt, n_steps, device=self.device)
        
        # Use BrownianInterval which is more robust for long time horizons
        bm = torchsde.BrownianInterval(
            t0=ts[0],
            t1=ts[-1],
            size=(batch_size, 2 * self.n_rois),
            device=self.device,
            dtype=y0.dtype,
        )
        
        # Use torchsde for proper SDE integration
        sdeint_kwargs = {"method": method, "bm": bm}
        if dt_min is not None:
            sdeint_kwargs["dt"] = dt_min
        
        trajectory = torchsde.sdeint(
            self.sde_func,
            y0,
            ts,
            **sdeint_kwargs
        )
        # Shape: (n_steps, batch, 2*n_rois)
        # Extract real part (first n_rois dimensions)
        traj_real = trajectory[:, :, :self.n_rois]
        traj_imag = trajectory[:, :, self.n_rois:]
        
        if return_complex:
            result = torch.complex(traj_real, traj_imag)
        else:
            result = traj_real
        
        # Permute: (n_steps, batch, n_rois) -> (batch, n_rois, n_steps)
        result = result.permute(1, 2, 0)
        
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
    