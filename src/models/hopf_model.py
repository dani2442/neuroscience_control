"""Coupled Hopf Model for brain dynamics simulation.

Complex oscillatory neural dynamics using coupled nonlinear oscillators.
Native complex-valued SDE: state, drift, diffusion, and Brownian motion
are all complex.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base_model import BaseNeuroscienceModel
import torchsde


class HopfSDEFunc(nn.Module):
    """SDE function for Hopf oscillator dynamics (torchsde-compatible).

    State is complex: ``(batch, n_rois)`` with ``dtype=complex64/128``.

    .. math::
        dz = \\bigl[(a + i\\omega - |z|^2)\\,z
             + G\\sum_j C_{ij}(z_j - z_i)\\bigr]\\,dt
             + \\sigma\\,dW

    where :math:`W` is a complex Brownian motion.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, n_rois, a, g, kappa, omega, structural_connectivity, noise_sigma=0.5):
        super().__init__()
        self.n_rois = n_rois
        self.noise_sigma = noise_sigma
        self.a = a
        self.kappa = kappa
        self.global_coupling = g
        self.omega = omega
        self.structural_connectivity = structural_connectivity

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: z·(a + iω − |z|²) + G·Σ_j C_ij(z_j−z_i)  (complex)."""
        z_sq = self.kappa*torch.abs(y) ** 2  # |z|², real
        omega = self.omega.unsqueeze(0)
        local = y * (self.a*self.kappa - z_sq + 1j * omega)
        sc = self.structural_connectivity.to(y.dtype)
        coupled_sum = y @ sc.T
        row_sum = sc.sum(dim=1, keepdim=True).T
        coupling = self.global_coupling * (coupled_sum - y * row_sum)
        return local + coupling

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.noise_sigma * torch.ones_like(y)


class CoupledHopfModel(BaseNeuroscienceModel):
    """Coupled Hopf oscillator model.

    dz/dt = z·(a + iω − |z|²) + G·Σ_j C_ij(z_j−z_i) + σ·dW
    """

    def __init__(
        self,
        n_rois: int,
        structural_connectivity: Optional[torch.Tensor] = None,
        initial_a: float = -0.02,
        initial_g: float = 0.5,
        initial_kappa: float = 0.5,
        omega: Optional[torch.Tensor] = None,
        noise_sigma: float = 0.5,
        device: str = "cpu",
        learnable_a: bool = True,
        learnable_g: bool = True,
        learnable_kappa: bool = True,
        learnable_omega: bool = False,
    ):
        super().__init__(n_rois, device)
        self.noise_sigma = noise_sigma
        self.learnable_a = learnable_a
        self.learnable_g = learnable_g
        self.learnable_omega = learnable_omega

        # Structural connectivity (default: identity)
        sc = structural_connectivity.to(device) if structural_connectivity is not None else torch.eye(n_rois, device=device)
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('structural_connectivity', sc)

        # Bifurcation parameter
        a_init = torch.tensor(initial_a, device=device)
        self.a = nn.Parameter(a_init) if learnable_a else self.register_buffer('a', a_init) or self.a

        kappa_init = torch.tensor(initial_kappa, device=device)
        self.kappa = nn.Parameter(kappa_init) if learnable_kappa else self.register_buffer('kappa', kappa_init) or self.kappa

        # Global coupling
        g_init = torch.tensor(initial_g, device=device)
        self.g = nn.Parameter(g_init) if learnable_g else self.register_buffer('g', g_init) or self.g

        # Intrinsic frequencies
        if omega is None:
            omega = torch.linspace(0.04, 0.07, n_rois, device=device) * 2 * torch.pi
        else:
            omega = omega.to(device)
        self.omega = nn.Parameter(omega) if learnable_omega else self.register_buffer('omega', omega) or self.omega

        self.sde_func = HopfSDEFunc(n_rois, self.a, self.g, self.kappa, self.omega, self.structural_connectivity, noise_sigma)
        self.to(device)

    def _update_sde_func_params(self):
        self.sde_func.a = self.a
        self.sde_func.global_coupling = self.g
        self.sde_func.omega = self.omega
        self.sde_func.structural_connectivity = self.structural_connectivity

    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int = 100,
        dt: float = 0.72,
        method: str = "euler",
        dt_min: Optional[float] = 0.05,
    ) -> torch.Tensor:
        """Simulate brain dynamics via SDE integration.

        Args:
            initial_state: Complex tensor (batch, n_rois).
            n_steps: Number of time steps.
            dt: Time step size.
            method: SDE solver ('euler', 'milstein', ...).
            dt_min: Sub-step for SDE solver.

        Returns:
            Complex timeseries (batch, n_rois, n_steps).
        """
        self._update_sde_func_params()

        z = initial_state.to(self.device)
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))

        ts = torch.linspace(0, (n_steps - 1) * dt, n_steps, device=self.device)

        sdeint_kwargs = {"method": method}
        if dt_min is not None:
            sdeint_kwargs["dt"] = dt_min

        trajectory = torchsde.sdeint(self.sde_func, z, ts, **sdeint_kwargs)
        # (n_steps, batch, n_rois), complex → (batch, n_rois, n_steps)
        return trajectory.permute(1, 2, 0)

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'a': self.a.detach().clone(),
            'g': self.g.detach().clone(),
            'omega': self.omega.detach().clone(),
        }

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "n_rois": int(self.n_rois),
            "noise_sigma": float(self.noise_sigma),
            "learnable_a": bool(self.learnable_a),
            "learnable_g": bool(self.learnable_g),
            "learnable_omega": bool(self.learnable_omega),
            "initial_a": float(self.a.detach().mean().cpu().item()),
            "initial_g": float(self.g.detach().cpu().item()),
        }

    def set_parameters(self, a=None, g=None, omega=None):
        """Set model parameters in-place."""
        def _to_tensor(val):
            if isinstance(val, torch.Tensor):
                return val.to(self.device)
            return torch.tensor(val, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            if a is not None:
                val = _to_tensor(a)
                target = self.a
                target.copy_(val) if isinstance(target, nn.Parameter) else setattr(self, 'a', val)
            if g is not None:
                val = _to_tensor(g)
                target = self.g
                target.copy_(val) if isinstance(target, nn.Parameter) else setattr(self, 'g', val)
            if omega is not None:
                val = _to_tensor(omega)
                target = self.omega
                target.copy_(val) if isinstance(target, nn.Parameter) else setattr(self, 'omega', val)
    
