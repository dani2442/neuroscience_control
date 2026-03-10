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
        dz = \\bigl[(\\kappa a + i\\omega - \\kappa|z|^2)\\,z
             + G\\sum_j C_{ij}(z_j - z_i)\\bigr]\\,dt
             + \\sigma\\,dW

    where :math:`W` is a complex Brownian motion.

    When control inputs are present the coupling sum extends over
    ``n_rois + n_control_dims`` augmented nodes (the auxiliary control
    nodes are constant in time and do not evolve).
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, n_rois, a, g, kappa, omega, structural_connectivity, noise_sigma=0.5, fc_matrix=None,
                 control_coupling=None, n_control_dims=0):
        super().__init__()
        self.n_rois = n_rois
        self.noise_sigma = noise_sigma
        self.a = a
        self.kappa = kappa
        self.global_coupling = g
        self.omega = omega
        self.structural_connectivity = structural_connectivity
        self.fc_matrix = fc_matrix  # Learnable FC matrix (nn.Parameter or None)
        self.control_coupling = control_coupling  # (n_rois, n_control_dims) or None
        self.n_control_dims = n_control_dims
        # Set by model.forward() before each sdeint call
        self.control_input: Optional[torch.Tensor] = None  # (batch, n_control_dims) complex

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: z·(κa + iω − κ|z|²) + G·Σ_j C̃_ij(z̃_j−z_i)  (complex)."""
        z_sq = self.kappa*torch.abs(y) ** 2  # |z|², real
        omega = self.omega.unsqueeze(0)
        local = y * (self.a*self.kappa - z_sq + 1j * omega)
        # Use learnable FC matrix if available, otherwise structural connectivity
        sc = self.fc_matrix if self.fc_matrix is not None else self.structural_connectivity
        sc = sc.to(y.dtype)

        # --- Augmented-state coupling (paper formulation) ---
        if self.n_control_dims > 0 and self.control_input is not None and self.control_coupling is not None:
            # Extend the coupling matrix: [sc | control_coupling] -> (n_rois, n_rois + m)
            cc = self.control_coupling.to(y.dtype)  # (n_rois, m)
            sc_ext = torch.cat([sc, cc], dim=1)  # (n_rois, n_rois + m)
            # Augmented state: [y | u]  u is constant control (batch, m)
            u = self.control_input.to(y.dtype)  # (batch, m)
            z_aug = torch.cat([y, u], dim=1)  # (batch, n_rois + m)
            coupled_sum = z_aug @ sc_ext.T  # (batch, n_rois)
            row_sum = sc_ext.sum(dim=1, keepdim=True).T  # (1, n_rois)
            coupling = self.global_coupling * (coupled_sum - y * row_sum)
        else:
            coupled_sum = y @ sc.T
            row_sum = sc.sum(dim=1, keepdim=True).T
            coupling = self.global_coupling * (coupled_sum - y * row_sum)
        return local + coupling

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.noise_sigma * torch.ones_like(y)


class CoupledHopfModel(BaseNeuroscienceModel):
    """Coupled Hopf oscillator model.

    dz/dt = z·(κa + iω − κ|z|²) + G·Σ_j C_ij(z_j−z_i) + σ·dW
    """

    def __init__(
        self,
        n_rois: int,
        structural_connectivity: Optional[torch.Tensor] = None,
        initial_a: float = -0.02,
        initial_g: float = 0.5,
        initial_kappa: float = 0.1,
        omega: Optional[torch.Tensor] = None,
        noise_sigma: float = 0.5,
        device: str = "cpu",
        learnable_a: bool = True,
        learnable_g: bool = True,
        learnable_kappa: bool = False,
        learnable_omega: bool = False,
        learnable_fc: bool = False,
        n_control_dims: int = 0,
    ):
        super().__init__(n_rois, device)
        self.noise_sigma = noise_sigma
        self.learnable_a = learnable_a
        self.learnable_g = learnable_g
        self.learnable_omega = learnable_omega
        self.learnable_fc = learnable_fc
        self.n_control_dims = n_control_dims

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

        # Learnable FC matrix: initialized from structural connectivity
        if learnable_fc:
            self.fc_matrix = nn.Parameter(sc.clone())
        else:
            self.fc_matrix = None

        # Control coupling: learnable weights from control nodes to brain ROIs
        if n_control_dims > 0:
            self.control_coupling = nn.Parameter(
                torch.randn(n_rois, n_control_dims, device=device) * 0.01
            )
        else:
            self.control_coupling = None

        self.sde_func = HopfSDEFunc(
            n_rois, self.a, self.g, self.kappa, self.omega,
            self.structural_connectivity, noise_sigma,
            fc_matrix=self.fc_matrix,
            control_coupling=self.control_coupling,
            n_control_dims=n_control_dims,
        )
        self.to(device)

    def _update_sde_func_params(self):
        self.sde_func.a = self.a
        self.sde_func.global_coupling = self.g
        self.sde_func.omega = self.omega
        self.sde_func.structural_connectivity = self.structural_connectivity
        self.sde_func.fc_matrix = self.fc_matrix
        self.sde_func.control_coupling = self.control_coupling

    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int = 100,
        dt: float = 0.72,
        sde_type: str = "ito",
        method: str = "euler",
        dt_min: Optional[float] = 0.1,
        use_adjoint: bool = False,
        adjoint_method: Optional[str] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simulate brain dynamics via SDE integration.

        Args:
            initial_state: Complex tensor (batch, n_rois).
            n_steps: Number of time steps.
            dt: Time step size.
            sde_type: SDE interpretation ('ito' or 'stratonovich').
            method: SDE solver ('euler', 'milstein', ...).
            dt_min: Sub-step for SDE solver.
            use_adjoint: Use torchsde adjoint solver.
            adjoint_method: Adjoint SDE solver method (defaults to ``method``).
            control: Optional control input (batch, n_control_dims). Constant in time.

        Returns:
            Complex timeseries (batch, n_rois, n_steps).
        """
        self._update_sde_func_params()

        z = initial_state.to(self.device)
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))
        self.sde_func.sde_type = sde_type

        # Set control input on SDE func (constant in time)
        if control is not None and self.n_control_dims > 0:
            ctrl = control.to(self.device)
            if not torch.is_complex(ctrl):
                ctrl = torch.complex(ctrl, torch.zeros_like(ctrl))
            self.sde_func.control_input = ctrl
        else:
            self.sde_func.control_input = None

        ts = torch.linspace(0, (n_steps - 1) * dt, n_steps, device=self.device)

        sdeint_kwargs = {"method": method}
        if dt_min is not None:
            sdeint_kwargs["dt"] = dt_min

        if use_adjoint:
            resolved_adjoint_method = adjoint_method or "adjoint_reversible_heun"
            sdeint_kwargs["adjoint_method"] = resolved_adjoint_method
            trajectory = torchsde.sdeint_adjoint(
                self.sde_func,
                z,
                ts,
                **sdeint_kwargs,
            )
        else:
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
            "learnable_fc": bool(self.learnable_fc),
            "initial_a": float(self.a.detach().mean().cpu().item()),
            "initial_g": float(self.g.detach().cpu().item()),
            "n_control_dims": int(self.n_control_dims),
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
    
