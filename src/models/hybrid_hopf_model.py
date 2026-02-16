"""Hybrid mechanistic–neural Hopf Model for brain dynamics simulation.

Combines Hopf local oscillator dynamics with a learnable graph-coupling
network ψ_θ.  The SDE reads:

    dz_i = [(a + iω_i − |z_i|²) z_i
            + G Σ_j ψ_θ(z_j, z_i, C_ij)] dt + σ dW_i

where ψ_θ : ℂ³ → ℂ is a small MLP that replaces the fixed linear
diffusive coupling C_ij(z_j − z_i) of the classical Hopf model.

Native complex-valued SDE: state, drift, diffusion, and Brownian motion
are all complex.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base_model import BaseNeuroscienceModel
import torchsde


class CouplingNetwork(nn.Module):
    r"""Learnable edge-wise coupling function ψ_θ(z_j, z_i, C_ij) → ℂ.

    Internally converts complex inputs to a real representation
    (Re z_j, Im z_j, Re z_i, Im z_i, C_ij) ∈ ℝ⁵, processes through
    a real-valued MLP, and maps the 2-d output back to ℂ.
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        # Input: (Re z_j, Im z_j, Re z_i, Im z_i, C_ij) → 5
        # Output: (Re ψ, Im ψ) → 2
        in_dim = 5
        out_dim = 2
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.Tanh()])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z_j: torch.Tensor,
        z_i: torch.Tensor,
        c_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ψ_θ for every edge in a vectorised fashion.

        Args:
            z_j: Complex tensor (..., ) – source node states.
            z_i: Complex tensor (..., ) – target node states.
            c_ij: Real tensor (..., ) – structural connectivity weights.

        Returns:
            Complex tensor (..., ) – coupling contribution per edge.
        """
        features = torch.stack(
            [z_j.real, z_j.imag, z_i.real, z_i.imag, c_ij],
            dim=-1,
        )  # (..., 5)
        out = self.net(features)  # (..., 2)
        return torch.complex(out[..., 0], out[..., 1])


class HybridHopfSDEFunc(nn.Module):
    r"""SDE function for the hybrid Hopf model (torchsde-compatible).

    .. math::
        dz_i = \bigl[(a + i\omega_i - |z_i|^2)\,z_i
               + G\sum_j \psi_\theta(z_j, z_i, C_{ij})\bigr]\,dt
               + \sigma\,dW_i
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        n_rois: int,
        a,
        g,
        kappa,
        omega,
        structural_connectivity: torch.Tensor,
        coupling_net: CouplingNetwork,
        noise_sigma: float = 0.5,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.noise_sigma = noise_sigma
        self.a = a
        self.global_coupling = g
        self.kappa = kappa
        self.omega = omega
        self.structural_connectivity = structural_connectivity
        self.coupling_net = coupling_net

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: z_i·(κa + iω_i − κ|z_i|²) + G·Σ_j ψ_θ(z_j, z_i, C_ij)."""
        # --- local Hopf term ---
        z_sq = torch.abs(y) ** 2  # (batch, n_rois), real
        omega = self.omega.unsqueeze(0)  # (1, n_rois)
        local = y * (self.kappa * self.a - self.kappa * z_sq + 1j * omega)  # (batch, n_rois)

        # --- learned coupling term ---
        batch, n = y.shape
        # Expand to all (i, j) pairs: (batch, n_target, n_source)
        z_j = y.unsqueeze(1).expand(batch, n, n)  # source: dim 2
        z_i = y.unsqueeze(2).expand(batch, n, n)  # target: dim 1
        sc = self.structural_connectivity.unsqueeze(0).expand(batch, n, n)  # (batch, n, n) real

        psi = self.coupling_net(z_j, z_i, sc.to(y.real.dtype))  # (batch, n, n) complex

        coupling = self.global_coupling * psi.sum(dim=2)  # sum over j → (batch, n)
        return local + coupling

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.noise_sigma * torch.ones_like(y)


class HybridHopfModel(BaseNeuroscienceModel):
    r"""Hybrid mechanistic–neural Hopf oscillator model.

    dz_i = [z_i·(a + iω_i − |z_i|²) + G·Σ_j ψ_θ(z_j, z_i, C_ij)] dt + σ dW_i

    where ψ_θ is a learned coupling function.
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
        coupling_hidden_dim: int = 32,
        coupling_n_layers: int = 2,
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
        self.coupling_hidden_dim = coupling_hidden_dim
        self.coupling_n_layers = coupling_n_layers

        # Structural connectivity (default: identity)
        sc = (
            structural_connectivity.to(device)
            if structural_connectivity is not None
            else torch.eye(n_rois, device=device)
        )
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("structural_connectivity", sc)

        # Bifurcation parameter
        a_init = torch.tensor(initial_a, device=device)
        if learnable_a:
            self.a = nn.Parameter(a_init)
        else:
            self.register_buffer("a", a_init)

        # Global coupling
        g_init = torch.tensor(initial_g, device=device)
        if learnable_g:
            self.g = nn.Parameter(g_init)
        else:
            self.register_buffer("g", g_init)

        # kappa
        kappa_init = torch.tensor(initial_kappa, device=device)
        if learnable_kappa:
            self.kappa = nn.Parameter(kappa_init)
        else:
            self.register_buffer("kappa", kappa_init)

        # Intrinsic frequencies
        if omega is None:
            omega_init = torch.linspace(0.04, 0.07, n_rois, device=device) * 2 * torch.pi
        else:
            omega_init = omega.to(device)
        if learnable_omega:
            self.omega = nn.Parameter(omega_init)
        else:
            self.register_buffer("omega", omega_init)

        # Coupling network
        self.coupling_net = CouplingNetwork(
            hidden_dim=coupling_hidden_dim,
            n_layers=coupling_n_layers,
        )

        self.sde_func = HybridHopfSDEFunc(
            n_rois,
            self.a,
            self.g,
            self.kappa,
            self.omega,
            self.structural_connectivity,
            self.coupling_net,
            noise_sigma,
        )
        self.to(device)

    def _update_sde_func_params(self):
        self.sde_func.a = self.a
        self.sde_func.global_coupling = self.g
        self.sde_func.omega = self.omega
        self.sde_func.structural_connectivity = self.structural_connectivity
        self.sde_func.coupling_net = self.coupling_net

    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int = 100,
        dt: float = 0.72,
        method: str = "euler",
        dt_min: Optional[float] = 0.1,
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
            "a": self.a.detach().clone(),
            "g": self.g.detach().clone(),
            "omega": self.omega.detach().clone(),
        }

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "n_rois": int(self.n_rois),
            "noise_sigma": float(self.noise_sigma),
            "learnable_a": bool(self.learnable_a),
            "learnable_g": bool(self.learnable_g),
            "learnable_omega": bool(self.learnable_omega),
            "coupling_hidden_dim": int(self.coupling_hidden_dim),
            "coupling_n_layers": int(self.coupling_n_layers),
            "initial_a": float(self.a.detach().mean().cpu().item()),
            "initial_g": float(self.g.detach().cpu().item()),
        }

    def set_parameters(self, a=None, g=None, omega=None):
        """Set model parameters in-place."""
        with torch.no_grad():
            if a is not None:
                if isinstance(self.a, nn.Parameter):
                    self.a.copy_(torch.as_tensor(a, device=self.device))
                else:
                    self.a = torch.as_tensor(a, device=self.device)
            if g is not None:
                val = torch.tensor(g, device=self.device)
                if isinstance(self.g, nn.Parameter):
                    self.g.copy_(val)
                else:
                    self.g = val
            if omega is not None:
                val = torch.as_tensor(omega, device=self.device).float()
                if isinstance(self.omega, nn.Parameter):
                    self.omega.copy_(val)
                else:
                    self.omega = val
