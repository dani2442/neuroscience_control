"""Hybrid Hopf + Neural Drift model.

The SDE reads:
    dz_i = [f_hopf(z) + f_theta(z)] dt + sigma dW_i

where:
    f_hopf(z)_i = (a + i*omega_i - |z_i|^2) * z_i + G * sum_j C_ij (z_j - z_i)
    f_theta(z)  = DriftNetwork(z)   (near-zero init, same arch as NeuralSDE)
    sigma       = learnable scalar > 0 (via softplus)

Learnable: a, G, sigma, f_theta weights.
Fixed:     omega (from FFT), kappa=1, SC (structural connectivity buffer).
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
import torchsde

from .base_model import BaseNeuroscienceModel
from .neural_sde import DriftNetwork


class HybridHopfNeuralSDEFunc(nn.Module):
    """torchsde-compatible SDE function for the hybrid model."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        n_rois: int,
        a: nn.Parameter,
        g: nn.Parameter,
        log_sigma: nn.Parameter,
        log_omega: nn.Parameter,
        structural_connectivity: torch.Tensor,
        drift_net: DriftNetwork,
        n_control_dims: int = 0,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.a = a
        self.global_coupling = g
        self.log_sigma = log_sigma
        self.log_omega = log_omega
        self.structural_connectivity = structural_connectivity
        self.drift_net = drift_net
        self.n_control_dims = n_control_dims
        self._control: Optional[torch.Tensor] = None

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # --- Hopf local term ---
        z_sq = torch.abs(y) ** 2                          # (batch, n_rois)
        omega = torch.exp(self.log_omega).unsqueeze(0)    # (1, n_rois), always positive
        local = y * (self.a - z_sq + 1j * omega)

        # --- Hopf coupling term ---
        sc = self.structural_connectivity.to(y.dtype)
        coupled_sum = y @ sc.T
        row_sum = sc.sum(dim=1, keepdim=True).T
        coupling = self.global_coupling * (coupled_sum - y * row_sum)

        # --- Neural correction ---
        correction = self.drift_net(y, self._control)

        return local + coupling + correction

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigma = torch.nn.functional.softplus(self.log_sigma)
        return sigma * torch.ones_like(y)


class HybridHopfNeuralModel(BaseNeuroscienceModel):
    """Hybrid Hopf + learned drift correction model.

    Starts as a pure Hopf (drift_net near zero) and learns corrections
    where the mechanistic model falls short.
    """

    def __init__(
        self,
        n_rois: int,
        structural_connectivity: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
        initial_a: float = -0.02,
        initial_g: float = 0.5,
        initial_sigma: float = 0.5,
        drift_hidden_dim: int = 64,
        drift_n_layers: int = 2,
        device: str = "cpu",
        n_control_dims: int = 0,
    ):
        super().__init__(n_rois, device)

        # SC: fixed buffer, row-normalized
        sc = (
            structural_connectivity.to(device)
            if structural_connectivity is not None
            else torch.eye(n_rois, device=device)
        )
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("structural_connectivity", sc)

        # omega: learnable per-ROI frequency stored as log(omega)
        if omega is None:
            omega = torch.linspace(0.04, 0.07, n_rois, device=device) * 2 * torch.pi
        self.log_omega = nn.Parameter(torch.log(omega.to(device).clamp(min=1e-6)))

        self.a = nn.Parameter(torch.tensor(initial_a, device=device))
        self.g = nn.Parameter(torch.tensor(initial_g, device=device))
        # log_sigma so that sigma = softplus(log_sigma) > 0
        inv_softplus_val = torch.log(torch.exp(torch.tensor(initial_sigma)) - 1).item()
        self.log_sigma = nn.Parameter(torch.tensor(inv_softplus_val, device=device))

        # Neural drift correction — near-zero output init
        self.drift_net = DriftNetwork(n_rois, drift_hidden_dim, drift_n_layers,
                                      n_control_dims=n_control_dims)
        with torch.no_grad():
            self.drift_net.net[-1].weight.mul_(0.01)
            self.drift_net.net[-1].bias.zero_()

        self.sde_func = HybridHopfNeuralSDEFunc(
            n_rois=n_rois,
            a=self.a,
            g=self.g,
            log_sigma=self.log_sigma,
            log_omega=self.log_omega,
            structural_connectivity=self.structural_connectivity,
            drift_net=self.drift_net,
            n_control_dims=n_control_dims,
        )
        self.to(device)

    def _update_sde_func(self) -> None:
        self.sde_func.a = self.a
        self.sde_func.global_coupling = self.g
        self.sde_func.log_sigma = self.log_sigma
        self.sde_func.log_omega = self.log_omega

    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int = 100,
        dt: float = 0.72,
        sde_type: str = "ito",
        method: str = "euler",
        dt_min: Optional[float] = 0.05,
        use_adjoint: bool = False,
        adjoint_method: Optional[str] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._update_sde_func()
        self.sde_func.sde_type = sde_type
        self.sde_func._control = control

        z = initial_state.to(self.device)
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))

        ts = torch.linspace(0, (n_steps - 1) * dt, n_steps, device=self.device)
        kwargs: Dict[str, Any] = {"method": method}
        if dt_min is not None:
            kwargs["dt"] = dt_min

        if use_adjoint:
            kwargs["adjoint_method"] = adjoint_method or "euler"
            traj = torchsde.sdeint_adjoint(self.sde_func, z, ts, **kwargs)
        else:
            traj = torchsde.sdeint(self.sde_func, z, ts, **kwargs)

        return traj.permute(1, 2, 0)  # (batch, n_rois, n_steps)

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        sigma = torch.nn.functional.softplus(self.log_sigma)
        omega = torch.exp(self.log_omega)
        return {
            "a": self.a.detach().clone(),
            "g": self.g.detach().clone(),
            "sigma": sigma.detach().clone(),
            "omega_mean_hz": (omega / (2 * torch.pi)).mean().detach().clone(),
        }

    def get_model_config(self) -> Dict[str, Any]:
        sigma = torch.nn.functional.softplus(self.log_sigma)
        linear_layers = [m for m in self.drift_net.net if isinstance(m, nn.Linear)]
        return {
            "n_rois": int(self.n_rois),
            "initial_a": float(self.a.detach().cpu()),
            "initial_g": float(self.g.detach().cpu()),
            "initial_sigma": float(sigma.detach().cpu()),
            "drift_hidden_dim": linear_layers[0].out_features if linear_layers else 64,
            "drift_n_layers": len(linear_layers) - 1,
            "n_control_dims": int(self.sde_func.n_control_dims),
        }
