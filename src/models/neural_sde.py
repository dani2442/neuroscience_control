"""Neural SDE Model for brain dynamics simulation.

Uses neural networks to parameterize SDE drift and diffusion for
learnable complex brain dynamics.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base_model import BaseNeuroscienceModel
import torchsde


class DriftNetwork(nn.Module):
    """Neural network for the SDE drift term."""

    def __init__(self, state_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionNetwork(nn.Module):
    """Neural network for the SDE diffusion term (diagonal noise)."""

    def __init__(self, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralSDEFunc(nn.Module):
    """SDE function for torchsde. dX = f(X)dt + g(X)dW.

    State dimension is 2*n_rois (real || imag) for complex dynamics.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        structural_connectivity: Optional[torch.Tensor] = None,
        coupling_strength: float = 0.1,
    ):
        super().__init__()
        self.n_rois = n_rois
        state_dim = 2 * n_rois

        self.drift_net = DriftNetwork(state_dim, hidden_dim, n_layers)
        self.diffusion_net = DiffusionNetwork(state_dim, hidden_dim // 2)

        if structural_connectivity is not None:
            self.register_buffer('sc', structural_connectivity)
            self.coupling = nn.Parameter(torch.tensor(coupling_strength))
        else:
            self.sc = None
            self.coupling = None

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        drift = self.drift_net(y)
        if self.sc is not None:
            yr, yi = y[:, :self.n_rois], y[:, self.n_rois:]
            cr = self.coupling * (yr @ self.sc.T)
            ci = self.coupling * (yi @ self.sc.T)
            drift = drift + torch.cat([cr, ci], dim=1)
        return drift

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.diffusion_net(y)


class NeuralSDE(BaseNeuroscienceModel):
    """Neural SDE model for brain dynamics with complex states.

    Internally uses 2×n_rois real representation; external interface is complex.
    """

    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        structural_connectivity: Optional[torch.Tensor] = None,
        coupling_strength: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(n_rois, device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.sde_func = NeuralSDEFunc(
            n_rois=n_rois,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            structural_connectivity=structural_connectivity,
            coupling_strength=coupling_strength,
        )
        self.to(device)

    def forward(
        self,
        initial_state: torch.Tensor,
        n_steps: int = 100,
        dt: float = 0.72,
        method: str = "euler",
        dt_min: Optional[float] = 0.1,
    ) -> torch.Tensor:
        """Simulate brain dynamics via Neural SDE.

        Args:
            initial_state: Complex tensor (batch, n_rois).
            n_steps: Number of time steps.
            dt: Time step size.
            method: SDE solver method.
            dt_min: Sub-step for SDE solver.

        Returns:
            Complex timeseries (batch, n_rois, n_steps).
        """
        z = initial_state.to(self.device)
        if not torch.is_complex(z):
            z = torch.complex(z, torch.zeros_like(z))
        batch_size = z.shape[0]
        y0 = torch.cat([z.real, z.imag], dim=1)  # (batch, 2*n_rois)

        ts = torch.linspace(0, (n_steps - 1) * dt, n_steps, device=self.device)
        bm = torchsde.BrownianInterval(
            t0=ts[0], t1=ts[-1],
            size=(batch_size, 2 * self.n_rois),
            device=self.device, dtype=y0.dtype,
        )

        sdeint_kwargs = {"method": method, "bm": bm}
        if dt_min is not None:
            sdeint_kwargs["dt"] = dt_min

        trajectory = torchsde.sdeint(self.sde_func, y0, ts, **sdeint_kwargs)
        # (n_steps, batch, 2*n_rois) → complex (batch, n_rois, n_steps)
        return torch.complex(
            trajectory[:, :, :self.n_rois],
            trajectory[:, :, self.n_rois:],
        ).permute(1, 2, 0)

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        params = {}
        if self.sde_func.coupling is not None:
            params['coupling'] = self.sde_func.coupling.detach().clone()
        return params

    def get_model_config(self) -> Dict[str, Any]:
        coupling_strength = 0.1
        if self.sde_func.coupling is not None:
            coupling_strength = float(self.sde_func.coupling.detach().cpu().item())
        return {
            "n_rois": int(self.n_rois),
            "hidden_dim": int(self.hidden_dim),
            "n_layers": int(self.n_layers),
            "has_structural_connectivity": self.sde_func.sc is not None,
            "coupling_strength": coupling_strength,
        }
    
