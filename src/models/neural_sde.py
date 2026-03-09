"""Neural SDE Model for brain dynamics simulation.

Uses neural networks to parameterize SDE drift and diffusion for
learnable complex brain dynamics.  The SDE state, drift, diffusion,
and Brownian motion are all **complex-valued**.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base_model import BaseNeuroscienceModel
import torchsde


class DriftNetwork(nn.Module):
    """Neural network for the SDE drift term.

    Accepts complex input ``(batch, n_rois)`` and returns complex output
    of the same shape.  Internally the complex tensor is flattened to a
    ``2 * n_rois`` real representation, processed by a real-valued MLP,
    and reshaped back to complex.

    When ``n_control_dims > 0`` the input dimension is
    ``2 * (n_rois + n_control_dims)`` (augmented state) but the output
    remains ``2 * n_rois`` (only biological nodes evolve).
    """

    def __init__(self, n_rois: int, hidden_dim: int = 64, n_layers: int = 2,
                 n_control_dims: int = 0):
        super().__init__()
        self.n_rois = n_rois
        self.n_control_dims = n_control_dims
        state_dim = 2 * (n_rois + n_control_dims)  # input: augmented real repr
        out_dim = 2 * n_rois  # output: only brain nodes
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, control: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: complex (batch, n_rois)
        x_real = torch.view_as_real(x).reshape(x.shape[0], -1)  # (batch, 2*n_rois)
        if self.n_control_dims > 0:
            if control is not None:
                ctrl = control.to(x.dtype)
                if not torch.is_complex(ctrl):
                    ctrl = torch.complex(ctrl, torch.zeros_like(ctrl))
                ctrl_real = torch.view_as_real(ctrl).reshape(x.shape[0], -1)  # (batch, 2*m)
            else:
                # No control provided: pad with zeros (zero-control = baseline)
                ctrl_real = torch.zeros(
                    x.shape[0], 2 * self.n_control_dims, dtype=x_real.dtype, device=x_real.device
                )
            x_real = torch.cat([x_real, ctrl_real], dim=1)  # (batch, 2*(n_rois+m))
        out = self.net(x_real)  # (batch, 2*n_rois)
        return torch.view_as_complex(out.reshape(x.shape[0], self.n_rois, 2))


class DiffusionNetwork(nn.Module):
    """Neural network for the SDE diffusion term (diagonal noise).

    Accepts complex input ``(batch, n_rois)`` and returns complex output
    of the same shape.  Internally uses a real-valued MLP with
    ``Softplus`` output activation so that both real and imaginary
    components of the diffusion coefficient are non-negative.
    """

    def __init__(self, n_rois: int, hidden_dim: int = 32):
        super().__init__()
        self.n_rois = n_rois
        state_dim = 2 * n_rois
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = torch.view_as_real(x).reshape(x.shape[0], -1)  # (batch, 2*n_rois)
        out = self.net(x_real)  # (batch, 2*n_rois)
        return torch.view_as_complex(out.reshape(x.shape[0], self.n_rois, 2))


class NeuralSDEFunc(nn.Module):
    """SDE function for torchsde.  dZ = f(Z)dt + g(Z)dW.

    The state ``Z`` is complex ``(batch, n_rois)``; drift ``f`` and
    diffusion ``g`` return complex tensors of the same shape.
    Complex Brownian motion :math:`W = W_1 + iW_2` is handled by
    ``torchsde`` when the initial condition is complex.
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
        n_control_dims: int = 0,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.n_control_dims = n_control_dims
        # Set by model.forward() before each sdeint call
        self.control_input: Optional[torch.Tensor] = None

        self.drift_net = DriftNetwork(n_rois, hidden_dim, n_layers, n_control_dims=n_control_dims)
        self.diffusion_net = DiffusionNetwork(n_rois, hidden_dim // 2)

        if structural_connectivity is not None:
            self.register_buffer('sc', structural_connectivity)
            self.coupling = nn.Parameter(torch.tensor(coupling_strength))
        else:
            self.sc = None
            self.coupling = None

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        drift = self.drift_net(y, control=self.control_input)
        if self.sc is not None:
            sc = self.sc.to(y.dtype)
            coupling = self.coupling * (y @ sc.T)
            drift = drift + coupling
        return drift

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.diffusion_net(y)


class NeuralSDE(BaseNeuroscienceModel):
    """Neural SDE model for brain dynamics with native complex states.

    The SDE state, drift, diffusion, and Brownian motion are complex-valued.
    External interface: complex in, complex out.
    """

    def __init__(
        self,
        n_rois: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        structural_connectivity: Optional[torch.Tensor] = None,
        coupling_strength: float = 0.1,
        device: str = "cpu",
        n_control_dims: int = 0,
    ):
        super().__init__(n_rois, device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_control_dims = n_control_dims

        self.sde_func = NeuralSDEFunc(
            n_rois=n_rois,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            structural_connectivity=structural_connectivity,
            coupling_strength=coupling_strength,
            n_control_dims=n_control_dims,
        )
        self.to(device)

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
        """Simulate brain dynamics via Neural SDE.

        Args:
            initial_state: Complex tensor (batch, n_rois).
            n_steps: Number of time steps.
            dt: Time step size.
            sde_type: SDE interpretation ('ito' or 'stratonovich').
            method: SDE solver method.
            dt_min: Sub-step for SDE solver.
            use_adjoint: Use torchsde adjoint solver.
            adjoint_method: Adjoint SDE solver method (defaults to ``method``).
            control: Optional control input (batch, n_control_dims). Constant in time.

        Returns:
            Complex timeseries (batch, n_rois, n_steps).
        """
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
            "n_control_dims": int(self.n_control_dims),
        }
    
