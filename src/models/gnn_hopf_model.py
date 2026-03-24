"""GNN-Hopf Model: Graph-Neural-Network–style Hopf oscillator.

A memory-efficient alternative to the hybrid Hopf model.  Instead of
evaluating a neural network per *edge* (O(ROI²)), this model:

1.  Learns the coupling matrix **C** (initialized from structural
    connectivity, analogous to ``learnable_fc`` in the plain Hopf model).
2.  Performs a standard linear diffusive aggregation:
        m_i = Σ_j C_ij (z_j − z_i)          — one matrix-multiply, O(ROI²) FLOPS but no per-edge NN
3.  Applies a complex-valued node-wise MLP **ψ_θ** to the aggregated
    message *and* the local node state:
        coupling_i = ψ_θ(m_i, z_i)           — O(ROI) network evaluations

The SDE reads:

    dz_i = [(κa + iω_i − κ|z_i|²) z_i
            + G · ψ_θ(m_i, z_i)] dt + σ dW_i

This is analogous to a single message-passing layer in a Graph Neural
Network: aggregate neighbours linearly, then transform with a learnable
node function.

Native complex-valued SDE: state, drift, diffusion, and Brownian motion
are all complex.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base_model import BaseNeuroscienceModel
from .hybrid_hopf_model import ComplexLinear, PhasePreservingActivation
import torchsde


# ---------------------------------------------------------------------------
# Node-wise MLP (applied after aggregation, O(ROI))
# ---------------------------------------------------------------------------

class NodeNetwork(nn.Module):
    r"""Learnable per-node transform ψ_θ(m_i, z_i) → ℂ.

    A complex-valued MLP operating on each node independently.  It receives
    a 2-d complex input ``[m_i, z_i]`` (aggregated message + local state)
    and maps it to a scalar complex coupling value.

    All weights and biases are complex; the activation is phase-preserving.
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        # Input: (m_i, z_i) ∈ ℂ²  →  output ∈ ℂ¹
        in_dim = 2
        out_dim = 1
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.extend([ComplexLinear(d, hidden_dim), PhasePreservingActivation()])
            d = hidden_dim
        layers.append(ComplexLinear(d, out_dim))
        self.net = nn.Sequential(*layers)

        # Near-zero init for the output layer so the model starts close to
        # the pure-Hopf regime and gradually learns the neural correction.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(
        self,
        m_i: torch.Tensor,
        z_i: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ψ_θ for every node in a vectorised fashion.

        Args:
            m_i: Complex tensor (batch, n_rois) – aggregated neighbour messages.
            z_i: Complex tensor (batch, n_rois) – local node states.

        Returns:
            Complex tensor (batch, n_rois) – per-node coupling output.
        """
        features = torch.stack([m_i, z_i], dim=-1)  # (batch, n_rois, 2) complex
        out = self.net(features)  # (batch, n_rois, 1) complex
        return out.squeeze(-1)  # (batch, n_rois)


# ---------------------------------------------------------------------------
# SDE function
# ---------------------------------------------------------------------------

class GNNHopfSDEFunc(nn.Module):
    r"""SDE function for the GNN-Hopf model (torchsde-compatible).

    .. math::
        m_i &= \sum_j C_{ij}(z_j - z_i) \\
        dz_i &= \bigl[(\kappa a + i\omega_i - \kappa|z_i|^2)\,z_i
               + G\,\psi_\theta(m_i,\, z_i)\bigr]\,dt + \sigma\,dW_i

    When control inputs are present the aggregation extends over
    ``n_rois + n_control_dims`` augmented nodes.
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
        coupling_matrix: nn.Parameter,
        node_net: NodeNetwork,
        noise_sigma: float = 0.5,
        control_coupling: Optional[torch.Tensor] = None,
        n_control_dims: int = 0,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.noise_sigma = noise_sigma
        self.a = a
        self.global_coupling = g
        self.kappa = kappa
        self.omega = omega
        self.coupling_matrix = coupling_matrix
        self.node_net = node_net
        self.control_coupling = control_coupling
        self.n_control_dims = n_control_dims
        self.control_input: Optional[torch.Tensor] = None

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: (κa + iω − κ|z|²)z + G·ψ_θ(m, z)."""
        # --- local Hopf term ---
        z_sq = torch.abs(y) ** 2  # (batch, n_rois), real
        omega = self.omega.unsqueeze(0)  # (1, n_rois)
        local = y * (self.kappa * self.a - self.kappa * z_sq + 1j * omega)

        # --- linear diffusive aggregation m_i = Σ_j C_ij (z_j − z_i) ---
        C = self.coupling_matrix  # (n_rois, n_rois) or extended
        batch = y.shape[0]

        if self.n_control_dims > 0 and self.control_input is not None and self.control_coupling is not None:
            # Augmented-state coupling: [C | control_coupling]
            cc = self.control_coupling.to(y.dtype)  # (n_rois, m)
            C_ext = torch.cat([C.to(y.dtype), cc], dim=1)  # (n_rois, n_rois + m)
            u = self.control_input.to(y.dtype)  # (batch, m)
            z_aug = torch.cat([y, u], dim=1)  # (batch, n_rois + m)
            coupled_sum = z_aug @ C_ext.T  # (batch, n_rois)
            row_sum = C_ext.sum(dim=1, keepdim=True).T  # (1, n_rois)
            m = coupled_sum - y * row_sum  # (batch, n_rois)
        else:
            C_c = C.to(y.dtype)
            coupled_sum = y @ C_c.T  # (batch, n_rois)
            row_sum = C_c.sum(dim=1, keepdim=True).T  # (1, n_rois)
            m = coupled_sum - y * row_sum  # (batch, n_rois)

        # --- node-wise neural transform ---
        psi = self.node_net(m, y)  # (batch, n_rois)
        coupling = self.global_coupling * psi

        return local + coupling

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.noise_sigma * torch.ones_like(y)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GNNHopfModel(BaseNeuroscienceModel):
    r"""GNN-style Hopf oscillator model.

    Memory-efficient hybrid that applies a node-wise MLP after linear
    aggregation instead of an edge-wise MLP:

        m_i  = Σ_j C_ij (z_j − z_i)          (learned C, initialised from SC)
        dz_i = [(κa + iω_i − κ|z_i|²) z_i + G · ψ_θ(m_i, z_i)] dt + σ dW_i
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
        node_hidden_dim: int = 16,
        node_n_layers: int = 1,
        device: str = "cpu",
        learnable_a: bool = True,
        learnable_g: bool = True,
        learnable_kappa: bool = True,
        learnable_omega: bool = False,
        n_control_dims: int = 0,
    ):
        super().__init__(n_rois, device)
        self.noise_sigma = noise_sigma
        self.learnable_a = learnable_a
        self.learnable_g = learnable_g
        self.learnable_kappa = learnable_kappa
        self.learnable_omega = learnable_omega
        self.node_hidden_dim = node_hidden_dim
        self.node_n_layers = node_n_layers
        self.n_control_dims = n_control_dims

        # Structural connectivity → initial coupling matrix
        sc = (
            structural_connectivity.to(device)
            if structural_connectivity is not None
            else torch.eye(n_rois, device=device)
        )
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
        # Keep a frozen copy for reference / regularisation
        self.register_buffer("structural_connectivity", sc)
        # Learnable coupling matrix C, initialised from normalised SC
        self.coupling_matrix = nn.Parameter(sc.clone())

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

        # Node-wise neural network ψ_θ
        self.node_net = NodeNetwork(
            hidden_dim=node_hidden_dim,
            n_layers=node_n_layers,
        )

        # Control coupling: learnable weights from control nodes to brain ROIs
        if n_control_dims > 0:
            self.control_coupling = nn.Parameter(
                torch.randn(n_rois, n_control_dims, device=device) * 0.01
            )
        else:
            self.control_coupling = None

        self.sde_func = GNNHopfSDEFunc(
            n_rois,
            self.a,
            self.g,
            self.kappa,
            self.omega,
            self.coupling_matrix,
            self.node_net,
            noise_sigma,
            control_coupling=self.control_coupling,
            n_control_dims=n_control_dims,
        )
        self.to(device)

    def _update_sde_func_params(self):
        self.sde_func.a = self.a
        self.sde_func.global_coupling = self.g
        self.sde_func.kappa = self.kappa
        self.sde_func.omega = self.omega
        self.sde_func.coupling_matrix = self.coupling_matrix
        self.sde_func.node_net = self.node_net
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
            "a": self.a.detach().clone(),
            "g": self.g.detach().clone(),
            "omega": self.omega.detach().clone(),
            "coupling_matrix": self.coupling_matrix.detach().clone(),
        }

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "n_rois": int(self.n_rois),
            "noise_sigma": float(self.noise_sigma),
            "learnable_a": bool(self.learnable_a),
            "learnable_g": bool(self.learnable_g),
            "learnable_kappa": bool(self.learnable_kappa),
            "learnable_omega": bool(self.learnable_omega),
            "node_hidden_dim": int(self.node_hidden_dim),
            "node_n_layers": int(self.node_n_layers),
            "initial_a": float(self.a.detach().mean().cpu().item()),
            "initial_g": float(self.g.detach().cpu().item()),
            "initial_kappa": float(self.kappa.detach().cpu().item()),
            "n_control_dims": int(self.n_control_dims),
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
