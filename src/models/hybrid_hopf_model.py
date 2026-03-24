"""Hybrid mechanistic–neural Hopf Model for brain dynamics simulation.

Low-rank factored coupling with node-wise neural networks.  The SDE reads:

    k_j = φ_key(z_j)                             — key network  (O(n))
    m_i = Σ_j C_ij (k_j − k_i)                   — diffusive aggregation
    dz_i = [(κa + iω_i − κ|z_i|²) z_i
            + G · ψ_out(m_i, z_i)] dt + σ dW_i    — output network (O(n))

The coupling matrix C is optionally low-rank:  C ≈ L Rᵀ  (rank *r*).
  • Full-rank (``coupling_rank=None``):  C is n×n, initialised from SC.
  • Low-rank  (``coupling_rank=r``):     L, R ∈ ℝ^{n×r}, initialised
    from the top-*r* SVD of SC.  Aggregation costs O(n·r·d_k) instead
    of O(n²·d_k), and stores 2nr parameters instead of n².

Memory comparison at n=100, r=10, d_k=4, hidden=16:
  Old edge-wise model     →  10 000 NN evals/step, O(n² · h · T) memory
  New low-rank node model →  200 NN evals/step,    O(n · h · T) memory

Native complex-valued SDE: state, drift, diffusion, Brownian motion
are all complex.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .base_model import BaseNeuroscienceModel
import torchsde


# ── Complex building blocks (shared with gnn_hopf_model) ──────────────

class ComplexLinear(nn.Module):
    """Fully complex linear layer: out = W z + b with W, b ∈ ℂ."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.cfloat) * (in_features ** -0.5)
        )
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class PhasePreservingActivation(nn.Module):
    r"""Harmonic activation:  f(z) = (z / |z|) · tanh(|z|).

    Preserves phase while applying a bounded nonlinearity to the
    magnitude, following the harmonic-network convention.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = z.abs().clamp(min=1e-8)
        return z / mag * torch.tanh(mag)


# ── Node-wise networks ────────────────────────────────────────────────

def _build_complex_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int,
                        near_zero_output: bool = True) -> nn.Sequential:
    """Helper: build a complex-valued MLP with phase-preserving activations."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers.extend([ComplexLinear(d, hidden_dim), PhasePreservingActivation()])
        d = hidden_dim
    layers.append(ComplexLinear(d, out_dim))
    net = nn.Sequential(*layers)
    if near_zero_output:
        with torch.no_grad():
            net[-1].weight.mul_(0.01)
            net[-1].bias.zero_()
    return net


class KeyNetwork(nn.Module):
    r"""Per-node key encoder  φ_key : ℂ → ℂ^{d_k}.

    Enriches each node's scalar complex state into a ``d_k``-dimensional
    complex embedding *before* the linear aggregation step.
    """

    def __init__(self, d_key: int = 4, hidden_dim: int = 16, n_layers: int = 1):
        super().__init__()
        self.net = _build_complex_mlp(1, d_key, hidden_dim, n_layers, near_zero_output=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, n) complex  →  (batch, n, d_k) complex."""
        return self.net(z.unsqueeze(-1))  # (batch, n, 1) → (batch, n, d_k)


class OutputNetwork(nn.Module):
    r"""Per-node output transform  ψ_out : ℂ^{d_k + 1} → ℂ.

    Maps the aggregated message concatenated with the local state to a
    scalar complex coupling correction.
    """

    def __init__(self, d_key: int = 4, hidden_dim: int = 16, n_layers: int = 1):
        super().__init__()
        self.net = _build_complex_mlp(d_key + 1, 1, hidden_dim, n_layers, near_zero_output=True)

    def forward(self, m_i: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        """m_i: (batch, n, d_k), z_i: (batch, n)  →  (batch, n) complex."""
        features = torch.cat([m_i, z_i.unsqueeze(-1)], dim=-1)  # (batch, n, d_k+1)
        return self.net(features).squeeze(-1)  # (batch, n)


# ── Low-rank coupling helpers ────────────────────────────────────────

def _svd_init(sc: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialise L, R from the top-*rank* SVD of *sc*."""
    U, S, Vh = torch.linalg.svd(sc.float(), full_matrices=False)
    sqrt_S = S[:rank].sqrt()
    L = U[:, :rank] * sqrt_S.unsqueeze(0)   # (n, r)
    R = Vh[:rank, :].T * sqrt_S.unsqueeze(0) # (n, r)
    return L, R


# ── SDE function ──────────────────────────────────────────────────────

class HybridHopfSDEFunc(nn.Module):
    r"""SDE function for the low-rank hybrid Hopf model.

    .. math::
        k_j &= \phi_\text{key}(z_j) \\
        m_i &= \sum_j C_{ij}(k_j - k_i) \\
        dz_i &= \bigl[(\kappa a + i\omega_i - \kappa|z_i|^2)\,z_i
               + G\,\psi_\text{out}(m_i,\, z_i)\bigr]\,dt + \sigma\,dW_i

    ``C`` is either a full (n, n) matrix or factored as ``L @ R.T``.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        n_rois: int,
        a, g, kappa, omega,
        key_net: KeyNetwork,
        output_net: OutputNetwork,
        coupling_L: Optional[nn.Parameter],
        coupling_R: Optional[nn.Parameter],
        coupling_full: Optional[nn.Parameter],
        noise_sigma: float = 0.5,
        control_coupling: Optional[nn.Parameter] = None,
        n_control_dims: int = 0,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.noise_sigma = noise_sigma
        self.a = a
        self.global_coupling = g
        self.kappa = kappa
        self.omega = omega
        self.key_net = key_net
        self.output_net = output_net
        self.coupling_L = coupling_L
        self.coupling_R = coupling_R
        self.coupling_full = coupling_full
        self.control_coupling = control_coupling
        self.n_control_dims = n_control_dims
        self.control_input: Optional[torch.Tensor] = None

    # ── helpers for the two coupling modes ──

    def _aggregate_full(self, K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Full-rank aggregation: m_i = Σ_j C_ij (k_j − k_i).

        K: (batch, n, d_k),  C_full: (n, n)  →  (batch, n, d_k).
        """
        C = self.coupling_full
        assert C is not None
        batch, n, d_k = K.shape

        # Optionally extend C with control coupling columns
        if self.n_control_dims > 0 and self.control_input is not None and self.control_coupling is not None:
            cc = self.control_coupling  # (n, m)
            C_ext = torch.cat([C, cc], dim=1)  # (n, n + m)
            u = self.control_input.to(y.dtype)  # (batch, m)
            K_ctrl = self.key_net(u)  # (batch, m, d_k)
            K_aug = torch.cat([K, K_ctrl], dim=1)  # (batch, n+m, d_k)
        else:
            C_ext = C
            K_aug = K

        # m_i = Σ_j C_ij k_j  −  k_i Σ_j C_ij
        C_c = C_ext.to(K.dtype)  # (n, n_total)
        Ck = torch.einsum("ij,bjd->bid", C_c, K_aug)  # (batch, n, d_k)
        row_sum = C_c.sum(dim=1)  # (n_total →) (n,)
        m = Ck - K[:, :n] * row_sum.unsqueeze(0).unsqueeze(-1)  # (batch, n, d_k)
        return m

    def _aggregate_lowrank(self, K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Low-rank aggregation via C ≈ L R^T.

        m_i = Σ_j [LR^T]_ij (k_j − k_i)
            = L (R^T K_sum) − (L R^T 1) ⊙ k_i

        K: (batch, n, d_k), L: (n, r), R: (n, r)  →  (batch, n, d_k).
        """
        L = self.coupling_L
        R = self.coupling_R
        assert L is not None and R is not None

        # Control augmentation: extend R with control rows
        if self.n_control_dims > 0 and self.control_input is not None and self.control_coupling is not None:
            cc = self.control_coupling  # (n, m)
            # For low-rank, control columns are added as: C_ext = [L R^T | cc]
            # We handle the control part separately as a dense (n, m) block.
            u = self.control_input.to(y.dtype)  # (batch, m)
            K_ctrl = self.key_net(u)  # (batch, m, d_k)
            cc_c = cc.to(K.dtype)  # (n, m)
            ctrl_Ck = torch.einsum("im,bmd->bid", cc_c, K_ctrl)  # (batch, n, d_k)
            ctrl_rowsum = cc_c.sum(dim=1)  # (m →) (n,)
        else:
            ctrl_Ck = None
            ctrl_rowsum = None

        L_c = L.to(K.dtype)  # (n, r)
        R_c = R.to(K.dtype)  # (n, r)

        # Efficient low-rank path: O(n · r · d_k)
        RtK = torch.einsum("jr,bjd->brd", R_c, K)  # (batch, r, d_k)
        LRtK = torch.einsum("ir,brd->bid", L_c, RtK)  # (batch, n, d_k)

        # Row sums of C = L R^T:  (L R^T) 1 = L (R^T 1)
        Rt_ones = R_c.sum(dim=0)  # (r,)
        LRt_ones = L_c @ Rt_ones  # (n,)

        m = LRtK - K * LRt_ones.unsqueeze(0).unsqueeze(-1)  # (batch, n, d_k)

        # Add control contributions
        if ctrl_Ck is not None:
            m = m + ctrl_Ck - K * ctrl_rowsum.unsqueeze(0).unsqueeze(-1)

        return m

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift: (κa + iω − κ|z|²)z + G · ψ_out(m, z)."""
        # Soft-clamp state magnitude
        mag = y.abs().clamp(min=1e-8)
        scale = torch.where(mag > 10.0, 10.0 / mag, torch.ones_like(mag))
        y = y * scale

        # --- local Hopf term ---
        z_sq = torch.abs(y) ** 2
        omega = self.omega.unsqueeze(0)
        local = y * (self.kappa * self.a - self.kappa * z_sq + 1j * omega)

        # --- key embeddings ---
        K = self.key_net(y)  # (batch, n, d_k)

        # --- diffusive aggregation ---
        if self.coupling_full is not None:
            m = self._aggregate_full(K, y)
        else:
            m = self._aggregate_lowrank(K, y)

        # --- node-wise output network ---
        psi = self.output_net(m, y)  # (batch, n)
        coupling = self.global_coupling * psi

        return local + coupling

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.noise_sigma * torch.ones_like(y)


# ── Top-level model ──────────────────────────────────────────────────

class HybridHopfModel(BaseNeuroscienceModel):
    r"""Low-rank hybrid Hopf oscillator model.

    k_j = φ_key(z_j)
    m_i = Σ_j C_ij (k_j − k_i)        (C learned, optionally low-rank)
    dz_i = [(κa + iω_i − κ|z_i|²) z_i + G · ψ_out(m_i, z_i)] dt + σ dW_i

    Args:
        coupling_rank: Rank of the low-rank approximation of C.
            ``None`` → full n×n learnable matrix (like the old model).
            An integer *r* → C ≈ L R^T with L, R ∈ ℝ^{n×r}.
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
        coupling_hidden_dim: int = 16,
        coupling_n_layers: int = 1,
        coupling_rank: Optional[int] = 10,
        d_key: int = 4,
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
        self.coupling_hidden_dim = coupling_hidden_dim
        self.coupling_n_layers = coupling_n_layers
        self.coupling_rank = coupling_rank
        self.d_key = d_key
        self.n_control_dims = n_control_dims

        # Structural connectivity (default: identity)
        sc = (
            structural_connectivity.to(device)
            if structural_connectivity is not None
            else torch.eye(n_rois, device=device)
        )
        sc = sc / (sc.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("structural_connectivity", sc)

        # ── coupling matrix C ──
        if coupling_rank is not None:
            r = min(coupling_rank, n_rois)
            L_init, R_init = _svd_init(sc, r)
            self.coupling_L = nn.Parameter(L_init.to(device))
            self.coupling_R = nn.Parameter(R_init.to(device))
            self.coupling_full = None
            self._effective_rank = r
        else:
            self.coupling_L = None
            self.coupling_R = None
            self.coupling_full = nn.Parameter(sc.clone().to(device))
            self._effective_rank = n_rois

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

        # Node-wise networks
        self.key_net = KeyNetwork(d_key=d_key, hidden_dim=coupling_hidden_dim, n_layers=coupling_n_layers)
        self.output_net = OutputNetwork(d_key=d_key, hidden_dim=coupling_hidden_dim, n_layers=coupling_n_layers)

        # Control coupling
        if n_control_dims > 0:
            self.control_coupling = nn.Parameter(
                torch.randn(n_rois, n_control_dims, device=device) * 0.01
            )
        else:
            self.control_coupling = None

        self.sde_func = HybridHopfSDEFunc(
            n_rois,
            self.a, self.g, self.kappa, self.omega,
            self.key_net, self.output_net,
            self.coupling_L, self.coupling_R, self.coupling_full,
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
        self.sde_func.key_net = self.key_net
        self.sde_func.output_net = self.output_net
        self.sde_func.coupling_L = self.coupling_L
        self.sde_func.coupling_R = self.coupling_R
        self.sde_func.coupling_full = self.coupling_full
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
        d = {
            "a": self.a.detach().clone(),
            "g": self.g.detach().clone(),
            "omega": self.omega.detach().clone(),
        }
        if self.coupling_full is not None:
            d["coupling_matrix"] = self.coupling_full.detach().clone()
        else:
            d["coupling_L"] = self.coupling_L.detach().clone()
            d["coupling_R"] = self.coupling_R.detach().clone()
        return d

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "n_rois": int(self.n_rois),
            "noise_sigma": float(self.noise_sigma),
            "learnable_a": bool(self.learnable_a),
            "learnable_g": bool(self.learnable_g),
            "learnable_kappa": bool(self.learnable_kappa),
            "learnable_omega": bool(self.learnable_omega),
            "coupling_hidden_dim": int(self.coupling_hidden_dim),
            "coupling_n_layers": int(self.coupling_n_layers),
            "coupling_rank": self.coupling_rank,
            "d_key": int(self.d_key),
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
