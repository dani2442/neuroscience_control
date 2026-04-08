"""Runnable demo for the PE design module."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exp_design.basis import FourierBasis, LegendreBasis
from src.exp_design.criteria import CRITERIA
from src.exp_design.design import PersistentExcitationDesign
from src.exp_design.plotting import plot_design_summary


def build_basis(name: str, n_basis: int, horizon: float):
    """Construct one of the supported basis families."""
    key = name.strip().lower()
    if key == "fourier":
        return FourierBasis(n_basis=n_basis, horizon=horizon)
    if key == "legendre":
        return LegendreBasis(n_basis=n_basis, horizon=horizon)
    raise ValueError(f"Unsupported basis: {name}")


def _budget_tag(value: float) -> str:
    """Return a filename-safe tag for the derivative budget."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_demo(
    *,
    basis_name: str = "fourier",
    d: int = 1,
    horizon: float = 1.0,
    n_basis: int = 9,
    pe_order: int = 3,
    sobolev_order: int = 1,
    derivative_bound: float = 1.0,
    n_grid: int = 501,
    output_dir: str | Path = "paper/images/exp_design",
    criteria: tuple[str, ...] = CRITERIA,
    methods: tuple[str, ...] = ("backprop", "sdp"),
) -> dict[str, dict]:
    """Run the comparison and save a single summary figure."""
    if d <= 0:
        raise ValueError("d must be positive.")
    if isinstance(methods, str):
        methods = (methods,)
    methods = tuple(methods)
    if not methods:
        raise ValueError("methods must contain at least one optimization method.")
    invalid_methods = tuple(method for method in methods if method not in {"backprop", "sdp"})
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")

    basis = build_basis(basis_name, n_basis=n_basis, horizon=horizon)
    design = PersistentExcitationDesign(
        basis=basis,
        pe_order=pe_order,
        sobolev_order=sobolev_order,
        signal_dim=d,
        derivative_bound=derivative_bound,
        n_grid=n_grid,
    )
    results = design.compare_criteria(
        criteria=criteria,
        methods=methods,
        backprop_kwargs={"steps": 600, "lr": 5e-2, "restarts": 6, "seed": 0},
        sdp_kwargs={"refine_steps": 250, "refine_lr": 5e-2},
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_tag = _budget_tag(derivative_bound)
    summary_path = output_dir / f"{basis_name}_d{d}_C{budget_tag}_summary.pdf"
    plot_design_summary(
        results,
        grid=design.grid_np,
        save_path=summary_path,
        figsize=(4.6 * (d + 1), 4.0),
    )

    print(f"Saved summary figure to {summary_path}")
    print(f"Signal dimension: d={d}")
    print(
        "Constraint weights:",
        {order: float(weight) for order, weight in enumerate(design.constraint_weights_np)},
    )
    for criterion, method_results in results.items():
        print(f"\n{criterion.upper()}-optimal")
        for method, result in method_results.items():
            print(
                f"  {method:10s} "
                f"H^{sobolev_order}={design.sobolev_norm(result.coefficients):.6f} "
                f"lambda_min={result.metrics['lambda_min']:.6f} "
                f"log_det={result.metrics['log_det']:.6f} "
                f"trace_inv={result.metrics['trace_inv']:.6f}"
            )
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=("fourier", "legendre"), default="fourier")
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--output-dir", default="paper/images/exp_design")
    parser.add_argument("--n-basis", type=int, default=9)
    parser.add_argument("--pe-order", type=int, default=3)
    parser.add_argument("--sobolev-order", type=int, default=3)
    parser.add_argument("--derivative-bound", type=float, default=100.0)
    parser.add_argument("--n-grid", type=int, default=1001)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("backprop", "sdp"),
        default=("backprop",),
    )
    args = parser.parse_args()

    run_demo(
        basis_name=args.basis,
        d=args.d,
        output_dir=args.output_dir,
        n_basis=args.n_basis,
        pe_order=args.pe_order,
        sobolev_order=args.sobolev_order,
        derivative_bound=args.derivative_bound,
        n_grid=args.n_grid,
        methods=tuple(args.methods),
    )


if __name__ == "__main__":
    main()
