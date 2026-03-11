#!/usr/bin/env python3
"""
Model Comparison Script — thin wrapper.

All logic lives in :pymod:`src.utils.paper_pipeline`.  Prefer::

    python examples/train_models.py compare [OPTIONS]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paper_pipeline import run_compare_models


def main(argv: list[str] | None = None):
    """Parse CLI args and delegate to :func:`run_compare_models`."""
    parser = argparse.ArgumentParser(
        description="Compare trained Hopf and Neural SDE models",
    )
    parser.add_argument("--data-path", type=str, default="data/ts_young/ts_young_TR0.72.mat")
    parser.add_argument("--hopf-checkpoint", type=str, default="checkpoints/hopf_best.pt")
    parser.add_argument("--nsde-checkpoint", type=str, default="checkpoints/nsde_best.pt")
    parser.add_argument("--wandb-project", type=str, default="neuroscience-control")
    parser.add_argument("--run-name", type=str, default="model_comparison")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-simulations", type=int, default=10)
    parser.add_argument("--output-path", type=str, default="results/comparison_results.json")
    args = parser.parse_args(argv)

    return run_compare_models(
        data_path=args.data_path,
        hopf_checkpoint=args.hopf_checkpoint,
        nsde_checkpoint=args.nsde_checkpoint,
        device=args.device,
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        n_simulations=args.n_simulations,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
