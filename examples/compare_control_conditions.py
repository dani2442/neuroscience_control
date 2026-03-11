#!/usr/bin/env python3
"""
Compare model FC under LSD control conditions — thin wrapper.

All logic lives in :pymod:`src.utils.paper_pipeline`.  Prefer::

    python examples/train_models.py compare-conditions [OPTIONS]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paper_pipeline import run_compare_conditions


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and delegate to :func:`run_compare_conditions`."""
    parser = argparse.ArgumentParser(
        description="Compare model FC under LSD conditions (u=0,1,2).",
    )
    parser.add_argument("--checkpoints", nargs="+", type=str, default=None)
    parser.add_argument("--lsd-data-dir", type=str, default="data/lsd")
    parser.add_argument("--n-steps", type=int, default=240)
    parser.add_argument("--n-paths", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-labels", nargs="+", type=str, default=None)
    parser.add_argument("--ctrl-values", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)

    run_compare_conditions(
        checkpoints=args.checkpoints,
        lsd_data_dir=args.lsd_data_dir,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        device=args.device,
        model_labels=args.model_labels,
        ctrl_values=args.ctrl_values,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
