#!/usr/bin/env python3
r"""
Update LaTeX paper tables — thin wrapper.

All logic lives in :pymod:`src.utils.paper_pipeline`.  Prefer::

    python examples/train_models.py update-tables [OPTIONS]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paper_pipeline import run_update_tables


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and delegate to :func:`run_update_tables`."""
    parser = argparse.ArgumentParser(
        description="Patch paper LaTeX tables with fresh metrics.",
    )
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to the paper metrics JSON")
    parser.add_argument("--tex", type=str, default=None,
                        help="Explicit .tex file to patch")
    parser.add_argument("--table-label", type=str, default=None,
                        help="LaTeX table label (e.g. tab:model_comparison)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print patched table rows without writing")
    args = parser.parse_args(argv)

    run_update_tables(
        metrics_json=args.metrics,
        tex_target=args.tex,
        table_label=args.table_label,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
