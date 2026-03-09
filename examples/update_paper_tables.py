#!/usr/bin/env python3
"""
Update LaTeX paper tables from training metrics JSON files.

Reads results/lsd_paper_metrics.json (produced by train_models.py paper)
and rewrites the numeric cells in the LSD model-comparison table
(Table~\ref{tab:lsd_model_comparison}) inside
paper/sections/03_results.tex.

Usage
-----
    python examples/update_paper_tables.py
    python examples/update_paper_tables.py --metrics results/lsd_paper_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Metric keys that appear in the JSON              LaTeX column name
# ---------------------------------------------------------------------------
METRIC_COLS: list[tuple[str, str]] = [
    ("fc_correlation",          r"FC corr $\uparrow$"),
    ("fc_mse",                  r"FC MSE $\downarrow$"),
    ("phase_fc_correlation",    r"phFC corr $\uparrow$"),
    ("phfcd_ks",                r"phFCD KS $\downarrow$"),
    ("metastability_diff",      r"Meta $|\Delta|$ $\downarrow$"),
    ("temporal_correlation",    r"TS corr $\uparrow$"),
    ("power_spectrum_distance", r"TS PSD $\downarrow$"),
]

# Map JSON model key → display name in the table
MODEL_DISPLAY: dict[str, str] = {
    "hopf_grid":          "Coupled Hopf (grid search)",
    "hopf_backprop":      "Coupled Hopf (gradient opt.)",
    "hybrid_hopf_backprop": "Hybrid Hopf (gradient opt.)",
    "nsde_backprop":      "Neural SDE (gradient opt.)",
}

# Best metric per column (True → higher better, False → lower better)
HIGHER_BETTER: dict[str, bool] = {
    "fc_correlation": True,
    "fc_mse": False,
    "phase_fc_correlation": True,
    "phfcd_ks": False,
    "metastability_diff": False,
    "temporal_correlation": True,
    "power_spectrum_distance": False,
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val: float | None, key: str) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r"\textemdash"
    if key == "metastability_diff" and abs(val) < 0.01:
        # Scientific notation for very small values
        exp = math.floor(math.log10(max(abs(val), 1e-12)))
        mantissa = val / 10**exp
        return rf"${mantissa:.1f}\times 10^{{{exp}}}$"
    return f"{val:.3f}"


def _bold(s: str) -> str:
    return rf"\bf{{{s}}}"


def _build_table_body(metrics: dict[str, dict[str, float]]) -> str:
    """Return LaTeX table rows (\\midrule … \\bottomrule exclusive)."""
    col_keys = [mk for mk, _ in METRIC_COLS]

    # Determine best value per metric column across all models
    best: dict[str, float] = {}
    for mk in col_keys:
        vals = [m.get(mk) for m in metrics.values() if m.get(mk) is not None and not math.isnan(m.get(mk, float("nan")))]
        if not vals:
            continue
        best[mk] = max(vals) if HIGHER_BETTER.get(mk, True) else min(vals)

    rows: list[str] = []
    for model_key, display_name in MODEL_DISPLAY.items():
        if model_key not in metrics:
            continue
        m = metrics[model_key]
        cells = [display_name]
        for mk in col_keys:
            val = m.get(mk)
            formatted = _fmt(val, mk)
            if val is not None and not math.isnan(float(val)) and best.get(mk) is not None:
                if abs(float(val) - best[mk]) < 1e-6:
                    formatted = _bold(formatted)
            cells.append(formatted)
        rows.append("    " + " & ".join(cells) + r" \\")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# LaTeX patch helper
# ---------------------------------------------------------------------------

_MIDRULE = r"\midrule"
_BOTTOMRULE = r"\bottomrule"


def _patch_table(tex: str, table_label: str, new_body: str) -> str:
    """Replace the row block between \\midrule and \\bottomrule for *table_label*."""
    # Find the table environment containing the label
    label_pattern = re.compile(
        r"(\\label\{" + re.escape(table_label) + r"\}.*?"
        r"\\midrule\n)"          # everything up to and including \midrule\n
        r"(.*?)"                 # the row block we want to replace
        r"(\n\s*\\bottomrule)",  # the \bottomrule line
        re.DOTALL,
    )
    def _replacement(m: re.Match) -> str:
        return m.group(1) + new_body + m.group(3)
    patched, n_subs = label_pattern.subn(_replacement, tex)
    if n_subs == 0:
        print(f"  Warning: table '{table_label}' not found in LaTeX source.", file=sys.stderr)
    return patched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Patch paper LaTeX tables with fresh metrics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/lsd_paper_metrics.json"),
        help="Path to the paper metrics JSON (from train_models.py paper).",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("paper/sections/03_results.tex"),
        help="Path to the results LaTeX file to patch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print patched table rows without writing the file.",
    )
    args = parser.parse_args(argv)

    if not args.metrics.exists():
        print(f"Metrics file not found: {args.metrics}")
        print("Train models first with: python examples/train_models.py paper --dataset-type lsd ...")
        sys.exit(1)

    with args.metrics.open(encoding="utf-8") as fh:
        metrics: dict[str, dict[str, float]] = json.load(fh)

    print(f"Loaded metrics for: {list(metrics.keys())}")

    body = _build_table_body(metrics)
    print("\nGenerated table rows:")
    print(body)

    if args.dry_run:
        return

    if not args.tex.exists():
        print(f"LaTeX file not found: {args.tex}")
        sys.exit(1)

    original = args.tex.read_text(encoding="utf-8")
    patched = _patch_table(original, "tab:lsd_model_comparison", body)

    if patched == original:
        print("No changes made (table not found or already up to date).")
    else:
        args.tex.write_text(patched, encoding="utf-8")
        print(f"Updated {args.tex}")


if __name__ == "__main__":
    main()
