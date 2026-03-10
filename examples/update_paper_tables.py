#!/usr/bin/env python3
r"""
Update LaTeX paper tables from training metrics JSON files.

Reads results/<dataset>_paper_metrics_<timestamp>.json (produced by train_models.py paper)
and updates the corresponding model-comparison table.

Dataset-to-table mapping:
  - ts_young → tab:model_comparison
  - lsd → tab:lsd_model_comparison

Default output targets:
    - tab:model_comparison      → paper/sections/ts_young_model_table.tex
    - tab:lsd_model_comparison  → paper/sections/03_results.tex

If the JSON contains <metric>_std keys, they will be formatted as
"value {\scriptsize \pm std}" in the LaTeX table.

Usage
-----
    python examples/update_paper_tables.py
    python examples/update_paper_tables.py --metrics results/ts_young_paper_metrics_20260310_123456.json
    python examples/update_paper_tables.py --metrics results/lsd_paper_metrics_20260310_123456.json --table-label tab:lsd_model_comparison
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

def _fmt(val: float | None, key: str, std: float | None = None) -> str:
    """Format a metric value, optionally with std in smaller font."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r"\textemdash"
    if key == "metastability_diff" and abs(val) < 0.01:
        # Scientific notation for very small values
        exp = math.floor(math.log10(max(abs(val), 1e-12)))
        mantissa = val / 10**exp
        base = rf"${mantissa:.1f}\times 10^{{{exp}}}$"
        if std is not None and not math.isnan(std):
            std_exp = math.floor(math.log10(max(abs(std), 1e-12)))
            std_mantissa = std / 10**std_exp
            base += rf" {{\scriptsize $\pm$ {std_mantissa:.1f}$\times 10^{{{std_exp}}}$}}"
        return base
    base = f"{val:.3f}"
    if std is not None and not math.isnan(std):
        base += rf" {{\scriptsize $\pm$ {std:.3f}}}"
    return base


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
            std = m.get(f"{mk}_std")  # Check for std
            formatted = _fmt(val, mk, std)
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

TABLE_TARGET_BY_LABEL: dict[str, Path] = {
    "tab:model_comparison": Path("paper/sections/ts_young_model_table.tex"),
    "tab:lsd_model_comparison": Path("paper/sections/03_results.tex"),
}


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


def _patch_tabular_body(tex: str, new_body: str) -> str:
    """Replace row block between \\midrule and \\bottomrule in a tabular snippet."""
    tabular_pattern = re.compile(
        r"(\\midrule\n)"
        r"(.*?)"
        r"(\n\s*\\bottomrule)",
        re.DOTALL,
    )

    def _replacement(m: re.Match) -> str:
        return m.group(1) + new_body + m.group(3)

    patched, n_subs = tabular_pattern.subn(_replacement, tex)
    if n_subs == 0:
        print("  Warning: could not find \\midrule/\\bottomrule block in target file.", file=sys.stderr)
    return patched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Patch paper LaTeX tables with fresh metrics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to the paper metrics JSON (from train_models.py paper).",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=None,
        help="Path to the LaTeX target to patch. If not provided, inferred from --table-label (or dataset_type).",
    )
    parser.add_argument(
        "--table-label",
        type=str,
        default=None,
        help="Explicit LaTeX table label to update (e.g., tab:model_comparison). If not provided, infers from dataset_type in JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print patched table rows without writing the file.",
    )
    args = parser.parse_args(argv)

    # Auto-discover latest metrics file if not specified
    if args.metrics is None:
        results_dir = Path("results")
        candidates = sorted(results_dir.glob("*_paper_metrics_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No metrics JSON found in results/. Please specify --metrics or run training first.")
            sys.exit(1)
        args.metrics = candidates[0]
        print(f"Auto-detected latest metrics file: {args.metrics}")

    if not args.metrics.exists():
        print(f"Metrics file not found: {args.metrics}")
        print("Train models first with: python examples/train_models.py paper --dataset-type <dataset> ...")
        sys.exit(1)

    with args.metrics.open(encoding="utf-8") as fh:
        data = json.load(fh)
    
    # Handle both old format (dict of metrics) and new format (with dataset_type metadata)
    if "metrics" in data and "dataset_type" in data:
        metrics = data["metrics"]
        dataset_type = data["dataset_type"]
        print(f"Dataset type: {dataset_type}")
    else:
        # Old format - assume it's the metrics directly
        metrics = data
        dataset_type = None
        print("Warning: Old JSON format detected (no dataset_type metadata)")

    print(f"Loaded metrics for: {list(metrics.keys())}")

    # Determine which table to update
    if args.table_label:
        table_label = args.table_label
    elif dataset_type == "lsd":
        table_label = "tab:lsd_model_comparison"
    else:
        # Default to ts_young table
        table_label = "tab:model_comparison"
    
    print(f"Target table: {table_label}")

    tex_target = args.tex if args.tex is not None else TABLE_TARGET_BY_LABEL.get(table_label)
    if tex_target is None:
        tex_target = Path("paper/sections/03_results.tex")
    print(f"Target file: {tex_target}")

    body = _build_table_body(metrics)
    print("\nGenerated table rows:")
    print(body)

    if args.dry_run:
        return

    if not tex_target.exists():
        print(f"LaTeX file not found: {tex_target}")
        sys.exit(1)

    original = tex_target.read_text(encoding="utf-8")
    if tex_target.name == "03_results.tex":
        patched = _patch_table(original, table_label, body)
    else:
        patched = _patch_tabular_body(original, body)

    if patched == original:
        print("No changes made (table not found or already up to date).")
    else:
        tex_target.write_text(patched, encoding="utf-8")
        print(f"Updated {tex_target}")


if __name__ == "__main__":
    main()
