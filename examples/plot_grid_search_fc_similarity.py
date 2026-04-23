#!/usr/bin/env python3
"""Plot a Hopf grid-search landscape using FC similarity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/grid_search/grid_search_results.json"),
        help="Path to a grid_search_results.json file.",
    )
    parser.add_argument(
        "--metric",
        default="fc_correlation",
        help="Metric key to plot from each result entry.",
    )
    parser.add_argument(
        "--metric-label",
        default="FC similarity",
        help="Display name used in the colorbar and summary text.",
    )
    parser.add_argument(
        "--x-param",
        default="initial_g",
        help="Parameter to place on the horizontal axis.",
    )
    parser.add_argument(
        "--y-param",
        default="initial_a",
        help="Parameter to place on the vertical axis.",
    )
    parser.add_argument(
        "--reduce-param",
        default="initial_kappa",
        help="Parameter to collapse so the plot stays 2D.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("results/grid_search/fc_similarity_landscape"),
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--higher-is-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether larger metric values are better when reducing over the third parameter.",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in data or not isinstance(data["results"], list):
        raise ValueError(f"Unexpected grid-search payload in {path}")
    return data


def format_param(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    if abs(value) < 0.005:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_metric_surface(
    results: list[dict[str, Any]],
    *,
    x_param: str,
    y_param: str,
    reduce_param: str,
    metric: str,
    higher_is_better: bool,
) -> tuple[list[float], list[float], np.ndarray, dict[tuple[int, int], float], dict[str, Any]]:
    x_values = sorted({float(entry["params"][x_param]) for entry in results})
    y_values = sorted({float(entry["params"][y_param]) for entry in results})

    def metric_value(entry: dict[str, Any]) -> float:
        return float(entry["metrics"][metric])

    chooser = max if higher_is_better else min
    surface = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    reduced_values: dict[tuple[int, int], float] = {}

    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            candidates = [
                entry
                for entry in results
                if math.isclose(float(entry["params"][x_param]), x_value)
                and math.isclose(float(entry["params"][y_param]), y_value)
            ]
            if not candidates:
                continue
            best_cell = chooser(candidates, key=metric_value)
            surface[y_index, x_index] = metric_value(best_cell)
            if reduce_param in best_cell["params"]:
                reduced_values[(x_index, y_index)] = float(best_cell["params"][reduce_param])

    best_entry = chooser(results, key=metric_value)
    return x_values, y_values, surface, reduced_values, best_entry


def make_plot(
    *,
    x_values: list[float],
    y_values: list[float],
    surface: np.ndarray,
    reduced_values: dict[tuple[int, int], float],
    best_entry: dict[str, Any],
    metric: str,
    metric_label: str,
    x_param: str,
    y_param: str,
    reduce_param: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    fig.subplots_adjust(left=0.22, right=0.83, top=0.80, bottom=0.18)

    cmap = plt.get_cmap("coolwarm")
    image = ax.imshow(
        surface,
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        extent=(-0.5, len(x_values) - 0.5, -0.5, len(y_values) - 0.5),
        vmin=float(np.nanmin(surface)),
        vmax=float(np.nanmax(surface)),
        aspect="auto",
    )
    ax.set_box_aspect(1)

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([format_param(value) for value in x_values], fontsize=11)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=6)
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([format_param(value) for value in y_values], fontsize=11)
    ax.tick_params(axis="y", pad=6)

    ax.set_xlim(-0.5, len(x_values) - 0.5)
    ax.set_ylim(-0.5, len(y_values) - 0.5)
    ax.grid(color="white", linewidth=0.9, alpha=0.18)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#444444")

    x_grid, y_grid = np.meshgrid(range(len(x_values)), range(len(y_values)))
    ax.scatter(
        x_grid.flatten(),
        y_grid.flatten(),
        s=62,
        facecolor="white",
        edgecolor="#333333",
        linewidth=0.9,
        zorder=3,
    )

    best_x = x_values.index(float(best_entry["params"][x_param]))
    best_y = y_values.index(float(best_entry["params"][y_param]))
    best_score = float(best_entry["metrics"][metric])
    best_reduce = best_entry["params"].get(reduce_param)

    ax.add_patch(
        Circle(
            (best_x, best_y),
            radius=0.30,
            facecolor="none",
            edgecolor="#d4a61f",
            linewidth=2.3,
            zorder=4,
        )
    )
    ax.scatter(
        [best_x],
        [best_y],
        marker="*",
        s=250,
        facecolor="#111111",
        edgecolor="#d4a61f",
        linewidth=1.2,
        zorder=5,
    )

    ax.text(
        0.5,
        1.18,
        "Coupling strength (G)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=17,
    )
    ax.annotate(
        "",
        xy=(0.72, 1.08),
        xytext=(0.22, 1.08),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#111111"),
    )
    ax.text(0.18, 1.08, "Low", transform=ax.transAxes, ha="right", va="center", fontsize=11)
    ax.text(0.75, 1.08, "High", transform=ax.transAxes, ha="left", va="center", fontsize=11)

    ax.text(
        -0.30,
        0.50,
        "Bifurcation\nparameter (a)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
    )
    ax.annotate(
        "",
        xy=(-0.17, 0.90),
        xytext=(-0.17, 0.14),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#111111"),
    )
    ax.text(-0.19, 0.92, "High", transform=ax.transAxes, ha="right", va="center", fontsize=11)
    ax.text(-0.19, 0.12, "Low", transform=ax.transAxes, ha="right", va="center", fontsize=11)

    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.08, shrink=0.82)
    cbar.ax.set_title(metric_label, fontsize=15, pad=10)
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=10)

    selected_reduced_values = sorted({format_param(value) for value in reduced_values.values()})
    best_summary = (
        f"Star: best {metric_label.lower()} = {best_score:.3f} at "
        f"G={format_param(float(best_entry['params'][x_param]))}, "
        f"a={format_param(float(best_entry['params'][y_param]))}, "
        f"kappa={format_param(float(best_reduce)) if best_reduce is not None else 'n/a'}."
    )
    reduce_summary = (
        f"Each cell keeps the best {metric_label.lower()} over {reduce_param} "
        f"in {{{', '.join(selected_reduced_values)}}}."
    )
    fig.text(0.50, 0.09, best_summary, ha="center", va="center", fontsize=10.5)
    fig.text(0.50, 0.055, reduce_summary, ha="center", va="center", fontsize=10.5)

    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


def main() -> None:
    args = parse_args()
    payload = load_results(args.results_path)
    results = payload["results"]

    x_values, y_values, surface, reduced_values, best_entry = build_metric_surface(
        results,
        x_param=args.x_param,
        y_param=args.y_param,
        reduce_param=args.reduce_param,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
    )
    figure = make_plot(
        x_values=x_values,
        y_values=y_values,
        surface=surface,
        reduced_values=reduced_values,
        best_entry=best_entry,
        metric=args.metric,
        metric_label=args.metric_label,
        x_param=args.x_param,
        y_param=args.y_param,
        reduce_param=args.reduce_param,
    )

    output_stem = args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
