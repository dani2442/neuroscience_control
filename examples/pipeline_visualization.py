"""Render the BOLD -> FFT -> Hilbert pipeline for one subject."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mat_data
from src.utils import plot_signal_pipeline, prepare_signal_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "ts_young" / "ts_young_TR0.72.mat",
        help="Path to a .mat file containing timeseries_all.",
    )
    parser.add_argument("--subject", type=int, default=0, help="Subject index to visualize.")
    parser.add_argument(
        "--rois",
        type=int,
        nargs="*",
        default=[0, 25, 50],
        help="ROI indices to stack in the BOLD and filtered views.",
    )
    parser.add_argument(
        "--focus-roi",
        type=int,
        default=25,
        help="ROI used for the FFT and Hilbert panels.",
    )
    parser.add_argument("--dt", type=float, default=0.72, help="Sampling interval in seconds.")
    parser.add_argument("--f-lo", type=float, default=0.008, help="Lower band-pass cutoff in Hz.")
    parser.add_argument("--f-hi", type=float, default=0.08, help="Upper band-pass cutoff in Hz.")
    parser.add_argument(
        "--max-timepoints",
        type=int,
        default=240,
        help="Maximum number of samples shown on the time-domain panels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper" / "images" / "bold_fft_hilbert_pipeline.png",
        help="Path where the figure will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in an interactive window instead of only saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_mat_data(str(args.dataset))
    raw_timeseries = data["timeseries_all"].transpose(2, 0, 1)

    pipeline = prepare_signal_pipeline(
        raw_timeseries,
        subject_index=args.subject,
        roi_indices=args.rois,
        focus_roi=args.focus_roi,
        dt=args.dt,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        max_timepoints=args.max_timepoints,
    )
    plot_signal_pipeline(
        pipeline,
        title="Data Pipeline: BOLD -> FFT band-pass -> Hilbert transform",
        save_path=str(args.output),
    )

    print(f"Saved pipeline figure to {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
