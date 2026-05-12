"""Shared CLI argument definitions for training and postprocessing scripts."""

from __future__ import annotations

import argparse

# Names of dataset-related config fields that map 1-to-1 to CLI flags.
DATASET_ARG_NAMES = (
    "dataset_type",
    "data_path",
    "lsd_data_dir",
    "abide_data_dir",
    "abide_n_subjects",
    "abide_pipeline",
    "abide_band_pass_filtering",
    "abide_global_signal_regression",
    "abide_quality_checked",
    "adhd200_data_dir",
    "adhd200_n_subjects",
    "adhd200_local_pattern",
    "max_subjects",
    "nilearn_dataset",
    "nilearn_data_dir",
    "nilearn_n_subjects",
    "openneuro_dataset",
    "openneuro_tag",
    "openneuro_target_dir",
    "openneuro_include",
    "openneuro_exclude",
    "datalad_source",
    "datalad_dataset_dir",
    "datalad_get_paths",
    "bids_root",
    "bids_relative_path",
    "bids_derivatives_dir",
    "bids_task",
    "bids_space",
    "bids_desc",
    "bids_subject_ids",
    "bids_runs_per_subject",
    "atlas_n_rois",
    "atlas_yeo_networks",
    "atlas_resolution_mm",
    "atlas_smoothing_fwhm",
)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset-selection arguments to *parser*."""
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=[
            "ts_young",
            "mat",
            "lsd",
            "abide",
            "adhd200",
            "nilearn",
            "openneuro",
            "datalad",
            "bids",
        ],
        help="Dataset backend to use (default: TrainingConfig.dataset_type).",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to local .mat dataset.")
    parser.add_argument(
        "--lsd-data-dir",
        type=str,
        default=None,
        help="Directory containing LSD .mat files.",
    )
    parser.add_argument(
        "--abide-data-dir",
        type=str,
        default=None,
        help="ABIDE PCP cache/download directory.",
    )
    parser.add_argument(
        "--abide-n-subjects",
        type=int,
        default=None,
        help="Limit ABIDE subjects fetched before ROI extraction.",
    )
    parser.add_argument(
        "--abide-pipeline",
        type=str,
        default=None,
        choices=["ccs", "cpac", "dparsf", "niak"],
        help="ABIDE PCP preprocessing pipeline.",
    )
    parser.add_argument(
        "--abide-band-pass-filtering",
        action="store_true",
        default=None,
        help="Use ABIDE PCP band-pass filtered derivatives.",
    )
    parser.add_argument(
        "--abide-global-signal-regression",
        action="store_true",
        default=None,
        help="Use ABIDE PCP derivatives with global signal regression.",
    )
    parser.add_argument(
        "--no-abide-quality-checked",
        dest="abide_quality_checked",
        action="store_false",
        default=None,
        help="Disable nilearn's ABIDE quality-checked subject filter.",
    )
    parser.add_argument(
        "--adhd200-data-dir",
        type=str,
        default=None,
        help="ADHD-200 cache/download directory.",
    )
    parser.add_argument(
        "--adhd200-n-subjects",
        type=int,
        default=None,
        help="Limit ADHD-200 subjects fetched before ROI extraction.",
    )
    parser.add_argument(
        "--adhd200-local-pattern",
        type=str,
        default=None,
        help=(
            "Optional glob under --adhd200-data-dir for locally downloaded "
            "full ADHD-200 PCP NIfTI files."
        ),
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit number of subjects loaded (None = all).",
    )
    parser.add_argument(
        "--nilearn-dataset",
        type=str,
        default=None,
        choices=["development_fmri", "adhd", "adhd200", "abide", "abide_pcp"],
        help="nilearn fetcher dataset name.",
    )
    parser.add_argument(
        "--nilearn-data-dir",
        type=str,
        default=None,
        help="Cache directory for nilearn data.",
    )
    parser.add_argument(
        "--nilearn-n-subjects",
        type=int,
        default=None,
        help="Limit nilearn fetched subjects.",
    )
    parser.add_argument(
        "--no-nilearn-reduce-confounds",
        action="store_true",
        help="Disable nilearn fetch_development_fmri confound reduction.",
    )
    parser.add_argument(
        "--openneuro-dataset",
        type=str,
        default=None,
        help="OpenNeuro dataset id (e.g. ds000030).",
    )
    parser.add_argument(
        "--openneuro-tag",
        type=str,
        default=None,
        help="OpenNeuro dataset tag/revision.",
    )
    parser.add_argument(
        "--openneuro-target-dir",
        type=str,
        default=None,
        help="Directory for OpenNeuro download.",
    )
    parser.add_argument(
        "--openneuro-include",
        nargs="+",
        default=None,
        help="OpenNeuro include glob patterns.",
    )
    parser.add_argument(
        "--openneuro-exclude",
        nargs="+",
        default=None,
        help="OpenNeuro exclude glob patterns.",
    )
    parser.add_argument(
        "--datalad-source",
        type=str,
        default=None,
        help="DataLad dataset source URL/path.",
    )
    parser.add_argument(
        "--datalad-dataset-dir",
        type=str,
        default=None,
        help="Local DataLad dataset checkout dir.",
    )
    parser.add_argument(
        "--datalad-get-paths",
        nargs="+",
        default=None,
        help="DataLad paths/globs to materialize with `datalad get`.",
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default=None,
        help="Root path for direct BIDS loading.",
    )
    parser.add_argument(
        "--bids-relative-path",
        type=str,
        default=None,
        help="Subpath under downloaded dataset root.",
    )
    parser.add_argument(
        "--bids-derivatives-dir",
        type=str,
        default=None,
        help="Relative derivatives directory containing preprocessed BOLD files.",
    )
    parser.add_argument("--bids-task", type=str, default=None, help="BIDS task label filter.")
    parser.add_argument("--bids-space", type=str, default=None, help="BIDS space label filter.")
    parser.add_argument("--bids-desc", type=str, default=None, help="BIDS desc label filter.")
    parser.add_argument(
        "--bids-subject-ids",
        nargs="+",
        default=None,
        help="Optional subject list (sub-XXX or XXX).",
    )
    parser.add_argument(
        "--bids-runs-per-subject",
        type=int,
        default=None,
        help="BOLD runs to use per subject.",
    )
    parser.add_argument("--atlas-n-rois", type=int, default=None, help="Schaefer atlas ROI count.")
    parser.add_argument(
        "--atlas-yeo-networks",
        type=int,
        default=None,
        help="Schaefer atlas network count.",
    )
    parser.add_argument(
        "--atlas-resolution-mm",
        type=int,
        default=None,
        help="Schaefer atlas resolution.",
    )
    parser.add_argument(
        "--atlas-smoothing-fwhm",
        type=float,
        default=None,
        help="Optional smoothing passed to nilearn masker.",
    )
