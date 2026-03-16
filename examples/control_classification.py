"""Offline ABIDE autism-vs-control classification benchmark.

This script replaces the exploratory notebook with a cache-first workflow:

1. Index the locally available ABIDE PCP files already stored in `data/abide`.
2. Extract Schaefer ROI timeseries once and cache the preprocessed arrays.
3. Benchmark stronger classification pipelines with nested CV.
4. Save tabular results, a summary figure, and the best refit model.

The script is designed to work without network access. It does not rely on
`nilearn.fetch_abide_pcp`, which would try to download missing subjects.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent

# Avoid matplotlib trying to write into a non-writable HOME config directory.
MPL_DIR = ROOT / "data/.mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import logm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.metrics import compute_static_fc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


ABIDE_DIR = ROOT / "data/abide/ABIDE_pcp"
FUNC_DIR = ABIDE_DIR / "cpac/nofilt_noglobal"
PHENO_CSV = ABIDE_DIR / "Phenotypic_V1_0b_preprocessed1.csv"
ATLAS_DATA_DIR = ROOT / "data/nilearn"
NILEARN_CACHE_DIR = ROOT / "data/nilearn_cache"


@dataclass
class CacheBundle:
    """Cached offline features for the locally available ABIDE cohort."""

    subjects: pd.DataFrame
    timeseries: np.ndarray
    fc_matrices: np.ndarray
    atlas_n_rois: int


@dataclass
class ExperimentSpec:
    """Experiment definition used by the benchmark loop."""

    name: str
    input_key: str
    description: str
    estimator_factory: Callable[[argparse.Namespace], BaseEstimator]


class UpperTriangularVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize the strict upper triangle of a batch of matrices."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "UpperTriangularVectorizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected a 3D batch of matrices, got shape {X.shape}.")
        idx = np.triu_indices(X.shape[1], k=1)
        return np.asarray([mat[idx] for mat in X], dtype=np.float32)


class MatrixLogVectorizer(BaseEstimator, TransformerMixin):
    """Apply a matrix logarithm, then vectorize the strict upper triangle."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MatrixLogVectorizer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected a 3D batch of matrices, got shape {X.shape}.")
        idx = np.triu_indices(X.shape[1], k=1)
        feats = np.empty((X.shape[0], idx[0].size), dtype=np.float32)
        for i, mat in enumerate(X):
            feats[i] = np.real_if_close(logm(mat), tol=1000).real[idx]
        return feats


class CovariateAugmentedOASTangent(BaseEstimator, TransformerMixin):
    """Concatenate OAS tangent-space features with imputed subject covariates."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CovariateAugmentedOASTangent":
        timeseries, covariates = self._split_inputs(X)
        self.covariance_ = Covariances(estimator="oas")
        cov_matrices = self.covariance_.fit_transform(timeseries)
        self.tangent_ = TangentSpace(metric="riemann")
        self.tangent_.fit(cov_matrices)
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(covariates)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        timeseries, covariates = self._split_inputs(X)
        cov_matrices = self.covariance_.transform(timeseries)
        tangent_features = self.tangent_.transform(cov_matrices).astype(np.float32)
        covariates = self.imputer_.transform(covariates).astype(np.float32)
        return np.concatenate([tangent_features, covariates], axis=1, dtype=np.float32)

    @staticmethod
    def _split_inputs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("Expected an object array with shape (n_samples, 2).")
        timeseries = np.stack(X[:, 0]).astype(np.float32)
        covariates = np.stack(X[:, 1]).astype(np.float32)
        return timeseries, covariates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-rois", type=int, default=100, help="Number of Schaefer atlas ROIs.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=ROOT / "data/abide/abide_local_schaefer100_cache.joblib",
        help="Path to the cached ROI-timeseries/FC bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "examples",
        help="Directory where result CSV/plots/model artifacts are written.",
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Ignore any saved cache and recompute ROI timeseries/FC matrices.",
    )
    parser.add_argument(
        "--site-mode",
        choices=("all", "mixed"),
        default="all",
        help=(
            "Use all locally available sites or only sites containing both classes. "
            "'mixed' is less confounded when a site contains only controls or only autism."
        ),
    )
    parser.add_argument(
        "--site-filter",
        nargs="*",
        default=None,
        help="Optional explicit site whitelist, e.g. --site-filter PITT OLIN",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional stratified subject cap applied after loading the cached cohort.",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="Outer CV splits.")
    parser.add_argument("--inner-cv-splits", type=int, default=3, help="Inner CV splits for model selection.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment whitelist. Names must match the printed experiment ids exactly.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for the inner GridSearchCV. Outer CV stays sequential.",
    )
    return parser.parse_args()


def load_local_subject_manifest() -> pd.DataFrame:
    """Build an offline manifest from the local ABIDE mirror and phenotypic CSV."""

    if not FUNC_DIR.exists():
        raise FileNotFoundError(f"Missing local ABIDE directory: {FUNC_DIR}")
    if not PHENO_CSV.exists():
        raise FileNotFoundError(f"Missing ABIDE phenotypic CSV: {PHENO_CSV}")

    func_files = sorted(FUNC_DIR.glob("*_func_preproc.nii.gz"))
    if not func_files:
        raise FileNotFoundError(f"No local ABIDE preprocessed files found under {FUNC_DIR}")

    phenotypic = pd.read_csv(PHENO_CSV).set_index("FILE_ID")

    rows: list[dict[str, object]] = []
    for func_path in func_files:
        file_id = func_path.name.replace("_func_preproc.nii.gz", "")
        if file_id not in phenotypic.index:
            continue
        row = phenotypic.loc[file_id]
        rows.append(
            {
                "file_id": file_id,
                "path": str(func_path),
                "label": int(row["DX_GROUP"] == 1),
                "group_name": "autism" if int(row["DX_GROUP"] == 1) else "control",
                "site": str(row["SITE_ID"]),
                "age": float(row["AGE_AT_SCAN"]) if pd.notna(row["AGE_AT_SCAN"]) else np.nan,
                "sex": float(row["SEX"]) if pd.notna(row["SEX"]) else np.nan,
                "fiq": float(row["FIQ"]) if pd.notna(row["FIQ"]) else np.nan,
                "mean_fd": float(row["func_mean_fd"]) if pd.notna(row["func_mean_fd"]) else np.nan,
            }
        )

    subjects = pd.DataFrame(rows).sort_values(["site", "file_id"]).reset_index(drop=True)
    if subjects.empty:
        raise RuntimeError("No phenotypic rows matched the local ABIDE files.")
    return subjects


def cache_is_compatible(cache: dict[str, object], subjects: pd.DataFrame, atlas_n_rois: int) -> bool:
    """Check whether an existing cache matches the current offline manifest."""

    cached_subjects = cache.get("subjects")
    if not isinstance(cached_subjects, pd.DataFrame):
        return False
    timeseries = cache.get("timeseries", cache.get("ts_stack"))
    fc_matrices = cache.get("fc_matrices", cache.get("fc"))
    if timeseries is None or fc_matrices is None:
        return False
    cached_rois = int(cache.get("atlas_n_rois", np.asarray(timeseries).shape[1]))
    if cached_rois != atlas_n_rois:
        return False
    cached_ids = cached_subjects["file_id"].tolist()
    current_ids = subjects["file_id"].tolist()
    return set(cached_ids) == set(current_ids)


def reorder_cached_bundle(
    cache: dict[str, object],
    subjects: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reorder cached arrays so they match the current manifest order."""

    cached_subjects = cache["subjects"].copy().reset_index(drop=True)
    timeseries = np.asarray(cache.get("timeseries", cache.get("ts_stack")), dtype=np.float32)
    fc_matrices = np.asarray(cache.get("fc_matrices", cache.get("fc")), dtype=np.float32)

    order_lookup = {file_id: idx for idx, file_id in enumerate(cached_subjects["file_id"].tolist())}
    order = [order_lookup[file_id] for file_id in subjects["file_id"].tolist()]

    reordered_subjects = cached_subjects.set_index("file_id").loc[subjects["file_id"]].reset_index()
    return reordered_subjects, timeseries[order], fc_matrices[order]


def compute_fc_stack(timeseries: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Compute correlation FC matrices and regularize them to remain SPD.

    Processes subjects in batches so peak memory stays bounded to
    ``batch_size`` subjects at a time instead of the full cohort.
    """
    n, n_rois = timeseries.shape[0], timeseries.shape[1]
    fc = np.empty((n, n_rois, n_rois), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_ts = torch.from_numpy(timeseries[start:end])
        fc_batch = compute_static_fc(batch_ts).numpy().astype(np.float32)
        del batch_ts
        fc[start:end] = fc_batch
    for i in range(n):
        fc[i] = 0.5 * (fc[i] + fc[i].T)
        eigvals = np.linalg.eigvalsh(fc[i])
        if eigvals.min() <= 1e-6:
            fc[i] += (abs(float(eigvals.min())) + 1e-4) * np.eye(n_rois, dtype=np.float32)
    return fc


def extract_subject_timeseries(subjects: pd.DataFrame, atlas_n_rois: int) -> list[np.ndarray]:
    """Extract per-subject Schaefer ROI timeseries as `(n_rois, T)` arrays."""

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=atlas_n_rois, data_dir=str(ATLAS_DATA_DIR), verbose=0)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize="zscore_sample",
        memory=str(NILEARN_CACHE_DIR),
        verbose=0,
    )

    extracted: list[np.ndarray] = []
    for idx, func_path in enumerate(subjects["path"], start=1):
        ts = masker.fit_transform(func_path).astype(np.float32).T
        extracted.append(ts)
        if idx % 5 == 0 or idx == len(subjects):
            print(f"Extracted ROI timeseries for {idx}/{len(subjects)} subjects")
    return extracted


def stack_trimmed_timeseries(timeseries_list: list[np.ndarray]) -> np.ndarray:
    """Trim a list of `(n_rois, T_i)` arrays to a shared length and stack them.

    Pre-allocates the output to avoid holding both the list and a full copy in
    memory simultaneously; each source array is cleared after being written.
    """
    min_t = min(ts.shape[1] for ts in timeseries_list)
    n_rois = timeseries_list[0].shape[0]
    out = np.empty((len(timeseries_list), n_rois, min_t), dtype=np.float32)
    for i, ts in enumerate(timeseries_list):
        out[i] = ts[:, :min_t]
        timeseries_list[i] = None  # release reference as we go
    return out


def save_cache_bundle(
    subjects: pd.DataFrame,
    timeseries: np.ndarray,
    cache_path: Path,
    atlas_n_rois: int,
) -> CacheBundle:
    """Persist a normalized cache bundle and return the in-memory object."""

    fc_matrices = compute_fc_stack(timeseries)
    cache = {
        "subjects": subjects.reset_index(drop=True),
        "timeseries": timeseries,
        "fc_matrices": fc_matrices,
        "atlas_n_rois": atlas_n_rois,
        "min_timepoints": int(timeseries.shape[-1]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, cache_path)
    print(f"Saved preprocessing cache to {cache_path}")
    return CacheBundle(
        subjects=subjects.reset_index(drop=True),
        timeseries=timeseries,
        fc_matrices=fc_matrices,
        atlas_n_rois=atlas_n_rois,
    )


def extract_and_cache_subject_timeseries(
    subjects: pd.DataFrame,
    cache_path: Path,
    atlas_n_rois: int,
) -> CacheBundle:
    """Extract Schaefer ROI timeseries from the local ABIDE files and persist the cache."""

    timeseries_list = extract_subject_timeseries(subjects, atlas_n_rois)
    timeseries = stack_trimmed_timeseries(timeseries_list)
    return save_cache_bundle(subjects, timeseries, cache_path, atlas_n_rois)


def _apply_max_subjects_to_manifest(subjects: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Optionally subsample the full manifest before any extraction or cache lookup.

    This keeps both the cache and peak memory bounded when --max-subjects is set,
    rather than loading/extracting the full cohort and filtering afterwards.
    """
    if args.max_subjects is None or args.max_subjects >= len(subjects):
        return subjects

    stratify = subjects["site"].astype(str) + "__" + subjects["label"].astype(str)
    try:
        kept_idx, _ = train_test_split(
            subjects.index.to_numpy(),
            train_size=args.max_subjects,
            stratify=stratify.to_numpy(),
            random_state=args.random_state,
        )
    except ValueError:
        kept_idx, _ = train_test_split(
            subjects.index.to_numpy(),
            train_size=args.max_subjects,
            stratify=subjects["label"].to_numpy(),
            random_state=args.random_state,
        )
    return subjects.loc[np.sort(kept_idx)].reset_index(drop=True)


def load_or_build_cache(args: argparse.Namespace) -> CacheBundle:
    """Load a compatible preprocessing cache or compute it from local files."""

    subjects = load_local_subject_manifest()
    subjects = _apply_max_subjects_to_manifest(subjects, args)

    if args.cache_path.exists() and not args.force_recompute_cache:
        cache = joblib.load(args.cache_path)
        timeseries_key = cache.get("timeseries", cache.get("ts_stack"))
        if isinstance(cache.get("subjects"), pd.DataFrame) and timeseries_key is not None:
            cached_subjects = cache["subjects"].copy().reset_index(drop=True)
            cached_timeseries = np.asarray(timeseries_key, dtype=np.float32)
            cached_rois = int(cache.get("atlas_n_rois", cached_timeseries.shape[1]))

            if cached_rois == args.atlas_rois:
                cached_ids = set(cached_subjects["file_id"].tolist())
                current_ids = set(subjects["file_id"].tolist())
                reusable_ids = sorted(cached_ids & current_ids)

                if reusable_ids:
                    if current_ids == cached_ids:
                        print(f"Loaded preprocessing cache from {args.cache_path}")
                    else:
                        missing = subjects[~subjects["file_id"].isin(cached_ids)].reset_index(drop=True)
                        print(
                            f"Updating preprocessing cache from {len(cached_subjects)} to {len(subjects)} subjects "
                            f"by extracting {len(missing)} missing subjects"
                        )

                        existing_lookup = {
                            file_id: cached_timeseries[idx]
                            for idx, file_id in enumerate(cached_subjects["file_id"].tolist())
                            if file_id in current_ids
                        }
                        missing_timeseries = extract_subject_timeseries(missing, args.atlas_rois)
                        for file_id, ts in zip(missing["file_id"].tolist(), missing_timeseries):
                            existing_lookup[file_id] = ts

                        ordered_timeseries = [existing_lookup[file_id] for file_id in subjects["file_id"].tolist()]
                        stacked = stack_trimmed_timeseries(ordered_timeseries)
                        return save_cache_bundle(subjects, stacked, args.cache_path, args.atlas_rois)

                    reordered_subjects, timeseries, fc_matrices = reorder_cached_bundle(cache, subjects)
                    return CacheBundle(
                        subjects=reordered_subjects,
                        timeseries=timeseries,
                        fc_matrices=fc_matrices,
                        atlas_n_rois=cached_rois,
                    )

        print("Existing cache is incompatible with the current local manifest; recomputing from scratch.")

    return extract_and_cache_subject_timeseries(subjects, args.cache_path, args.atlas_rois)


def summarize_sites(subjects: pd.DataFrame) -> pd.DataFrame:
    """Return a site-by-label table useful for confound inspection."""

    return pd.crosstab(subjects["site"], subjects["group_name"]).sort_index()


def filter_subjects(subjects: pd.DataFrame, args: argparse.Namespace) -> pd.Index:
    """Select subjects after optional site filtering and subject subsampling."""

    mask = pd.Series(True, index=subjects.index)

    if args.site_filter:
        allowed = {site.upper() for site in args.site_filter}
        mask &= subjects["site"].str.upper().isin(allowed)

    filtered = subjects.loc[mask].copy()
    if filtered.empty:
        raise ValueError("Filtering removed every subject from the offline cohort.")

    if args.site_mode == "mixed":
        valid_sites = filtered.groupby("site")["label"].nunique()
        valid_sites = valid_sites[valid_sites >= 2].index
        filtered = filtered[filtered["site"].isin(valid_sites)].copy()
        if filtered.empty:
            raise ValueError("No sites with both autism and control subjects remain after filtering.")

    if args.max_subjects is not None and args.max_subjects < len(filtered):
        stratify = filtered["site"].astype(str) + "__" + filtered["label"].astype(str)
        try:
            kept_idx, _ = train_test_split(
                filtered.index.to_numpy(),
                train_size=args.max_subjects,
                stratify=stratify.to_numpy(),
                random_state=args.random_state,
            )
        except ValueError:
            kept_idx, _ = train_test_split(
                filtered.index.to_numpy(),
                train_size=args.max_subjects,
                stratify=filtered["label"].to_numpy(),
                random_state=args.random_state,
            )
        filtered = filtered.loc[np.sort(kept_idx)].copy()

    if filtered["label"].nunique() < 2:
        raise ValueError("The selected cohort contains only one class.")

    return filtered.index


def build_feature_views(bundle: CacheBundle, keep_index: pd.Index) -> dict[str, np.ndarray]:
    """Create the different inputs consumed by the benchmarked models."""

    subjects = bundle.subjects.loc[keep_index].reset_index(drop=True)
    keep_positions = keep_index.to_numpy()

    covariates = subjects[["age", "sex", "fiq", "mean_fd"]].to_numpy(dtype=np.float32)
    timeseries = bundle.timeseries[keep_positions]
    timeseries_with_covariates = np.empty((len(subjects), 2), dtype=object)
    timeseries_with_covariates[:, 0] = list(timeseries)
    timeseries_with_covariates[:, 1] = list(covariates)

    return {
        "subjects": subjects,
        "labels": subjects["label"].to_numpy(),
        "covariates": covariates,
        "timeseries": timeseries,
        "timeseries_with_covariates": timeseries_with_covariates,
        "fc_matrices": bundle.fc_matrices[keep_positions],
    }


def build_cv(args: argparse.Namespace) -> tuple[StratifiedKFold, StratifiedKFold]:
    """Outer and inner CV objects used for nested evaluation."""

    outer_cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    inner_cv = StratifiedKFold(n_splits=args.inner_cv_splits, shuffle=True, random_state=args.random_state)
    return outer_cv, inner_cv


def scoring_dict() -> dict[str, object]:
    """Shared scoring configuration across experiments."""

    return {
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1": "f1",
        "roc_auc": "roc_auc",
    }


def make_search(
    pipeline: Pipeline,
    param_grid: dict[str, list[object]],
    args: argparse.Namespace,
    inner_cv: StratifiedKFold,
) -> BaseEstimator:
    """Wrap a pipeline in GridSearchCV only when parameters are supplied."""

    if not param_grid:
        return pipeline
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=args.n_jobs,
        refit=True,
    )


def make_experiments(args: argparse.Namespace, inner_cv: StratifiedKFold) -> list[ExperimentSpec]:
    """Define baseline and improved classifiers."""

    def baseline_fc_logreg(_: argparse.Namespace) -> BaseEstimator:
        return Pipeline(
            [
                ("upper", UpperTriangularVectorizer()),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=5000)),
            ]
        )

    def baseline_log_fc_logreg(_: argparse.Namespace) -> BaseEstimator:
        return Pipeline(
            [
                ("log_upper", MatrixLogVectorizer()),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=5000)),
            ]
        )

    def improved_log_fc_logreg(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("log_upper", MatrixLogVectorizer()),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    def fc_tangent_logreg(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("tangent", TangentSpace(metric="riemann")),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    def oas_tangent_logreg(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("tangent", TangentSpace(metric="riemann")),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    def oas_tangent_svc(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("tangent", TangentSpace(metric="riemann")),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear")),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    def oas_tangent_lda(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("tangent", TangentSpace(metric="riemann")),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
            },
            cfg,
            inner_cv,
        )

    def oas_tangent_covariates_logreg(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("features", CovariateAugmentedOASTangent()),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    def oas_tangent_covariates_svc(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("features", CovariateAugmentedOASTangent()),
                ("select", SelectKBest(score_func=f_classif)),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear")),
            ]
        )
        return make_search(
            pipeline,
            {
                "select__k": [25, 50, 100, 200, 400],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    def oas_mdm(_: argparse.Namespace) -> BaseEstimator:
        return Pipeline(
            [
                ("cov", Covariances(estimator="oas")),
                ("clf", MDM(metric=dict(mean="riemann", distance="riemann"))),
            ]
        )

    def covariates_only(cfg: argparse.Namespace) -> BaseEstimator:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000)),
            ]
        )
        return make_search(
            pipeline,
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            cfg,
            inner_cv,
        )

    return [
        ExperimentSpec(
            name="baseline_fc_logreg",
            input_key="fc_matrices",
            description="Upper-triangle FC features + logistic regression",
            estimator_factory=baseline_fc_logreg,
        ),
        ExperimentSpec(
            name="baseline_log_fc_logreg",
            input_key="fc_matrices",
            description="Log-Euclidean FC vectorization + logistic regression",
            estimator_factory=baseline_log_fc_logreg,
        ),
        ExperimentSpec(
            name="improved_log_fc_logreg",
            input_key="fc_matrices",
            description="Log-Euclidean FC features with univariate selection + nested model search",
            estimator_factory=improved_log_fc_logreg,
        ),
        ExperimentSpec(
            name="fc_tangent_logreg",
            input_key="fc_matrices",
            description="Riemannian tangent features from FC matrices + nested model search",
            estimator_factory=fc_tangent_logreg,
        ),
        ExperimentSpec(
            name="oas_tangent_logreg",
            input_key="timeseries",
            description="OAS covariance -> tangent space -> logistic regression",
            estimator_factory=oas_tangent_logreg,
        ),
        ExperimentSpec(
            name="oas_tangent_svc",
            input_key="timeseries",
            description="OAS covariance -> tangent space -> linear SVC",
            estimator_factory=oas_tangent_svc,
        ),
        ExperimentSpec(
            name="oas_tangent_lda",
            input_key="timeseries",
            description="OAS covariance -> tangent space -> shrinkage LDA",
            estimator_factory=oas_tangent_lda,
        ),
        ExperimentSpec(
            name="oas_tangent_covariates_logreg",
            input_key="timeseries_with_covariates",
            description="OAS tangent features augmented with age/sex/FIQ/motion covariates + logistic regression",
            estimator_factory=oas_tangent_covariates_logreg,
        ),
        ExperimentSpec(
            name="oas_tangent_covariates_svc",
            input_key="timeseries_with_covariates",
            description="OAS tangent features augmented with age/sex/FIQ/motion covariates + linear SVC",
            estimator_factory=oas_tangent_covariates_svc,
        ),
        ExperimentSpec(
            name="oas_mdm",
            input_key="timeseries",
            description="OAS covariance -> Riemannian minimum-distance-to-mean",
            estimator_factory=oas_mdm,
        ),
        ExperimentSpec(
            name="covariates_only",
            input_key="covariates",
            description="Age/sex/FIQ/motion covariates only",
            estimator_factory=covariates_only,
        ),
    ]


def summarize_best_params(fitted_estimators: list[BaseEstimator]) -> str:
    """Compress fold-wise best params into a short string for the result table."""

    params: list[str] = []
    for estimator in fitted_estimators:
        if hasattr(estimator, "best_params_"):
            params.append(json.dumps(getattr(estimator, "best_params_"), sort_keys=True))
    if not params:
        return ""
    most_common, count = Counter(params).most_common(1)[0]
    return f"{most_common} (x{count})"


def evaluate_experiments(
    views: dict[str, np.ndarray],
    experiments: list[ExperimentSpec],
    args: argparse.Namespace,
    outer_cv: StratifiedKFold,
) -> pd.DataFrame:
    """Run nested CV for each experiment and return a result table."""

    y = views["labels"]
    scoring = scoring_dict()
    rows: list[dict[str, object]] = []

    for spec in experiments:
        estimator = spec.estimator_factory(args)
        X = views[spec.input_key]
        print(f"\nRunning {spec.name}")

        cv_scores = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring,
            n_jobs=1,
            return_estimator=True,
            error_score=np.nan,
        )

        row: dict[str, object] = {
            "name": spec.name,
            "description": spec.description,
            "input_key": spec.input_key,
            "best_params": summarize_best_params(list(cv_scores["estimator"])),
        }
        for metric in scoring:
            metric_scores = np.asarray(cv_scores[f"test_{metric}"], dtype=float)
            row[metric] = float(np.nanmean(metric_scores))
            row[f"{metric}_std"] = float(np.nanstd(metric_scores))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "roc_auc", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def select_experiments(experiments: list[ExperimentSpec], requested: list[str] | None) -> list[ExperimentSpec]:
    """Return either the full experiment set or a validated whitelist."""

    if not requested:
        return experiments

    available = {spec.name for spec in experiments}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(
            "Unknown experiment names: "
            f"{', '.join(missing)}. Available experiments: {', '.join(sorted(available))}"
        )

    requested_set = set(requested)
    return [spec for spec in experiments if spec.name in requested_set]


def refit_best_model(
    results: pd.DataFrame,
    experiments: list[ExperimentSpec],
    views: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[str, BaseEstimator]:
    """Refit the top-ranked experiment on the full filtered cohort."""

    best_name = str(results.iloc[0]["name"])
    best_spec = next(spec for spec in experiments if spec.name == best_name)
    estimator = best_spec.estimator_factory(args)
    estimator.fit(views[best_spec.input_key], views["labels"])
    return best_name, estimator


def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    """Save a compact balanced-accuracy summary figure."""

    ordered = results.sort_values(["balanced_accuracy", "roc_auc"], ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.6 * len(ordered))))
    bars = ax.barh(
        ordered["name"],
        ordered["balanced_accuracy"],
        xerr=ordered["balanced_accuracy_std"],
        color="#4c72b0",
        edgecolor="black",
        capsize=4,
    )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Balanced Accuracy (outer CV)")
    ax.set_title("Offline ABIDE Autism vs Control Classification")
    ax.set_xlim(0.3, 1.0)

    for bar, (_, row) in zip(bars, ordered.iterrows()):
        y_mid = bar.get_y() + bar.get_height() / 2
        ax.text(
            float(row["balanced_accuracy"]) + 0.01,
            y_mid,
            f"AUC {row['roc_auc']:.2f} | Acc {row['accuracy']:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_metadata_summary(
    views: dict[str, np.ndarray],
    results: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """Persist a small JSON summary for reproducibility."""

    subjects = views["subjects"]
    summary = {
        "n_subjects": int(len(subjects)),
        "class_counts": subjects["group_name"].value_counts().to_dict(),
        "site_table": summarize_sites(subjects).to_dict(),
        "site_mode": args.site_mode,
        "site_filter": args.site_filter,
        "max_subjects": args.max_subjects,
        "best_model": str(results.iloc[0]["name"]),
        "best_balanced_accuracy": float(results.iloc[0]["balanced_accuracy"]),
        "best_roc_auc": float(results.iloc[0]["roc_auc"]),
    }
    output_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    args.cache_path = args.cache_path.resolve()
    args.output_dir = args.output_dir.resolve()

    bundle = load_or_build_cache(args)
    keep_index = filter_subjects(bundle.subjects, args)
    views = build_feature_views(bundle, keep_index)

    subjects = views["subjects"]
    site_table = summarize_sites(subjects)
    print("\nSelected cohort summary")
    print(f"Subjects: {len(subjects)}")
    print(f"Class counts: {subjects['group_name'].value_counts().to_dict()}")
    print("Site x label table:")
    print(site_table)

    if (site_table == 0).any().any():
        print(
            "\nWarning: at least one selected site contains only one class. "
            "Use --site-mode mixed for a less site-confounded evaluation."
        )

    outer_cv, inner_cv = build_cv(args)
    experiments = select_experiments(make_experiments(args, inner_cv), args.experiments)
    results = evaluate_experiments(views, experiments, args, outer_cv)

    results_csv = args.output_dir / "control_classification_results.csv"
    summary_png = args.output_dir / "classification_results.png"
    summary_json = args.output_dir / "control_classification_summary.json"
    best_model_path = args.output_dir / "control_classification_best_model.joblib"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_csv, index=False)
    plot_results(results, summary_png)
    write_metadata_summary(views, results, summary_json, args)

    best_name, best_estimator = refit_best_model(results, experiments, views, args)
    joblib.dump(
        {
            "name": best_name,
            "estimator": best_estimator,
            "subjects": subjects,
            "results": results,
            "args": vars(args),
        },
        best_model_path,
    )

    print("\nTop models")
    printable = results[["name", "balanced_accuracy", "accuracy", "roc_auc", "f1", "best_params"]]
    print(printable.to_string(index=False, justify="left", col_space=12))

    print("\nSaved artifacts")
    print(results_csv)
    print(summary_png)
    print(summary_json)
    print(best_model_path)


if __name__ == "__main__":
    main()
