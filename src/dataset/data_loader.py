"""Data loading utilities for neuroscience timeseries data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


def load_mat_data(filepath: str) -> Dict[str, np.ndarray]:
    """Load .mat file. Returns dict with FC_all, FC_mean, timeseries_all."""
    data = loadmat(filepath)
    return {
        "FC_all": data["FC_all"],
        "FC_mean": data["FC_mean"],
        "timeseries_all": data["timeseries_all"],
    }


# Condition-name -> integer control value mapping for LSD dataset.
LSD_CONDITION_MAP: Dict[str, int] = {
    "PLACEBO": 0,
    "LSD": 2,
    "LSD+KET": 1,
}

_LSD_ROI_PARTS = [
    "Thalamus",
    "Visual_Cortex",
    "Posterior_Cingulate_Cortex",
    "Temporal_Gyrus",
]

_SUBJECT_PATTERN = re.compile(r"sub-[A-Za-z0-9]+")


def _normalize_cond_name(name: str) -> str:
    return name.upper().replace(" ", "").replace("_", "").replace("+", "")


def _extract_subject_id(path: Path) -> str:
    for part in path.parts:
        match = _SUBJECT_PATTERN.fullmatch(part)
        if match is not None:
            return match.group(0)
    match = _SUBJECT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not infer BIDS subject id from path: {path}")
    return match.group(0)


def _locate_confounds_file(bold_file: Path) -> Path | None:
    """Try common fMRIPrep confounds sidecar filenames for a BOLD file."""
    candidates = (
        bold_file.name.replace("_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv"),
        bold_file.name.replace("_desc-preproc_bold.nii", "_desc-confounds_timeseries.tsv"),
        bold_file.name.replace("_desc-preproc_bold.nii.gz", "_desc-confounds_regressors.tsv"),
        bold_file.name.replace("_desc-preproc_bold.nii", "_desc-confounds_regressors.tsv"),
    )
    for cand in candidates:
        conf_path = bold_file.with_name(cand)
        if conf_path.exists():
            return conf_path
    return None


def _stack_and_trim_timeseries(timeseries_list: Sequence[np.ndarray]) -> np.ndarray:
    """Stack ROI-timeseries arrays (n_rois, T_i) by trimming to shared min T."""
    if not timeseries_list:
        raise ValueError("No timeseries were extracted.")

    n_rois = int(timeseries_list[0].shape[0])
    min_t = min(int(ts.shape[1]) for ts in timeseries_list)
    if min_t < 2:
        raise ValueError("Need at least 2 timepoints per run to build a dataset.")

    trimmed: list[np.ndarray] = []
    for idx, ts in enumerate(timeseries_list):
        if ts.ndim != 2:
            raise ValueError(f"Expected 2D timeseries for subject {idx}, got shape {ts.shape}.")
        if int(ts.shape[0]) != n_rois:
            raise ValueError(
                "Inconsistent number of ROIs across subjects/runs "
                f"(expected {n_rois}, got {ts.shape[0]} at index {idx})."
            )
        trimmed.append(ts[:, :min_t].astype(np.float32, copy=False))
    return np.stack(trimmed, axis=0)


def _extract_with_nilearn_masker(
    bold_files: Sequence[Path],
    confounds_files: Sequence[Path | None] | None,
    atlas_maps: str,
    *,
    smoothing_fwhm: float | None,
) -> np.ndarray:
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize=False,
        detrend=True,
        smoothing_fwhm=smoothing_fwhm,
        memory_level=0,
        verbose=0,
    )

    all_ts: list[np.ndarray] = []
    for idx, bold_path in enumerate(bold_files):
        confounds_path = None
        if confounds_files is not None and idx < len(confounds_files):
            confounds_path = confounds_files[idx]
        confounds_arg = str(confounds_path) if confounds_path is not None else None
        ts = masker.fit_transform(str(bold_path), confounds=confounds_arg)
        # fit_transform returns (n_timepoints, n_rois); transpose to (n_rois, T)
        all_ts.append(np.asarray(ts.T, dtype=np.float32))

    return _stack_and_trim_timeseries(all_ts)


def load_lsd_data(
    data_dir: str = "data/lsd",
    condition_map: Dict[str, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load LSD pharmacological dataset.

    Returns
    -------
    timeseries : ndarray, shape ``(total_subjects, n_rois, n_timepoints)``
        Concatenated timeseries across all conditions.
    control : ndarray, shape ``(total_subjects,)``
        Integer control label for each subject (see *condition_map*).
    patient_ids : ndarray, shape ``(total_subjects,)``
        Integer patient identifier (0..n_patients-1) for each row, so that
        all conditions for the same physical patient share the same id.
    condition_names : list[str]
        Ordered condition names for reference.
    """
    data_dir = Path(data_dir)
    data_ts = loadmat(str(data_dir / "time_series_data.mat"))
    data_names = loadmat(str(data_dir / "condition_names.mat"))

    if condition_map is None:
        condition_map = LSD_CONDITION_MAP

    n_conditions = data_names["conditionNames"].shape[1]
    condition_names: list[str] = []
    all_ts: list[np.ndarray] = []
    all_ctrl: list[np.ndarray] = []
    all_patient_ids: list[np.ndarray] = []

    for i in range(n_conditions):
        cond_name = str(data_names["conditionNames"][0, i][0])
        condition_names.append(cond_name)
        # Each condition: (n_subjects, n_rois, n_timepoints)
        cond_data = np.array([data_ts["pyDict"][0, i][part][0, 0] for part in _LSD_ROI_PARTS]).transpose(
            1,
            0,
            2,
        )
        n_subs = int(cond_data.shape[0])

        # Map condition name to control value (case-insensitive match).
        # Sort keys longest-first so "LSD+KET" matches before "LSD".
        ctrl_val = None
        cond_norm = _normalize_cond_name(cond_name)
        for key in sorted(condition_map, key=len, reverse=True):
            key_norm = _normalize_cond_name(key)
            if key_norm == cond_norm or key_norm in cond_norm:
                ctrl_val = condition_map[key]
                break
        if ctrl_val is None:
            # Fallback: use condition index
            ctrl_val = i

        all_ts.append(cond_data)
        all_ctrl.append(np.full(n_subs, ctrl_val, dtype=np.int64))
        all_patient_ids.append(np.arange(n_subs, dtype=np.int64))

    timeseries = np.concatenate(all_ts, axis=0)  # (total_subjects, n_rois, n_timepoints)
    control = np.concatenate(all_ctrl, axis=0)  # (total_subjects,)
    patient_ids = np.concatenate(all_patient_ids, axis=0)  # (total_subjects,)
    return timeseries, control, patient_ids, condition_names


def compute_fc_from_timeseries(timeseries: torch.Tensor) -> torch.Tensor:
    """Compute FC (Pearson correlation) from real part of timeseries.

    Args:
        timeseries: ``(n_subjects, n_rois, n_timepoints)`` real or complex.

    Returns:
        FC matrices ``(n_subjects, n_rois, n_rois)``.
    """
    ts = timeseries.real if torch.is_complex(timeseries) else timeseries
    t_len = int(ts.shape[2])
    if t_len < 2:
        raise ValueError("Need at least 2 timepoints to compute FC.")
    ts_centered = ts - ts.mean(dim=2, keepdim=True)
    std = ts_centered.std(dim=2, keepdim=True) + 1e-8
    ts_norm = ts_centered / std
    return torch.bmm(ts_norm, ts_norm.transpose(1, 2)) / (t_len - 1)


def fft_bandpass_3d(x: torch.Tensor, dt: float, f_lo: float, f_hi: float) -> torch.Tensor:
    """FFT brick-wall bandpass on last dimension. x: (..., T) real."""
    t_len = x.shape[-1]
    x_fft = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(t_len, d=dt).to(device=x.device, dtype=x.dtype)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    for _ in range(x_fft.ndim - 1):
        mask = mask.unsqueeze(0)
    return torch.fft.irfft(x_fft * mask, n=t_len, dim=-1)


def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """Analytic signal via Hilbert transform on last dimension. x: (..., T) real -> complex."""
    t_len = x.shape[-1]
    x_fft = torch.fft.fft(x, dim=-1)
    h = torch.zeros(t_len, device=x.device, dtype=x_fft.dtype)
    if t_len % 2 == 0:
        h[0] = 1.0
        h[t_len // 2] = 1.0
        h[1 : t_len // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (t_len + 1) // 2] = 2.0
    for _ in range(x_fft.ndim - 1):
        h = h.unsqueeze(0)
    return torch.fft.ifft(x_fft * h, dim=-1)


class NeuroscienceDataset(Dataset):
    """PyTorch Dataset for neuroscience timeseries data.

    Stores **complex** timeseries (analytic signal via Hilbert transform).
    FC matrices remain real (empirical targets from .mat file or computed).

    Attributes:
        timeseries: Complex tensor (n_subjects, n_rois, n_timepoints)
        fc_matrices: Real tensor (n_subjects, n_rois, n_rois)
        fc_mean: Real tensor (n_rois, n_rois)
        control: Real tensor (n_subjects, n_control_dims) - per-subject control input
        ts: Time array (n_timepoints,)
        dt: TR in seconds
        n_control_dims: int - number of control dimensions (0 means no control)
    """

    def __init__(
        self,
        filepath: str,
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = True,
        denoise_f_lo: float = 0.008,
        denoise_f_hi: float = 0.08,
    ):
        self.device = device
        self.dt = dt

        data = load_mat_data(filepath)

        if max_subjects is not None:
            n_total = int(data["timeseries_all"].shape[2])
            if max_subjects <= 0 or max_subjects > n_total:
                raise ValueError(f"max_subjects must be in [1, {n_total}], got {max_subjects}")
            data["timeseries_all"] = data["timeseries_all"][:, :, :max_subjects]
            data["FC_all"] = data["FC_all"][:, :, :max_subjects]
            data["FC_mean"] = data["FC_all"].mean(axis=2)

        # (ROIs, timepoints, subjects) -> (subjects, ROIs, timepoints)
        timeseries = torch.tensor(
            data["timeseries_all"].transpose(2, 0, 1),
            dtype=torch.float32,
            device=device,
        )
        self.fc_matrices = torch.tensor(
            data["FC_all"].transpose(2, 0, 1),
            dtype=torch.float32,
            device=device,
        )
        self.fc_mean = torch.tensor(data["FC_mean"], dtype=torch.float32, device=device)

        self._init_timeseries(timeseries, normalize, fourier_denoise, dt, denoise_f_lo, denoise_f_hi)

        # No control for the standard ts_young dataset
        n_subjects = int(self.timeseries.shape[0])
        self.control = torch.zeros(n_subjects, 0, device=device)
        self.n_control_dims = 0
        self.patient_ids = None

    # ------------------------------------------------------------------
    # Shared initialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_raw_timeseries(
        cls,
        raw_ts: np.ndarray,
        *,
        normalize: bool,
        device: str,
        max_subjects: Optional[int],
        dt: float,
        fourier_denoise: bool,
        denoise_f_lo: float,
        denoise_f_hi: float,
        control: np.ndarray | None = None,
        patient_ids: np.ndarray | None = None,
    ) -> "NeuroscienceDataset":
        if raw_ts.ndim != 3:
            raise ValueError(f"Expected raw_ts shape (subjects, rois, timepoints), got {raw_ts.shape}.")

        if max_subjects is not None:
            n_total = int(raw_ts.shape[0])
            if max_subjects <= 0 or max_subjects > n_total:
                raise ValueError(f"max_subjects must be in [1, {n_total}], got {max_subjects}")
            raw_ts = raw_ts[:max_subjects]
            if control is not None:
                control = control[:max_subjects]
            if patient_ids is not None:
                patient_ids = patient_ids[:max_subjects]

        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.device = device
        obj.dt = dt

        timeseries = torch.tensor(raw_ts, dtype=torch.float32, device=device)
        obj._init_timeseries(timeseries, normalize, fourier_denoise, dt, denoise_f_lo, denoise_f_hi)

        obj.fc_matrices = compute_fc_from_timeseries(obj.timeseries)
        obj.fc_mean = obj.fc_matrices.mean(dim=0)

        if control is None:
            n_subjects = int(obj.timeseries.shape[0])
            obj.control = torch.zeros(n_subjects, 0, device=device)
            obj.n_control_dims = 0
        else:
            control_arr = np.asarray(control)
            if control_arr.ndim == 1:
                control_arr = control_arr[:, None]
            if int(control_arr.shape[0]) != int(obj.timeseries.shape[0]):
                raise ValueError(
                    f"control has {control_arr.shape[0]} rows but timeseries has {obj.timeseries.shape[0]} subjects."
                )
            obj.control = torch.tensor(control_arr, dtype=torch.float32, device=device)
            obj.n_control_dims = int(obj.control.shape[1])

        # Patient-level grouping (e.g. LSD: 25 patients × 3 conditions = 75 rows).
        # When set, data splitting should be done at the patient level.
        if patient_ids is not None:
            obj.patient_ids = torch.tensor(np.asarray(patient_ids), dtype=torch.long, device=device)
        else:
            obj.patient_ids = None

        return obj

    def _init_timeseries(
        self,
        timeseries: torch.Tensor,
        normalize: bool,
        fourier_denoise: bool,
        dt: float,
        denoise_f_lo: float,
        denoise_f_hi: float,
    ) -> None:
        """Z-score, bandpass, Hilbert transform and set shape attributes."""
        if normalize:
            mean = timeseries.mean(dim=2, keepdim=True)
            std = timeseries.std(dim=2, keepdim=True) + 1e-8
            timeseries = (timeseries - mean) / std

        if fourier_denoise:
            timeseries = fft_bandpass_3d(timeseries, dt, denoise_f_lo, denoise_f_hi)

        self.timeseries = hilbert_transform(timeseries)
        self.n_subjects, self.n_rois, self.n_timepoints = self.timeseries.shape
        self.ts = torch.linspace(
            0,
            (self.n_timepoints - 1) * dt,
            self.n_timepoints,
            device=timeseries.device,
        )

    # ------------------------------------------------------------------
    # Alternate constructor: LSD pharmacological dataset
    # ------------------------------------------------------------------

    @classmethod
    def from_lsd(
        cls,
        data_dir: str = "data/lsd",
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = True,
        denoise_f_lo: float = 0.008,
        denoise_f_hi: float = 0.08,
        condition_map: Optional[Dict[str, int]] = None,
    ) -> "NeuroscienceDataset":
        """Create a dataset from the LSD pharmacological experiment."""
        raw_ts, raw_ctrl, raw_patient_ids, cond_names = load_lsd_data(data_dir, condition_map)
        obj = cls._from_raw_timeseries(
            raw_ts,
            normalize=normalize,
            device=device,
            max_subjects=max_subjects,
            dt=dt,
            fourier_denoise=fourier_denoise,
            denoise_f_lo=denoise_f_lo,
            denoise_f_hi=denoise_f_hi,
            control=raw_ctrl,
            patient_ids=raw_patient_ids,
        )
        n_patients = len(set(raw_patient_ids.tolist()))
        print(f"LSD dataset: conditions={cond_names}, control_values={sorted(set(raw_ctrl.tolist()))}, "
              f"n_patients={n_patients}, n_timeseries={len(raw_ts)}")
        return obj

    # ------------------------------------------------------------------
    # Alternate constructor: nilearn fetchers + atlas extraction
    # ------------------------------------------------------------------

    @classmethod
    def from_nilearn(
        cls,
        dataset_name: str = "development_fmri",
        data_dir: str = "data/nilearn",
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = True,
        denoise_f_lo: float = 0.008,
        denoise_f_hi: float = 0.08,
        atlas_n_rois: int = 100,
        atlas_yeo_networks: int = 7,
        atlas_resolution_mm: int = 1,
        atlas_smoothing_fwhm: float | None = None,
        n_subjects: int | None = None,
        reduce_confounds: bool = True,
    ) -> "NeuroscienceDataset":
        """Create a dataset from nilearn fetchers using Schaefer atlas ROI extraction."""
        try:
            from nilearn import datasets as nilearn_datasets
        except ModuleNotFoundError as exc:
            raise ImportError(
                "nilearn is required for dataset_type='nilearn'. Install with: pip install nilearn"
            ) from exc

        dataset_key = dataset_name.strip().lower()
        fetch_n_subjects = n_subjects if n_subjects is not None else max_subjects

        if dataset_key in {"development_fmri", "development"}:
            fetched = nilearn_datasets.fetch_development_fmri(
                n_subjects=fetch_n_subjects,
                reduce_confounds=reduce_confounds,
                data_dir=data_dir,
                verbose=1,
            )
        elif dataset_key == "adhd":
            fetched = nilearn_datasets.fetch_adhd(
                n_subjects=fetch_n_subjects if fetch_n_subjects is not None else 30,
                data_dir=data_dir,
                verbose=1,
            )
        else:
            raise ValueError(
                "Unsupported nilearn dataset_name. Supported: 'development_fmri', 'adhd'. "
                f"Got '{dataset_name}'."
            )

        func_files = [Path(p) for p in fetched.func]
        confounds_files_raw = getattr(fetched, "confounds", None)
        confounds_files: list[Path | None] | None = None
        if confounds_files_raw is not None:
            confounds_files = [Path(p) if p else None for p in confounds_files_raw]

        atlas = nilearn_datasets.fetch_atlas_schaefer_2018(
            n_rois=atlas_n_rois,
            yeo_networks=atlas_yeo_networks,
            resolution_mm=atlas_resolution_mm,
            data_dir=data_dir,
            verbose=1,
        )

        raw_ts = _extract_with_nilearn_masker(
            bold_files=func_files,
            confounds_files=confounds_files,
            atlas_maps=str(atlas.maps),
            smoothing_fwhm=atlas_smoothing_fwhm,
        )

        obj = cls._from_raw_timeseries(
            raw_ts,
            normalize=normalize,
            device=device,
            max_subjects=max_subjects,
            dt=dt,
            fourier_denoise=fourier_denoise,
            denoise_f_lo=denoise_f_lo,
            denoise_f_hi=denoise_f_hi,
        )
        print(
            "nilearn dataset loaded: "
            f"name={dataset_name}, runs={len(func_files)}, atlas_rois={atlas_n_rois}, "
            f"subjects_used={obj.n_subjects}"
        )
        return obj

    # ------------------------------------------------------------------
    # Alternate constructor: BIDS derivatives (fMRIPrep expected)
    # ------------------------------------------------------------------

    @classmethod
    def from_bids(
        cls,
        bids_root: str,
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = True,
        denoise_f_lo: float = 0.008,
        denoise_f_hi: float = 0.08,
        derivatives_dir: str = "derivatives/fmriprep",
        task: str | None = None,
        space: str | None = "MNI152NLin2009cAsym",
        desc: str | None = "preproc",
        subject_ids: Sequence[str] | None = None,
        n_runs_per_subject: int = 1,
        atlas_n_rois: int = 100,
        atlas_yeo_networks: int = 7,
        atlas_resolution_mm: int = 1,
        atlas_smoothing_fwhm: float | None = None,
        atlas_data_dir: str = "data/nilearn",
    ) -> "NeuroscienceDataset":
        """Create a dataset from BIDS derivatives by extracting atlas ROI timeseries."""
        try:
            from nilearn import datasets as nilearn_datasets
        except ModuleNotFoundError as exc:
            raise ImportError(
                "nilearn is required for BIDS extraction. Install with: pip install nilearn"
            ) from exc

        root = Path(bids_root)
        search_root = root / derivatives_dir
        if not search_root.exists():
            search_root = root

        bold_candidates = sorted(search_root.rglob("*_bold.nii.gz")) + sorted(
            search_root.rglob("*_bold.nii")
        )

        if not bold_candidates:
            raise FileNotFoundError(
                "No BOLD files found. Expected preprocessed BIDS derivatives under "
                f"'{search_root}' (e.g. *_desc-preproc_bold.nii.gz)."
            )

        filtered: list[Path] = []
        wanted_subjects = {s if s.startswith("sub-") else f"sub-{s}" for s in subject_ids or []}
        for bold_file in bold_candidates:
            name = bold_file.name
            if task and f"_task-{task}_" not in name:
                continue
            if space and f"_space-{space}_" not in name:
                continue
            if desc and f"_desc-{desc}_" not in name:
                continue
            subj = _extract_subject_id(bold_file)
            if wanted_subjects and subj not in wanted_subjects:
                continue
            filtered.append(bold_file)

        if not filtered:
            raise FileNotFoundError(
                "No BOLD files matched filters. "
                f"task={task!r}, space={space!r}, desc={desc!r}, subject_ids={subject_ids!r}."
            )

        grouped: dict[str, list[Path]] = defaultdict(list)
        for bold_file in filtered:
            grouped[_extract_subject_id(bold_file)].append(bold_file)

        selected: list[Path] = []
        for subj in sorted(grouped):
            selected.extend(sorted(grouped[subj])[: max(1, n_runs_per_subject)])
            if max_subjects is not None and len(selected) >= max_subjects:
                break

        if not selected:
            raise ValueError("No runs selected from filtered BOLD files.")

        confounds = [_locate_confounds_file(bold_path) for bold_path in selected]

        atlas = nilearn_datasets.fetch_atlas_schaefer_2018(
            n_rois=atlas_n_rois,
            yeo_networks=atlas_yeo_networks,
            resolution_mm=atlas_resolution_mm,
            data_dir=atlas_data_dir,
            verbose=1,
        )

        raw_ts = _extract_with_nilearn_masker(
            bold_files=selected,
            confounds_files=confounds,
            atlas_maps=str(atlas.maps),
            smoothing_fwhm=atlas_smoothing_fwhm,
        )

        obj = cls._from_raw_timeseries(
            raw_ts,
            normalize=normalize,
            device=device,
            max_subjects=max_subjects,
            dt=dt,
            fourier_denoise=fourier_denoise,
            denoise_f_lo=denoise_f_lo,
            denoise_f_hi=denoise_f_hi,
        )
        print(
            "BIDS dataset loaded: "
            f"root={root}, selected_runs={len(selected)}, atlas_rois={atlas_n_rois}, "
            f"subjects_used={obj.n_subjects}"
        )
        return obj

    # ------------------------------------------------------------------
    # Alternate constructor: OpenNeuro downloader + BIDS extraction
    # ------------------------------------------------------------------

    @classmethod
    def from_openneuro(
        cls,
        dataset: str,
        target_dir: str = "data/openneuro",
        tag: str | None = None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = True,
        denoise_f_lo: float = 0.008,
        denoise_f_hi: float = 0.08,
        bids_relative_path: str | None = None,
        derivatives_dir: str = "derivatives/fmriprep",
        task: str | None = None,
        space: str | None = "MNI152NLin2009cAsym",
        desc: str | None = "preproc",
        subject_ids: Sequence[str] | None = None,
        n_runs_per_subject: int = 1,
        atlas_n_rois: int = 100,
        atlas_yeo_networks: int = 7,
        atlas_resolution_mm: int = 1,
        atlas_smoothing_fwhm: float | None = None,
        atlas_data_dir: str = "data/nilearn",
    ) -> "NeuroscienceDataset":
        """Download OpenNeuro data with openneuro-py, then load as BIDS derivatives."""
        try:
            import openneuro
        except ModuleNotFoundError as exc:
            raise ImportError(
                "openneuro-py is required for dataset_type='openneuro'. "
                "Install with: pip install openneuro-py"
            ) from exc

        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        include_arg = list(include) if include else None
        exclude_arg = list(exclude) if exclude else None

        openneuro.download(
            dataset=dataset,
            tag=tag,
            target_dir=target_path,
            include=include_arg,
            exclude=exclude_arg,
        )

        dataset_root = target_path / dataset
        if not dataset_root.exists():
            dataset_root = target_path
        if bids_relative_path:
            dataset_root = dataset_root / bids_relative_path

        return cls.from_bids(
            bids_root=str(dataset_root),
            normalize=normalize,
            device=device,
            max_subjects=max_subjects,
            dt=dt,
            fourier_denoise=fourier_denoise,
            denoise_f_lo=denoise_f_lo,
            denoise_f_hi=denoise_f_hi,
            derivatives_dir=derivatives_dir,
            task=task,
            space=space,
            desc=desc,
            subject_ids=subject_ids,
            n_runs_per_subject=n_runs_per_subject,
            atlas_n_rois=atlas_n_rois,
            atlas_yeo_networks=atlas_yeo_networks,
            atlas_resolution_mm=atlas_resolution_mm,
            atlas_smoothing_fwhm=atlas_smoothing_fwhm,
            atlas_data_dir=atlas_data_dir,
        )

    # ------------------------------------------------------------------
    # Alternate constructor: DataLad install/get + BIDS extraction
    # ------------------------------------------------------------------

    @classmethod
    def from_datalad(
        cls,
        source: str,
        dataset_dir: str = "data/datalad_dataset",
        get_paths: Sequence[str] | None = None,
        normalize: bool = True,
        device: str = "cpu",
        max_subjects: Optional[int] = None,
        dt: float = 0.72,
        fourier_denoise: bool = True,
        denoise_f_lo: float = 0.008,
        denoise_f_hi: float = 0.08,
        bids_relative_path: str | None = None,
        derivatives_dir: str = "derivatives/fmriprep",
        task: str | None = None,
        space: str | None = "MNI152NLin2009cAsym",
        desc: str | None = "preproc",
        subject_ids: Sequence[str] | None = None,
        n_runs_per_subject: int = 1,
        atlas_n_rois: int = 100,
        atlas_yeo_networks: int = 7,
        atlas_resolution_mm: int = 1,
        atlas_smoothing_fwhm: float | None = None,
        atlas_data_dir: str = "data/nilearn",
    ) -> "NeuroscienceDataset":
        """Install/get a DataLad dataset and load BIDS derivatives from it."""
        try:
            from datalad import api as datalad_api
        except ModuleNotFoundError as exc:
            raise ImportError(
                "datalad is required for dataset_type='datalad'. Install with: pip install datalad"
            ) from exc

        ds_path = Path(dataset_dir)
        ds_path.parent.mkdir(parents=True, exist_ok=True)

        if not ds_path.exists() or not (ds_path / ".git").exists():
            datalad_api.install(source=source, path=str(ds_path), get_data=False)

        if get_paths:
            resolved_get_paths = [str(ds_path / path) for path in get_paths]
            datalad_api.get(path=resolved_get_paths, dataset=str(ds_path), recursive=True)

        bids_root = ds_path / bids_relative_path if bids_relative_path else ds_path
        return cls.from_bids(
            bids_root=str(bids_root),
            normalize=normalize,
            device=device,
            max_subjects=max_subjects,
            dt=dt,
            fourier_denoise=fourier_denoise,
            denoise_f_lo=denoise_f_lo,
            denoise_f_hi=denoise_f_hi,
            derivatives_dir=derivatives_dir,
            task=task,
            space=space,
            desc=desc,
            subject_ids=subject_ids,
            n_runs_per_subject=n_runs_per_subject,
            atlas_n_rois=atlas_n_rois,
            atlas_yeo_networks=atlas_yeo_networks,
            atlas_resolution_mm=atlas_resolution_mm,
            atlas_smoothing_fwhm=atlas_smoothing_fwhm,
            atlas_data_dir=atlas_data_dir,
        )

    def __len__(self) -> int:
        return self.n_subjects

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.timeseries[idx], self.fc_matrices[idx]


def load_dataset(cfg: Any, device: str) -> "NeuroscienceDataset":
    """Load dataset according to *cfg* and print a summary.

    *cfg* is duck-typed: any object with ``dataset_type``, ``data_path``,
    ``max_subjects``, ``tr``, ``fourier_denoise``, ``denoise_f_lo``,
    ``denoise_f_hi`` attributes (i.e. a ``TrainingConfig`` subclass).
    """
    dataset_type = getattr(cfg, "dataset_type", "ts_young")
    dataset_kwargs = dict(
        normalize=True,
        device=device,
        max_subjects=cfg.max_subjects,
        dt=cfg.tr,
        fourier_denoise=cfg.fourier_denoise,
        denoise_f_lo=cfg.denoise_f_lo,
        denoise_f_hi=cfg.denoise_f_hi,
    )

    if dataset_type == "lsd":
        dataset = NeuroscienceDataset.from_lsd(
            data_dir=getattr(cfg, "lsd_data_dir", "data/lsd"),
            **dataset_kwargs,
        )
    elif dataset_type == "nilearn":
        dataset = NeuroscienceDataset.from_nilearn(
            dataset_name=cfg.nilearn_dataset,
            data_dir=cfg.nilearn_data_dir,
            n_subjects=cfg.nilearn_n_subjects,
            reduce_confounds=cfg.nilearn_reduce_confounds,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            **dataset_kwargs,
        )
    elif dataset_type == "openneuro":
        dataset = NeuroscienceDataset.from_openneuro(
            dataset=cfg.openneuro_dataset,
            target_dir=cfg.openneuro_target_dir,
            tag=cfg.openneuro_tag,
            include=cfg.openneuro_include,
            exclude=cfg.openneuro_exclude,
            bids_relative_path=cfg.bids_relative_path,
            derivatives_dir=cfg.bids_derivatives_dir,
            task=cfg.bids_task,
            space=cfg.bids_space,
            desc=cfg.bids_desc,
            subject_ids=cfg.bids_subject_ids,
            n_runs_per_subject=cfg.bids_runs_per_subject,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            atlas_data_dir=cfg.nilearn_data_dir,
            **dataset_kwargs,
        )
    elif dataset_type == "datalad":
        dataset = NeuroscienceDataset.from_datalad(
            source=cfg.datalad_source,
            dataset_dir=cfg.datalad_dataset_dir,
            get_paths=cfg.datalad_get_paths,
            bids_relative_path=cfg.bids_relative_path,
            derivatives_dir=cfg.bids_derivatives_dir,
            task=cfg.bids_task,
            space=cfg.bids_space,
            desc=cfg.bids_desc,
            subject_ids=cfg.bids_subject_ids,
            n_runs_per_subject=cfg.bids_runs_per_subject,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            atlas_data_dir=cfg.nilearn_data_dir,
            **dataset_kwargs,
        )
    elif dataset_type == "bids":
        if not cfg.bids_root:
            raise ValueError("dataset_type='bids' requires cfg.bids_root to be set.")
        dataset = NeuroscienceDataset.from_bids(
            bids_root=cfg.bids_root,
            derivatives_dir=cfg.bids_derivatives_dir,
            task=cfg.bids_task,
            space=cfg.bids_space,
            desc=cfg.bids_desc,
            subject_ids=cfg.bids_subject_ids,
            n_runs_per_subject=cfg.bids_runs_per_subject,
            atlas_n_rois=cfg.atlas_n_rois,
            atlas_yeo_networks=cfg.atlas_yeo_networks,
            atlas_resolution_mm=cfg.atlas_resolution_mm,
            atlas_smoothing_fwhm=cfg.atlas_smoothing_fwhm,
            atlas_data_dir=cfg.nilearn_data_dir,
            **dataset_kwargs,
        )
    elif dataset_type in ("ts_young", "mat"):
        dataset = NeuroscienceDataset(filepath=cfg.data_path, **dataset_kwargs)
    else:
        raise ValueError(
            "Unsupported dataset_type. Expected one of "
            "{'ts_young', 'mat', 'lsd', 'nilearn', 'openneuro', 'datalad', 'bids'}, "
            f"got {dataset_type!r}."
        )

    print("Loaded dataset:")
    print(f"  - subjects={dataset.n_subjects}  ROIs={dataset.n_rois}  T={dataset.n_timepoints}")
    print(f"  - FC shape={dataset.fc_mean.shape}  dtype={dataset.timeseries.dtype}  dt={dataset.dt}s")
    print(f"  - control_dims={dataset.n_control_dims}")
    return dataset
