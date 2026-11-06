"""Group-level quantities, estimated from training data only.

Several inputs are averages over subjects rather than properties of one
recording: the connectome prior ``C0`` that seeds the coupling of the
Hopf-family models, the intrinsic frequencies ``omega``, and the grid-search FC
target. Any held-out data averaged into them leaks into the fit. Held-out data
here means validation and test subjects, and also the second temporal half of
the training subjects, which is the within-subject (``test_intra``) test set.

:func:`training_view` builds these quantities. It returns a view of the
dataset restricted to the training subjects and, within them, to the training
segment ``[0, t_stop)``. ``fc_mean`` on that view is the mean Pearson FC of
those segments. Do not use ``dataset.fc_mean`` of the full dataset for
training: it is the mean over every subject's full recording.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type annotations
    from .data_loader import NeuroscienceDataset


class GroupLeakageError(RuntimeError):
    """A group-level quantity would include held-out data."""


def training_segment_stop(n_timepoints: int, use_full_timeseries: bool = False) -> int:
    """End (exclusive) of the training segment of every training recording.

    The first temporal half is used for training. The second half is the
    within-subject test set, except with ``use_full_timeseries``, where the whole
    recording is used for training.
    """
    n_timepoints = int(n_timepoints)
    return n_timepoints if use_full_timeseries else n_timepoints // 2


def _as_index_array(idx: Iterable[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(idx) if not isinstance(idx, np.ndarray) else idx, dtype=np.int64)
    return arr.reshape(-1)


def training_view(
    dataset: "NeuroscienceDataset",
    train_idx: Iterable[int] | np.ndarray,
    *,
    t_stop: int,
    held_out_idx: Optional[Iterable[int] | np.ndarray] = None,
) -> "NeuroscienceDataset":
    """Shallow copy of *dataset* restricted to the training data.

    Args:
        dataset: The full dataset.
        train_idx: The training subjects, in the order the caller uses them.
        t_stop: End (exclusive) of the training segment, normally
            :func:`training_segment_stop`.
        held_out_idx: Subjects that must not contribute, e.g. the validation
            and test subjects. Overlap with *train_idx* raises
            :class:`GroupLeakageError`.

    Returns:
        A view whose ``timeseries`` is ``dataset.timeseries[train_idx, :, :t_stop]``,
        whose ``fc_matrices`` are the per-subject Pearson FC of those segments,
        and whose ``fc_mean`` is their mean. ``omega`` estimated from
        ``view.timeseries`` (as ``build_model`` does) therefore uses the same
        segments.
    """
    from .data_loader import compute_fc_from_timeseries

    train = _as_index_array(train_idx)
    n_subjects = int(dataset.timeseries.shape[0])
    n_timepoints = int(dataset.timeseries.shape[2])

    if train.size == 0:
        raise GroupLeakageError("training_view needs at least one training subject.")
    if len(np.unique(train)) != train.size:
        raise GroupLeakageError(f"Duplicate subjects in train_idx: {train.tolist()}")
    if train.min() < 0 or train.max() >= n_subjects:
        raise GroupLeakageError(
            f"train_idx out of range [0, {n_subjects}): {train.tolist()}"
        )
    if held_out_idx is not None:
        overlap = sorted(set(train.tolist()) & set(_as_index_array(held_out_idx).tolist()))
        if overlap:
            raise GroupLeakageError(
                f"Held-out subjects {overlap} are in the training set of a group-level "
                "estimate. The connectome prior and omega must use training subjects only."
            )
    t_stop = int(t_stop)
    if not 2 <= t_stop <= n_timepoints:
        raise GroupLeakageError(f"t_stop must be in [2, {n_timepoints}], got {t_stop}")

    view = copy.copy(dataset)
    view.timeseries = dataset.timeseries[train, :, :t_stop]
    view.fc_matrices = compute_fc_from_timeseries(view.timeseries)
    view.fc_mean = view.fc_matrices.mean(dim=0)
    for attr in ("control", "patient_ids"):
        value = getattr(dataset, attr, None)
        if value is not None:
            setattr(view, attr, value[train])
    view.n_subjects = int(train.size)
    view.n_timepoints = t_stop
    view.ts = dataset.ts[:t_stop] if getattr(dataset, "ts", None) is not None else None
    view.group_prior_info = {
        "subjects": sorted(int(i) for i in train),
        "t_stop": t_stop,
        "n_timepoints": n_timepoints,
    }
    return view
