"""Metrics module for model evaluation."""

from .fc_metrics import (
    compute_static_fc,
    FCCorrelation,
    FCMSE,
)
from .timeseries_metrics import (
    AutocorrelationDistance,
    PowerSpectrumDistance,
    TemporalCorrelation,
    L2Timeseries,
    AmplitudeLoss,
    OmegaLoss,
    compute_ref_amplitude,
    compute_ref_omega,
)
from .dynamics_metrics import (
    FCD,
    PhFCD,
    Metastability,
    PhaseFC,
    ks_distance_2samp,
    metastability_value,
    phase_coherence_fc,
    phase_coherence_matrix,
    phfcd_matrix,
    phfcd_distribution,
)
from .metrics_store import MetricsStore

__all__ = [
    # Low-level helpers
    "compute_static_fc",
    "compute_ref_amplitude",
    "compute_ref_omega",
    "ks_distance_2samp",
    "metastability_value",
    "phase_coherence_fc",
    "phase_coherence_matrix",
    "phfcd_matrix",
    "phfcd_distribution",
    # nn.Module metric/loss classes
    "FCCorrelation",
    "FCMSE",
    "PowerSpectrumDistance",
    "TemporalCorrelation",
    "AutocorrelationDistance",
    "FCD",
    "PhFCD",
    "Metastability",
    "PhaseFC",
    "L2Timeseries",
    "AmplitudeLoss",
    "OmegaLoss",
    # Utilities
    "MetricsStore",
]
