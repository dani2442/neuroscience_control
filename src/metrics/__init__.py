"""Metrics module for model evaluation."""

from .fc_metrics import (
    compute_static_fc,
    fc_correlation,
    fc_mse,
    FCCorrelation,
    FCMSE,
)
from .timeseries_metrics import (
    AutocorrelationDistance,
    PowerSpectrumDistance,
    TemporalCorrelation,
)
from .dynamics_metrics import (
    FCD,
    PhFCD,
    Metastability,
    PhaseFC,
    ks_distance_2samp,
    metastability_value,
    phase_coherence_fc,
    phase_coherence_fc_correlation,
    phase_coherence_matrix,
    phfcd_matrix,
    phfcd_distribution,
)
from .metrics_store import MetricsStore

__all__ = [
    # Low-level helpers
    "compute_static_fc",
    "fc_correlation",
    "fc_mse",
    "ks_distance_2samp",
    "metastability_value",
    "phase_coherence_fc",
    "phase_coherence_fc_correlation",
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
    # Utilities
    "MetricsStore",
]
