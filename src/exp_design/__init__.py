"""Persistent-excitation experiment design utilities."""

from .basis import Basis, FourierBasis, LegendreBasis
from .criteria import CRITERIA, canonicalize_criterion
from .design import DesignResult, PersistentExcitationDesign
from .plotting import plot_design_summary

__all__ = [
    "Basis",
    "CRITERIA",
    "DesignResult",
    "FourierBasis",
    "LegendreBasis",
    "PersistentExcitationDesign",
    "canonicalize_criterion",
    "plot_design_summary",
]
