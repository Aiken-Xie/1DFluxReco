"""Source-free 1D FEM solutions with element data retained for later recovery."""

from .problem import Problem
from .solver import FEMSolver
from .flux_reco import FluxReconstruction

__version__ = "0.1.0"
__all__ = ["Problem", "FEMSolver", "FluxReconstruction"]
