"""Source-free 1D FEM solutions with element data retained for later recovery."""

from .problem import Problem
from .solver import Evaluation, FEMSolution, solve

__version__ = "0.1.0"
__all__ = ["Problem", "Evaluation", "FEMSolution", "solve"]
