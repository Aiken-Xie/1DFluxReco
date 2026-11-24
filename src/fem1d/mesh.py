"""Uniform interval meshes with separate element endpoints and solution nodes."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UniformMesh:
    """An affine uniform mesh; local node ordering is always left to right.

    Python element/node indices are zero-based. Degree 2 adds a solution node
    at each element midpoint without splitting the original element.
    """

    elements: int
    degree: int
    vertices: NDArray[np.float64] = field(init=False, repr=False)
    nodes: NDArray[np.float64] = field(init=False, repr=False)
    connectivity: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.elements, bool) or not isinstance(self.elements, Integral) or self.elements < 1:
            raise ValueError("elements must be a positive integer")
        if isinstance(self.degree, bool) or not isinstance(self.degree, Integral) or self.degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        object.__setattr__(self, "elements", int(self.elements))
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "vertices", np.linspace(0.0, 1.0, self.elements + 1))
        object.__setattr__(self, "nodes", np.linspace(0.0, 1.0, self.degree * self.elements + 1))
        connectivity = self.degree * np.arange(self.elements)[:, None] + np.arange(self.degree + 1)
        object.__setattr__(self, "connectivity", connectivity.astype(np.int64))

    @property
    def h(self) -> float:
        """Full original element length, including for quadratic elements."""
        return 1.0 / self.elements
