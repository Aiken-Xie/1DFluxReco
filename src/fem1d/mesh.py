"""Uniform interval meshes with separate element endpoints and solution nodes."""

import numpy as np


class UniformMesh:
    """An affine uniform mesh; local node ordering is always left to right.

    Python element/node indices are zero-based. Degree 2 adds a solution node
    at each element midpoint without splitting the original element.
    """

    def __init__(self, elements, degree):
        self.elements = elements
        self.degree = degree
        self.vertices = np.linspace(0.0, 1.0, self.elements + 1)
        self.nodes = np.linspace(0.0, 1.0, self.degree * self.elements + 1)
        connectivity = self.degree * np.arange(self.elements)[:, None] + np.arange(self.degree + 1)
        self.connectivity = connectivity

    @property
    def h(self):
        """Full original element length, including for quadratic elements."""
        return 1.0 / self.elements
