"""Uniform interval meshes and the solution nodes belonging to each element."""

import numpy as np


class UniformMesh:
    """Uniform mesh; local node ordering is always left to right.

    Python element/node indices are zero-based. Degree 2 adds a solution node
    at each element midpoint to accommodate quadratic elements.
    """

    def __init__(self, N, degree):
        # N: number of elements in [0, 1], a positive integer.
        # degree: polynomial degree, 1 for linear or 2 for quadratic elements.
        self.N = N
        self.degree = degree
        self.h = 1.0 / N  # Full element length for either degree.

        # Coordinates of all solution nodes: shape (N*degree+1,).
        # Includes midpoint nodes for quadratic elements.
        self.nodes = np.linspace(0.0, 1.0, self.degree * self.N + 1)

        # Shape (N, degree+1); row e lists the global node indices in element e.
        # Two elements: linear [[0, 1], [1, 2]], quadratic [[0, 1, 2], [2, 3, 4]].
        self.elements = (
            self.degree * np.arange(self.N)[:, None]
            + np.arange(self.degree + 1)
        )
