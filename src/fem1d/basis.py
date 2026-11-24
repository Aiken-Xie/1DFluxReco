"""Reference Lagrange basis functions and Gauss quadrature on [-1, 1]."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def reference_basis(xi, degree):
    """Return N, N_xi, N_xixi, each shaped (local_nodes, evaluation_points).

    Scalar xi is treated as one evaluation point. Derivatives are with respect
    to xi; physical derivatives require factors 2/h and (2/h)**2.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    one = np.ones_like(xi)
    if degree == 1:
        return (
            np.array([(1 - xi) / 2, (1 + xi) / 2]),
            np.array([-one / 2, one / 2]),
            np.zeros((2, xi.size)),
        )
    return (
        np.array([xi * (xi - 1) / 2, 1 - xi**2, xi * (xi + 1) / 2]),
        np.array([xi - 0.5, -2 * xi, xi + 0.5]),
        np.array([one, -2 * one, one]),
    )


def gauss_rule(order=3):
    """Return nodes and weights for integration over the reference interval."""
    return leggauss(order)
