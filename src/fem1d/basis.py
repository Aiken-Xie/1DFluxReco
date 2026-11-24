"""Reference Lagrange basis functions and Gauss quadrature on [-1, 1]."""

from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray


def reference_basis(
    xi: ArrayLike, degree: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return N, N_xi, N_xixi, each shaped (local_nodes, evaluation_points).

    Scalar xi is treated as one evaluation point. Derivatives are with respect
    to xi; physical derivatives require factors 2/h and (2/h)**2.
    """
    if isinstance(degree, bool) or not isinstance(degree, Integral) or degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    if xi.ndim != 1 or not np.all(np.isfinite(xi)) or np.any(np.abs(xi) > 1):
        raise ValueError("xi must be a finite scalar or 1D array within [-1, 1]")
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


def gauss_rule(order: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return nodes and weights for integration over the reference interval."""
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    return leggauss(order)
