"""Conservative element forms, stabilization, and sparse global assembly."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .basis import gauss_rule, reference_basis
from .mesh import UniformMesh
from .problem import Problem

METHODS = ("galerkin", "ead", "supg")


def fitting_function(alpha: float) -> float:
    """Stable coth(alpha) - 1/alpha, with its continuous value at zero."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha < 1e-3:
        return alpha / 3 - alpha**3 / 45 + 2 * alpha**5 / 945
    return float(1 / np.tanh(alpha) - 1 / alpha)


def stabilization(problem: Problem, h: float, method: str) -> tuple[float, float]:
    """Return active (tau, kappa_bar) coefficients for the selected method.

    EAD uses kappa_bar only; SUPG uses tau only. The fitted choices satisfy
    kappa_bar(EAD) = a**2 * tau(SUPG), using the full element length for both
    degrees. This is a comparison parameter, not a quadratic optimality claim.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if method == "galerkin":
        return 0.0, 0.0
    alpha = problem.a * h / (2 * problem.kappa)
    fit = fitting_function(alpha)
    if method == "ead":
        return 0.0, problem.a * h * fit / 2
    return h * fit / (2 * problem.a), 0.0


def local_system(
    problem: Problem, h: float, degree: int, method: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Return (k, f, tau, kappa_bar) before any Dirichlet elimination.

    Rows are test indices, columns are trial indices. The Galerkin advection
    entry is -integral(a*N_i,x*N_j), as in the reference report. It differs
    from the homework's advective form at element boundary contributions.
    """
    tau, kappa_bar = stabilization(problem, h, method)
    xi, weights = gauss_rule(3)
    N, Nxi, Nxixi = reference_basis(xi, degree)
    Nx, Nxx = (2 / h) * Nxi, (2 / h)**2 * Nxixi
    k = np.zeros((degree + 1, degree + 1))
    for q, weight in enumerate(weights):
        integrand = (
            -problem.a * np.outer(Nx[:, q], N[:, q])
            + (problem.kappa + kappa_bar) * np.outer(Nx[:, q], Nx[:, q])
            + problem.a * tau * np.outer(
                Nx[:, q], problem.a * Nx[:, q] - problem.kappa * Nxx[:, q]
            )
        )
        k += weight * (h / 2) * integrand
    return k, np.zeros(degree + 1), tau, kappa_bar


@dataclass(frozen=True)
class AssembledSystem:
    mesh: UniformMesh
    problem: Problem
    method: str
    element_matrices: NDArray[np.float64]
    element_loads: NDArray[np.float64]
    tau_e: NDArray[np.float64]
    kappa_bar_e: NDArray[np.float64]
    matrix: csr_matrix
    load: NDArray[np.float64]


def assemble(problem: Problem, mesh: UniformMesh, method: str) -> AssembledSystem:
    """Assemble once from identical elements on this uniform constant problem."""
    k, f, tau, kappa_bar = local_system(problem, mesh.h, mesh.degree, method)
    element_matrices = np.repeat(k[None, :, :], mesh.elements, axis=0)
    element_loads = np.repeat(f[None, :], mesh.elements, axis=0)
    local_nodes = mesh.degree + 1
    rows = np.repeat(mesh.connectivity, local_nodes, axis=1).ravel()
    columns = np.tile(mesh.connectivity, (1, local_nodes)).ravel()
    matrix = coo_matrix(
        (element_matrices.ravel(), (rows, columns)),
        shape=(mesh.nodes.size, mesh.nodes.size),
    ).tocsr()
    load = np.zeros(mesh.nodes.size)
    np.add.at(load, mesh.connectivity.ravel(), element_loads.ravel())
    return AssembledSystem(
        mesh, problem, method, element_matrices, element_loads,
        np.full(mesh.elements, tau), np.full(mesh.elements, kappa_bar), matrix, load,
    )
