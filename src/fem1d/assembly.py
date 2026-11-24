"""Conservative element forms, stabilization, and sparse global assembly."""

import numpy as np
from scipy.sparse import coo_matrix

from .basis import gauss_rule, reference_basis

METHODS = ("galerkin", "ead", "supg")


def fitting_function(alpha):
    """Stable coth(alpha) - 1/alpha, with its continuous value at zero."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha < 1e-3:
        return alpha / 3 - alpha**3 / 45 + 2 * alpha**5 / 945
    return float(1 / np.tanh(alpha) - 1 / alpha)


def stabilization(problem, h, method):
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


def local_system(problem, h, degree, method):
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


class AssembledSystem:
    def __init__(self, mesh, problem, method, element_matrices, element_loads,
                 tau_e, kappa_bar_e, matrix, load):
        self.mesh = mesh
        self.problem = problem
        self.method = method
        self.element_matrices = element_matrices
        self.element_loads = element_loads
        self.tau_e = tau_e
        self.kappa_bar_e = kappa_bar_e
        self.matrix = matrix
        self.load = load


def assemble(problem, mesh, method):
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
