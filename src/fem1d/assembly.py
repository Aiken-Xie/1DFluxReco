"""Conservative element forms, stabilization, and dense global assembly."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def shape_functions(xi, degree):
    """Return N, N_xi, N_xixi, each shaped (local_nodes, evaluation_points).

    xi is a 1D NumPy array: quadrature points during assembly, or sampling
    points when evaluating the solution. Derivatives are with respect to xi;
    physical derivatives require factors 2/h and (2/h)**2.
    """
    one = np.ones(len(xi))
    if degree == 1:
        return (
            np.array([(1 - xi) / 2, (1 + xi) / 2]),
            np.array([-one / 2, one / 2]),
            np.zeros((2, xi.size)),
        )
    elif degree == 2:
        return (
            np.array([xi * (xi - 1) / 2, 1 - xi**2, xi * (xi + 1) / 2]),
            np.array([xi - 0.5, -2 * xi, xi + 0.5]),
            np.array([one, -2 * one, one]),
        )
    else:
        raise NotImplementedError("Only linear and quadratic elements are implemented")


def element_stiffness(problem, h, degree, method):
    """Return (k, tau, kappa_bar) before any Dirichlet elimination.

    Rows are test indices, columns are trial indices. The Galerkin advection
    entry is -integral(a*N_i,x*N_j), as in the reference report. It differs
    from the homework's advective form at element boundary contributions.
    """
    # EAD uses artificial diffusion; SUPG uses tau. Galerkin uses neither.
    tau, kappa_bar = 0.0, 0.0
    if method != "galerkin":
        alpha = problem.a * h / (2 * problem.kappa)
        fit = 1 / np.tanh(alpha) - 1 / alpha
        if method == "ead":
            kappa_bar = problem.a * h * fit / 2
        else:
            tau = h * fit / (2 * problem.a)

    xi, weights = leggauss(3)  # Each has shape (3,).
    N, Nxi, Nxixi = shape_functions(xi, degree)  # Each: (degree+1, 3).
    Nx, Nxx = (2 / h) * Nxi, (2 / h)**2 * Nxixi
    k = np.zeros((degree + 1, degree + 1))  # (degree+1, degree+1).
    for q, weight in enumerate(weights):
        integrand = (
            -problem.a * np.outer(Nx[:, q], N[:, q])
            + (problem.kappa + kappa_bar) * np.outer(Nx[:, q], Nx[:, q])
            + problem.a * tau * np.outer(
                Nx[:, q], problem.a * Nx[:, q] - problem.kappa * Nxx[:, q]
            )
        )
        k += weight * (h / 2) * integrand
    return k, tau, kappa_bar


class AssemblyData:
    def __init__(self, mesh, problem, method, element_matrices,
                 tau_e, kappa_bar_e, stiffness_matrix):
        self.mesh = mesh        # UniformMesh object.
        self.problem = problem  # Problem object.
        self.method = method    # String: one of ("galerkin", "ead", "supg").
        # N = mesh.N, p = mesh.degree, n = N*p + 1 global nodes.
        self.element_matrices = element_matrices  # (N, p+1, p+1).
        self.tau_e = tau_e                        # (N,).
        self.kappa_bar_e = kappa_bar_e            # (N,).
        self.stiffness_matrix = stiffness_matrix  # Dense global stiffness matrix (n, n).


def assemble_stiffness(problem, mesh, method):
    """Assemble once from identical elements on this uniform constant problem."""
    k, tau, kappa_bar = element_stiffness(problem, mesh.h, mesh.degree, method)
    element_matrices = np.repeat(k[None, :, :], mesh.N, axis=0)
    stiffness_matrix = np.zeros((mesh.nodes.size, mesh.nodes.size))
    for nodes in mesh.elements:
        # Add the local matrix into this element's global rows and columns.
        stiffness_matrix[np.ix_(nodes, nodes)] += k
    return AssemblyData(
        mesh, problem, method, element_matrices,
        np.full(mesh.N, tau), np.full(mesh.N, kappa_bar), stiffness_matrix,
    )
