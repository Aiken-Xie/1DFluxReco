"""Dirichlet elimination and FEM results retaining all data needed for recovery."""

import numpy as np
from scipy.sparse.linalg import spsolve

from .assembly import assemble
from .basis import gauss_rule, reference_basis
from .mesh import UniformMesh


class Evaluation:
    """Element-local arrays, including when xi was a scalar.

    J_phys is the direct physical flux, not a reconstructed or stabilized flux.
    """

    def __init__(self, x, u, du_dx, d2u_dx2, J_phys):
        self.x = x
        self.u = u
        self.du_dx = du_dx
        self.d2u_dx2 = d2u_dx2
        self.J_phys = J_phys


class FEMSolution:
    """Original FEM field and unmodified conservative assembly data.

    Element residuals are k_e @ d[connectivity_e] - f_e after the solve.
    Only assembled free-node residuals vanish; boundary rows retain reactions.
    """

    def __init__(self, problem, mesh, method, coefficients, element_matrices,
                 element_loads, tau_e, kappa_bar_e, matrix, load, free_dofs,
                 residual, element_residuals):
        self.problem = problem
        self.mesh = mesh
        self.method = method
        self.coefficients = coefficients
        self.element_matrices = element_matrices
        self.element_loads = element_loads
        self.tau_e = tau_e
        self.kappa_bar_e = kappa_bar_e
        self.matrix = matrix
        self.load = load
        self.free_dofs = free_dofs
        self.residual = residual
        self.element_residuals = element_residuals

    @property
    def residual_inf(self):
        """Maximum assembled free-node residual; zero when no free nodes exist."""
        return float(np.max(np.abs(self.residual[self.free_dofs]), initial=0.0))

    @property
    def scaled_residual_inf(self):
        """Scale by max(1, ||K||_inf ||d||_inf + ||F||_inf)."""
        matrix_norm = float(np.max(np.asarray(abs(self.matrix).sum(axis=1))))
        scale = max(1.0, matrix_norm * np.linalg.norm(self.coefficients, np.inf)
                    + np.linalg.norm(self.load, np.inf))
        return self.residual_inf / scale

    @property
    def max_nodal_error(self):
        return float(np.max(np.abs(self.coefficients - self.problem.exact_solution(self.mesh.nodes))))

    def evaluate(self, element, xi):
        """Evaluate in one element; endpoint derivatives are never averaged.

        xi is a scalar or 1D array in [-1, 1]. Element numbering is zero-based.
        """
        N, Nxi, Nxixi = reference_basis(xi, self.mesh.degree)
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        local = self.coefficients[self.mesh.connectivity[element]]
        u = local @ N
        du_dx = (2 / self.mesh.h) * (local @ Nxi)
        d2u_dx2 = (2 / self.mesh.h)**2 * (local @ Nxixi)
        x = self.mesh.vertices[element] + (self.mesh.h / 2) * (xi + 1)
        return Evaluation(x, u, du_dx, d2u_dx2, self.problem.a * u - self.problem.kappa * du_dx)

    def l2_error(self, quadrature_order=20):
        """Field L2 error; increase quadrature_order to check layer resolution."""
        xi, weights = gauss_rule(quadrature_order)
        squared = 0.0
        for e in range(self.mesh.elements):
            values = self.evaluate(e, xi)
            squared += (self.mesh.h / 2) * np.dot(
                weights, (values.u - self.problem.exact_solution(values.x))**2
            )
        return float(np.sqrt(squared))


def solve(problem, *, elements, degree, method="galerkin"):
    """Solve on a uniform mesh of [0, 1], preserving the original K, F, and k_e."""
    mesh = UniformMesh(elements, degree)
    system = assemble(problem, mesh, method)
    d = np.zeros(mesh.nodes.size)
    boundary = np.array([0, mesh.nodes.size - 1])
    d[boundary] = problem.boundary_values
    free = np.arange(1, mesh.nodes.size - 1, dtype=np.int64)
    if free.size:
        rhs = system.load[free] - system.matrix[free][:, boundary] @ d[boundary]
        d[free] = spsolve(system.matrix[free][:, free], rhs, use_umfpack=False)
    residual = system.matrix @ d - system.load
    element_residuals = np.einsum(
        "eij,ej->ei", system.element_matrices, d[mesh.connectivity]
    ) - system.element_loads
    return FEMSolution(
        problem=problem, mesh=mesh, method=method, coefficients=d,
        element_matrices=system.element_matrices, element_loads=system.element_loads,
        tau_e=system.tau_e, kappa_bar_e=system.kappa_bar_e,
        matrix=system.matrix, load=system.load, free_dofs=free,
        residual=residual, element_residuals=element_residuals,
    )
