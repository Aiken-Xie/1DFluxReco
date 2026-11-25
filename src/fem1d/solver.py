"""Dense FEM solve and element evaluation, retaining data for flux recovery."""

import numpy as np

from .assembly import assemble_stiffness, shape_functions
from .mesh import UniformMesh


class FEMSolver:
    """Assemble and solve the source-free problem, then evaluate the FEM field."""

    def __init__(self, problem, N, degree, method="galerkin"):
        self.problem = problem  # Problem object.
        self.mesh = UniformMesh(N, degree)  # UniformMesh object.
        self.method = method    # String: one of ("galerkin", "ead", "supg").

    def solve(self):
        """Store the assembled system and solution; return the nodal coefficients."""
        assembly = assemble_stiffness(self.problem, self.mesh, self.method)
        # N = self.mesh.N, p = self.mesh.degree, n = N*p + 1 global nodes.
        self.element_matrices = assembly.element_matrices  # (N, p+1, p+1).
        self.tau_e = assembly.tau_e                        # (N,).
        self.kappa_bar_e = assembly.kappa_bar_e            # (N,).
        self.stiffness_matrix = assembly.stiffness_matrix  # Dense (n, n).
        self.coefficients = np.zeros(self.mesh.nodes.size)  # (n,).
        self.free_dofs = np.arange(1, self.mesh.nodes.size - 1)  # (n-2,).

        boundary = np.array([0, self.mesh.nodes.size - 1])  # (2,).
        self.coefficients[boundary] = self.problem.boundary_values
        if self.free_dofs.size:
            free = self.free_dofs
            # Only prescribed boundary values contribute to rhs: shape (n-2,).
            rhs = -self.stiffness_matrix[np.ix_(free, boundary)] @ self.coefficients[boundary]
            self.coefficients[free] = np.linalg.solve(
                self.stiffness_matrix[np.ix_(free, free)], rhs
            )

        # With no forcing, R = Kd and R^e = k^e d^e.
        self.residual = self.stiffness_matrix @ self.coefficients  # (n,).
        self.element_residuals = np.einsum(  # (N, p+1).
            "eij,ej->ei", self.element_matrices, self.coefficients[self.mesh.elements]
        )
        return self.coefficients

    def evaluate(self, element, xi):
        """Return (x, u, du_dx, d2u_dx2, J_phys) in the selected elements.

        xi is a 1D NumPy array in [-1, 1]. element is a zero-based index or a
        1D NumPy array of indices. Outputs have shape (len(xi),) for one index,
        or (number_of_elements, len(xi)) for an array. Endpoint derivatives
        are never averaged. J_phys uses the physical diffusivity.
        """
        N, Nxi, Nxixi = shape_functions(xi, self.mesh.degree)
        local = self.coefficients[self.mesh.elements[element]]
        u = local @ N
        du_dx = (2 / self.mesh.h) * (local @ Nxi)
        d2u_dx2 = (2 / self.mesh.h)**2 * (local @ Nxixi)
        x = np.asarray(element)[..., None] * self.mesh.h + (self.mesh.h / 2) * (xi + 1)
        J_phys = self.problem.a * u - self.problem.kappa * du_dx
        return x, u, du_dx, d2u_dx2, J_phys
