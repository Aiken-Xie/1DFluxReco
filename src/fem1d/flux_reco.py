"""Source-free flux reconstruction from the report's linear and quadratic formulas."""

import numpy as np

from .assembly import element_stiffness, shape_functions


class FluxReconstruction:
    """Recover a physical flux from solved nodal values and their FEM formulation."""

    def __init__(self, solution, mesh, problem, method):
        self.solution = solution  # Original nodal values: shape (mesh.N*mesh.degree+1,).
        self.mesh = mesh          # UniformMesh object.
        self.problem = problem    # Problem object.
        self.method = method      # String: one of ("galerkin", "ead", "supg").
        self.coefficients = None  # Set by reconstruct(): shape (mesh.N, mesh.degree+1).

    def reconstruct(self):
        """Store and return the element-local postprocessed coefficients."""
        # N = self.mesh.N, p = self.mesh.degree. These are local, not global, coefficients.
        # Fix the arbitrary additive constant by setting each element's first coefficient to zero.
        self.coefficients = np.zeros((self.mesh.N, self.mesh.degree + 1))  # (N, p+1).
        if self.mesh.degree == 1:
            face_xi = np.array([0.0])  # (1,); the element midpoint.
        elif self.mesh.degree == 2:
            face_xi = np.array([-0.5, 0.5])  # (2,); the element quarter points.
        else:
            raise NotImplementedError("Only linear and quadratic elements are implemented")

        a, kappa, h = self.problem.a, self.problem.kappa, self.mesh.h
        k, _, _ = element_stiffness(self.problem, h, self.mesh.degree, self.method)
        N_faces, _, _ = shape_functions(face_xi, self.mesh.degree)  # (p+1, p).
        local = self.solution[self.mesh.elements]  # (N, p+1).
        residual = local @ k.T  # (N, p+1); each row is an element residual.
        u = local @ N_faces    # (N, p); original FEM field at the faces.
        if self.mesh.degree == 1:
            # d_tilde_2 = (h/kappa)*(a*u(0) - R_1), with d_tilde_1 = 0.
            self.coefficients[:, 1] = (h / kappa) * (a * u[:, 0] - residual[:, 0]) #d_tilde_2
        elif self.mesh.degree == 2:
            # d_tilde_2 = h/(2*kappa)*(a*u(-1/2) - R_1), with d_tilde_1 = 0.
            # d_tilde_3 = d_tilde_2 + h/(2*kappa)*(a*u(1/2) + R_3).
            self.coefficients[:, 1] = (h / (2 * kappa)) * (a * u[:, 0] - residual[:, 0]) #d_tilde_2
            self.coefficients[:, 2] = (
                self.coefficients[:, 1] + (h / (2 * kappa)) * (a * u[:, 1] + residual[:, 2]) #d_tilde_3
            )

        return self.coefficients

    def evaluate(self, element, xi):
        """Return J_phys_hat at the requested reference coordinates after reconstruction.

        xi is a 1D NumPy array in [-1, 1]. element is one zero-based index or a
        1D NumPy array of indices. The result has shape (len(xi),) for one index,
        or (number_of_elements, len(xi)) for an array.
        Advection uses the original FEM field; diffusion uses the element-local
        postprocessed gradient and physical kappa.
        """
        N, Nxi, _ = shape_functions(xi, self.mesh.degree)
        u = self.solution[self.mesh.elements[element]] @ N
        du_post_dx = (2 / self.mesh.h) * (self.coefficients[element] @ Nxi)
        J_phys_hat = self.problem.a * u - self.problem.kappa * du_post_dx
        return J_phys_hat
