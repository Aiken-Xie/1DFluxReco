"""Dirichlet elimination and FEM results retaining all data needed for recovery."""

from dataclasses import dataclass
from numbers import Integral
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .assembly import assemble
from .basis import gauss_rule, reference_basis
from .mesh import UniformMesh
from .problem import Problem


@dataclass(frozen=True)
class Evaluation:
    """Element-local arrays, including when xi was a scalar.

    J_phys is the direct physical flux, not a reconstructed or stabilized flux.
    """

    x: NDArray[np.float64]
    u: NDArray[np.float64]
    du_dx: NDArray[np.float64]
    d2u_dx2: NDArray[np.float64]
    J_phys: NDArray[np.float64]


@dataclass(frozen=True)
class FEMSolution:
    """Original FEM field and unmodified conservative assembly data.

    Element residuals are k_e @ d[connectivity_e] - f_e after the solve.
    Only assembled free-node residuals vanish; boundary rows retain reactions.
    """

    problem: Problem
    mesh: UniformMesh
    method: str
    coefficients: NDArray[np.float64]
    element_matrices: NDArray[np.float64]
    element_loads: NDArray[np.float64]
    tau_e: NDArray[np.float64]
    kappa_bar_e: NDArray[np.float64]
    matrix: csr_matrix
    load: NDArray[np.float64]
    free_dofs: NDArray[np.int64]
    residual: NDArray[np.float64]
    element_residuals: NDArray[np.float64]

    @property
    def residual_inf(self) -> float:
        """Maximum assembled free-node residual; zero when no free nodes exist."""
        return float(np.max(np.abs(self.residual[self.free_dofs]), initial=0.0))

    @property
    def scaled_residual_inf(self) -> float:
        """Scale by max(1, ||K||_inf ||d||_inf + ||F||_inf)."""
        matrix_norm = float(np.max(np.asarray(abs(self.matrix).sum(axis=1))))
        scale = max(1.0, matrix_norm * np.linalg.norm(self.coefficients, np.inf)
                    + np.linalg.norm(self.load, np.inf))
        return self.residual_inf / scale

    @property
    def max_nodal_error(self) -> float:
        return float(np.max(np.abs(self.coefficients - self.problem.exact(self.mesh.nodes))))

    def evaluate(self, element: int, xi: ArrayLike) -> Evaluation:
        """Evaluate in one element; endpoint derivatives are never averaged.

        xi is a scalar or 1D array in [-1, 1]. Element numbering is zero-based.
        """
        if isinstance(element, bool) or not isinstance(element, Integral) or not 0 <= element < self.mesh.elements:
            raise ValueError("element must be an integer within the mesh")
        N, Nxi, Nxixi = reference_basis(xi, self.mesh.degree)
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        local = self.coefficients[self.mesh.connectivity[element]]
        u = local @ N
        du_dx = (2 / self.mesh.h) * (local @ Nxi)
        d2u_dx2 = (2 / self.mesh.h)**2 * (local @ Nxixi)
        x = self.mesh.vertices[element] + (self.mesh.h / 2) * (xi + 1)
        return Evaluation(x, u, du_dx, d2u_dx2, self.problem.a * u - self.problem.kappa * du_dx)

    def l2_error(self, quadrature_order: int = 20) -> float:
        """Field L2 error; increase quadrature_order to check layer resolution."""
        xi, weights = gauss_rule(quadrature_order)
        squared = 0.0
        for e in range(self.mesh.elements):
            values = self.evaluate(e, xi)
            squared += (self.mesh.h / 2) * np.dot(
                weights, (values.u - self.problem.exact(values.x))**2
            )
        return float(np.sqrt(squared))


def solve(problem: Problem, *, elements: int, degree: int, method: str = "galerkin") -> FEMSolution:
    """Solve on a uniform mesh of [0, 1], preserving the original K, F, and k_e."""
    mesh = UniformMesh(elements, degree)
    system = assemble(problem, mesh, method)
    d = np.zeros(mesh.nodes.size)
    boundary = np.array([0, mesh.nodes.size - 1])
    d[boundary] = problem.boundary_values
    free = np.arange(1, mesh.nodes.size - 1, dtype=np.int64)
    if free.size:
        rhs = system.load[free] - system.matrix[free][:, boundary] @ d[boundary]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                d[free] = spsolve(system.matrix[free][:, free], rhs, use_umfpack=False)
            except MatrixRankWarning as exc:
                raise RuntimeError("The reduced FEM matrix is singular") from exc
        if not np.all(np.isfinite(d)):
            raise RuntimeError("The FEM solve returned non-finite coefficients")
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
