import numpy as np
import pytest

from fem1d import Problem
from fem1d.assembly import METHODS, assemble, fitting_function, local_system, stabilization
from fem1d.mesh import UniformMesh


def analytic_matrix(problem, h, degree, method):
    """Closed-form expectations from report Section 4 and Appendix A.

    Production assembly integrates basis functions instead of using these arrays.
    """
    a, kappa = problem.a, problem.kappa
    alpha = a * h / (2 * kappa)
    fit = 1 / np.tanh(alpha) - 1 / alpha
    added = a * h / 2 * fit if method == "ead" else 0.
    tau = h / (2 * a) * fit if method == "supg" else 0.
    diffusion = kappa + added + a*a*tau
    if degree == 1:
        return a / 2 * np.array([[1., 1.], [-1., -1.]]) + diffusion / h * np.array([[1., -1.], [-1., 1.]])
    return (
        a / 6 * np.array([[3., 4., -1.], [-4., 0., 4.], [1., -4., -3.]])
        + diffusion / (3*h) * np.array([[7., -8., 1.], [-8., 16., -8.], [1., -8., 7.]])
        + 4*a*kappa*tau / h**2 * np.array([[1., -2., 1.], [0., 0., 0.], [-1., 2., -1.]])
    )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("a,kappa,h", [(1., .05, .25), (.7, .3, .17), (2.3, .007, .023)])
def test_quadrature_matches_independent_analytic_matrices(degree, method, a, kappa, h):
    problem = Problem(a=a, kappa=kappa)
    matrix, load, _, _ = local_system(problem, h, degree, method)
    np.testing.assert_allclose(matrix, analytic_matrix(problem, h, degree, method), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(load, 0)
    np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=1e-13*np.linalg.norm(matrix, np.inf))


def test_linear_equivalence_and_quadratic_supg_diffusion_residual():
    problem, h = Problem(), .25
    for degree in (1, 2):
        ead = local_system(problem, h, degree, "ead")[0]
        supg, _, tau, _ = local_system(problem, h, degree, "supg")
        if degree == 1:
            np.testing.assert_allclose(supg, ead, rtol=1e-13, atol=1e-13)
        else:
            expected = 4*problem.a*problem.kappa*tau/h**2 * np.array([[1., -2., 1.], [0., 0., 0.], [-1., 2., -1.]])
            np.testing.assert_allclose(supg-ead, expected, atol=1e-13)
            assert np.linalg.norm(supg-ead) > 1e-3


@pytest.mark.parametrize("degree", [1, 2])
def test_original_conservative_matrix_and_element_storage(degree):
    mesh = UniformMesh(4, degree)
    problem = Problem()
    system = assemble(problem, mesh, "galerkin")
    independently_assembled = np.zeros(system.matrix.shape)
    for connectivity, element_matrix in zip(mesh.connectivity, system.element_matrices):
        independently_assembled[np.ix_(connectivity, connectivity)] += element_matrix
    np.testing.assert_allclose(system.matrix.toarray(), independently_assembled, atol=1e-14)
    # A constant field has nonzero endpoint contributions in the conservative form.
    constant_residual = system.matrix @ np.ones(mesh.nodes.size)
    expected = np.zeros(mesh.nodes.size)
    expected[[0, -1]] = [problem.a, -problem.a]
    np.testing.assert_allclose(constant_residual, expected, atol=2e-14)


@pytest.mark.parametrize("alpha", [0., 1e-12, 1e-8, 1e-4, 1e3])
def test_stable_fitting_limits(alpha):
    value = fitting_function(alpha)
    assert np.isfinite(value) and value >= 0
    expected = alpha/3-alpha**3/45 if alpha < 1e-3 else 1-1/alpha
    assert value == pytest.approx(expected, rel=1e-14, abs=1e-25)


def test_matched_stabilization_uses_full_element_length():
    problem, h = Problem(), .25
    tau, unused_diffusion = stabilization(problem, h, "supg")
    unused_time, added = stabilization(problem, h, "ead")
    assert unused_diffusion == unused_time == 0
    assert added == pytest.approx(problem.a**2*tau)
    assert tau == pytest.approx(.07669591372657604)


@pytest.mark.parametrize("method,h", [("unknown", .1), ("ead", 0.), ("supg", np.nan)])
def test_invalid_assembly_input(method, h):
    with pytest.raises(ValueError):
        local_system(Problem(), h, 1, method)
