import numpy as np
import pytest

from fem1d.basis import gauss_rule, reference_basis
from fem1d.mesh import UniformMesh


@pytest.mark.parametrize("degree", [1, 2])
def test_nodal_interpolation_and_partition(degree):
    reference_nodes = np.linspace(-1, 1, degree + 1)
    np.testing.assert_array_equal(reference_basis(reference_nodes, degree)[0], np.eye(degree + 1))
    points = np.array([-1., -.73, -.2, 0., .17, .88, 1.])
    N, first, second = reference_basis(points, degree)
    np.testing.assert_allclose(N.sum(axis=0), 1, atol=2e-16)
    np.testing.assert_allclose(first.sum(axis=0), 0, atol=2e-16)
    np.testing.assert_allclose(second.sum(axis=0), 0, atol=2e-16)


@pytest.mark.parametrize("degree", [1, 2])
def test_polynomial_reproduction_and_derivatives(degree):
    # An independently specified polynomial must be reproduced, including derivatives.
    nodes = np.linspace(-1, 1, degree + 1)
    points = np.linspace(-1, 1, 11)
    N, first, second = reference_basis(points, degree)
    coefficients = 2 + 3 * nodes
    exact = 2 + 3 * points
    exact_first = np.full(points.size, 3.)
    exact_second = np.zeros(points.size)
    if degree == 2:
        coefficients += 4 * nodes**2
        exact += 4 * points**2
        exact_first += 8 * points
        exact_second += 8
    np.testing.assert_allclose(coefficients @ N, exact, atol=2e-15)
    np.testing.assert_allclose(coefficients @ first, exact_first, atol=2e-15)
    np.testing.assert_allclose(coefficients @ second, exact_second, atol=2e-15)


@pytest.mark.parametrize("degree", [1, 2])
def test_uniform_connectivity_and_physical_mapping(degree):
    mesh = UniformMesh(3, degree)
    assert mesh.nodes.size == degree * 3 + 1
    local = np.linspace(-1, 1, degree + 1)
    for e, connectivity in enumerate(mesh.connectivity):
        expected = mesh.vertices[e] + mesh.h / 2 * (local + 1)
        np.testing.assert_allclose(mesh.nodes[connectivity], expected, atol=1e-15)
    for e in range(2):
        assert mesh.connectivity[e, -1] == mesh.connectivity[e + 1, 0]
    if degree == 2:
        assert all(np.count_nonzero(mesh.connectivity == row[1]) == 1 for row in mesh.connectivity)


def test_gauss_rule_and_scalar_evaluation():
    points, weights = gauss_rule(3)
    for power in range(6):
        expected = 0 if power % 2 else 2 / (power + 1)
        assert np.dot(weights, points**power) == pytest.approx(expected, abs=1e-15)
    assert reference_basis(0., 2)[0].shape == (3, 1)


@pytest.mark.parametrize("elements,degree", [(0, 1), (-1, 2), (2.5, 1), (True, 1), (2, 0), (2, 3), (2, 1.)])
def test_invalid_mesh(elements, degree):
    with pytest.raises(ValueError):
        UniformMesh(elements, degree)


@pytest.mark.parametrize("points,degree", [([1.1], 1), ([np.nan], 2), ([[0.]], 2), ([0.], 3)])
def test_invalid_basis_input(points, degree):
    with pytest.raises(ValueError):
        reference_basis(points, degree)
