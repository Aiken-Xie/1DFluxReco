"""Compute L2 errors from saved FEM and postprocessed coefficients, without solving again."""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fem1d import FEMSolver, FluxReconstruction, Problem

INPUT = Path(__file__).resolve().parents[1] / "outputs" / "baseline"
QUADRATURE_ORDER = 20


def main():
    metadata = json.loads((INPUT / "metadata.json").read_text())
    problem = Problem(**metadata["problem"])
    cases = metadata["cases"]
    J_exact = problem.exact_flux()
    error_names = ("solution_L2", "direct_flux_L2", "effective_flux_L2", "reconstructed_flux_L2")
    # Each array has shape (number of saved cases,).
    errors = {name: np.zeros(len(cases)) for name in error_names}
    xi, weights = np.polynomial.legendre.leggauss(QUADRATURE_ORDER)  # Each: (20,).

    with np.load(INPUT / "solutions.npz", allow_pickle=False) as saved:
        progress = tqdm(cases, desc="Computing L2 errors", unit="case", mininterval=0)
        for i, case in enumerate(progress):
            key, degree, h = case["id"], case["degree"], case["h"]
            progress.set_description(key)
            # Restore saved coefficients without solving or reconstructing again.
            solver = FEMSolver(problem, N=case["N"], degree=degree, method=case["method"])
            solver.coefficients = saved[f"{key}_coefficients"]  # (N*degree+1,).
            reconstruction = FluxReconstruction(solver.coefficients, solver.mesh, problem, case["method"])
            reconstruction.coefficients = saved[f"{key}_postprocessed_coefficients"]  # (N, degree+1).
            kappa_effective = (
                problem.kappa + saved[f"{key}_kappa_bar_e"]
                + problem.a**2 * saved[f"{key}_tau_e"]
            )  # (N,).

            element_indices = np.arange(solver.mesh.N)
            x, u, du_dx, _, J_phys = solver.evaluate(element_indices, xi)  # Each: (N, 20).
            J_phys_hat = reconstruction.evaluate(element_indices, xi)    # (N, 20).
            # Comparison flux; quadratic SUPG also has a second-derivative term.
            J_effective = problem.a * u - kappa_effective[:, None] * du_dx
            differences = dict(
                solution_L2=u - problem.exact_solution(x),
                direct_flux_L2=J_phys - J_exact,
                effective_flux_L2=J_effective - J_exact,
                reconstructed_flux_L2=J_phys_hat - J_exact,
            )
            for name, difference in differences.items():
                # Integrate over all elements, including dx = (h/2) dxi.
                errors[name][i] = np.sqrt((h / 2) * np.sum(weights * difference**2))

    print(f"{'Case':<22} {'u L2':>12} {'J_phys L2':>12} {'J_eff L2':>12} {'J_hat L2':>12}")
    for i, case in enumerate(cases):
        values = " ".join(f"{errors[name][i]:12.5e}" for name in error_names)
        print(f"{case['id']:<22} {values}")


if __name__ == "__main__":
    main()
