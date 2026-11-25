"""Evaluate interior nodal control-volume balances from saved FEM and recovered coefficients."""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fem1d import FEMSolver, FluxReconstruction, Problem

INPUT = Path(__file__).resolve().parents[1] / "outputs" / "baseline"


def main():
    metadata = json.loads((INPUT / "metadata.json").read_text())
    problem = Problem(**metadata["problem"])
    cases = metadata["cases"]
    flux_names = ("direct", "reconstructed")
    J_exact = problem.exact_flux()
    flux_scale = problem.a * abs(problem.boundary_values[1] - problem.boundary_values[0])
    # Summary arrays: (number of cases,). Per-case defects: (N*degree-1,).
    max_defects = {name: np.zeros(len(cases)) for name in flux_names}
    scaled_max_defects = {name: np.zeros(len(cases)) for name in flux_names}
    max_face_errors = {name: np.zeros(len(cases)) for name in flux_names}
    defects, node_positions = {}, {}

    with np.load(INPUT / "solutions.npz", allow_pickle=False) as saved:
        progress = tqdm(cases, desc="Computing conservation", unit="case", mininterval=0)
        for i, case in enumerate(progress):
            key, degree = case["id"], case["degree"]
            progress.set_description(key)
            # Restore saved coefficients without solving or reconstructing again.
            solver = FEMSolver(problem, N=case["N"], degree=degree, method=case["method"])
            solver.coefficients = saved[f"{key}_coefficients"]  # (N*degree+1,).
            reconstruction = FluxReconstruction(solver.coefficients, solver.mesh, problem, case["method"])
            reconstruction.coefficients = saved[f"{key}_postprocessed_coefficients"]  # (N, degree+1).

            if degree == 1:
                face_xi = np.array([0.0])  # Element midpoint.
            elif degree == 2:
                face_xi = np.array([-0.5, 0.5])  # Element quarter points.
            else:
                raise NotImplementedError("Only linear and quadratic elements are implemented")

            element_indices = np.arange(solver.mesh.N)
            _, _, _, _, J_phys = solver.evaluate(element_indices, face_xi)
            face_fluxes = dict(
                direct=J_phys,  # (N, degree).
                reconstructed=reconstruction.evaluate(element_indices, face_xi),
            )

            node_positions[key] = solver.mesh.nodes[1:-1]  # Interior nodes only; no boundary volumes.
            defects[key] = {}
            for name, flux in face_fluxes.items():
                # Ordered faces run left to right; adjacent faces bound one interior nodal volume.
                # For quadratic elements this includes both midpoint-node and shared-vertex volumes.
                ordered_flux = flux.ravel()  # (N*degree,).
                # Accuracy at every unique control-volume face, distinct from volume balance.
                max_face_errors[name][i] = np.max(np.abs(ordered_flux - J_exact))
                J_left = ordered_flux[:-1]
                J_right = ordered_flux[1:]
                defects[key][name] = J_right - J_left  # Net outward flux; zero for no forcing.
                max_defects[name][i] = np.max(np.abs(defects[key][name]), initial=0.0)
                scaled_max_defects[name][i] = max_defects[name][i] / flux_scale

    print(
        f"{'Case':<22} {'Direct max':>13} {'Recovered max':>14} "
        f"{'Direct scaled':>14} {'Recovered scaled':>17} "
        f"{'Direct face error':>18} {'Recovered face error':>21}"
    )
    for i, case in enumerate(cases):
        print(
            f"{case['id']:<22} {max_defects['direct'][i]:13.5e} "
            f"{max_defects['reconstructed'][i]:14.5e} "
            f"{scaled_max_defects['direct'][i]:14.5e} "
            f"{scaled_max_defects['reconstructed'][i]:17.5e} "
            f"{max_face_errors['direct'][i]:18.5e} "
            f"{max_face_errors['reconstructed'][i]:21.5e}"
        )


if __name__ == "__main__":
    main()
