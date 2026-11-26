"""Run all six method/degree combinations and save results under outputs/baseline.

Run with: python scripts/solve_and_reco.py (after installing .[scripts]).
Settings are ordinary Python values so this example also serves as a template.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fem1d import FEMSolver, FluxReconstruction, Problem

PROBLEM = Problem(a=1., kappa=.05, boundary_values=(0., 10.))
ELEMENT_COUNTS = (4, 8, 16, 32, 64, 128)
METHODS = ("galerkin", "ead", "supg")
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "baseline"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Solve every case and collect its numerical results.
    solutions = {}
    cases, arrays = [], {}
    case_settings = [
        (degree, method, N)
        for degree in (1, 2)
        for method in METHODS
        for N in ELEMENT_COUNTS
    ]
    progress = tqdm(case_settings, desc="Solving cases", unit="case", mininterval=0)
    for degree, method, N in progress:
        progress.set_description(f"degree={degree}, {method}, N={N}")
        result = FEMSolver(PROBLEM, N=N, degree=degree, method=method)
        result.solve()
        solutions[degree, method, N] = result
        key = f"p{degree}_{method}_n{N}"
        cases.append(dict(
            id=key, method=method, degree=degree, N=N,
            dofs=result.coefficients.size,
            h=result.mesh.h, cell_Pe=PROBLEM.a*result.mesh.h/PROBLEM.kappa,
            tau=float(result.tau_e[0]), artificial_kappa=float(result.kappa_bar_e[0]),
        ))
        for name, value in dict(
            nodes=result.mesh.nodes, coefficients=result.coefficients,
            elements=result.mesh.elements,
            element_matrices=result.element_matrices,
            element_residuals=result.element_residuals,
            tau_e=result.tau_e, kappa_bar_e=result.kappa_bar_e,
        ).items():
            arrays[f"{key}_{name}"] = value

    # Reconstruct the flux for each solved case before saving or plotting.
    progress = tqdm(case_settings, desc="Reconstructing fluxes", unit="case", mininterval=0)
    for degree, method, N in progress:
        progress.set_description(f"reconstruction: degree={degree}, {method}, N={N}")
        result = solutions[degree, method, N]
        reconstruction = FluxReconstruction(
            solution=result.coefficients, mesh=result.mesh, problem=PROBLEM, method=method
        )
        reconstruction.reconstruct()

        # Evaluate the flux at control-volume faces only when preparing the saved output.
        if degree == 1:
            face_xi = np.array([0.0])
        elif degree == 2:
            face_xi = np.array([-0.5, 0.5])
        element_indices = np.arange(N)
        face_coordinates, _, _, _, _ = result.evaluate(element_indices, face_xi)
        face_fluxes = reconstruction.evaluate(element_indices, face_xi)  # (N, degree).
        boundary_fluxes = np.zeros((N, degree+1))
        boundary_fluxes[:, 0] = -result.element_residuals[:, 0]
        boundary_fluxes[:, -1] = -result.element_residuals[:, -1]

        key = f"p{degree}_{method}_n{N}"
        for name, value in dict(
            postprocessed_coefficients=reconstruction.coefficients,
            boundary_fluxes=boundary_fluxes,
            face_xi=face_xi,
            face_coordinates=face_coordinates,
            reconstructed_face_fluxes=face_fluxes,
        ).items():
            arrays[f"{key}_{name}"] = value

    # Save the numerical arrays and case metadata.
    np.savez_compressed(OUTPUT / "solutions.npz", **arrays)
    metadata = dict(
        problem=dict(a=PROBLEM.a, kappa=PROBLEM.kappa, boundary_values=PROBLEM.boundary_values),
        cases=cases,
    )
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Plot the solutions
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey="row", layout="constrained")
    exact_x = np.linspace(0., 1., 1001)
    profiles = np.linspace(-1., 1., 51)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ELEMENT_COUNTS)))
    for degree in (1, 2):
        for column, method in enumerate(METHODS):
            axis = axes[degree-1, column]
            axis.plot(exact_x, PROBLEM.exact_solution(exact_x), color="black", lw=1.4, label="Exact")
            for index, (N, color) in enumerate(zip(ELEMENT_COUNTS, colors)):
                result = solutions[degree, method, N]
                x, u, _, _, _ = result.evaluate(np.arange(N), profiles)  # (N, len(profiles)).
                zorder = 2 + len(ELEMENT_COUNTS) - index  # Draw smaller N on top.
                axis.plot(x.ravel(), u.ravel(), color=color,
                          lw=1.2, label=f"N = {N}", zorder=zorder)
                axis.plot(result.mesh.nodes, result.coefficients, "o", color=color,
                          ms=2.5, zorder=zorder)
            axis.set_title(f"{'Linear' if degree == 1 else 'Quadratic'} / {method.upper() if method != 'galerkin' else 'Galerkin'}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
            if column == 0:
                axis.set_ylabel("u")
            if degree == 2:
                axis.set_xlabel("x")
    figure.suptitle("Source-free advection-diffusion: a = 1, kappa = 0.05, u(0) = 0, u(1) = 10")
    figure.savefig(OUTPUT / "profiles.png", dpi=180)
    plt.close(figure)

    print(f"Saved {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
