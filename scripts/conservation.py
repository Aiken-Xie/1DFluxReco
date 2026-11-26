"""Evaluate interior nodal control-volume balances from saved FEM and recovered coefficients."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    # Summary arrays: (number of cases,). Per-case defects: (N*degree-1,).
    max_defects = {name: np.zeros(len(cases)) for name in flux_names}
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

    print(
        f"{'Case':<22} {'Direct max':>13} {'Recovered max':>14} "
        f"{'Direct face error':>18} {'Recovered face error':>21}"
    )
    for i, case in enumerate(cases):
        print(
            f"{case['id']:<22} {max_defects['direct'][i]:13.5e} "
            f"{max_defects['reconstructed'][i]:14.5e} "
            f"{max_face_errors['direct'][i]:18.5e} "
            f"{max_face_errors['reconstructed'][i]:21.5e}"
        )

    figures = INPUT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12})
    # Separate panels keep the roundoff plateau visible for each element degree.
    for values, filename, ylabel in (
        (max_defects, "conservation_refinement.png", r"$E_{\mathrm{conservation}}$"),
        (max_face_errors, "face_flux_refinement.png", r"$E_{\mathrm{face}}$"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        for ax, degree in zip(axes, (1, 2)):
            for method, label, color in zip(
                ("galerkin", "ead", "supg"), ("Galerkin", "EAD", "SUPG"),
                ("#0072B2", "#D55E00", "#009E73"),
            ):
                indices = [i for i, case in enumerate(cases)
                           if case["degree"] == degree and case["method"] == method]
                counts = np.array([cases[i]["N"] for i in indices])
                direct_style = "s-" if values is max_face_errors and method == "ead" else "o-"
                ax.loglog(counts, values["direct"][indices], direct_style, color=color,
                          fillstyle="none", label=f"{label}: direct")
                ax.loglog(counts, values["reconstructed"][indices], "x--", color=color,
                          label=f"{label}: reconstructed")
                if values is max_face_errors and method == "supg":
                    second_order_anchor = 0.25 * values["direct"][indices[-1]]
                    ax.loglog(counts, second_order_anchor * (counts[-1] / counts)**2,
                              color="0.35", linestyle=":", label="Second-order reference")
                    if degree == 2:
                        fourth_order_anchor = 0.35 * values["reconstructed"][indices[-2]]
                        ax.loglog(counts, fourth_order_anchor * (counts[-2] / counts)**4,
                                  color="0.35", linestyle="-.", label="Fourth-order reference")
            ax.set(title="Linear elements" if degree == 1 else "Quadratic elements",
                   xlabel="Number of elements N", ylabel=ylabel)
            ax.set_xticks(counts, labels=counts)
            ax.grid(True, which="major", alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        if values is max_face_errors:
            quadratic_handles, quadratic_labels = axes[1].get_legend_handles_labels()
            handles.append(quadratic_handles[-1])
            labels.append(quadratic_labels[-1])
            figure.legend(handles, labels, fontsize=10, ncols=4, loc="outside lower center")
        else:
            figure.legend(handles, labels, fontsize=12, ncols=3, loc="outside lower center")
        figure.savefig(figures / filename, dpi=220)
        plt.close(figure)
    return cases, max_defects, max_face_errors


if __name__ == "__main__":
    main()
