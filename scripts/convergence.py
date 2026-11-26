"""Compute L2 errors from saved FEM and postprocessed coefficients, without solving again."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
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

    methods = ("galerkin", "ead", "supg")
    labels = ("Galerkin", "EAD", "SUPG")
    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "x")
    rates = {}
    figures = INPUT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12})
    solution_figure, solution_axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    solution_figure.suptitle("Convergence of Linear vs Quadratic Elements")
    flux_figure, flux_axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    print("\nObserved rates, N=64 to 128 (absolute L2 errors)")
    print(f"{'Degree / method':<22} {'u':>10} {'J_phys':>10} {'J_hat':>10}")
    for row, degree in enumerate((1, 2)):
        finest_errors = {}
        for col, (method, label, color, marker) in enumerate(zip(methods, labels, colors, markers)):
            indices = [i for i, case in enumerate(cases)
                       if case["degree"] == degree and case["method"] == method]
            counts = np.array([cases[i]["N"] for i in indices])
            h = np.array([cases[i]["h"] for i in indices])
            rates[degree, method] = {
                name: np.log(errors[name][indices[:-1]] / errors[name][indices[1:]])
                / np.log(h[:-1] / h[1:]) for name in error_names
            }
            last_rates = rates[degree, method]
            print(f"{str(degree) + ' / ' + label:<22} "
                  f"{last_rates['solution_L2'][-1]:10.3f} "
                  f"{last_rates['direct_flux_L2'][-1]:10.3f} "
                  f"{last_rates['reconstructed_flux_L2'][-1]:10.3f}")
            solution_axes[row].loglog(counts, errors["solution_L2"][indices],
                                      color=color, marker=marker, fillstyle="none", label=label)
            finest_errors[method] = errors["solution_L2"][indices[-1]]
            ax = flux_axes[row, col]
            direct_errors = errors["direct_flux_L2"][indices]
            reconstructed_errors = errors["reconstructed_flux_L2"][indices]
            ax.loglog(counts, direct_errors, "o-",
                      color="#0072B2", fillstyle="none", label="Direct physical")
            ax.loglog(counts, reconstructed_errors, "x--",
                      color="#D55E00", label="Reconstructed physical")
            reference_anchor = 0.35 * min(direct_errors[-1], reconstructed_errors[-1])
            ax.loglog(counts, reference_anchor * (counts[-1] / counts)**degree,
                      color="0.35", linestyle="-." if degree == 1 else ":",
                      label="First-order reference" if degree == 1 else "Second-order reference")
            ax.set(title=f"{'Linear' if degree == 1 else 'Quadratic'} / {label}",
                   xlabel="Number of elements N", ylabel="Flux L2 error")
            ax.set_xticks(counts, labels=counts)
            ax.grid(True, which="major", alpha=0.2)
        references = ((2, "galerkin", ":"),) if degree == 1 else (
            (2, "ead", ":"),
            (3, "galerkin", "--"),
        )
        for order, method, linestyle in references:
            # Offset the guide below the corresponding error at the finest mesh.
            reference_errors = 0.35 * finest_errors[method] * (counts[-1] / counts)**order
            solution_axes[row].loglog(
                counts, reference_errors, color="0.35", linestyle=linestyle,
                label="Second-order reference" if order == 2 else "Third-order reference",
            )
        solution_axes[row].set(title="Linear elements" if degree == 1 else "Quadratic elements",
                               xlabel="Number of elements N", ylabel="Solution L2 error")
        solution_axes[row].set_xticks(counts, labels=counts)
        solution_axes[row].grid(True, which="major", alpha=0.2)
    handles, labels = solution_axes[1].get_legend_handles_labels()
    solution_figure.legend(handles, labels, fontsize=11, ncols=5, loc="outside lower center")
    handles, labels = flux_axes[0, 0].get_legend_handles_labels()
    quadratic_handles, quadratic_labels = flux_axes[1, 0].get_legend_handles_labels()
    flux_figure.legend(handles + quadratic_handles[-1:], labels + quadratic_labels[-1:],
                       fontsize=11, ncols=4, loc="outside lower center")
    solution_figure.savefig(figures / "solution_convergence.png", dpi=220)
    flux_figure.savefig(figures / "flux_convergence.png", dpi=220)
    plt.close(solution_figure)
    plt.close(flux_figure)
    return cases, errors, rates


if __name__ == "__main__":
    main()
