"""Run all six method/degree combinations and save results under outputs/baseline.

Run with: python examples/baseline.py (after installing .[examples]).
Settings are ordinary Python values so this example also serves as a template.
"""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from fem1d import Problem, __version__, solve

PROBLEM = Problem(a=1., kappa=.05, boundary_values=(0., 10.))
ELEMENT_COUNTS = (4, 16)
METHODS = ("galerkin", "ead", "supg")
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "baseline"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey="row", layout="constrained")
    exact_x = np.linspace(0., 1., 1001)
    profiles = np.linspace(-1., 1., 51)
    colors = ("#d55e00", "#0072b2")
    rows, arrays = [], {}
    for degree in (1, 2):
        for column, method in enumerate(METHODS):
            axis = axes[degree-1, column]
            axis.plot(exact_x, PROBLEM.exact(exact_x), color="black", lw=1.4, label="Exact")
            for elements, color in zip(ELEMENT_COUNTS, colors):
                result = solve(PROBLEM, elements=elements, degree=degree, method=method)
                if result.scaled_residual_inf > 1e-12:
                    raise RuntimeError(f"Baseline residual check failed: {method}, degree={degree}, N={elements}")
                rows.append(dict(
                    method=method, degree=degree, elements=elements,
                    dofs=result.coefficients.size, a=PROBLEM.a, kappa=PROBLEM.kappa,
                    g_left=PROBLEM.boundary_values[0], g_right=PROBLEM.boundary_values[1],
                    h=result.mesh.h, cell_Pe=PROBLEM.a*result.mesh.h/PROBLEM.kappa,
                    tau=float(result.tau_e[0]), artificial_kappa=float(result.kappa_bar_e[0]),
                    L2=result.l2_error(), max_nodal_error=result.max_nodal_error,
                    residual_inf=result.residual_inf, scaled_residual_inf=result.scaled_residual_inf,
                    min_u=float(result.coefficients.min()), max_u=float(result.coefficients.max()),
                ))
                key = f"p{degree}_{method}_n{elements}"
                for name, value in dict(
                    nodes=result.mesh.nodes, coefficients=result.coefficients,
                    connectivity=result.mesh.connectivity,
                    element_matrices=result.element_matrices,
                    element_residuals=result.element_residuals,
                    tau_e=result.tau_e, kappa_bar_e=result.kappa_bar_e,
                ).items():
                    arrays[f"{key}_{name}"] = value
                evaluations = [result.evaluate(e, profiles) for e in range(elements)]
                axis.plot(np.concatenate([v.x for v in evaluations]),
                          np.concatenate([v.u for v in evaluations]), color=color,
                          lw=1.2, label=f"N = {elements}")
                axis.plot(result.mesh.nodes, result.coefficients, "o", color=color, ms=2.5)
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
    with (OUTPUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUTPUT / "solutions.npz", **arrays)
    metadata = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        problem=asdict(PROBLEM), elements=list(ELEMENT_COUNTS), degrees=[1, 2], methods=list(METHODS),
        quadrature=dict(assembly=3, field_error=20),
        peclet_convention="cell_Pe = a*h/kappa; alpha = cell_Pe/2; h is the full element length",
        versions=dict(python=platform.python_version(), fem1d=__version__, numpy=np.__version__,
                      scipy=scipy.__version__, matplotlib=matplotlib.__version__),
    )
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(rows)} cases to {OUTPUT}")
    print(f"Maximum scaled free residual: {max(row['scaled_residual_inf'] for row in rows):.3e}")


if __name__ == "__main__":
    main()
