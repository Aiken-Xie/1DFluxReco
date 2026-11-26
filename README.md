# 1DFluxReco

This repository contains the code for the final project in MECEE6106, *Finite Element Method for Fluid Flow and Fluid-Structure Interactions*. The project studies conservative flux reconstruction for a source-free, one-dimensional advection–diffusion problem. It compares Galerkin, exact artificial diffusion (EAD), and streamline upwind Petrov–Galerkin (SUPG) with linear and quadratic elements.

## Repository structure

```text
src/fem1d/              Finite element solver and flux reconstruction
  problem.py            Problem parameters and exact solution and flux
  mesh.py               Uniform mesh and element-to-node mapping
  assembly.py           Shape functions and stiffness matrix assembly
  solver.py             Finite element solve and direct flux evaluation
  flux_reco.py          Conservative flux reconstruction and evaluation
scripts/
  solve_and_reco.py     Solve and reconstruct the 36 refinement cases
  convergence.py        Compute and plot solution and flux L2 errors
  conservation.py       Check control-volume balances and face-flux accuracy
pyproject.toml          Package setup and dependencies
```

## Install and run

Use Python 3.12 or newer. From the repository directory, create a virtual environment, install the package with its script dependencies, and run the scripts in order.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[scripts]"

.venv/bin/python scripts/solve_and_reco.py
.venv/bin/python scripts/convergence.py
.venv/bin/python scripts/conservation.py
```

The first script saves the solution and reconstruction coefficients, case settings, and solution profiles in `outputs/baseline/`. The analysis scripts read the saved coefficients, print the numerical results, and save the refinement plots in `outputs/baseline/figures/`.

## Python API

```python
import numpy as np

from fem1d import FEMSolver, FluxReconstruction, Problem

problem = Problem(a=1.0, kappa=0.05, boundary_values=(0.0, 10.0))
solver = FEMSolver(problem, N=16, degree=2, method="supg")
coefficients = solver.solve()

xi = np.array([-0.5, 0.0, 0.5])
x, u, du_dx, d2u_dx2, J_phys = solver.evaluate(element=0, xi=xi)

reconstruction = FluxReconstruction(
    solution=coefficients,
    mesh=solver.mesh,
    problem=problem,
    method="supg",
)
postprocessed_coefficients = reconstruction.reconstruct()
J_phys_hat = reconstruction.evaluate(element=0, xi=xi)
```
