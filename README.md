# 1DFluxReco

## Install and run

From the repository directory, using Python 3.12 or newer:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[scripts]"

.venv/bin/python scripts/solve_and_reco.py
.venv/bin/python scripts/convergence.py
.venv/bin/python scripts/conservation.py
```

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
