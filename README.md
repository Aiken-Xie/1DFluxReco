# 1DFluxReco

Galerkin, exact artificial diffusion (EAD), and full SUPG for

$$
a u_{,x}-\kappa u_{,xx}=0,\qquad 0<x<1,
\qquad u(0)=g_L,\quad u(1)=g_R.
$$

The first implementation supports positive constant `a` and `kappa`, uniform meshes,
and continuous linear or quadratic Lagrange elements. Defaults are `a=1`,
`kappa=0.05`, and endpoint values `(0, 10)`. Flux reconstruction supports both
element degrees and all three methods. There is no forcing.

The mathematical reference is the external project report,
[`../report/project_report.tex`](../report/project_report.tex), including its section
sources and Appendix A. The report and research notes remain outside this Git
repository. The solver runs independently of those files.

## Install and run

Use Python 3.12 or newer for the first command (`python3.12` is an example of the
interpreter name). From this repository:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[scripts]"
.venv/bin/python scripts/solve_and_reco.py
```

The runtime dependency is NumPy. The `scripts` extra adds Matplotlib and tqdm.
Matplotlib uses its default plotting backend.

## Python API

```python
import numpy as np

from fem1d import FEMSolver, Problem

problem = Problem(a=1.0, kappa=0.05, boundary_values=(0.0, 10.0))
solver = FEMSolver(problem, N=16, degree=2, method="supg")
coefficients = solver.solve()
x, u, du_dx, d2u_dx2, J_phys = solver.evaluate(
    element=0, xi=np.array([-0.5, 0.0, 0.5])
)

print(coefficients)
print(u, du_dx, J_phys)
```

`N` is the number of elements; `degree` is `1` or `2`;
`method` is `"galerkin"`, `"ead"`, or `"supg"`.
Endpoint values can be any finite pair, including equal values. Element indices
in Python start at zero, while the report uses mathematical element numbering.

`evaluate` accepts a one-dimensional NumPy array of reference coordinates
in `[-1, 1]`. It returns the tuple `(x, u, du_dx, d2u_dx2, J_phys)`, with each
array shaped `(len(xi),)` for one element index. Passing a NumPy array of element
indices evaluates all of them at once, returning arrays shaped
`(number_of_elements, len(xi))`. For a single point, pass an array such as
`np.array([0.0])`. Evaluation is local
to the specified element: derivatives and fluxes are not averaged at shared
endpoints. `J_phys = a*u - kappa*du_dx` always uses the physical diffusivity.

Error integration is handled by `scripts/convergence.py`. The exact solution,
available through `problem.exact_solution(x)`, uses shifted exponentials
and `expm1` to handle small and large Péclet
numbers without overflowing exponentials.

## Assembly and the reconstruction interface

The solver assembles the report's **conservative** element form directly, with
test indices in rows and trial indices in columns:

$$
k_{G,ij}^e=\int_{\Omega^e}
\left[-aN_{i,x}N_j+\kappa N_{i,x}N_{j,x}\right]dx.
$$

EAD adds `kappa_bar * N_i,x * N_j,x`. SUPG adds
`a*tau * N_i,x * (a*N_j,x - kappa*N_j,xx)`. Three-point Gauss quadrature evaluates
these polynomial integrals exactly up to floating-point arithmetic.

The stabilization uses the full original element length `h` for both degrees:

$$
\alpha=\frac{ah}{2\kappa},\qquad
\tau=\frac{h}{2a}\left(\coth\alpha-\frac1\alpha\right),\qquad
\bar\kappa=a^2\tau.
$$

Linear EAD and SUPG are equivalent with this matched choice. Quadratic SUPG retains its full
second-derivative term; the chosen parameter is the project's comparison formula,
not a claim of quadratic optimality.

Call `solver.solve()` before evaluating the field or computing errors. The method
returns the nodal coefficients and stores the following data on `FEMSolver` for
later reconstruction:

| Data | Meaning |
| --- | --- |
| `problem`, `method`, `mesh` | Parameters and mesh: `mesh.N` is the element count, `mesh.nodes` are all solution nodes, `mesh.elements[e]` lists the global node indices for element `e`, and `mesh.h` is the full element length. Element endpoints are calculated as `e*h` and `(e+1)*h`. |
| `coefficients` | Solved global nodal values, including prescribed endpoints |
| `element_matrices` | Original element matrices of shape `(N, degree+1, degree+1)` |
| `tau_e`, `kappa_bar_e` | Active per-element stabilization coefficients: EAD uses only `kappa_bar_e`, SUPG only `tau_e`, Galerkin neither |
| `stiffness_matrix`, `free_dofs` | Original dense global NumPy matrix and free-node indices |
| `residual` | Full assembled `stiffness_matrix @ coefficients`, retaining boundary reactions |
| `element_residuals` | Elementwise `k_e @ coefficients[mesh.elements[e]]`, evaluated after the solve |

Assembly adds each element matrix into a dense global matrix using
`stiffness_matrix[np.ix_(nodes, nodes)] += k`. Dirichlet elimination solves the reduced
system with `np.linalg.solve()` and preserves all original
assembly data. With no forcing, no load vectors are stored or assembled; the
reduced right-hand side is `-K_free,boundary @ d_boundary`.

Recovery uses the retained conservative residuals computed from the original
element matrices and solved nodal coefficients.

## Flux reconstruction

```python
from fem1d import FluxReconstruction

reconstruction = FluxReconstruction(
    solution=solver.coefficients, mesh=solver.mesh, problem=problem, method="supg"
)
postprocessed_coefficients = reconstruction.reconstruct()
J_phys_hat = reconstruction.evaluate(
    element=0, xi=np.array([-0.5, 0.0, 0.5])
)
```

`FluxReconstruction` implements the closed-form coefficient differences in report
Sections 4 and 5. Its inputs are the solved nodal-value array, mesh, problem, and
method, with no dependency on a solver object. `reconstruct()` computes the element
residuals as `k_e @ solution[mesh.elements[e]]` using that method's stiffness matrix,
then determines the postprocessed coefficients without another matrix solve.
The reconstructed flux is `a*u - kappa*du_post_dx`, with `u` from the unchanged
original FEM solution.

| Data | Meaning and shape |
| --- | --- |
| `solution` | Original global FEM coefficients, `(N*degree+1,)` |
| `mesh`, `problem`, `method` | The mesh and problem objects, and the formulation name |
| `coefficients` | Initially `None`; after `reconstruct()`, element-local postprocessed coefficients of shape `(N, degree+1)`, with the first coefficient of each element zero to fix the additive constant |

`reconstruct()` does not evaluate or cache fluxes. `evaluate(element, xi)` returns
only the reconstructed physical flux array, shaped `(len(xi),)` for one element
or `(number_of_elements, len(xi))` for a NumPy array of element indices.
The example evaluates control-volume faces separately when saving its results.

The conservation conditions apply to the interior nodal control volumes bounded
by midpoint faces for linear elements or quarter-point faces for quadratic
elements. The numerical endpoint data are distinct from pointwise traces
of the recovered polynomial at original element endpoints. Boundary control
volumes remain outside the current recovery scope.

## Solve and reconstruct

`scripts/solve_and_reco.py` runs all six method/degree combinations with `N=4,16`,
giving 12 cases in one loop, with a progress bar showing degree, method, and
element count. After all solves, a second loop reconstructs each case's flux.
All numerical results are saved before plotting begins.
Edit its Python settings to explore other parameters. It writes
to `outputs/baseline/`, regardless of the current working directory:

- `solutions.npz`: nodes, coefficients, `elements` (the node-index array for each element), original element matrices,
  element residuals, and stabilization arrays, plus `postprocessed_coefficients`,
  `boundary_fluxes`, `face_xi`, `face_coordinates`, and `reconstructed_face_fluxes`.
  Keys use prefixes such as
  `p2_supg_n16_`; all arrays load with `np.load(..., allow_pickle=False)`.
- `metadata.json`: shared problem settings (`a`, `kappa`, boundary values)
  and a `cases` list with each case's method, degree, element count `N`, number
  of degrees of freedom, mesh spacing, cell Péclet number, and stabilization
  coefficients. Each case's `id` is the prefix of its arrays in the NPZ file.
- `profiles.png`: linear and quadratic solution profiles versus the exact field.

Numerical arrays live in the NPZ file; problem and case settings live in JSON.
There is no separate CSV summary.

Generated outputs and the local environment are ignored by Git. Rerunning the
example replaces these files.

## Errors from saved solutions

After the baseline run:

```bash
.venv/bin/python scripts/convergence.py
```

This script loads `solutions.npz` and `metadata.json` without running the solver
or reconstruction again. It compares the saved fields against the exact continuum
solution and its constant physical flux, using 20-point quadrature.
The saved coefficients are loaded into `FEMSolver` and `FluxReconstruction`, and
their `evaluate()` methods supply the fields and physical fluxes. The integrals
include the element Jacobian `h/2`.

It keeps four error arrays in memory, each of shape `(number_of_cases,)`,
and prints the results.
It does not save postprocessing NPZ files:

| Array | Quantity compared with the exact continuum solution or physical flux |
| --- | --- |
| `solution_L2` | Original FEM field `u` |
| `direct_flux_L2` | Direct physical flux `a*u - kappa*du_dx` |
| `effective_flux_L2` | Comparison flux `a*u - (kappa+kappa_bar+a**2*tau)*du_dx` |
| `reconstructed_flux_L2` | Reconstructed physical flux `a*u - kappa*du_post_dx` |

Stabilization changes the FEM coefficients; it is not added to the diffusivity
when evaluating the direct physical flux. The effective-diffusivity comparison
includes the artificial diffusion. For quadratic SUPG, it does not include the
full second-derivative stabilization contribution, and is not identified with
the recovered physical flux. All flux errors use the physical continuum flux
as their reference. The cases follow the order in `metadata.json`.

## Conservation from saved solutions

```bash
.venv/bin/python scripts/conservation.py
```

This script loads the saved original and postprocessed coefficients and evaluates
the direct and reconstructed physical fluxes at control-volume faces. It uses
physical `kappa` in both definitions; it does not reuse saved residuals or fluxes.
It restores the saved coefficients in the FEM and reconstruction objects and
calls their `evaluate()` methods instead of duplicating shape-function formulas.
Faces lie at reference `xi=0` for linear elements and `xi=-0.5, 0.5` for quadratic
elements. Ordering these faces from left to right gives the left and right
boundaries of every interior nodal volume, including both quadratic node types.

For each interior node it computes `J_right - J_left`, which should be zero for
our source-free problem. It keeps the signed defects and node positions in memory
and prints the maximum absolute defect for each case, both unscaled and divided
by `a*abs(u_right-u_left)`. It also prints the maximum face-flux error
`max(abs(J_face - J_exact))` for both physical fluxes, using each unique
control-volume face once. Face errors measure accuracy; volume defects measure
conservation. The normalization scale uses our unequal prescribed boundary values.
Boundary-node volumes are excluded. The script does not run a solve or recovery,
and does not save any files.

## Research workflow and layout

Use research scripts to explore the results and check correctness independently:
compare with the exact solution, evaluate the matrices derived in the report,
inspect residuals, and examine mesh refinement. The baseline script saves
the solution and element residual arrays for this purpose. There is no automated test suite or
dependency on saved homework results.

```text
src/fem1d/
  problem.py    # Parameters and exact reference field/flux
  mesh.py       # Uniform geometry and node indices for each element
  assembly.py   # Reference basis, quadrature, element forms, and dense assembly
  solver.py     # FEMSolver with solve(), retained results, and element evaluation
  flux_reco.py  # FluxReconstruction for linear and quadratic elements
scripts/       # Reproducible research scripts
outputs/        # Generated results, ignored by Git
```

The next step is the refinement study, convergence rates, and diagnostic plots.
