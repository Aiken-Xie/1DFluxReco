# One-dimensional FEM solver

Galerkin, exact artificial diffusion (EAD), and full SUPG for

$$
a u_{,x}-\kappa u_{,xx}=0,\qquad 0<x<1,
\qquad u(0)=g_L,\quad u(1)=g_R.
$$

The first implementation supports positive constant `a` and `kappa`, uniform meshes,
and continuous linear or quadratic Lagrange elements. Defaults are `a=1`,
`kappa=0.05`, and endpoint values `(0, 10)`. There is no forcing or reconstruction
in this version.

The mathematical reference is the external project report,
[`../report/project_report.tex`](../report/project_report.tex), including its section
sources and Appendix A. The report and research notes remain outside this Git
repository. The solver and tests run independently of those files.

## Install and run

Use Python 3.12 or newer for the first command (`python3.12` is an example of the
interpreter name). From this repository:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,examples]"
.venv/bin/python -m pytest
.venv/bin/python examples/baseline.py
```

Runtime dependencies are NumPy and SciPy. The `dev` extra adds pytest, and the
`examples` extra adds Matplotlib. The example uses a noninteractive plotting
backend and needs no display server.

## Python API

```python
from fem1d import Problem, solve

problem = Problem(a=1.0, kappa=0.05, boundary_values=(0.0, 10.0))
solution = solve(problem, elements=16, degree=2, method="supg")
values = solution.evaluate(element=0, xi=[-0.5, 0.0, 0.5])

print(solution.coefficients)
print(solution.scaled_residual_inf)
print(solution.l2_error())
print(values.u, values.du_dx, values.J_phys)
```

`degree` is `1` or `2`; `method` is `"galerkin"`, `"ead"`, or `"supg"`.
Endpoint values can be any finite pair, including equal values. Element indices
in Python start at zero, while the report uses mathematical element numbering.

`evaluate` accepts a scalar or one-dimensional array of reference coordinates
in `[-1, 1]`. It returns an `Evaluation` with one-dimensional arrays `x`, `u`,
`du_dx`, `d2u_dx2`, and `J_phys`, including for a single point. Evaluation is local
to the specified element: derivatives and fluxes are not averaged at shared
endpoints. `J_phys = a*u - kappa*du_dx` always uses the physical diffusivity.

`l2_error(quadrature_order=20)` integrates the field error against the exact
solution. Increasing the order provides a quadrature-resolution check. The exact
solution uses shifted exponentials and `expm1` to handle small and large Péclet
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

A series expansion evaluates the fitting function near zero. Linear EAD and
SUPG are equivalent with this matched choice. Quadratic SUPG retains its full
second-derivative term; the chosen parameter is the project's comparison formula,
not a claim of quadratic optimality.

`FEMSolution` retains the following data for later reconstruction:

| Data | Meaning |
| --- | --- |
| `problem`, `method`, `mesh` | Parameters, original element vertices, solution nodes, connectivity, and full element length |
| `coefficients` | Solved global nodal values, including prescribed endpoints |
| `element_matrices`, `element_loads` | Original arrays of shape `(elements, degree+1, degree+1)` and `(elements, degree+1)` |
| `tau_e`, `kappa_bar_e` | Active per-element stabilization coefficients: EAD uses only `kappa_bar_e`, SUPG only `tau_e`, Galerkin neither |
| `matrix`, `load`, `free_dofs` | Original sparse global matrix, original load, and free-node indices |
| `residual` | Full assembled `matrix @ coefficients - load`, retaining boundary reactions |
| `element_residuals` | Elementwise `k_e @ coefficients[connectivity_e] - f_e`, evaluated after the solve |

Dirichlet elimination solves a separate reduced system and preserves all original
assembly data. The reduced right-hand side includes the endpoint contributions
even though the original distributed load is zero. `residual_inf` measures only
free equations and is zero for a single linear element with no free nodes.
`scaled_residual_inf` divides that value by
`max(1, ||K||_inf * ||d||_inf + ||F||_inf)`.

The homework used an advective element form. Its assembled free equations agree
with this conservative form for the present constant-coefficient problem, but
the element residuals differ. Future recovery must use the retained conservative
residuals, not residuals from boundary-modified matrices.

## Baseline example

`examples/baseline.py` runs all six method/degree combinations with `N=4,16`,
giving 12 cases. Edit its Python settings to explore other parameters. It writes
to `outputs/baseline/`, regardless of the current working directory:

- `summary.csv`: parameters, degrees of freedom, active stabilization coefficients,
  field and nodal errors, free residuals, and solution extrema.
- `solutions.npz`: nodes, coefficients, connectivity, original element matrices,
  element residuals, and stabilization arrays. Keys use prefixes such as
  `p2_supg_n16_`; all arrays load with `np.load(..., allow_pickle=False)`.
- `profiles.png`: linear and quadratic solution profiles versus the exact field.
- `metadata.json`: run settings, quadrature choices, and dependency versions.

Generated outputs and the local environment are ignored by Git. Rerunning the
example replaces these files; the timestamp in the metadata records each run.

## Verification and layout

Tests compare quadrature assembly with independent analytic element matrices,
check basis interpolation and derivatives, verify boundary and residual identities,
exercise single-element and constant-solution cases, and reproduce 12 saved
homework cases. The homework fixture is committed under `tests/data/`; tests
never import or execute the original homework code.

```text
src/fem1d/
  problem.py    # Parameters and exact reference field/flux
  mesh.py       # Uniform geometry and solution-node connectivity
  basis.py      # Reference basis and Gauss quadrature
  assembly.py   # Conservative element forms and sparse assembly
  solver.py     # Reduced solve, retained result, and element evaluation
examples/       # Reproducible research scripts
tests/          # Independent algebra and numerical regressions
outputs/        # Generated results, ignored by Git
```

The next milestone is a separate `recovery.py` that consumes `FEMSolution`.
Flux-error diagnostics and the full convergence campaign follow afterward.
