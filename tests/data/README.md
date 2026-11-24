# Homework regression fixture

`homework_baseline.csv` contains the 12 unmodified rows with element counts 4
and 16 from the pre-existing homework results, covering both degrees and all
three methods. The original file is `../../HW3/results/advection_results.csv`
relative to this repository's root, produced by the homework `code/q3.py`.

Parameters are `a=1`, `kappa=0.05`, endpoint values `(0, 10)`, and zero forcing.
`cell_Pe` is `a*h/kappa` using the full original element length. `L2` used
20-point elementwise Gauss quadrature. The homework's element advection form
differs from the new solver's conservative form, but their free equations and
solutions agree for this problem.

These saved values are regression expectations, not regenerated test output.
Tests compare the field and nodal errors with an absolute floor of 1e-12;
roundoff-scale nodal errors and residuals are not expected to be bitwise equal.
