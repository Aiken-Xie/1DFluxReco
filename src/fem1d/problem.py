"""The constant-coefficient boundary-driven problem on [0, 1]."""

import numpy as np


class Problem:
    """Solve a*u_x - kappa*u_xx = 0 with two prescribed endpoint values."""

    def __init__(self, a=1.0, kappa=0.05, boundary_values=(0.0, 10.0)):
        self.a = float(a)
        self.kappa = float(kappa)
        self.boundary_values = tuple(float(value) for value in boundary_values)

    def exact_solution(self, x):
        """Evaluate the exact field without subtracting nearly equal exponentials.

        All exponential arguments are nonpositive, including at large a/kappa.
        """
        x = np.asarray(x, dtype=float)
        lam = self.a / self.kappa
        fraction = (
            np.exp(-lam * (1 - x)) * (-np.expm1(-lam * x))
            / (-np.expm1(-lam))
        )
        left, right = self.boundary_values
        return left + (right - left) * fraction

    def exact_flux(self):
        """Return the constant exact physical flux a*u - kappa*u_x."""
        lam = self.a / self.kappa
        left, right = self.boundary_values
        return float(
            self.a * left
            - self.a * (right - left) * np.exp(-lam) / (-np.expm1(-lam))
        )
