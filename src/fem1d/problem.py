"""The constant-coefficient boundary-driven problem on [0, 1]."""

import numpy as np


class Problem:
    """Solve a*u_x - kappa*u_xx = 0 with two prescribed endpoint values."""

    def __init__(self, a=1.0, kappa=0.05, boundary_values=(0.0, 10.0)):
        a, kappa = float(a), float(kappa)
        if not np.isfinite(a) or a <= 0:
            raise ValueError("a must be finite and positive")
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        values = tuple(float(value) for value in boundary_values)
        if len(values) != 2 or not np.all(np.isfinite(values)):
            raise ValueError("boundary_values must contain two finite values")
        self.a = a
        self.kappa = kappa
        self.boundary_values = values

    def exact(self, x):
        """Evaluate the exact field without subtracting nearly equal exponentials.

        All exponential arguments are nonpositive, including at large a/kappa.
        """
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("x must be finite and within [0, 1]")
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
