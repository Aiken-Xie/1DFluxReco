"""The constant-coefficient boundary-driven problem on [0, 1]."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Problem:
    """Solve a*u_x - kappa*u_xx = 0 with two prescribed endpoint values."""

    a: float = 1.0
    kappa: float = 0.05
    boundary_values: tuple[float, float] = (0.0, 10.0)

    def __post_init__(self) -> None:
        a, kappa = float(self.a), float(self.kappa)
        if not np.isfinite(a) or a <= 0:
            raise ValueError("a must be finite and positive")
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        values = tuple(float(value) for value in self.boundary_values)
        if len(values) != 2 or not np.all(np.isfinite(values)):
            raise ValueError("boundary_values must contain two finite values")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "boundary_values", values)

    def exact(self, x: ArrayLike) -> NDArray[np.float64]:
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

    def exact_flux(self) -> float:
        """Return the constant exact physical flux a*u - kappa*u_x."""
        lam = self.a / self.kappa
        left, right = self.boundary_values
        return float(
            self.a * left
            - self.a * (right - left) * np.exp(-lam) / (-np.expm1(-lam))
        )
