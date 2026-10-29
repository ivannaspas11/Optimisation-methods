import numpy as np

from src.filters import bilateral_filter
from src.metrics import mse

LO2 = np.array([0.3, 0.01])
HI2 = np.array([5.0, 1.0])


def make_objective2(noisy, clean, pad_mode="reflect"):
    """f(theta) = MSE(bilateral(noisy, s, r), clean) + штраф за вихід за межі."""
    def f(theta):
        point = np.asarray(theta, dtype=float)
        clamped = np.clip(point, LO2, HI2)
        value = mse(bilateral_filter(noisy, clamped[0], clamped[1], pad_mode), clean)
        gap = point - clamped
        return value + 1000.0 * float(gap @ gap)
    return f