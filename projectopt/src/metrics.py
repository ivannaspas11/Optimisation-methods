import math
import numpy as np


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    value = mse(a, b)
    if value <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / value)