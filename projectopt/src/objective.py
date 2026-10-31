from src.filters import gaussian_blur
from src.metrics import mse

LO, HI = 0.05, 8.0


def make_objective(noisy, clean):
    def f(sigma):
        clamped = min(max(sigma, LO), HI)
        value = mse(gaussian_blur(noisy, clamped), clean)
        return value + 1000.0 * (sigma - clamped) ** 2
    return f