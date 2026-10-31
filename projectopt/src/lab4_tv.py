import numpy as np

from src.operators import grad_xy, divergence


def tv_energy(I, noisy, lam_tv, eps):
    """E(I) = ½‖I − I_шумне‖² + λ_TV Σ sqrt(|∇I|² + ε²)."""
    gx, gy = grad_xy(I)
    return 0.5 * float(np.sum((I - noisy) ** 2)) + \
           lam_tv * float(np.sum(np.sqrt(gx * gx + gy * gy + eps ** 2)))


def tv_grad(I, noisy, lam_tv, eps):
    """∇E = (I − I_шумне) − λ_TV · div( ∇I / sqrt(|∇I|² + ε²) )."""
    gx, gy = grad_xy(I)
    d = np.sqrt(gx * gx + gy * gy + eps ** 2)
    return (I - noisy) - lam_tv * divergence(gx / d, gy / d)


def tv_nesterov(noisy, lam_tv, eps, max_iter=1000, tol=1e-6):
    """Нестеров (сильно опуклий варіант): μ = 1, L = 1 + 8 λ_TV / ε."""
    L = 1.0 + 8.0 * lam_tv / eps
    beta = (np.sqrt(L) - 1.0) / (np.sqrt(L) + 1.0)
    h = 1.0 / L
    I = noisy.copy()
    I_prev = I.copy()
    k = 0
    for k in range(1, max_iter + 1):
        Y = I + beta * (I - I_prev)
        g = tv_grad(Y, noisy, lam_tv, eps)
        g_norm = float(np.linalg.norm(g))
        I_prev, I = I, Y - h * g
        if g_norm <= tol:
            break
    return {"I": I, "iterations": k, "L": L, "beta": beta,
            "E": tv_energy(I, noisy, lam_tv, eps), "grad": g_norm}