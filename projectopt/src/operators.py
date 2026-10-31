import numpy as np


def grad_xy(u):
    """Прямі різниці: праворуч і вниз. Повертає дві таблиці."""
    gx = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]      # останній стовпець лишається нулем
    gy = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]      # останній рядок лишається нулем
    return gx, gy


def divergence(px, py):
    """Оператор, спряжений до градієнта зі знаком мінус."""
    d = np.zeros_like(px)
    d[:, 0] += px[:, 0]
    d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    d[:, -1] -= px[:, -2]
    d[0, :] += py[0, :]
    d[1:-1, :] += py[1:-1, :] - py[:-2, :]
    d[-1, :] -= py[-2, :]
    return d


def laplacian(u):
    """Лапласіан: сума чотирьох сусідів мінус учетверо взятий центр."""
    return divergence(*grad_xy(u))


def energy(I, noisy, lam):
    """E(I) = ½‖I − I_шумне‖² + (λ/2)‖∇I‖²."""
    gx, gy = grad_xy(I)
    return 0.5 * float(np.sum((I - noisy) ** 2)) + \
           0.5 * lam * float(np.sum(gx * gx + gy * gy))


def energy_grad(I, noisy, lam):
    """∇E(I) = (I − I_шумне) − λ·ΔI."""
    return (I - noisy) - lam * laplacian(I)


# ---------- точний розв'язок через косинусне перетворення (DCT-II) ----------

def _dct_matrix(n):
    """Ортонормована матриця DCT-II (те саме, що scipy.fft.dct, norm='ortho')."""
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    C = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * m + 1) * k / (2.0 * n))
    C[0, :] /= np.sqrt(2.0)
    return C


def tikhonov_exact(noisy, lam):
    """Точний мінімум енергії Тихонова: I*_ij = DCT(noisy)_ij / (1 + λ μ_ij)."""
    h, w = noisy.shape
    Ch, Cw = _dct_matrix(h), _dct_matrix(w)
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    mu = (2 - 2 * np.cos(np.pi * i / h)) + (2 - 2 * np.cos(np.pi * j / w))
    spectrum = Ch @ noisy @ Cw.T                    # пряме 2D-DCT
    return Ch.T @ (spectrum / (1 + lam * mu)) @ Cw  # зворотне 2D-DCT