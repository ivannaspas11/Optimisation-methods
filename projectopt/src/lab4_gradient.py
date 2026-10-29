import math
import numpy as np

from src.operators import energy, energy_grad


def gradient_descent(noisy, lam, L, kind="gd", beta=None, step=None,
                     max_iter=200, tol=1e-6, x_ref=None):
    """kind: 'gd' — сталий крок; 'armijo' — підбір кроку;
             'hb' — важка кулька; 'nag' — метод Нестерова."""
    h = step if step is not None else 1.0 / L
    I = noisy.copy()
    I_prev = I.copy()
    n_f = n_grad = 0
    trial = h
    rows = []
    converged = False

    for k in range(1, max_iter + 1):
        if kind == "nag":
            b = (k - 1) / (k + 2) if beta is None else beta
            Y = I + b * (I - I_prev)
            g = energy_grad(Y, noisy, lam); n_grad += 1
            g_norm = float(np.linalg.norm(g))
            I_new = Y - h * g
        else:
            g = energy_grad(I, noisy, lam); n_grad += 1
            g_norm = float(np.linalg.norm(g))

            if kind == "armijo":
                s = trial
                f_I = energy(I, noisy, lam); n_f += 1
                g2 = float(np.sum(g * g))
                for _ in range(40):                      # ділимо крок навпіл
                    candidate = I - s * g
                    f_c = energy(candidate, noisy, lam); n_f += 1
                    if f_c <= f_I - 1e-4 * s * g2:
                        break
                    s *= 0.5
                I_new = candidate
                trial = s * 1.5                          # помірне нарощування
                h = s
            elif kind == "hb":
                I_new = I - h * g + beta * (I - I_prev)
            else:
                I_new = I - h * g

        e_k = energy(I, noisy, lam); n_f += 1
        rows.append({"k": k - 1, "E": e_k, "grad": g_norm, "h": h,
                     "n_f": n_f, "n_grad": n_grad,
                     "dist": float(np.linalg.norm(I - x_ref)) if x_ref is not None else None})

        if g_norm <= tol:
            converged = True
            break

        I_prev, I = I, I_new

    return {"I": I, "iterations": len(rows) - 1, "n_f": n_f, "n_grad": n_grad,
            "E": energy(I, noisy, lam), "grad": rows[-1]["grad"],
            "converged": converged, "rows": rows}