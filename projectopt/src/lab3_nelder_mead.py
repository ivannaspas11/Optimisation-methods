import numpy as np

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])

ALPHA, GAMMA, RHO, SIGMA = 1.0, 2.0, 0.5, 0.5


def nelder_mead(oracle, x0, step, tol=1e-7, max_iter=200):
    """Деформований симплекс. Повертає найкращу точку і таблицю ітерацій."""
    x0 = np.asarray(x0, dtype=float)
    simplex = [x0,
               np.clip(x0 + np.array([step, 0.0]), LO, HI),
               np.clip(x0 + np.array([0.0, 0.4 * step]), LO, HI)]
    values = [oracle.value(p) for p in simplex]
    rows = []

    for k in range(1, max_iter + 1):
        order = np.argsort(values)                       # сортуємо вершини
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        best, middle, worst = simplex
        f_best, f_middle, f_worst = values

        centroid = (best + middle) / 2.0                 # центр без найгіршої
        x_r = np.clip(centroid + ALPHA * (centroid - worst), LO, HI)
        f_r = oracle.value(x_r)

        if f_r < f_best:                                 # 1. розтягування
            x_e = np.clip(centroid + GAMMA * (x_r - centroid), LO, HI)
            f_e = oracle.value(x_e)
            if f_e < f_r:
                simplex[2], values[2], op = x_e, f_e, "розтягування"
            else:
                simplex[2], values[2], op = x_r, f_r, "відображення"
        elif f_r < f_middle:                             # 2. просте прийняття
            simplex[2], values[2], op = x_r, f_r, "відображення"
        else:                                            # 3. стиснення
            if f_r < f_worst:
                x_c = np.clip(centroid + RHO * (x_r - centroid), LO, HI)
                op = "зовнішнє стиснення"
            else:
                x_c = np.clip(centroid + RHO * (worst - centroid), LO, HI)
                op = "внутрішнє стиснення"
            f_c = oracle.value(x_c)
            if f_c < min(f_r, f_worst):
                simplex[2], values[2] = x_c, f_c
            else:                                        # 4. редукція
                for i in (1, 2):
                    simplex[i] = np.clip(best + SIGMA * (simplex[i] - best), LO, HI)
                    values[i] = oracle.value(simplex[i])
                op = "редукція"

        i_best = int(np.argmin(values))
        size = max(float(np.linalg.norm(p - simplex[i_best])) for p in simplex)
        rows.append({"k": k, "op": op, "x": simplex[i_best].copy(),
                     "f": values[i_best], "size": size, "n_f": oracle.n_f})

        if size <= tol and (max(values) - min(values)) <= tol:
            break

    i_best = int(np.argmin(values))
    return {"x": simplex[i_best], "f": values[i_best], "iterations": k,
            "n_f": oracle.n_f, "rows": rows}