import numpy as np

from src.lab1_svenn import svenn
from src.lab2_search import golden_section
from src.oracle import Oracle

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])


def line_minimize(oracle, x, direction, delta=0.1, eps=1e-4, alpha_max=1000.0, stats=None):
    """Мінімум уздовж прямої: Свенн дає відрізок, золотий переріз уточнює."""
    line = Oracle(lambda a: oracle.value(np.clip(x + a * direction, LO, HI)))

    bracket = svenn(line, 0.0, delta, -alpha_max, alpha_max)
    result = golden_section(line, bracket["a"], bracket["b"], eps)

    alpha = result["x"]
    x_new = np.clip(x + alpha * direction, LO, HI)
    f_new = oracle.value(x_new)                   # перевірка нової точки (1 звернення)
    if stats is not None:                         # розклад вартості (дослід 2)
        stats.append({"svenn": bracket["n_f"],
                      "golden": result["n_f"] - bracket["n_f"],
                      "check": 1})
    return x_new, f_new, alpha


def powell(oracle, x0, tol=1e-4, max_cycles=50, stats=None):
    """Метод спряжених напрямків (Пауелл)."""
    x = np.asarray(x0, dtype=float)
    f_x = oracle.value(x)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    rows = []

    for cycle in range(1, max_cycles + 1):
        x_start = x.copy()
        f_start = f_x
        gains, alphas = [], []

        for i in range(2):                       # мінімум уздовж кожного напрямку
            x_new, f_new, alpha = line_minimize(oracle, x, directions[i], stats=stats)
            gains.append(f_x - f_new)
            alphas.append(alpha)
            x, f_x = x_new, f_new

        shift = x - x_start
        norm = float(np.linalg.norm(shift))
        rows.append({"cycle": cycle, "x": x.copy(), "f": f_x, "shift": norm,
                     "alpha1": alphas[0], "alpha2": alphas[1], "n_f": oracle.n_f})

        if norm < tol or abs(f_start - f_x) <= 1e-12 * (1.0 + abs(f_x)):
            break

        x, f_x, _ = line_minimize(oracle, x, shift / norm, stats=stats)  # новий напрямок
        directions[int(np.argmax(gains))] = shift / norm                 # заміна напрямку
        if cycle % 3 == 0:                                               # перезапуск
            directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    return {"x": x, "f": f_x, "cycles": cycle, "n_f": oracle.n_f, "rows": rows}