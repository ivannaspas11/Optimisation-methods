import math

TAU = (math.sqrt(5.0) - 1.0) / 2.0        # 0.6180339887...


def dichotomy(oracle, a, b, eps=1e-5, delta=None, max_iter=500, strict=True):
    """Дихотомія: 2 звернення на ітерацію.

    strict=False потрібен лише для досліду 2 (delta >= eps).
    """
    if delta is None:
        delta = eps / 3.0
    if strict and not 0.0 < delta < eps:
        raise ValueError("має виконуватися 0 < delta < eps")

    rows = []
    k = 0
    while (b - a) > eps and k < max_iter:
        x1 = (a + b - delta) / 2.0
        x2 = (a + b + delta) / 2.0
        f1 = oracle.value(x1)                      # звернення 1
        f2 = oracle.value(x2)                      # звернення 2

        if f1 <= f2:
            decision = "f(x1) <= f(x2)  →  b = x2"
            b = x2
        else:
            decision = "f(x1) >  f(x2)  →  a = x1"
            a = x1

        k += 1
        rows.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                     "f1": f1, "f2": f2, "L": b - a, "n_f": oracle.n_f,
                     "decision": decision})

    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "a": a, "b": b,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}


def golden_section(oracle, a, b, eps=1e-5, max_iter=500):
    """Золотий переріз: 1 звернення на ітерацію (друга точка успадковується)."""
    x1 = b - TAU * (b - a)
    x2 = a + TAU * (b - a)
    f1 = oracle.value(x1)                          # два звернення лише
    f2 = oracle.value(x2)                          # на самому початку

    rows = []
    k = 0
    while (b - a) > eps and k < max_iter:
        if f1 <= f2:
            decision = "f(x1) <= f(x2)  →  b = x2, успадкували x1"
            b = x2
            x2, f2 = x1, f1                        # ← фокус: без обчислення
            x1 = b - TAU * (b - a)
            f1 = oracle.value(x1)                  # єдине нове звернення
            new_point = x1
        else:
            decision = "f(x1) >  f(x2)  →  a = x1, успадкували x2"
            a = x1
            x1, f1 = x2, f2                        # ← фокус: без обчислення
            x2 = a + TAU * (b - a)
            f2 = oracle.value(x2)                  # єдине нове звернення
            new_point = x2

        k += 1
        rows.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                     "f1": f1, "f2": f2, "L": b - a, "n_f": oracle.n_f,
                     "new": new_point, "decision": decision})

    x_best = (a + b) / 2.0
    return {"x": x_best, "f": oracle.value(x_best), "a": a, "b": b,
            "iterations": k, "n_f": oracle.n_f, "rows": rows}