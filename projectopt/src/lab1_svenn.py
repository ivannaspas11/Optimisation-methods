def svenn(oracle, x0, delta0, lo=0.05, hi=8.0, max_iter=40):
    rows = []

    def ask(x, k, step, p, comment):
        value = oracle.value(x)
        rows.append({"k": k, "x": x, "f": value, "delta": step,
                     "p": p, "comment": comment})
        return value

    # --- дія 1: три пробні точки ---
    f0 = ask(x0, 0, delta0, 0, "початкова точка x0")
    xr = min(x0 + delta0, hi)
    fr = ask(xr, 0, delta0, 0, "пробний крок праворуч")
    xl = max(x0 - delta0, lo)
    fl = ask(xl, 0, -delta0, 1, "пробний крок ліворуч")

    if fl >= f0 <= fr:
        return {"a": min(xl, xr), "b": max(xl, xr), "p": 0,
                "iterations": 0, "n_f": oracle.n_f, "rows": rows,
                "status": "мінімум одразу між пробними точками"}

    if fl < f0 > fr:
        rows[-1]["comment"] += "  ⚠ порушено унімодальність"

    if fr <= f0:
        p, step, x_cur, f_cur = 0, delta0, xr, fr
        rows[-1]["comment"] += "  →  йдемо праворуч (p=0)"
    else:
        p, step, x_cur, f_cur = 1, -delta0, xl, fl
        rows[-1]["comment"] += "  →  йдемо ліворуч (p=1)"

    # --- дія 2: розгін з подвоєнням кроку ---
    x_prev = x0
    x_new = x_cur
    k = 1
    while k <= max_iter:
        x_new = min(max(x_cur + (2.0 ** k) * step, lo), hi)
        if x_new == x_cur:                       # уперлися в межу
            break
        f_new = ask(x_new, k, (2.0 ** k) * step, p, f"розгін: x + 2^{k}·Δ")
        if f_new >= f_cur:
            rows[-1]["comment"] += "  →  f зросла, зупинка"
            break
        x_prev, x_cur, f_cur = x_cur, x_new, f_new
        k += 1

    # --- дія 3: відрізок по передостанній і останній точках ---
    a, b = min(x_prev, x_new), max(x_prev, x_new)
    return {"a": max(a, lo), "b": min(b, hi), "p": p, "iterations": k,
            "n_f": oracle.n_f, "rows": rows, "status": "відрізок знайдено"}