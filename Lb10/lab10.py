"""ЛР 10. Метод Хука-Дживса (прямий пошук, без похідних).
F = x1^2 + 2 x2^2 + 2 x1 x2 - 9 x1 - 13 x2,  x0 = (-3; -2)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X0 = np.array([-3.0, -2.0])
X_STAR = np.array([2.5, 2.0])          # аналітичний мінімум - лише для контролю похибки
EPS = 0.01


# ======================= цільові функції та лічильники =======================
class Obj:
    """Обгортка: рахує звернення до F (NF) всередині виклику; градієнт - лише для ЛР 8-9."""
    def __init__(self, f, g=None):
        self._f, self._g, self.nf, self.ng, self.hist = f, g, 0, 0, []
    def F(self, x):
        self.nf += 1
        v = self._f(np.asarray(x, float))
        self.hist.append(v)
        return v
    def grad(self, x):
        self.ng += 1
        return self._g(np.asarray(x, float))

f_main = lambda x: x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1] - 9*x[0] - 13*x[1]
g_main = lambda x: np.array([2*x[0] + 2*x[1] - 9, 2*x[0] + 4*x[1] - 13])
f_slide = lambda x: 2*x[0]**2 + 3*x[1]**2 + x[0]*x[1]               # (самоперевірка)
f_ros = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
g_ros = lambda x: np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2), 200*(x[1] - x[0]**2)])
f_G = lambda x: 2*abs(x[0] - 1) + 3*abs(x[1] + 2) + 0.5*(x[0] - 1)**2      # негладка функція 
g_G = lambda x: np.array([2*np.sign(x[0] - 1) + (x[0] - 1), 3*np.sign(x[1] + 2)])   # субградієнт


# ======================= ХУК-ДЖИВС =======================
def explore(F, y, Fy, d, trace=None):
    """Дослідний пошук (крок 3): по кожній координаті x+d, інакше x-d, інакше лишаємо.
    Другу координату пробуємо вже з оновленої точки. Повертає (z, F(z))."""
    z, Fz = y.copy(), Fy
    for i in range(len(z)):
        for sgn, name in ((+1, "+"), (-1, "-")):
            t = z.copy(); t[i] += sgn*d; Ft = F(t)
            ok = Ft < Fz
            if trace is not None:
                trace.append((f"x{i+1}{name}", t.copy(), Ft, ok))
            if ok:
                z, Fz = t, Ft
                break                       # проба «-» не потрібна
    return z, Fz


def pattern_search(F, base_new, base_old):
    """Пошук за зразком (крок 5): y = z + (z - x(k)); одне обчислення F."""
    y = base_new + (base_new - base_old)
    return y, F(y)


def hooke_jeeves(obj, x0, d0=1.0, eps=EPS, max_iter=100000, verbose=False):
    F = obj.F
    base = np.array(x0, float); Fb = F(base)                     # кроки 1-2
    y, Fy, d, from_pattern = base.copy(), Fb, d0, False
    bases, patterns = [base.copy()], []
    rows, detail = [], []
    k = 0
    def row(act, src, p, Fp, to, Fto, probes):
        rows.append((k, act, src, p.copy(), Fp, d, None if to is None else to.copy(), Fto, probes, obj.nf))
    it = 0
    while it < max_iter:
        it += 1
        nf0 = obj.nf
        trace = [] if verbose else None
        z, Fz = explore(F, y, Fy, d, trace)                      # крок 3
        src = "зразок" if from_pattern else "база"
        row("дослідний", src, y, Fy, z, Fz, obj.nf - nf0)
        if verbose:
            detail.append(f"дослідний пошук (з точки «{src}»): y = {pt(y)}, F = {num(Fy)}, Δ = {num(d)}")
            for lab, t, Ft, ok in trace:
                detail.append(f"    {lab:<3} F{pt(t)} = {num(Ft)}  {'✓' if ok else '✗'}")
            detail.append(f"    -> z = {pt(z)}, F(z) = {num(Fz)};  NF = {obj.nf}")
        if Fz < Fb:                                              # крок 4: порівняння з БАЗОЮ
            old = base
            base, Fb = z, Fz; k += 1; bases.append(base.copy())
            y, Fy = pattern_search(F, base, old)                 # крок 5
            patterns.append(y.copy())
            from_pattern = True
            row("нова база → зразок", "", base, Fb, y, Fy, 1)
            if verbose:
                detail.append(f"нова база x({k}) = {pt(base)}, F = {num(Fb)}; зразок y = {pt(y)}, F(y) = {num(Fy)};  NF = {obj.nf}")
        elif from_pattern:                                       # крок 6: назад у базу
            y, Fy, from_pattern = base.copy(), Fb, False
            row("повернення в базу", "", base, Fb, None, None, 0)
            if verbose:
                detail.append(f"    -> F(z) = {num(Fz)} не менше за F(бази) = {num(Fb)}: повернення в базу {pt(base)}")
        else:                                                    # крок 7: дроблення
            d_old = d; d /= 2
            row("дроблення Δ", "", base, Fb, None, None, 0)
            if verbose:
                detail.append(f"    -> з бази не вдалося: Δ = {num(d_old)} → {num(d)}")
            if d < eps:
                break
    return dict(x=base, F=Fb, k=k, bases=np.array(bases), patterns=np.array(patterns) if patterns else np.zeros((0, len(x0))),
                rows=rows, detail=detail, nf=obj.nf, delta_end=d)


# ======================= ЛР 8-9 (для порівняння; використовують градієнт) =======================
def gradient_method(obj, x0, h0, metric="sph", eps=EPS, max_iter=5000):
    """ЛР 8: сферична / кубічна / октаедрична метрика, дроблення кроку навпіл."""
    x = np.array(x0, float); h = h0
    Fx = obj.F(x); g = obj.grad(x)
    traj, k, reason = [x.copy()], 0, "‖∇F‖ ≤ ε"
    while np.linalg.norm(g) > eps:
        if k >= max_iter:
            reason = f"ліміт {max_iter} ітерацій"; break
        if metric == "sph":   d = -g / np.linalg.norm(g)
        elif metric == "cub": d = -np.sign(g)
        else:
            d = np.zeros_like(g); j = np.argmax(np.abs(g)); d[j] = -np.sign(g[j])
        while True:
            xn = x + h*d; Fn = obj.F(xn)
            if Fn >= Fx:
                h /= 2
                if h < 1e-12: break
            else: break
        if h < 1e-12 and Fn >= Fx:
            reason = "крок h < 1e-12: спаду F немає"; break
        x, Fx = xn, Fn; g = obj.grad(x); traj.append(x.copy()); k += 1
    return x, k, np.array(traj), reason


def svenn(phi, delta0=0.01):
    pts, vals = [0.0], [phi(0.0)]; h, step = 0.0, delta0
    while True:
        h += step; v = phi(h); pts.append(h); vals.append(v)
        if v > vals[-2]: break
        step *= 2
    return (pts[-3] if len(pts) >= 3 else pts[0]), pts[-1]

def golden(phi, a, b, eps=1e-6):
    t = (np.sqrt(5) - 1) / 2
    x1, x2 = b - t*(b - a), a + t*(b - a); f1, f2 = phi(x1), phi(x2)
    while b - a > eps:
        if f1 < f2: b, x2, f2 = x2, x1, f1; x1 = b - t*(b - a); f1 = phi(x1)
        else:       a, x1, f1 = x1, x2, f2; x2 = a + t*(b - a); f2 = phi(x2)
    return (a + b) / 2

def fletcher_reeves(obj, x0, eps=EPS, restart=2, max_iter=5000, exact_beta0=False):
    """ЛР 9: Флетчер-Рівс (або антиградієнт + точний пошук при exact_beta0=True)."""
    x = np.array(x0, float); g = obj.grad(x); d = -g; traj, k = [x.copy()], 0
    while np.linalg.norm(g) > eps and k < max_iter:
        phi = lambda h: obj.F(x + h*d)
        a, b = svenn(phi); h = golden(phi, a, b)
        x = x + h*d; g_new = obj.grad(x)
        beta = 0.0 if exact_beta0 else (g_new @ g_new) / (g @ g)
        if (k + 1) % restart == 0: beta = 0.0
        d = -g_new + beta*d; g = g_new; traj.append(x.copy()); k += 1
    return x, k, np.array(traj)


# ======================= форматування =======================
def num(v, nd=4):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s

def pt(v, nd=4):
    return "(" + "; ".join(num(t, nd) for t in v) + ")"

def print_protocol(res):
    print(f"{'k':>2} {'дія':<28}{'точка':<22}{'F':>11}{'Δ':>8}  {'куди':<22}{'F':>11}{'проб':>5}{'NF':>5}")
    print("-" * 118)
    for (k, act, src, p, Fp, d, to, Fto, probes, nf) in res["rows"]:
        a = act + (f" із {src}" if src else "")
        print(f"{k:>2} {a:<28}{pt(p):<22}{num(Fp, 5):>11}{num(d, 5):>8}  "
              f"{(pt(to) if to is not None else '—'):<22}{(num(Fto, 5) if Fto is not None else '—'):>11}{probes:>5}{nf:>5}")

def err(x): return float(np.linalg.norm(np.asarray(x) - X_STAR))


# ======================= головна програма =======================
def main():
    SEP = "=" * 100
    print(SEP)
    print("П.1 Постановка і аналітика (лише для контролю похибки; метод похідних не використовує)")
    print("F(x1,x2) = x1^2 + 2x2^2 + 2x1x2 - 9x1 - 13x2,  x0 = (-3; -2),  F(x0) =", f_main(X0))
    print("Аналітичний мінімум: x* = (2.5; 2), F(x*) =", f_main(X_STAR))

    # --- П.2 самоперевірка на функції зі слайдів
    print("\n" + SEP)
    print("П.2 Самоперевірка реалізації на прикладі зі слайдів 8-10 (F = 2x1^2+3x2^2+x1x2, старт (5; -3), Δ0 = 1)")
    o = Obj(f_slide); r = hooke_jeeves(o, [5, -3], 1.0, 0.01)
    ref = [62, 81, 47, 36, 18, 31, 9, 8, 12]
    print("Перші 9 значень F:", [int(round(v)) for v in o.hist[:9]], " (за слайдами:", ref, ")")
    print(f"Баз: {r['k']}, NF = {r['nf']}, мінімум = {pt(r['x'])}  (за слайдами: 4 бази, NF = 53, (0; 0))")
    assert [int(round(v)) for v in o.hist[:9]] == ref and r["nf"] == 53 and r["k"] == 4

    # --- П.3 прогін Δ0 = 1
    print("\n" + SEP)
    print("П.3 Наша F, Δ0 = 1, ε = 0.01 - протокол прогону")
    o1 = Obj(f_main); r1 = hooke_jeeves(o1, X0, 1.0, EPS, verbose=True)
    print_protocol(r1)
    print(f"\nРезультат: x = {pt(r1['x'], 6)}, F = {num(r1['F'], 6)}, баз = {r1['k']}, "
          f"NF = {r1['nf']}, N∇F = 0, ‖x - x*‖ = {err(r1['x']):.3e}, Δ_кінц = {r1['delta_end']:.5f}")
    print("\nДетальний протокол (усі проби):")
    for line in r1["detail"]: print(line)

    # --- П.4 Δ0 = 0.7 і 0.5
    print("\n" + SEP)
    print("П.4 Порівняння стартових кроків: Δ0 = 1 / 0.7 / 0.5  (ε = 0.01)")
    runs = {1.0: r1}
    for d0 in (0.7, 0.5):
        o = Obj(f_main); runs[d0] = hooke_jeeves(o, X0, d0, EPS, verbose=(d0 == 0.7))
    print("\nПротокол Δ0 = 0.7:")
    print_protocol(runs[0.7])
    print("\nБазові точки Δ0 = 0.7:")
    for i, (b) in enumerate(runs[0.7]["bases"]):
        print(f"  x({i}) = {pt(b)}   F = {num(f_main(b), 5)}")
    print("\nПорівняння:")
    print(f"{'Δ0':>5}{'баз':>5}{'NF':>6}{'Δ_кінц':>10}{'x_знайд':>26}{'‖x−x*‖':>12}{'F':>14}")
    for d0, r in runs.items():
        print(f"{d0:>5}{r['k']:>5}{r['nf']:>6}{r['delta_end']:>10.5f}{pt(r['x'], 5):>26}{err(r['x']):>12.2e}{r['F']:>14.6f}")
    print("Ґратка: усі точки HJ мають вигляд x0 + Δ·(m, k), де Δ = Δ0·2^-j (m, k цілі). x* - x0 = (5.5; 4).")
    for d0 in (1.0, 0.7, 0.5):
        hit = None
        for j in range(0, 12):
            st = d0 / 2**j
            if abs(5.5/st - round(5.5/st)) < 1e-9 and abs(4/st - round(4/st)) < 1e-9:
                hit = (j, st); break
        print(f"   Δ0 = {d0}: " + (f"x* стає вузлом ґратки при Δ = Δ0/2^{hit[0]} = {hit[1]:g} -> метод може влучити точно"
                                   if hit else "x* не є вузлом ґратки Δ0/2^j за жодного j (5.5/Δ0 і 4/Δ0 не цілі) -> точно влучити неможливо"))

    # --- П.5 негладка G
    print("\n" + SEP)
    print("П.5 Негладка функція G = 2|x1-1| + 3|x2+2| + 0.5(x1-1)^2, x* = (1; -2), G(x*) = 0")
    xG = np.array([1.0, -2.0])
    gres = {}
    for start in ([5.0, -3.0], [5.3, -3.1]):
        print(f"\n--- старт {pt(start)}, G(старт) = {num(f_G(start), 4)} ---")
        oG = Obj(f_G); rG = hooke_jeeves(oG, start, 1.0, EPS); gres[tuple(start)] = rG
        if start[0] == 5.0:
            print_protocol(rG)
        print(f"Хук-Дживс: x = {pt(rG['x'], 6)}, G = {num(rG['F'], 6)}, баз = {rG['k']}, NF = {rG['nf']}, "
              f"‖x - x*‖ = {np.linalg.norm(rG['x'] - xG):.2e}")
        print("Градієнтні методи (у точках зламу беремо субградієнт, sign(0) = 0):")
        for name, m in (("сферична ЛР 8, h0=1", "sph"), ("кубічна ЛР 8, h0=1", "cub"), ("октаедрична ЛР 8, h0=1", "oct")):
            o = Obj(f_G, g_G)
            x, n, tr, why = gradient_method(o, start, 1.0, m)
            print(f"   {name:<24} n = {n:<4} x = {pt(x, 5):<20} ‖∇G‖ = {np.linalg.norm(g_G(x)):<6.3f} зупинка: {why}")
        o = Obj(f_G, g_G); xF, nF, _ = fletcher_reeves(o, start, max_iter=200)
        print(f"   {'Флетчер-Рівс ЛР 9':<24} n = {nF:<4} x = {pt(xF, 5):<20} ‖∇G‖ = {np.linalg.norm(g_G(xF)):<6.3f} зупинка: ліміт 200 ітерацій")
    print("\nПояснення. Біля мінімуму ‖∇G‖ не прямує до 0: він стрибає між значеннями з компонентами ±2 і ±3 (поза самим мінімумом ‖∇G‖ ≥ 2, на практиці 3-3.8),")
    print("тому критерій ‖∇G‖ ≤ ε не спрацьовує. Для старту (5; -3) кубічна й октаедрична метрики «спрацювали» лише тому, що кроки")
    print("по цілих вузлах потрапили точно в (1; -2), де прийнято sign(0) = 0 (градієнт = 0 за домовленістю) - це збіг, а не збіжність.")
    rG = gres[(5.0, -3.0)]

    # --- П.6 Розенброк
    print("\n" + SEP)
    print("П.6 Розенброк f = (1-x1)^2 + 100(x2-x1^2)^2, x* = (1; 1), старт (-1.2; 1), Δ0 = 1")
    ros = {}
    print(f"{'Метод':<34}{'баз':>5}{'NF':>7}{'N∇F':>7}{'x_знайд':>26}{'f':>12}")
    for eps in (1e-2, 1e-4):
        o = Obj(f_ros); r = hooke_jeeves(o, [-1.2, 1.0], 1.0, eps); ros[eps] = r
        print(f"{'Хук-Дживс, Δ0=1, ε=' + str(eps):<34}{r['k']:>5}{r['nf']:>7}{0:>7}{pt(r['x'], 4):>26}{r['F']:>12.2e}")
    o = Obj(f_ros, g_ros); xs8, n8, tr8, _ = gradient_method(o, [-1.2, 1.0], 1.0, "sph")
    print(f"{'сферична ЛР 8, h0=1':<34}{n8:>5}{o.nf:>7}{o.ng:>7}{pt(xs8, 4):>26}{f_ros(xs8):>12.2e}")
    o = Obj(f_ros, g_ros); xs9, n9, tr9 = fletcher_reeves(o, [-1.2, 1.0])
    print(f"{'Флетчер-Рівс ЛР 9 (перезапуск 2)':<34}{n9:>5}{o.nf:>7}{o.ng:>7}{pt(xs9, 4):>26}{f_ros(xs9):>12.2e}")
    print("Пастка критерію зупинки: з ε = 0.01 метод зупиняється, коли Δ < ε, хоч до x* = (1; 1) ще далеко:")
    for eps, r in ros.items():
        print(f"   ε = {eps:g}: ‖x - x*‖ = {np.linalg.norm(r['x'] - np.array([1, 1])):.4f}, f = {r['F']:.3e}, Δ_кінц = {r['delta_end']:.2e}")

    # --- П.7 порівняння з ЛР 8-9 + графік
    print("\n" + SEP)
    print("П.7 Зведена таблиця (наша F, старт (-3; -2)). NF+3·N∇F - якщо градієнт оцінювати різницями (n+1 = 3 значення F).")
    table = []
    lr8 = {}
    for h0 in (1.0, 0.7):
        for name, m in (("сферична", "sph"), ("кубічна", "cub"), ("октаедрична", "oct")):
            o = Obj(f_main, g_main); x, n, tr, _ = gradient_method(o, X0, h0, m)
            lr8[(m, h0)] = tr
            table.append((f"{name} ЛР 8, h0={h0}", n, o.nf, o.ng, err(x)))
    o = Obj(f_main, g_main); x, n, trFR = fletcher_reeves(o, X0); table.append(("Флетчер-Рівс ЛР 9", n, o.nf, o.ng, err(x)))
    o = Obj(f_main, g_main); x, n, trAG = fletcher_reeves(o, X0, exact_beta0=True); table.append(("антиградієнт+точний пошук ЛР 9", n, o.nf, o.ng, err(x)))
    for d0, r in runs.items():
        table.append((f"Хук-Дживс, Δ0={d0}", r["k"], r["nf"], 0, err(r["x"])))
    print(f"{'Метод':<34}{'n/баз':>6}{'NF':>6}{'N∇F':>6}{'NF+N∇F':>8}{'NF+3·N∇F':>10}{'‖x−x*‖':>12}")
    for (name, n, nf, ng, e) in table:
        print(f"{name:<34}{n:>6}{nf:>6}{ng:>6}{nf + ng:>8}{nf + 3*ng:>10}{e:>12.2e}")
    print("(n для градієнтних методів - ітерації; для Хука-Дживса - кількість нових базових точок k, без стартової - як на слайдах)")

    # графік 1: наша F
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    X, Y = np.meshgrid(np.linspace(-4.5, 4.5, 500), np.linspace(-3.5, 4.5, 500))
    Z = X**2 + 2*Y**2 + 2*X*Y - 9*X - 13*Y
    lv = [-24, -20, -16, -13, -9, -5, -1, 3, 6, 10, 14, 18, 22, 29, 37, 50]
    for ax in axs:
        cs = ax.contour(X, Y, Z, levels=lv, cmap="viridis"); ax.clabel(cs, fontsize=7)
        ax.plot(*X0, "k*", ms=14, label="Старт (-3; -2)"); ax.plot(*X_STAR, "rX", ms=11, label="Мінімум (2.5; 2)")
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax = axs[0]
    for d0, col, mk in ((1.0, "tab:green", "^"), (0.7, "tab:red", "o"), (0.5, "tab:purple", "s")):
        b = runs[d0]["bases"]
        ax.plot(*b.T, mk + "-", color=col, ms=5, lw=1.6, label=f"Хук–Дживс Δ0={d0}: {len(b) - 1} баз, NF={runs[d0]['nf']}")
    ax.set_title("Хук–Дживс: траєкторії базових точок"); ax.legend(fontsize=8, loc="upper left")
    ax = axs[1]
    ax.plot(*lr8[("sph", 1.0)].T, "o-", ms=3, color="tab:blue", label="сферична ЛР 8, h0=1 (n=19)")
    ax.plot(*trAG.T, "s-", ms=4, color="tab:orange", label="антиградієнт + точний пошук ЛР 9 (n=4)")
    ax.plot(*trFR.T, "v-", ms=6, lw=2, color="tab:olive", label="Флетчер–Рівс ЛР 9 (n=2)")
    ax.plot(*runs[1.0]["bases"].T, "^-", ms=6, lw=2, color="tab:green", label=f"Хук–Дживс Δ0=1 ({len(runs[1.0]['bases']) - 1} баз)")
    ax.set_title("Порівняння з методами ЛР 8–9"); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig("lr10_trajectories.png", dpi=140); plt.close(fig)

    # графік 2: Розенброк
    fig, ax = plt.subplots(figsize=(8, 6.5))
    X2, Y2 = np.meshgrid(np.linspace(-1.6, 1.6, 500), np.linspace(-0.6, 1.6, 500))
    ax.contour(X2, Y2, (1 - X2)**2 + 100*(Y2 - X2**2)**2, levels=np.logspace(-1, 3.3, 18), cmap="viridis", linewidths=.7)
    for eps, col in ((1e-2, "tab:red"), (1e-4, "tab:green")):
        b = ros[eps]["bases"]; ax.plot(*b.T, "o-", ms=3, color=col, label=f"ε={eps:g}: {len(b) - 1} баз, NF={ros[eps]['nf']}")
    ax.plot(-1.2, 1, "k*", ms=13, label="Старт (-1.2; 1)"); ax.plot(1, 1, "rX", ms=11, label="Мінімум (1; 1)")
    ax.set_title("Розенброк: траєкторії Хука–Дживса"); ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.savefig("lr10_rosenbrock.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # графік 3: негладка G
    fig, ax = plt.subplots(figsize=(8, 6.5))
    X3, Y3 = np.meshgrid(np.linspace(-1.5, 6, 500), np.linspace(-4.5, 0.5, 500))
    Z3 = 2*abs(X3 - 1) + 3*abs(Y3 + 2) + 0.5*(X3 - 1)**2
    cs = ax.contour(X3, Y3, Z3, levels=[0.5, 1, 2.5, 5, 7.5, 10, 14, 19, 25], cmap="viridis"); ax.clabel(cs, fontsize=7)
    b = rG["bases"]; ax.plot(*b.T, "^-", color="tab:green", lw=2, label=f"Хук–Дживс: {len(b) - 1} баз, NF={rG['nf']}")
    ax.plot(5, -3, "k*", ms=13, label="Старт (5; -3)"); ax.plot(1, -2, "rX", ms=11, label="Мінімум (1; -2)")
    ax.set_aspect("equal"); ax.set_title("Негладка функція G: лінії рівня з «кутами»"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.savefig("lr10_nonsmooth.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nГрафіки збережено: lr10_trajectories.png, lr10_rosenbrock.png, lr10_nonsmooth.png")


if __name__ == "__main__":
    main()