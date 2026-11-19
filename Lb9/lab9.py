"""ЛР 9. Метод спряжених градієнтів Флетчера-Рівса.
F = x1^2 + 2 x2^2 + 2 x1 x2 - 9 x1 - 13 x2,  x0 = (-3; -2),  eps = 0.01
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 0.01
X0 = np.array([-3.0, -2.0])
H = np.array([[2.0, 2.0], [2.0, 4.0]])
X_STAR = np.array([2.5, 2.0])
F_STAR = -24.25


# ---------- цільова функція (з лічильниками звернень) ----------
class Counter:
    def __init__(self, f, g):
        self._f, self._g, self.nf, self.ng = f, g, 0, 0
    def F(self, x):
        self.nf += 1
        return self._f(x)
    def grad(self, x):
        self.ng += 1
        return self._g(x)

f_quad = lambda x: x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1] - 9*x[0] - 13*x[1]
g_quad = lambda x: np.array([2*x[0] + 2*x[1] - 9, 2*x[0] + 4*x[1] - 13])
f_rosen = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
g_rosen = lambda x: np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                              200*(x[1] - x[0]**2)])


# ---------- ЛР 1: Свенн; ЛР 2-4: золотий переріз ----------
def svenn(phi, delta0=0.01, info=None):
    """h: 0, d, 3d, 7d, ... (крок подвоюється), доки phi спадає."""
    pts = [0.0]; vals = [phi(0.0)]
    h, step = 0.0, delta0
    while True:
        h += step
        v = phi(h)
        pts.append(h); vals.append(v)
        if v > vals[-2]:
            break
        step *= 2
    a = pts[-3] if len(pts) >= 3 else pts[0]
    if info is not None:
        info["pts"], info["vals"] = pts, vals
    return a, pts[-1]

def golden(phi, a, b, eps=1e-6, info=None):
    t = (np.sqrt(5) - 1) / 2
    it = 0
    x1, x2 = b - t*(b - a), a + t*(b - a)
    f1, f2 = phi(x1), phi(x2)
    while b - a > eps:
        it += 1
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - t*(b - a); f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + t*(b - a); f2 = phi(x2)
    if info is not None:
        info["a"], info["b"], info["iters"] = a, b, it
    return (a + b) / 2

def line_search(F, x, d):
    phi = lambda h: F(x + h*d)
    a, b = svenn(phi, 0.01)
    return golden(phi, a, b, 1e-6)


# ---------- Флетчер-Рівс ----------
def fletcher_reeves(obj, x0, eps=EPS, restart=2, max_iter=5000, restart_on=True,
                    exact_beta0=False, log=False):
    x = np.array(x0, float)
    g = obj.grad(x); d = -g
    traj, rows, k = [x.copy()], [], 0
    while np.linalg.norm(g) > eps and k < max_iter:
        Fx = obj.F(x) if log else None
        h = line_search(obj.F, x, d)
        x_new = x + h*d
        g_new = obj.grad(x_new)
        beta = (g_new @ g_new) / (g @ g)
        if exact_beta0:
            beta = 0.0
        if restart_on and (k + 1) % restart == 0:
            beta = 0.0
        if log:
            rows.append((k, x.copy(), Fx, g.copy(), np.linalg.norm(g), d.copy(), h, beta,
                         obj._f(x_new), g_new @ d))   # <g_new, d_old> для перевірки
        d = -g_new + beta*d
        x, g = x_new, g_new
        traj.append(x.copy()); k += 1
    return x, k, np.array(traj), rows


# ---------- ЛР 8: сферична метрика з дробленням (для порівняння) ----------
def gradient_spherical(obj, x0, h0, eps=EPS, max_iter=5000):
    x = np.array(x0, float); h = h0
    Fx = obj.F(x); g = obj.grad(x)
    traj, k = [x.copy()], 0
    while np.linalg.norm(g) > eps and k < max_iter:
        d = -g / np.linalg.norm(g)
        while True:
            xn = x + h*d; Fn = obj.F(xn)
            if Fn >= Fx: h /= 2
            else: break
            if h < 1e-15: break
        x, Fx = xn, Fn; g = obj.grad(x); traj.append(x.copy()); k += 1
    return x, k, np.array(traj)


def fmt(v, nd=5):
    return "(" + "; ".join(f"{t:.{nd}f}" for t in v) + ")"


def main():
    print("=" * 70)
    print("П.1 Аналітика")
    print("grad F = (2x1+2x2-9; 2x1+4x2-13), H =", H.tolist())
    xs = np.linalg.solve(H, np.array([9.0, 13.0]))
    print("x* =", xs, " F* =", f_quad(xs), " eigenvalues(H) =", np.linalg.eigvalsh(H))

    # П.2 перевірка лінійного пошуку
    g0 = g_quad(X0); d0 = -g0
    h_exact = -(g0 @ d0) / (d0 @ H @ d0)
    gd, dHd = g0 @ d0, d0 @ H @ d0
    obj = Counter(f_quad, g_quad)
    phi = lambda h: obj.F(X0 + h*d0)
    sv, gl = {}, {}
    a, b = svenn(phi, 0.01, sv)
    nf_sv = obj.nf
    h_num = golden(phi, a, b, 1e-6, gl)
    nf_gl = obj.nf - nf_sv
    print("\nП.2 Лінійний пошук на 1-му кроці: x0 = (-3; -2), d0 = -grad F = (19; 27)")
    print(f"phi(h) = F(x0 + h*d0) = {f_quad(X0):.0f} + ({gd:.0f})h + {dHd/2:.0f}h^2   "
          f"(<g,d> = {gd:.0f}, d'Hd = {dHd:.0f})")
    print("Метод Свенна, початковий крок 0.01 (крок подвоюється, поки phi спадає):")
    for hh, vv in zip(sv["pts"], sv["vals"]):
        print(f"   h = {hh:.2f}   phi = {vv:.4f}")
    print(f"   перше зростання на h = {sv['pts'][-1]:.2f} -> відрізок [a; b] = [{a:.2f}; {b:.2f}], "
          f"{nf_sv} обчислень F")
    print("Золотий переріз, eps = 1e-6:")
    print(f"   початковий відрізок: [{a:.6f}; {b:.6f}]")
    print(f"   кінцевий відрізок:   [{gl['a']:.6f}; {gl['b']:.6f}]  (довжина {gl['b']-gl['a']:.2e})")
    print(f"   h = {h_num:.6f}, ітерацій: {gl['iters']}, обчислень F: {nf_gl}")
    print(f"Перевірка: h* = -<g,d>/d'Hd = {h_exact:.6f};  |h - h*| = {abs(h_num - h_exact):.1e}")

    # П.3 Флетчер-Рівс
    obj = Counter(f_quad, g_quad)
    x_fr, n_fr, tr_fr, rows = fletcher_reeves(obj, X0, restart=2, log=True)
    # NF у log-режимі містить зайві F(x) для таблиці -> перерахуємо чисто
    obj_c = Counter(f_quad, g_quad); fletcher_reeves(obj_c, X0, restart=2)
    nf_fr, ng_fr = obj_c.nf, obj_c.ng
    print("\nП.3 Флетчер-Рівс")
    hdr = (f"{'k':>2} {'x(k)':<22}{'F(x(k))':>10} {'grad F':<22}{'|grad|':>9} "
           f"{'d(k)':<22}{'h_k':>9}{'beta':>9}{'F(k+1)':>11}")
    print(hdr)
    print("-" * len(hdr))
    for (k, x, Fx, g, ng, d, h, beta, Fn, _) in rows:
        print(f"{k:>2} {fmt(x):<22}{Fx:>10.4f} {fmt(g):<22}{ng:>9.4f} {fmt(d):<22}"
              f"{h:>9.6f}{beta:>9.6f}{Fn:>11.3e}")
    gf = g_quad(x_fr)
    print(f"{len(rows):>2} {fmt(x_fr):<22}{f_quad(x_fr):>10.2e} {fmt(gf):<22}{np.linalg.norm(gf):>9.2e}   -> стоп")
    print(f"n = {n_fr}, NF = {nf_fr}, Ngrad = {ng_fr}, ||x-x*|| = {np.linalg.norm(x_fr - xs):.3e}")

    # П.4 три перевірки
    print("\nП.4 Перевірки")
    x1 = tr_fr[1]; g1 = g_quad(x1); d0 = rows[0][5]; beta1 = rows[0][7]
    d1 = -g1 + beta1*d0
    print(f"1) <grad F(x1), d0>   = {g1 @ d0:.3e}   (має бути ~0)")
    print(f"2) d1' H d0           = {d1 @ H @ d0:.3e}   (має бути ~0)")
    print(f"3) <grad F(x1), d1>   = {g1 @ d1:.6f}  (має бути < 0);  -||g1||^2 = {-(g1 @ g1):.6f}")
    print(f"   кут між d0 і d1 = {np.degrees(np.arccos(d0@d1/np.linalg.norm(d0)/np.linalg.norm(d1))):.2f} град")
    print(f"   Кількість ітерацій = {n_fr}  ({'так, n=2' if n_fr == 2 else 'НЕ 2!'})")

    # П.5 антиградієнт + точний пошук
    obj_ag = Counter(f_quad, g_quad)
    x_ag, n_ag, tr_ag, _ = fletcher_reeves(obj_ag, X0, exact_beta0=True)
    print(f"\nП.5 Антиградієнт + точний пошук: n = {n_ag}, NF = {obj_ag.nf}, Ngrad = {obj_ag.ng}, "
          f"||x-x*|| = {np.linalg.norm(x_ag - xs):.3e}")
    for i in range(1, len(tr_ag) - 1):
        a_ = tr_ag[i] - tr_ag[i-1]; b_ = tr_ag[i+1] - tr_ag[i]
        c = a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_)
        print(f"   cos між кроками {i} і {i+1}: {c:.2e}")
    print("Сферична метрика ЛР 8 (для порівняння, дроблення кроку навпіл):")
    res8 = {}
    for h0 in (1.0, 0.7):
        o = Counter(f_quad, g_quad)
        x8, n8, tr8 = gradient_spherical(o, X0, h0)
        res8[h0] = (n8, o.nf, o.ng, np.linalg.norm(x8 - xs), tr8)
        print(f"   h0 = {h0}: n = {n8}, NF = {o.nf}, Ngrad = {o.ng}, ||x-x*|| = {np.linalg.norm(x8 - xs):.4e}")

    # П.6 графік
    print("\nП.6 Графік ліній рівня з трьома траєкторіями")
    fig, ax = plt.subplots(figsize=(9, 7))
    X, Y = np.meshgrid(np.linspace(-4.5, 4, 400), np.linspace(-3.5, 4.5, 400))
    Z = X**2 + 2*Y**2 + 2*X*Y - 9*X - 13*Y
    cs = ax.contour(X, Y, Z, levels=[-24, -20, -16, -13, -9, -5, -1, 3, 6, 10, 14, 18, 22, 29, 37, 50], cmap="viridis")
    ax.clabel(cs, fontsize=7)
    ax.plot(*res8[1.0][4].T, "o-", ms=3, label=f"Сферична ЛР 8, h0=1 (n={res8[1.0][0]})")
    ax.plot(*tr_ag.T, "s-", ms=4, label=f"Антиградієнт + точний пошук (n={n_ag})")
    ax.plot(*tr_fr.T, "^-", ms=6, lw=2, label=f"Флетчер–Рівс (n={n_fr})")
    ax.plot(*X0, "k*", ms=14, label="Початкова точка (-3; -2)")
    ax.plot(*X_STAR, "rX", ms=11, label="Точний мінімум (2.5; 2)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title("Лінії рівня F та траєкторії методів"); ax.legend(fontsize=8)
    fig.savefig("lr9_trajectories.png", dpi=150, bbox_inches="tight")
    print("Графік збережено: lr9_trajectories.png")

    # П.7 Розенброк
    print("\nП.7 Розенброк, старт (-1.2; 1), eps = 0.01")
    rows7 = []
    for name, kw in [("ФР, перезапуск кожні 2", dict(restart_on=True, restart=2)),
                     ("ФР, без перезапуску", dict(restart_on=False))]:
        o = Counter(f_rosen, g_rosen)
        x, n, _, _ = fletcher_reeves(o, [-1.2, 1.0], **kw)
        rows7.append((name, n, o.nf, o.ng, f_rosen(x)))
    o = Counter(f_rosen, g_rosen)
    x, n, _ = gradient_spherical(o, [-1.2, 1.0], 1.0)
    rows7.append(("Сферична ЛР 8 (h0=1, ліміт 5000)", n, o.nf, o.ng, f_rosen(x)))
    for r in rows7:
        print(f"  {r[0]:<36} n={r[1]:<6} NF={r[2]:<7} Ngrad={r[3]:<6} f={r[4]:.2e}")

    # П.8 зведена таблиця
    print("\nП.8 Зведена таблиця (наша F, старт (-3;-2))")
    print(f"{'Метод':<34}{'n':>4}{'NF':>6}{'Ngrad':>7}{'NF+Ngrad':>10}{'||x-x*||':>12}")
    t = [("Сферична ЛР 8, h0=1", res8[1.0][0], res8[1.0][1], res8[1.0][2], res8[1.0][3]),
         ("Сферична ЛР 8, h0=0.7", res8[0.7][0], res8[0.7][1], res8[0.7][2], res8[0.7][3]),
         ("Антиградієнт + точний пошук", n_ag, obj_ag.nf, obj_ag.ng, np.linalg.norm(x_ag - xs)),
         ("Флетчер–Рівс", n_fr, nf_fr, ng_fr, np.linalg.norm(x_fr - xs))]
    for r in t:
        print(f"{r[0]:<34}{r[1]:>4}{r[2]:>6}{r[3]:>7}{r[2]+r[3]:>10}{r[4]:>12.2e}")

if __name__ == "__main__":
    main()