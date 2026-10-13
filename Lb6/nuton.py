# -*- coding: utf-8 -*-
"""
ЛР 6. Метод Ньютона та Ньютона-Рафсона з регулюванням кроку.
f(x) = (x+1)^3 (x+2.5)(x-3.3) + 3.5

"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

os.makedirs("images", exist_ok=True)

SHOW = True   

def _finish(fig):
    """Закрити фігуру, якщо вікна не потрібні; інакше залишити для plt.show()."""
    if not SHOW:
        plt.close(fig)


# 1. Ф-ія та похідні  вручну
#    f   = (x+1)^3 (x^2 - 0.8x - 8.25) + 3.5
#    f'  = (x+1)^2 (5x^2 - 1.2x - 25.55)
#    f'' = (x+1)   (20x^2 + 6.4x - 52.3)

def f(x):
    return (x + 1) ** 3 * (x + 2.5) * (x - 3.3) + 3.5


def df(x):
    return (x + 1) ** 2 * (5 * x ** 2 - 1.2 * x - 25.55)


def d2f(x):
    return (x + 1) * (20 * x ** 2 + 6.4 * x - 52.3)


class Counter:
    """Лічильник звернень до f, f', f'' (лише ті, що потрібні алгоритму)."""
    def __init__(self):
        self.f = self.d1 = self.d2 = 0

    def F(self, x):
        self.f += 1
        return f(x)

    def D1(self, x):
        self.d1 += 1
        return df(x)

    def D2(self, x):
        self.d2 += 1
        return d2f(x)

    def total(self):
        return f"{self.f} f + {self.d1} f' + {self.d2} f''"



# 2. параметри

A0 = 2.36          # a0 зі Свенна
B0 = 2.42          # b0 зі Свенна
EPS = 1e-3
ALPHA = 0.25       # параметр α ∈ (0,1) у (3.10); при 0.5 повний крок майже на межі умови
T_MIN = 1e-8
MAX_IT = 50

# точний мінімум: корінь 5x^2 - 1.2x - 25.55 = 0 (x > 0)
X_EXACT = (1.2 + np.sqrt(1.2 ** 2 + 4 * 5 * 25.55)) / 10



# 3. Перевірка похідних чисельно

def check_derivatives(points, h=1e-5):
    print("=" * 78)
    print("ПЕРЕВІРКА ПОХІДНИХ: (f(x+h) - f(x-h)) / 2h,  h = 1e-5")
    print("=" * 78)
    print(f"{'x':>8} {'f\'(анал.)':>15} {'f\'(числ.)':>15} {'|Δ|':>10}"
          f" {'f\'\'(анал.)':>15} {'f\'\'(числ.)':>15} {'|Δ|':>10}")
    ok = True
    for x in points:
        a1 = df(x)
        n1 = (f(x + h) - f(x - h)) / (2 * h)
        a2 = d2f(x)
        n2 = (df(x + h) - df(x - h)) / (2 * h)
        e1, e2 = abs(a1 - n1), abs(a2 - n2)
        ok &= (e1 < 1e-4 * max(1, abs(a1))) and (e2 < 1e-4 * max(1, abs(a2)))
        print(f"{x:8.3f} {a1:15.6f} {n1:15.6f} {e1:10.2e} {a2:15.6f} {n2:15.6f} {e2:10.2e}")
    print("Похідні виведено правильно." if ok else "УВАГА: розбіжність!")
    print()


# 4. част А — класичний Ньютон 
def newton(x0, eps=EPS, max_it=MAX_IT, verbose=True, check_sign=False):
    c = Counter()
    rows, traj = [], [x0]
    x = x0
    status = "збіг"
    k = 0
    while True:
        d1 = c.D1(x)                                   # крок 2
        d2 = c.D2(x)
        rows.append((k, x, f(x), d1, abs(d1), d2))     # f тільки для таблиці
        if abs(d1) <= eps:                             # крок 3
            break
        if k >= max_it:
            status = "розбіжність (перевищено max_it)"
            break
        if d2 == 0 or (check_sign and d2 < 0):
            status = f"f'' = {d2:.3g} ≤ 0 — зупинка"
            break
        x = x - d1 / d2                                # крок 4
        traj.append(x)
        if not np.isfinite(x) or abs(x) > 1e8:
            status = "розбіжність (x → ∞)"
            break
        k += 1                                         # крок 5
    if verbose:
        print(f"{'k':>2} {'x_k':>16} {'f(x_k)':>16} {'f\'(x_k)':>16}"
              f" {'|f\'(x_k)|':>14} {'f\'\'(x_k)':>14}")
        for r in rows:
            print(f"{r[0]:>2} {r[1]:16.8f} {r[2]:16.8f} {r[3]:16.8f}"
                  f" {r[4]:14.8f} {r[5]:14.6f}")
    return x, k, c, rows, traj, status



# 5. част Б — Ньютон-Рафсон 

def newton_raphson(x0, eps=EPS, alpha=ALPHA, t_min=T_MIN, max_it=MAX_IT,
                   verbose=True):
    c = Counter()
    rows = []          # (k, xk, fk, d1, d2, t, xb, fb, d1b, cond)
    traj = [x0]
    x = x0
    fx, d1, d2 = c.F(x), c.D1(x), c.D2(x)             # крок 2
    status = "збіг"
    if abs(d1) <= eps:
        rows.append((0, x, fx, d1, d2, None, None, None, None, "стоп"))
        return x, 0, c, rows, traj, status
    k = 0
    done = False
    while k < max_it and not done:
        if d2 <= 1e-12:                                # захист 
            status = f"f'' = {d2:.3g} ≤ 0 — зупинка (змініть x0)"
            break
        t = 1.0                                        # крок 3
        first_try = True
        while True:
            xb = x - t * d1 / d2                       # крок 4
            d1b = c.D1(xb)                             # крок 5
            if abs(d1b) <= eps:
                fb = f(xb)                             # для таблиці (не рахуємо в Nf)
                rows.append((k, x if first_try else None, fx if first_try else None,
                             d1 if first_try else None, d2 if first_try else None,
                             t, xb, fb, d1b, "стоп"))
                x, d1 = xb, d1b
                traj.append(x)
                done = True
                break
            fb = c.F(xb)                               # крок 6
            promised = -alpha * t * d1 ** 2 / d2
            ok = (fb - fx) <= promised                 # крок 7, умова (3.10)
            rows.append((k, x if first_try else None, fx if first_try else None,
                         d1 if first_try else None, d2 if first_try else None,
                         t, xb, fb, d1b, "✓" if ok else "✗"))
            first_try = False
            if ok:
                x, fx, d1 = xb, fb, d1b                # кроки 8, 12
                traj.append(x)
                d2 = c.D2(x)                           # крок 11
                break
            t /= 2                                     # крок 10
            if t < t_min:                              # крок 9
                status = "дроблення не допомогло (t < t_min)"
                done = True
                break
        if done:
            break
        k += 1
    else:
        if not done:
            status = "перевищено max_it"
    if verbose:
        print(f"{'k':>2} {'x_k':>12} {'f(x_k)':>14} {'f\'(x_k)':>14} {'f\'\'(x_k)':>12}"
              f" {'t':>9} {'x̄':>14} {'f(x̄)':>14} {'f\'(x̄)':>14} {'(3.10)':>7}")
        s = lambda v, w, p: (f"{v:{w}.{p}f}" if v is not None else " " * w)
        for r in rows:
            tt = f"{r[5]:9.5g}" if r[5] is not None else " " * 9
            print(f"{r[0]:>2} {s(r[1],12,6)} {s(r[2],14,6)} {s(r[3],14,6)} {s(r[4],12,4)}"
                  f" {tt} {s(r[6],14,6)} {s(r[7],14,6)} {s(r[8],14,6)} {r[9]:>7}")
    return x, k, c, rows, traj, status



# 6. Графіки

def plot_phi_tangents(traj, fname="images/phi_tangents.png"):
    """φ(x) = f'(x) з дотичними першої та другої ітерацій (загальний вигляд + збільшення)."""
    x0, x1, x2 = traj[0], traj[1], traj[2]
    cols = ["tab:orange", "tab:green"]
    fig, (ax, az) = plt.subplots(1, 2, figsize=(14, 6))

    def draw(a, xs):
        a.plot(xs, df(xs), "b-", lw=2.5, label="φ(x) = f′(x)")
        a.axhline(0, color="k", lw=0.8)
        for i, xk in enumerate([x0, x1]):
            tang = df(xk) + d2f(xk) * (xs - xk)
            a.plot(xs, tang, "--", color=cols[i], lw=1.8,
                   label=f"дотична в x{i} = {xk:.6f}")
            a.plot(xk, df(xk), "o", color=cols[i], ms=7)
        a.plot(X_EXACT, 0, "*", color="red", ms=15, zorder=5,
               label=f"x* = {X_EXACT:.6f}")
        a.grid(alpha=0.3)
        a.set_xlabel("x")
        a.set_ylabel("φ(x)")

    # ліва панель загальний вигляд 
    xs = np.linspace(x0 - 0.01, x0 + 0.08, 400)
    draw(ax, xs)
    xn0 = x0 - df(x0) / d2f(x0)
    ax.plot(xn0, 0, "s", color=cols[0], ms=7)
    ax.annotate(f"x₁ = {xn0:.6f}", (xn0, 0), xytext=(xn0 + 0.012, -4.5),
                color=cols[0], fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cols[0]))
    ax.annotate(f"(x₀; f′(x₀))\n= ({x0:.2f}; {df(x0):.3f})", (x0, df(x0)),
                xytext=(x0 - 0.004, 4.5), color=cols[0], fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cols[0]))
    ax.annotate(f"(x₁; f′(x₁))\n= ({x1:.6f}; {df(x1):.3f})", (x1, df(x1)),
                xytext=(x1 + 0.008, df(x1) - 8), color=cols[1], fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cols[1]))
    ax.set_title("Загальний вигляд")
    ax.legend(loc="upper left", fontsize=9)

    # права панель збільшення біля кореня
    d = 0.0008
    xs = np.linspace(X_EXACT - d, X_EXACT + d, 400)
    draw(az, xs)
    xn1 = x1 - df(x1) / d2f(x1)
    az.plot(xn1, 0, "s", color=cols[1], ms=8)
    az.annotate(f"x₂ = {xn1:.7f}", (xn1, 0), xytext=(xn1 - 0.0006, -0.12),
                color=cols[1], fontsize=10,
                arrowprops=dict(arrowstyle="->", color=cols[1]))
    az.annotate(f"x* = {X_EXACT:.7f}", (X_EXACT, 0), xytext=(X_EXACT - 0.0006, 0.1),
                color="red", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="red"))
    az.set_xlim(X_EXACT - d, X_EXACT + d)
    az.set_ylim(-0.25, 0.25)
    az.set_title("Збільшення біля кореня")
    az.ticklabel_format(useOffset=False)

    fig.suptitle("Метод Ньютона: φ(x) = f′(x) і дотичні 1-ї та 2-ї ітерацій")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    _finish(fig)
    print(f"Збережено {fname}")


def plot_function(fname="images/f_and_df.png"):
    xs = np.linspace(-3.5, 4, 600)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.plot(xs, f(xs), "b")
    a1.axvline(X_EXACT, color="r", ls="--")
    a1.set_ylabel("f(x)")
    a1.grid(alpha=0.3)
    a1.set_title("f(x) = (x+1)³(x+2,5)(x−3,3)+3,5 та f′(x)")
    a2.plot(xs, df(xs), "b")
    a2.axhline(0, color="k", lw=0.8)
    a2.axvline(X_EXACT, color="r", ls="--")
    a2.set_ylabel("f′(x)")
    a2.set_xlabel("x")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    _finish(fig)
    print(f"Збережено {fname}")


# 8. допоміжне

def line(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    print(f"f(x) = (x+1)^3 (x+2.5)(x-3.3) + 3.5")
    print("f'(x)  = (x+1)^2 (5x^2 - 1.2x - 25.55)")
    print("f''(x) = (x+1) (20x^2 + 6.4x - 52.3)")
    print(f"x0 = a0 = {A0},  b0 = {B0},  ε = {EPS},  α = {ALPHA}")
    print(f"Точний мінімум (аналітично): x = {X_EXACT:.8f}, f = {f(X_EXACT):.8f}\n")

    check_derivatives([A0, B0, 0.5, 1.0, 2.0, 3.0, -2.0])

    # част А
    line("ЧАСТИНА А. Класичний Ньютон (табл. 4.8), x0 = a0")
    xA, nA, cA, rowsA, trajA, stA = newton(A0)
    print(f"\nx* = {xA:.8f}, f(x*) = {f(xA):.8f}, n = {nA}")
    print(f"Звернень: {cA.d1} f' + {cA.d2} f''  (f не потрібна)")
    print(f"|x* − x_точн| = {abs(xA - X_EXACT):.2e}")

    # част Б 
    line("ЧАСТИНА Б. Ньютон-Рафсон із регулюванням кроку (табл. 4.9), x0 = a0")
    xB, nB, cB, rowsB, trajB, stB = newton_raphson(A0)
    print(f"\nx* = {xB:.8f}, f(x*) = {f(xB):.8f}, n = {nB}")
    print(f"Звернень: {cB.total()}   статус: {stB}")
    print(f"|x* − x_точн| = {abs(xB - X_EXACT):.2e}")

    # графіки 
    line("ГРАФІКИ")
    plot_function()
    if len(trajA) >= 3:
        plot_phi_tangents(trajA)
    else:
        print("Замало ітерацій для графіка з двома дотичними — "
              "зменште ε або змініть x0.")
    

    # експеримент 
    line("ЕКСПЕРИМЕНТ: різні старти")
    starts = [("a0", A0), ("b0", B0),
              ("далеко: 0.5", 0.5), ("далеко: 1.2", 1.2),
              ("далеко: 3.0", 3.0), ("далеко: 5.0", 5.0),
              ("далеко: -1.0", -1.0), ("далеко: -3.0", -3.0)]
    print(f"{'старт':>14} {'x0':>6} | {'Ньютон: x*':>12} {'n':>3} {'f″(x*)':>10}"
          f" {'тип':>9} {'статус':<22} | {'Н-Рафсон: x*':>12} {'n':>3} {'тип':>9}"
          f" статус")

    def kind(x):
        """Тип стаціонарної точки за значеннями f у сусідніх точках (знак f'' поблизу
        x = -1 оманливий, бо там f'' = 0 і f' має подвійний корінь)."""
        if not np.isfinite(x) or abs(x) > 1e6 or abs(df(x)) > 1e-2:
            return "—"
        dl = 0.05
        fm, f0, fp = f(x - dl), f(x), f(x + dl)
        if f0 < fm and f0 < fp:
            return "мінімум"
        if f0 > fm and f0 > fp:
            return "МАКСИМУМ"
        return "перегин"

    for name, x0 in starts:
        xa, na, ca, _, _, sa = newton(x0, verbose=False)
        xb_, nb, cb, _, _, sb = newton_raphson(x0, verbose=False)
        d2a = d2f(xa) if np.isfinite(xa) and abs(xa) < 1e6 else float("nan")
        print(f"{name:>14} {x0:6.2f} | {xa:12.6f} {na:3d} {d2a:10.3f} {kind(xa):>9}"
              f" {sa[:22]:<22} | {xb_:12.6f} {nb:3d} {kind(xb_):>9} {sb[:40]}")
    print("\nКласичний Ньютон не перевіряє знак f″ і зупиняється за |f′| ≤ ε, тому з лівих "
          "стартів потрапляє у хибну стаціонарну точку:\nперегин x = -1 (f′ має подвійний корінь, "
          "збіжність там лише лінійна) або максимум x = -2.144.\nУ Ньютоні-Рафсоні є захист f″ > 0 "
          "(слайд 14): він зупиняється з повідомленням.")

    # табл. 4.1
    line("ЗВЕДЕНА ТАБЛИЦЯ 4.1")
    print(f"{'Метод':<34} {'x*':>10} {'n':>4} {'Звернення':<22} {'|x*-x_точн|':>12}")
    old = [
        ("Дихотомія (ЛР 2)", 2.38368, 7, "14 f"),
        ("Половинний поділ (ЛР 2)", 2.38391, 6, "13 f"),
        ("Золотий переріз (ЛР 3)", 2.383699, 14, "16 f"),
        ("Фібоначчі (ЛР 4)", 2.383754, 13, "14 f"),
        ("Пауелл (ЛР 5)", 2.383768, 2, "5 f"),
    ]
    for nm, xs_, n_, ev in old:
        print(f"{nm:<34} {xs_:10.6f} {n_:4d} {ev:<22} {abs(xs_ - X_EXACT):12.2e}")
    print(f"{'Ньютон (ЛР 6, частина А)':<34} {xA:10.6f} {nA:4d} "
          f"{(str(cA.d1) + ' f′ + ' + str(cA.d2) + ' f″'):<22} {abs(xA - X_EXACT):12.2e}")
    print(f"{'Ньютон-Рафсон (ЛР 6, частина Б)':<34} {xB:10.6f} {nB:4d} "
          f"{cB.total():<22} {abs(xB - X_EXACT):12.2e}")
    print(f"\nТочне значення: x = {X_EXACT:.8f}")

    if SHOW:
        plt.show()  


if __name__ == "__main__":
    main()