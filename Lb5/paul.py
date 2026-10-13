# -*- coding: utf-8 -*-
"""
ЛР 5. Метод Пауелла (квадратична апроксимація)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# вхідні дані
X1, DX = 2.36, 0.006
SF, SX = 0.01, 0.001
MAX_ITER = 50
EPS = 1e-12            # "нуль" для перевірок ділення / збігу точок

N_F = 0                # лічильник обчислень цільової функції


def f(x):
    """Цільова функція; кожен виклик рахується в Nf."""
    global N_F
    N_F += 1
    return (x + 1) ** 3 * (x + 2.5) * (x - 3.3) + 3.5


def f_plain(x):
    """Та сама функція БЕЗ лічильника - лише для графіка/перевірки."""
    return (x + 1) ** 3 * (x + 2.5) * (x - 3.3) + 3.5


# метод Пауелла 
def powell(x1, dx, sf, sx, max_iter=MAX_ITER, verbose=True):
    global N_F
    N_F = 0
    rows = []                      # рядки табл. 4.7
    notes = []                     # аварійні ситуації

    # Крок 1-2: x2 = x1 + dx, f1, f2
    x2 = x1 + dx
    f1, f2 = f(x1), f(x2)
    # Крок 3: третя точка в бік спадання
    x3 = x1 + 2 * dx if f1 > f2 else x1 - dx
    f3 = f(x3)                     # разом 3 обчислення
    start = (x1, x2, x3, f1, f2, f3)

    xt = ft = None
    converged = False
    for k in range(1, max_iter + 1):
        xs, fs = [x1, x2, x3], [f1, f2, f3]
        # Крок 4: найкраща і найгірша точки
        i_min = fs.index(min(fs))
        i_max = fs.index(max(fs))
        fmin, xmin = fs[i_min], xs[i_min]
        xmax = xs[i_max]

        # захист: точки мають бути різними, інакше парабола не існує
        if min(abs(x2 - x1), abs(x3 - x1), abs(x3 - x2)) < EPS:
            notes.append(f"k={k}: дві точки збіглися -> зупинка, перевірте критерій")
            converged = True
            break

        # Крок 5: коефіцієнти (2.30), (2.31)
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2)

        # Крок 6: вершина
        if a2 > EPS:                               # парабола гілками вгору
            xt = -a1 / (2 * a2) + (x1 + x2) / 2    # (2.32)
            mode = ""
        else:                                      # a2 <= 0: зауваження 2
            xt = 2 * xmin - xmax
            mode = " (a2<=0: x~=2xmin-xmax)"
            notes.append(f"k={k}: a2={a2:.6g} <= 0 -> x~ = 2*xmin - xmax")

        # збіг x~ з наявною точкою: f вдруге не рахуємо
        dup = [i for i in range(3) if abs(xt - xs[i]) < EPS]
        if dup:
            ft = fs[dup[0]]
            notes.append(f"k={k}: x~ збіглося з точкою x{dup[0]+1} -> f не перераховуємо")
        else:
            ft = f(xt)                             # 1 обчислення за виток

        # Крок 7: умови закінчення (обидві)
        cond_f = abs(fmin - ft) <= sf
        cond_x = abs(xmin - xt) <= sx
        stop = (cond_f and cond_x) or bool(dup)
        action = "СТОП" if stop else f"x{i_max+1} <- x~"

        rows.append(dict(k=k, x1=x1, x2=x2, x3=x3, f1=f1, f2=f2, f3=f3,
                         a1=a1, a2=a2, xt=xt, ft=ft,
                         df=abs(fmin - ft), dxm=abs(xmin - xt),
                         cf=cond_f, cx=cond_x, act=action, mode=mode))
        if verbose:
            print(f"k={k}: x~={xt:.6f} f~={ft:.6f} "
                  f"|fmin-f~|={abs(fmin-ft):.2e} {'ok' if cond_f else '--'}, "
                  f"|xmin-x~|={abs(xmin-xt):.2e} {'ok' if cond_x else '--'}{mode}")
        if stop:
            converged = True
            break

        # Крок 8: x_max <- x~  (точки не сортуємо)
        xs[i_max], fs[i_max] = xt, ft
        x1, x2, x3 = xs
        f1, f2, f3 = fs

    return dict(x_star=xt, f_star=ft, n=len(rows), nf=N_F, rows=rows,
                start=start, notes=notes, converged=converged)


# таблиця 
def print_table(res):
    """Табл. 4.7: обов'язкові стовпці + допоміжна таблиця (a1, a2, перевірки).
    Дві вужчі таблиці замість однієї широкої, щоб рядки не переносились у терміналі."""
    rows = res["rows"]
    print("\nТаблиця 4.7 (метод Пауелла)")
    h = f"{'k':>2} {'x1':>9} {'x2':>9} {'x3':>9} {'f(x1)':>12} {'f(x2)':>12} {'f(x3)':>12} {'x~':>9} {'f(x~)':>12}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['k']:>2} {r['x1']:>9.6f} {r['x2']:>9.6f} {r['x3']:>9.6f} "
              f"{r['f1']:>12.6f} {r['f2']:>12.6f} {r['f3']:>12.6f} "
              f"{r['xt']:>9.6f} {r['ft']:>12.6f}")

    print("\nДопоміжна таблиця (самоперевірка)")
    h2 = f"{'k':>2} {'a1':>10} {'a2':>10} {'|fmin-f~|':>10} {'(а)':>4} {'|xmin-x~|':>10} {'(б)':>4}  дія"
    print(h2); print("-" * len(h2))
    for r in rows:
        print(f"{r['k']:>2} {r['a1']:>10.4f} {r['a2']:>10.4f} {r['df']:>10.2e} "
              f"{'так' if r['cf'] else 'ні':>4} {r['dxm']:>10.2e} {'так' if r['cx'] else 'ні':>4}  {r['act']}")
    for n in res["notes"]:
        print("  *", n)


# графік 
def plot_first_iteration(res, path="graph_lr5.png"):
    r = res["rows"][0]
    xs3 = np.array([r["x1"], r["x2"], r["x3"]])
    fs3 = np.array([r["f1"], r["f2"], r["f3"]])
    a0, a1, a2 = r["f1"], r["a1"], r["a2"]
    par = lambda x: a0 + a1 * (x - r["x1"]) + a2 * (x - r["x1"]) * (x - r["x2"])

    lo = min(xs3.min(), r["xt"], res["x_star"]) - 0.3
    hi = max(xs3.max(), r["xt"], res["x_star"]) + 0.3
    xx = np.linspace(lo, hi, 600)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # ліво
    xw = np.linspace(-0.5, 4.0, 600)
    ax[0].plot(xw, f_plain(xw), "b", label="f(x)")
    ax[0].axvline(res["x_star"], color="g", ls=":", label=f"x* = {res['x_star']:.4f}")
    ax[0].scatter(xs3, fs3, c="r", zorder=5, label="стартові точки")
    ax[0].set_title("f(x) на ширшому інтервалі"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("f(x)")
    # право околиця точок і парабола 1-ї ітерації
    ax[1].plot(xx, f_plain(xx), "b", label="f(x)")
    ax[1].plot(xx, par(xx), "--", color="darkorange", label="парабола y(x), ітерація 1")
    ax[1].scatter(xs3, fs3, c="r", zorder=5, label="x1, x2, x3")
    ax[1].scatter([r["xt"]], [par(r["xt"])], c="k", marker="*", s=120, zorder=6,
                  label=f"вершина x~ = {r['xt']:.4f}")
    ins = ax[1].inset_axes([0.4, 0.5, 0.55, 0.4])
    xl = np.linspace(xs3.min() - 0.01, r["xt"] + 0.01, 300)
    ins.plot(xl, f_plain(xl), "b"); ins.plot(xl, par(xl), "--", color="darkorange")
    ins.scatter(xs3, fs3, c="r", zorder=5)
    ins.scatter([r["xt"]], [par(r["xt"])], c="k", marker="*", s=90, zorder=6)
    for i, (xv, fv, dy) in enumerate(zip(xs3, fs3, (14, 0, -14)), 1):
        ins.annotate(f"x{i}", (xv, fv), textcoords="offset points",
                     xytext=(-4, dy + 8), ha="center", fontsize=9,
                     arrowprops=dict(arrowstyle="-", lw=.5))
    ins.annotate("x~", (r["xt"], par(r["xt"])), textcoords="offset points",
                 xytext=(0, -16), ha="center", fontsize=9)
    ins.set_title("збільшено", fontsize=8); ins.tick_params(labelsize=7)
    ins.grid(alpha=.3)
    ax[1].indicate_inset_zoom(ins, edgecolor="gray")
    ax[1].set_title("Три стартові точки та парабола 1-ї ітерації")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    ax[1].grid(alpha=.3); ax[1].set_xlabel("x")
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print("Збережено", path)
    plt.show()


# блок-схема
def draw_flowchart(path="flowchart_lr5.png"):
    """Блок-схема алгоритму"""
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.set_xlim(-10, 100); ax.set_ylim(-8, 150); ax.axis("off")

    def term(x, y, t):
        ax.add_patch(FancyBboxPatch((x-9, y-3), 18, 6, boxstyle="round,pad=0,rounding_size=3", fc="w", ec="k"))
        ax.text(x, y, t, ha="center", va="center", fontsize=9)

    def proc(x, y, t, w=34, h=8):
        ax.add_patch(plt.Rectangle((x-w/2, y-h/2), w, h, fc="w", ec="k"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8)

    def data(x, y, t, w=34, h=7):
        ax.add_patch(Polygon([(x-w/2+3, y+h/2), (x+w/2+3, y+h/2), (x+w/2-3, y-h/2), (x-w/2-3, y-h/2)], fc="w", ec="k"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8)

    def dec(x, y, t, w=26, h=10):
        ax.add_patch(Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)], fc="w", ec="k"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8)

    def arr(p, q):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1))

    def line(*pts):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "k", lw=1)

    cx = 40
    term(cx, 146, "Початок")
    data(cx, 136, "1: x1, Δx, σx, σf")
    proc(cx, 125, "2: x2 = x1+Δx; f1=f(x1), f2=f(x2)")
    dec(cx, 113, "3: f1 > f2")
    proc(15, 100, "x3 = x1 − Δx", w=24); proc(65, 100, "x3 = x1 + 2Δx", w=26)
    proc(cx, 89, "3: f3 = f(x3)")
    proc(cx, 77, "4: fmin, xmin; fmax, xmax", h=8)
    proc(cx, 66, "5: a1, a2 за (2.30), (2.31)", h=8)
    dec(cx, 54, "6: a2 > 0 ?")
    proc(8, 41, "x~ = 2xmin − xmax", w=22); proc(66, 41, "x~ = −a1/(2a2) + (x1+x2)/2", w=40)
    proc(cx, 30, "6:f~ = f(x~)")
    dec(cx, 18, "7: |fmin−f~|≤σf\nі |xmin−x~|≤σx", w=44, h=14)
    proc(88, 18, "8: xmax = x~\nfmax = f~", w=20, h=10)
    data(cx, 6, "x* = x~, f(x*) = f~")
    term(cx, -2, "Кінець")

    arr((cx, 143), (cx, 139.5)); arr((cx, 132.5), (cx, 129)); arr((cx, 121), (cx, 118))
    line((cx-13, 113), (15, 113)); arr((15, 113), (15, 104))
    line((cx+13, 113), (65, 113)); arr((65, 113), (65, 104))
    ax.text(26, 115, "ні"); ax.text(53, 115, "так")
    line((15, 96), (15, 89)); arr((15, 89), (cx-17, 89))
    line((65, 96), (65, 89)); arr((65, 89), (cx+17, 89))
    arr((cx, 85), (cx, 81)); arr((cx, 73), (cx, 70)); arr((cx, 62), (cx, 59))
    line((cx-13, 54), (8, 54)); arr((8, 54), (8, 45)); ax.text(20, 56, "ні")
    line((cx+13, 54), (66, 54)); arr((66, 54), (66, 45)); ax.text(53, 56, "так")
    line((8, 37), (8, 30)); arr((8, 30), (cx-17, 30))
    line((66, 37), (66, 30)); arr((66, 30), (cx+17, 30))
    arr((cx, 26), (cx, 25))
    line((cx+22, 18), (78, 18)); ax.text(60, 20, "ні")
    line((88, 23), (88, 77)); arr((88, 77), (cx+17, 77))
    arr((cx, 11), (cx, 9.5)); ax.text(cx+1.5, 10, "так")
    arr((cx, 2.5), (cx, 1))
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print("Збережено", path)
    plt.close(fig)


# main 
if __name__ == "__main__":
    res = powell(X1, DX, SF, SX)
    print_table(res)
    print(f"\nРезультат: x* = {res['x_star']:.6f}, f(x*) = {res['f_star']:.6f}, "
          f"n = {res['n']}, Nf = {res['nf']} (перевірка Nf = n + 3 = {res['n'] + 3})")
    if not res["converged"]:
        print("УВАГА: ліміт ітерацій вичерпано, критерій не виконано!")

    # перевірка щільною сіткою (без лічильника)
    xx = np.linspace(X1 - 1, X1 + 3, 400001)
    xg = xx[np.argmin(f_plain(xx))]
    print(f"Контроль (сітка): мінімум ≈ {xg:.5f}, f ≈ {f_plain(xg):.5f}")

    # рядок для зведеної табл. 4.1
    print(f"\nЗведена табл. 4.1 -> Пауелл: x* = {res['x_star']:.6f}, "
          f"Nf = {res['nf']}, n = {res['n']}")

    draw_flowchart()
    plot_first_iteration(res)