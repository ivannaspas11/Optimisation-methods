import math
import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lr7_output")
os.makedirs(OUT, exist_ok=True)

A0, B0, EPS = 2.36, 2.42, 1e-3
X_EXACT = (1.2 + math.sqrt(512.44)) / 10


def f(x):
    return (x + 1) ** 3 * (x + 2.5) * (x - 3.3) + 3.5


def df(x):
    return (x + 1) ** 2 * (5 * x ** 2 - 1.2 * x - 25.55)


def d2f(x):
    return (x + 1) * (20 * x ** 2 + 6.4 * x - 52.3)


def secant(a0, b0, eps, pure=False, max_iter=100):
    xm, xk = a0, b0
    pm, pk = df(xm), df(xk)
    calls = 2
    if pm * pk >= 0:
        raise ValueError("f'(x0)*f'(x1) >= 0: корінь не локалізовано")
    rows = []
    for k in range(1, max_iter + 1):
        if abs(pk - pm) < 1e-14:
            raise ZeroDivisionError("f'(xk) = f'(xk-1): січна горизонтальна")
        xn = (pk * xm - pm * xk) / (pk - pm)
        pn = df(xn)
        calls += 1
        
        if pure:
            action = "x_{k-1} ← x_k, x_k ← x_{k+1}"
        else:
            action = "x_k ← x_{k+1}" if pn > 0 else "x_{k-1} ← x_{k+1}"
            
        rows.append(dict(k=k, xm=xm, pm=pm, xk=xk, pk=pk, xn=xn, pn=pn,
                         fn=f(xn), err=abs(xn - X_EXACT), action=action))
        
        if abs(pn) <= eps:
            rows[-1]["action"] = "|f'| <= eps → СТОП"
            return rows, xn, calls
            
        if pure:
            xm, pm, xk, pk = xk, pk, xn, pn
        elif pn > 0:
            xk, pk = xn, pn
        else:
            xm, pm = xn, pn
            
    return rows, xn, calls


def newton(x0, eps, max_iter=100):
    x, n, c1, c2 = x0, 0, 0, 0
    while True:
        g = df(x); c1 += 1
        if abs(g) <= eps:
            return x, n, c1, c2
        x = x - g / d2f(x); c2 += 1
        n += 1
        if n >= max_iter:
            return x, n, c1, c2


HEAD = ["k", "x_{k-1}", "f'(x_{k-1})", "x_k", "f'(x_k)", "x_{k+1}",
        "f'(x_{k+1})", "f(x_{k+1})", "|x_{k+1}-x_ex|", "Дія"]


def print_table(rows, title):
    print("\n" + title)
    w = [3, 11, 13, 11, 13, 11, 13, 13, 14, 28]
    print(" ".join(h.center(wi) for h, wi in zip(HEAD, w)))
    print("-" * (sum(w) + len(w)))
    for r in rows:
        v = [f"{r['k']}", f"{r['xm']:.6f}", f"{r['pm']:.6f}", f"{r['xk']:.6f}",
             f"{r['pk']:.6f}", f"{r['xn']:.6f}", f"{r['pn']:.6f}",
             f"{r['fn']:.6f}", f"{r['err']:.1e}", r["action"]]
        print(" ".join(s.rjust(wi) if i < 9 else s.ljust(wi)
                       for i, (s, wi) in enumerate(zip(v, w))))





def save_table_png(rows, name, title):
    fig, ax = plt.subplots(figsize=(15, 0.6 + 0.45 * (len(rows) + 2)))
    ax.axis("off")
    cells = [[str(r["k"]), f"{r['xm']:.6f}", f"{r['pm']:.6f}", f"{r['xk']:.6f}",
              f"{r['pk']:.6f}", f"{r['xn']:.6f}", f"{r['pn']:.6f}",
              f"{r['fn']:.6f}", r["action"].replace("x_{k-1}", "$x_{k-1}$").replace("x_k", "$x_k$").replace("x_{k+1}", "$x_{k+1}$")] for r in rows]
    cols = ["k", "$x_{k-1}$", "$f'(x_{k-1})$", "$x_k$", "$f'(x_k)$",
            "$x_{k+1}$", "$f'(x_{k+1})$", "$f(x_{k+1})$", "Дія"]
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    t.auto_set_column_width(list(range(len(cols))))
    ax.set_title(title, fontsize=11)
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_phi(rows):
    r1, r2 = rows[0], rows[1]
    xs = np.linspace(A0, B0, 400)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, df(xs), "b", lw=2, label=r"$\varphi(x)=f'(x)$")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(X_EXACT, color="r", ls=":", label=f"$x^*$={X_EXACT:.6f}")
    
    ax.plot([r1["xm"], r1["xk"]], [r1["pm"], r1["pk"]], "o-", color="orange", lw=1.8, label="січна 1: $x_0,x_1$")
    ax.plot([r1["xn"]], [0], "s", color="orange")
    
    ax.plot([r2["xm"], r2["xk"]], [r2["pm"], r2["pk"]], "o-", color="green", lw=1.8, label="січна 2: $x_k$-пара після перепозначення")
    ax.plot([r2["xn"]], [0], "s", color="green")
    
    for r, c, n in ((r1, "orange", 2), (r2, "green", 3)):
        ax.annotate(f"$x_{n}$={r['xn']:.6f}", (r["xn"], 0), textcoords="offset points", xytext=(0, -18 if n == 2 else 12), ha="center", color=c)
        
    ax.annotate(f"$\\varphi(x_0)$={r1['pm']:.3f}", (r1["xm"], r1["pm"]), textcoords="offset points", xytext=(5, 6))
    ax.annotate(f"$\\varphi(x_1)$={r1['pk']:.3f}", (r1["xk"], r1["pk"]), textcoords="offset points", xytext=(-70, -4))
    
    ax.set_xlabel("x"); ax.set_ylabel(r"$\varphi(x)=f'(x)$")
    ax.set_title(r"ЛР 7: графік $\varphi(x)=f'(x)$ на $[a_0,b_0]$ з першими двома січними")
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    
    axins = ax.inset_axes([0.55, 0.05, 0.4, 0.4]) 
    axins.plot(xs, df(xs), "b", lw=2)
    axins.axhline(0, color="k", lw=0.8)
    axins.axvline(X_EXACT, color="r", ls=":")
    axins.plot([r1["xm"], r1["xk"]], [r1["pm"], r1["pk"]], "o-", color="orange", lw=1.8)
    axins.plot([r1["xn"]], [0], "s", color="orange")
    axins.plot([r2["xm"], r2["xk"]], [r2["pm"], r2["pk"]], "o-", color="green", lw=1.8)
    axins.plot([r2["xn"]], [0], "s", color="green")
    axins.set_xlim(X_EXACT - 0.005, X_EXACT + 0.005)
    axins.set_ylim(-0.25, 0.25)
    axins.grid(alpha=0.3)
    axins.set_title("Масштаб біля $x^*$", fontsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="black")


    for r, c, n in ((r1, "orange", 2), (r2, "green", 3)):
        ax.annotate(f"$x_{n}$={r['xn']:.6f}", (r["xn"], 0), 
                    textcoords="offset points", 
                    xytext=(20, -15 if n == 2 else 15), 
                    ha="center", color=c)

    fig.savefig(os.path.join(OUT, "graph_phi_secants.png"), dpi=170, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    print(f"f'(a0)={df(A0):.6f}, f'(b0)={df(B0):.6f}, добуток={df(A0) * df(B0):.4f} < 0: {df(A0) * df(B0) < 0}")
    print(f"Точний мінімум: x_ex = {X_EXACT:.8f}, f(x_ex) = {f(X_EXACT):.8f}")

    rows_b, xb, cb = secant(A0, B0, EPS, pure=False)
    rows_p, xp, cp = secant(A0, B0, EPS, pure=True)

    print_table(rows_b, "Табл. 4.11 — метод січних за схемою посібника (з перепозначенням за знаком)")
    print_table(rows_p, "Експеримент — «чисті» січні (дві останні точки)")
    save_table_png(rows_b, "table_4_11_book.png", "Табл. 4.11 — метод січних (схема посібника)")
    save_table_png(rows_p, "table_pure.png", "«Чисті» січні")

    xn_, nn, n1, n2 = newton(A0, EPS)
    
    summary = [
        ("Ньютон, ε=0,001 (ЛР 6)", nn, "3 f' + 2 f'' (у звіті 3)", xn_),
        ("Ньютон-Рафсон із кроком, α=0,25 (ЛР 6)", 1, "2 f + 3 f' + 2 f''", 2.38371394),
        ("Січні за посібником, ε=0,001 (ЛР 7)", len(rows_b), f"{cb} f'", xb),
        ("«Чисті» січні, ε=0,001 (ЛР 7, експеримент)", len(rows_p), f"{cp} f'", xp),
    ]
    
    print("\nЗведена табл. 4.1")
    print(f"{'Метод':<48}{'Ітер.':>6}  {'Звернень':<22}{'x*':>12}{'|x*-x_ex|':>12}")
    for name, n, calls, x in summary:
        print(f"{name:<48}{n:>6}  {calls:<22}{x:>12.6f}{abs(x - X_EXACT):>12.1e}")
    plot_phi(rows_b)
    print(f"\nФайли збережено в: {OUT}")


if __name__ == "__main__":
    main()