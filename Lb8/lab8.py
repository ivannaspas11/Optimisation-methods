"""
Лабораторна робота: градієнтні методи зі сферичною, кубічною
та октаедричною метриками.
"""

from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


# Вхідні дані

X0 = np.array([-3.0, -2.0], dtype=float)
EPS = 0.01
H0_VALUES = (1.0, 0.7)
FD_H = 1e-5
H_MIN = 1e-10
MAX_ITER = 10000
OUTPUT_DIR = Path("lab_results")


#  Цільова функція і градієнт

def F(x):
    """Цільова функція."""
    x1, x2 = x
    return x1**2 + 2*x2**2 + 2*x1*x2 - 9*x1 - 13*x2


def grad(x):
    """Аналітичний градієнт: (dF/dx1, dF/dx2)."""
    x1, x2 = x
    return np.array([
        2*x1 + 2*x2 - 9,
        2*x1 + 4*x2 - 13
    ], dtype=float)


def finite_difference_gradient(x, h=FD_H):
    """Центральні скінченні різниці з PDF."""
    result = np.zeros(2, dtype=float)
    for i in range(2):
        ei = np.zeros(2, dtype=float)
        ei[i] = 1.0
        result[i] = (F(x + h*ei) - F(x - h*ei)) / (2*h)
    return result


# Напрямки трьох метрик 

def direction(g, metric):
    """
    Напрямок спуску з презентації:
    Сферична — L2; Кубічна — L-infinity; Октаедрична — L1.
    """
    if metric == "Сферична":
        norm = np.linalg.norm(g)
        return -g / norm
    if metric == "Кубічна":
        return -np.sign(g)
    if metric == "Октаедрична":
        # Якщо модулі похідних рівні, np.argmax обирає першу координату.
        j = int(np.argmax(np.abs(g)))
        d = np.zeros_like(g)
        d[j] = -np.sign(g[j])
        return d
    raise ValueError(f"Невідома метрика: {metric}")


#  Один прогін 

def run_method(metric, h0):
    x = X0.copy()
    fx = F(x)
    h = float(h0)

    # NF — кількість обчислень F; Ngrad — кількість обчислень градієнта.
    nF = 1  # початкове F(x0)
    nG = 0
    n_steps = 0
    total_splits = 0
    rows = []
    trajectory = [x.copy()]

    for _ in range(MAX_ITER):
        g = grad(x)
        nG += 1
        norm_g = float(np.linalg.norm(g))

        # Критерій зупинки з PDF: евклідова норма градієнта <= epsilon.
        if norm_g <= EPS:
            rows.append({
                "k": n_steps + 1,
                "x1": x[0], "x2": x[1], "F(xk)": fx,
                "grad_x1": g[0], "grad_x2": g[1], "norm_grad": norm_g,
                "d1": "", "d2": "", "h_accepted": h,
                "splits": 0, "F(xk+1)": ""
            })
            break

        d = direction(g, metric)
        h_trial = h
        splits = 0

        # Дробимо крок лише тоді, коли F у пробній точці не зменшується.
        while True:
            xb = x + h_trial*d
            fb = F(xb)
            nF += 1
            if fb < fx:
                break
            h_trial /= 2.0
            splits += 1
            total_splits += 1
            if h_trial < H_MIN:
                raise RuntimeError(
                    f"Крок став меншим за {H_MIN:g}: метрика={metric}, h0={h0}"
                )

        rows.append({
            "k": n_steps + 1,
            "x1": x[0], "x2": x[1], "F(xk)": fx,
            "grad_x1": g[0], "grad_x2": g[1], "norm_grad": norm_g,
            "d1": d[0], "d2": d[1], "h_accepted": h_trial,
            "splits": splits, "F(xk+1)": fb
        })

        # Крок 6: приймаємо точку. h зберігається і ніколи не повертається до h0.
        x = xb
        fx = fb
        h = h_trial
        n_steps += 1
        trajectory.append(x.copy())
    else:
        raise RuntimeError(f"Перевищено MAX_ITER для {metric}, h0={h0}")

    x_exact = np.array([2.5, 2.0])
    error = float(np.linalg.norm(x - x_exact))

    return {
        "metric": metric,
        "h0": h0,
        "x": x,
        "F": fx,
        "n": n_steps,
        "NF": nF,
        "Ngrad": nG,
        "splits": total_splits,
        "error": error,
        "grad_norm": float(np.linalg.norm(grad(x))),
        "rows": rows,
        "trajectory": np.array(trajectory)
    }



def save_iteration_csv(result, path):
    fields = [
        "k", "x1", "x2", "F(xk)", "grad_x1", "grad_x2", "norm_grad",
        "d1", "d2", "h_accepted", "splits", "F(xk+1)"
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])


def print_iteration_table(result):
    print(f"\n{'='*110}")
    print(f"{result['metric']} метрика, h0 = {result['h0']}")
    print(" k |       x^k=(x1,x2)       |       F(xk) |          grad F          | norm grad |       d       |    h    | дробл. | F(xk+1)")
    print("-"*110)
    for r in result["rows"]:
        x_text = f"({r['x1']: .6f}, {r['x2']: .6f})"
        g_text = f"({r['grad_x1']: .6f}, {r['grad_x2']: .6f})"
        if r["d1"] == "":
            d_text = "-"
            f_next = "-"
        else:
            d_text = f"({r['d1']: .4f}, {r['d2']: .4f})"
            f_next = f"{r['F(xk+1)']:.8f}"
        print(
            f"{r['k']:2d} | {x_text:23s} | {r['F(xk)']:12.6f} | "
            f"{g_text:25s} | {r['norm_grad']:9.5f} | {d_text:14s} | "
            f"{r['h_accepted']:7.5f} | {r['splits']:6d} | {f_next}"
        )
    print(
        f"Результат: x* = ({result['x'][0]:.8f}, {result['x'][1]:.8f}), "
        f"F = {result['F']:.8f}, ||grad F||2 = {result['grad_norm']:.8f}"
    )


def save_summary_csv(results, path):
    fields = ["Метрика", "h0", "n", "NF", "Ngrad", "Дроблень",
              "x1_final", "x2_final", "F_final", "||x-xточн||2",
              "||grad F||2"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Метрика": r["metric"], "h0": r["h0"], "n": r["n"],
                "NF": r["NF"], "Ngrad": r["Ngrad"], "Дроблень": r["splits"],
                "x1_final": r["x"][0], "x2_final": r["x"][1],
                "F_final": r["F"], "||x-xточн||2": r["error"],
                "||grad F||2": r["grad_norm"]
            })


def plot_trajectories(results, h0, path):
    """Лінії рівня F та траєкторії трьох метрик для одного h0."""
    chosen = [r for r in results if math.isclose(r["h0"], h0)]
    all_points = np.vstack([r["trajectory"] for r in chosen] + [X0, [2.5, 2.0]])
    xmin, xmax = all_points[:, 0].min() - 1.5, all_points[:, 0].max() + 1.5
    ymin, ymax = all_points[:, 1].min() - 1.5, all_points[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 350),
                         np.linspace(ymin, ymax, 350))
    zz = xx**2 + 2*yy**2 + 2*xx*yy - 9*xx - 13*yy

    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(-24.0, max(zz.max() * 0.25, 10), 18)
    contours = ax.contour(xx, yy, zz, levels=levels, cmap="viridis", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    markers = ["o", "s", "^"]
    for r, marker in zip(chosen, markers):
        p = r["trajectory"]
        ax.plot(p[:, 0], p[:, 1], marker=marker, markersize=3,
                linewidth=1.8, label=f"{r['metric']} (n={r['n']})")

    ax.scatter([X0[0]], [X0[1]], marker="*", s=150, color="black",
               label="Початкова точка (-3; -2)", zorder=5)
    ax.scatter([2.5], [2.0], marker="X", s=100, color="red",
               label="Точний мінімум (2.5; 2)", zorder=5)
    ax.set_title(f"Лінії рівня F та траєкторії методів, h0 = {h0}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)



def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("ЛАБОРАТОРНА РОБОТА: ГРАДІЄНТНІ МЕТОДИ")
    print("F(x1,x2) = x1^2 + 2*x2^2 + 2*x1*x2 - 9*x1 - 13*x2")
    print(f"x0 = {tuple(X0)}, epsilon = {EPS}, h0 = {H0_VALUES}")
    print("\nАналітичні похідні:")
    print("dF/dx1 = 2*x1 + 2*x2 - 9")
    print("dF/dx2 = 2*x1 + 4*x2 - 13")
    print("Точний мінімум: x* = (2.5, 2.0), F(x*) = -24.25")

    g_analytic = grad(X0)
    g_numeric = finite_difference_gradient(X0)


    print("\nПеревірка центральними скінченними різницями при h = 10^-5:")
    print(f"Точка перевірки: {tuple(X0)}")
    print(f"Аналітичний градієнт: {g_analytic}")
    print(f"Чисельний градієнт:   {g_numeric}")
    print(f"Абсолютна похибка:    {np.abs(g_analytic - g_numeric)}")

    results = []
    metrics = ("Сферична", "Кубічна", "Октаедрична")
    for h0 in H0_VALUES:
        for metric in metrics:
            result = run_method(metric, h0)
            results.append(result)
            safe_metric = {"Сферична": "sphere", "Кубічна": "cube",
                           "Октаедрична": "octa"}[metric]
            csv_path = OUTPUT_DIR / f"iterations_{safe_metric}_h0_{h0}.csv"
            save_iteration_csv(result, csv_path)
            print_iteration_table(result)

        plot_trajectories(
            results, h0, OUTPUT_DIR / f"trajectories_h0_{h0}.png"
        )

    summary_path = OUTPUT_DIR / "summary.csv"
    save_summary_csv(results, summary_path)

    print("\n" + "="*90)
    print("ЗВЕДЕНА ТАБЛИЦЯ")
    print(f"{'Метрика':14s} {'h0':>5s} {'n':>6s} {'NF':>6s} {'Ngrad':>7s} "
          f"{'||x-xточн||2':>16s} {'||grad F||2':>14s}")
    print("-"*90)
    for r in results:
        print(f"{r['metric']:14s} {r['h0']:5.1f} {r['n']:6d} {r['NF']:6d} "
              f"{r['Ngrad']:7d} {r['error']:16.8e} {r['grad_norm']:14.8e}")

    print("\nФайли збережено в папці:", OUTPUT_DIR.resolve())
    print(" - summary.csv: зведена таблиця для всіх 6 запусків")
    print(" - iterations_*.csv: таблиці ітерацій")
    print(" - trajectories_h0_1.0.png і trajectories_h0_0.7.png: графіки")


if __name__ == "__main__":
    main()
