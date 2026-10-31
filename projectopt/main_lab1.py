import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # зберігаємо графік у файл
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, X0, DELTA0, SIZE
from src.image import clean_image, add_noise
from src.oracle import Oracle
from src.filters import gaussian_blur, gaussian_kernel
from src.objective import make_objective, LO, HI
from src.metrics import mse, psnr
from src.lab1_svenn import svenn


def print_table(result):
    print(f"{'k':>2} {'x_k':>9} {'f(x_k)':>12} {'Delta_k':>9} {'p':>2}  Коментар")
    for r in result["rows"]:
        print(f"{r['k']:>2} {r['x']:>9.4f} {r['f']:>12.8f} {r['delta']:>9.4f} {r['p']:>2}  {r['comment']}")
    print(f"\nвідрізок = [{result['a']:.4f}, {result['b']:.4f}]")
    print(f"довжина  = {result['b'] - result['a']:.4f}")
    print(f"N_f      = {result['n_f']}")


def main():
    os.makedirs("outputs", exist_ok=True)

    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    print(f"MSE шумного = {mse(noisy, clean):.8f}   PSNR = {psnr(noisy, clean):.4f} дБ")

    # ---------- перевірки фільтра ----------
    flat = np.full((SIZE, SIZE), 0.5)
    print("\nрівне поле: max|blur - 0.5| =", np.abs(gaussian_blur(flat, 0.4431) - 0.5).max())
    kernel, r = gaussian_kernel(X0)
    print(f"ядро для sigma = {X0}: радіус {r}, ваги {np.round(kernel, 6)}")
    print(f"піксель (62,62) після розмиття: {gaussian_blur(noisy, X0)[62][62]:.6f}")

    # ---------- цільова функція в кількох точках ----------
    f = make_objective(noisy, clean)
    oracle = Oracle(f)
    print("\nf(sigma) у контрольних точках:")
    for s in (0.05, X0 - DELTA0, X0, X0 + DELTA0, 1.0, 2.0, 8.0):
        print(f"  sigma = {s:.4f}   f = {oracle.peek(s):.8f}")

    # ---------- основний запуск Свенна ----------
    print("\n===== Свенн: основний запуск =====")
    oracle.reset()
    result = svenn(oracle, X0, DELTA0, LO, HI)
    print_table(result)

    # ---------- графік ----------
    grid = np.linspace(0.05, 8.0, 200)
    values = [oracle.peek(s) for s in grid]           # peek не рахується
    plt.figure(figsize=(9, 5))
    plt.plot(grid, values, label="f(σ)")
    plt.scatter([r["x"] for r in result["rows"]], [r["f"] for r in result["rows"]],
                color="red", zorder=5, label="пробні точки Свенна")
    plt.axvspan(result["a"], result["b"], alpha=0.2, color="green",
                label=f"відрізок [{result['a']:.3f}, {result['b']:.3f}]")
    plt.xscale("log")
    plt.xlabel("σ — сила розмиття")
    plt.ylabel("f(σ) = MSE")
    plt.title("Лабораторна 1: локалізація мінімуму методом Свенна")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("outputs/lab1_svenn.png", dpi=150, bbox_inches="tight")
    print("\nграфік збережено: outputs/lab1_svenn.png")

    # ---------- три обов'язкові досліди ----------
    experiments = [
        ("Дослід 1: Δ0 у 10 разів менше", X0, DELTA0 / 10),
        ("Дослід 2: Δ0 у 10 разів більше", X0, DELTA0 * 10),
        ("Дослід 3: x0 = 3.0",             3.0, DELTA0),
    ]
    for title, x0, d0 in experiments:
        print(f"\n===== {title} (x0={x0}, Δ0={d0:.5f}) =====")
        oracle.reset()
        res = svenn(oracle, x0, d0, LO, HI)
        print_table(res)


if __name__ == "__main__":
    main()