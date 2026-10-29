import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, X0, DELTA0, EPS, SIZE
from src.image import clean_image, add_noise, save_png
from src.oracle import Oracle
from src.filters import gaussian_blur
from src.objective import make_objective, LO, HI
from src.metrics import mse, psnr
from src.lab1_svenn import svenn
from src.lab2_search import dichotomy, golden_section, TAU


def print_rows(rows, head=6, tail=1):
    print(f"{'k':>3} {'a_k':>9} {'b_k':>9} {'x1':>9} {'x2':>9} "
          f"{'f(x1)':>11} {'f(x2)':>11} {'L_k':>10}  Рішення")
    shown = rows[:head] + ([None] if len(rows) > head + tail else []) + rows[-tail:]
    for r in shown:
        if r is None:
            print("  …")
            continue
        print(f"{r['k']:>3} {r['a']:>9.6f} {r['b']:>9.6f} {r['x1']:>9.6f} {r['x2']:>9.6f} "
              f"{r['f1']:>11.8f} {r['f2']:>11.8f} {r['L']:>10.3e}  {r['decision']}")


def compression(L_end, L0, n_f):
    """Стиснення на одне звернення: (L_end / L0) ** (1 / N_f)."""
    return (L_end / L0) ** (1.0 / n_f)


def main():
    os.makedirs("outputs", exist_ok=True)

    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    oracle = Oracle(make_objective(noisy, clean))

    # ---------- відрізок зі Свенна ----------
    br = svenn(oracle, X0, DELTA0, LO, HI)
    a0, b0 = br["a"], br["b"]
    L0 = b0 - a0
    print(f"Свенн: [{a0:.4f}, {b0:.4f}]  L0 = {L0:.4f}  N_f = {br['n_f']}")
    print(f"tau = {TAU:.10f}   tau^2 = {TAU**2:.10f}   1 - tau = {1 - TAU:.10f}")

    # ---------- ручний перший крок ----------
    delta = EPS / 3
    x1d, x2d = (a0 + b0 - delta) / 2, (a0 + b0 + delta) / 2
    x1g, x2g = b0 - TAU * L0, a0 + TAU * L0
    print(f"\nдихотомія, 1-й крок: x1 = {x1d:.7f}, x2 = {x2d:.7f}, "
          f"f = {oracle.peek(x1d):.8f}, {oracle.peek(x2d):.8f}")
    print(f"золотий,   1-й крок: x1 = {x1g:.7f}, x2 = {x2g:.7f}, "
          f"f = {oracle.peek(x1g):.8f}, {oracle.peek(x2g):.8f}")

    # ---------- основні запуски (reset між методами обов'язковий!) ----------
    oracle.reset()
    res_d = dichotomy(oracle, a0, b0, EPS)
    oracle.reset()
    res_g = golden_section(oracle, a0, b0, EPS)

    print("\n===== Дихотомія =====")
    print_rows(res_d["rows"])
    print(f"x* = {res_d['x']:.6f}  f* = {res_d['f']:.8f}  "
          f"ітерацій = {res_d['iterations']}  N_f = {res_d['n_f']}")

    print("\n===== Золотий переріз =====")
    print_rows(res_g["rows"])
    print(f"x* = {res_g['x']:.6f}  f* = {res_g['f']:.8f}  "
          f"ітерацій = {res_g['iterations']}  N_f = {res_g['n_f']}")

    r1, r2 = res_g["rows"][0], res_g["rows"][1]
    print(f"\nуспадкування (крок 1→2): старе x1 = {r1['x1']:.10f}, "
          f"нове x2 = {r2['x2']:.10f}, різниця = {abs(r1['x1'] - r2['x2']):.2e}")

    # ---------- зведена таблиця ----------
    print("\nЗведена таблиця:")
    print(f"{'Метод':<16} {'σ*':>9} {'f(σ*)':>12} {'L кінець':>10} {'ітер':>5} {'N_f':>4} {'стиснення':>10}")
    for name, r in (("Дихотомія", res_d), ("Золотий переріз", res_g)):
        L_end = r["b"] - r["a"]
        print(f"{name:<16} {r['x']:>9.6f} {r['f']:>12.8f} {L_end:>10.3e} {r['iterations']:>5} "
              f"{r['n_f']:>4} {compression(L_end, L0, r['n_f']):>10.4f}")
    print(f"\nтеорія: дихотомія ≈ {math.log2(L0 / EPS):.1f},  "
          f"золотий ≈ {math.log(EPS / L0) / math.log(TAU):.1f} ітерацій")

    # ---------- відновлення зображення ----------
    sigma_star = res_g["x"]
    restored = gaussian_blur(noisy, sigma_star)
    save_png(restored, "outputs/lab2_restored.png")
    print(f"\nσ* = {sigma_star:.6f}")
    print(f"було:  MSE = {mse(noisy, clean):.8f}   PSNR = {psnr(noisy, clean):.4f} дБ")
    print(f"стало: MSE = {mse(restored, clean):.8f}   PSNR = {psnr(restored, clean):.4f} дБ")

    # ---------- графік збіжності ----------
    plt.figure(figsize=(8, 5))
    nd = [2 * r["k"] for r in res_d["rows"]]
    ng = [r["n_f"] for r in res_g["rows"]]
    plt.semilogy([0] + nd, [L0] + [r["L"] for r in res_d["rows"]], "o-", label="дихотомія")
    plt.semilogy([2] + ng, [L0] + [r["L"] for r in res_g["rows"]], "s-", label="золотий переріз")
    plt.axhline(EPS, color="gray", linestyle="--", label="ε")
    plt.xlabel("кількість звернень до оракула N_f")
    plt.ylabel("довжина відрізка L")
    plt.title("Лабораторна 2: стиснення відрізка за звернення")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("outputs/lab2_compare.png", dpi=150, bbox_inches="tight")

    # ---------- дослід 1: ε / 10 ----------
    print("\n===== Дослід 1: ε / 10 =====")
    eps2 = EPS / 10
    oracle.reset(); d2 = dichotomy(oracle, a0, b0, eps2)
    oracle.reset(); g2 = golden_section(oracle, a0, b0, eps2)
    print(f"дихотомія: ітер {res_d['iterations']} → {d2['iterations']} "
          f"(+{d2['iterations'] - res_d['iterations']}), N_f {res_d['n_f']} → {d2['n_f']}")
    print(f"золотий:   ітер {res_g['iterations']} → {g2['iterations']} "
          f"(+{g2['iterations'] - res_g['iterations']}), N_f {res_g['n_f']} → {g2['n_f']}")
    print(f"теорія: +{math.log2(10):.1f} (дихотомія), +{math.log(10) / math.log(1 / TAU):.1f} (золотий)")

    # ---------- дослід 2: delta = eps ----------
    print("\n===== Дослід 2: δ = ε =====")
    try:
        dichotomy(oracle, a0, b0, EPS, delta=EPS)
    except ValueError as e:
        print("із перевіркою:", e)
    oracle.reset()
    bad = dichotomy(oracle, a0, b0, EPS, delta=EPS, max_iter=60, strict=False)
    print(f"без перевірки (max_iter=60): ітерацій = {bad['iterations']}, "
          f"L = {bad['b'] - bad['a']:.6e}, ε = {EPS:.0e}, N_f = {bad['n_f']}")
    print("останні L:", [f"{r['L']:.6e}" for r in bad["rows"][-3:]])

    # ---------- дослід 3: ручний підбір ----------
    print("\n===== Дослід 3: ручний підбір σ =====")
    for s in (0.3, 0.5, 0.8, 1.5, sigma_star):
        img = gaussian_blur(noisy, s)
        print(f"σ = {s:.4f}   MSE = {mse(img, clean):.8f}   PSNR = {psnr(img, clean):.4f} дБ")


if __name__ == "__main__":
    main()