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


def fmt(x):
    """Число як у таблицях сайту: 4 знаки після коми, без зайвих нулів (0.008, 0.712)."""
    if x == 0:
        return "0"
    s = f"{x:.5f}" if abs(x) < 1e-4 else f"{x:.4f}"
    return s.rstrip("0").rstrip(".")


def site_decision(text):
    """Текст рішення в тому самому записі, що й на сайті."""
    text = text.replace("→", "->").replace("успадкували", "успадковано")
    return " ".join(text.split())


def print_full_table(rows, golden=False):
    """Друкує ВСІ рядки таблиці ітерацій (у форматі сайту)."""
    head = (f"{'k':>3} {'a_k':>8} {'b_k':>8} {'x1':>8} {'x2':>8} "
            f"{'f(x1)':>8} {'f(x2)':>8} {'L_k':>8}")
    if golden:
        head += f" {'Нова точка':>10}"
    print(head + "  Рішення")
    for r in rows:
        line = (f"{r['k']:>3} {fmt(r['a']):>8} {fmt(r['b']):>8} {fmt(r['x1']):>8} {fmt(r['x2']):>8} "
                f"{fmt(r['f1']):>8} {fmt(r['f2']):>8} {fmt(r['L']):>8}")
        if golden:
            line += f" {fmt(r['new']):>10}"
        print(line + "  " + site_decision(r["decision"]))


def compression(L_start, L_end, n_f):
    """Стиснення на одне звернення: (L_кінець / L_початок) ** (1 / N_f)."""
    return (L_end / L_start) ** (1.0 / n_f)


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
    print(f"Початковий інтервал з лаб. 1: [a0, b0] = [{a0:.6f}, {b0:.6f}], точність ε = {EPS:.0e}")
    print(f"tau = {TAU:.10f}   tau^2 = {TAU**2:.10f}   1 - tau = {1 - TAU:.10f}")

    # ---------- ручний перший крок ----------
    delta = EPS / 3
    x1d, x2d = (a0 + b0 - delta) / 2, (a0 + b0 + delta) / 2
    x1g, x2g = b0 - TAU * L0, a0 + TAU * L0
    print(f"\nдихотомія, 1-й крок: x1 = {x1d:.7f}, x2 = {x2d:.7f}, "
          f"f = {oracle.peek(x1d):.8f}, {oracle.peek(x2d):.8f}")
    print(f"золотий,   1-й крок: x1 = {x1g:.7f}, x2 = {x2g:.7f}, "
          f"f = {oracle.peek(x1g):.8f}, {oracle.peek(x2g):.8f}")

    # ---------- основні запуски ----------
    oracle.reset()
    res_d = dichotomy(oracle, a0, b0, EPS)
    oracle.reset()
    res_g = golden_section(oracle, a0, b0, EPS)

    print("\n===== Дихотомія =====")
    print_full_table(res_d["rows"])
    print(f"x* = {res_d['x']:.6f}  f* = {res_d['f']:.8f}  ітерацій = {res_d['iterations']}  N_f = {res_d['n_f']}")

    print("\n===== Золотий переріз =====")
    print_full_table(res_g["rows"], golden=True)
    print(f"x* = {res_g['x']:.6f}  f* = {res_g['f']:.8f}  ітерацій = {res_g['iterations']}  N_f = {res_g['n_f']}")

    # перевірка успадкування: x2 (нова) == x1 (стара)
    r1, r2 = res_g["rows"][0], res_g["rows"][1]
    print(f"\nуспадкування (крок 1→2): старе x1 = {r1['x1']:.10f}, нове x2 = {r2['x2']:.10f}, "
          f"різниця = {abs(r1['x1'] - r2['x2']):.2e}")

    # ---------- порівняння методів ----------
    nf_d, nf_g = res_d["n_f"], res_g["n_f"]
    saving = 100.0 * (nf_d - nf_g) / nf_d
    print("\nσ* (золотий переріз) =", f"{res_g['x']:.6f}")
    print(f"N_f дихотомії = {nf_d}     N_f золотого перерізу = {nf_g}  ({nf_g - nf_d:+d} звернень)     Економія = {saving:.1f}%")

    print("\nПорівняння методів:")
    print(f"{'Метод':<16} {'x*':>8} {'f(x*)':>8} {'Довжина [a,b]':>14} {'Ітерацій':>9} {'N_f':>5} "
          f"{'Стиснення (від L1)':>19} {'Стиснення (від L0)':>19}")
    for name, r in (("Дихотомія", res_d), ("Золотий переріз", res_g)):
        L_end = r["b"] - r["a"]
        L1 = r["rows"][0]["L"]
        print(f"{name:<16} {fmt(r['x']):>8} {fmt(r['f']):>8} {fmt(L_end):>14} {r['iterations']:>9} {r['n_f']:>5} "
              f"{compression(L1, L_end, r['n_f']):>19.4f} {compression(L0, L_end, r['n_f']):>19.4f}")
    print(f"\nтеорія: дихотомія ≈ {math.log2(L0 / EPS):.1f},  золотий ≈ {math.log(EPS / L0) / math.log(TAU):.1f} ітерацій")

    print(f"\nВисновок лаб. 2. Золотий переріз досяг тієї самої точності за {nf_g} звернень проти {nf_d} "
          f"у дихотомії (економія {saving:.1f}%), хоча виконав більше ітерацій "
          f"({res_g['iterations']} проти {res_d['iterations']}). Мірою вартості методу є "
          f"оракульна складність, а не кількість ітерацій.")

    # ---------- зображення ----------
    sigma_star = res_g["x"]
    restored = gaussian_blur(noisy, sigma_star)
    save_png(restored, "outputs/lab2_restored.png")
    print(f"\nσ* = {sigma_star:.6f}")
    print(f"було:  MSE = {mse(noisy, clean):.8f}   PSNR = {psnr(noisy, clean):.4f} дБ")
    print(f"стало: MSE = {mse(restored, clean):.8f}   PSNR = {psnr(restored, clean):.4f} дБ")

    # ---------- графік: L_k за ітераціями ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(len(res_d["rows"])), [r["L"] for r in res_d["rows"]],
                "o-", ms=4, lw=1.5, label=f"Дихотомія (N_f = {res_d['n_f']})")
    ax.semilogy(range(len(res_g["rows"])), [r["L"] for r in res_g["rows"]],
                "s--", ms=4, lw=1.5, label=f"Золотий переріз (N_f = {res_g['n_f']})")
    ax.axhline(EPS, color="gray", lw=1, ls=":", label=f"ε = {EPS:.0e}")
    ax.set_title("Лабораторна 2: скорочення довжини інтервалу L_k")
    ax.set_xlabel("k — номер ітерації")
    ax.set_ylabel("L_k — довжина інтервалу (логарифмічна шкала)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig("outputs/lab2_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- дослід 1: eps / 10 ----------
    print("\n===== Дослід 1: ε / 10 =====")
    eps2 = EPS / 10
    oracle.reset(); d2 = dichotomy(oracle, a0, b0, eps2)
    oracle.reset(); g2 = golden_section(oracle, a0, b0, eps2)
    print(f"дихотомія: ітер {res_d['iterations']} → {d2['iterations']} (+{d2['iterations'] - res_d['iterations']}), "
          f"N_f {res_d['n_f']} → {d2['n_f']}")
    print(f"золотий:   ітер {res_g['iterations']} → {g2['iterations']} (+{g2['iterations'] - res_g['iterations']}), "
          f"N_f {res_g['n_f']} → {g2['n_f']}")
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