import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, X0, DELTA0, EPS, SIMPLEX_EDGE, SIZE
from src.image import clean_image, add_noise, save_png
from src.oracle import Oracle
from src.filters import gaussian_blur, bilateral_filter
from src.objective import make_objective, LO, HI
from src.objective2 import make_objective2
from src.metrics import mse, psnr
from src.lab1_svenn import svenn
from src.lab2_search import golden_section
from src.lab3_nelder_mead import nelder_mead
from src.lab3_powell import powell


def main():
    os.makedirs("outputs", exist_ok=True)

    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)

    # ---------- σ* з лабораторної 2 (Свенн + золотий переріз) ----------
    o1 = Oracle(make_objective(noisy, clean))
    br = svenn(o1, X0, DELTA0, LO, HI)
    o1.reset()
    g = golden_section(o1, br["a"], br["b"], EPS)
    sigma_star = g["x"]
    psnr_lab2 = psnr(gaussian_blur(noisy, sigma_star), clean)
    print(f"σ* (лаб. 2) = {sigma_star:.6f}   PSNR лаб. 2 = {psnr_lab2:.4f} дБ")

    # ---------- перевірка: bilateral при великій σr ≈ гаусів ----------
    b_big = bilateral_filter(noisy, 1.0, 100.0)
    g_1 = gaussian_blur(noisy, 1.0)
    print(f"\nbilateral(σr=100) vs gaussian(σ=1): max|diff| = {np.abs(b_big - g_1).max():.4f}, "
          f"MSE між ними = {mse(b_big, g_1):.2e}")

    # ---------- початкова точка ----------
    theta0 = np.array([min(max(2 * sigma_star, 0.5), 4.0),
                       min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)])
    f2 = make_objective2(noisy, clean)
    oracle = Oracle(f2)
    print(f"\nθ0 = ({theta0[0]:.6f}, {theta0[1]:.6f})   f(θ0) = {oracle.peek(theta0):.8f}")
    print(f"початковий симплекс: x0 = ({theta0[0]:.4f}, {theta0[1]:.4f}), "
          f"x1 = ({theta0[0] + SIMPLEX_EDGE:.4f}, {theta0[1]:.4f}), "
          f"x2 = ({theta0[0]:.4f}, {theta0[1] + 0.4 * SIMPLEX_EDGE:.4f})")

    # ---------- Нелдер—Мід ----------
    oracle.reset()
    nm = nelder_mead(oracle, theta0, SIMPLEX_EDGE)
    print("\n===== Нелдер—Мід =====")
    print(f"{'k':>3}  {'операція':<20} {'σs':>8} {'σr':>8} {'f':>12} {'розмір':>9} {'N_f':>4}")
    for r in nm["rows"][:6] + nm["rows"][-1:]:
        print(f"{r['k']:>3}  {r['op']:<20} {r['x'][0]:>8.4f} {r['x'][1]:>8.4f} "
              f"{r['f']:>12.8f} {r['size']:>9.5f} {r['n_f']:>4}")
    print(f"θ* = ({nm['x'][0]:.6f}, {nm['x'][1]:.6f})  f* = {nm['f']:.10f}  "
          f"ітерацій = {nm['iterations']}  N_f = {nm['n_f']}")

    # ---------- Пауелл ----------
    oracle.reset()
    stats = []
    pw = powell(oracle, theta0, stats=stats)
    print("\n===== Пауелл =====")
    print(f"{'цикл':>4} {'σs':>8} {'σr':>8} {'f':>12} {'|зсув|':>9} {'α1':>9} {'α2':>9} {'N_f':>4}")
    for r in pw["rows"]:
        print(f"{r['cycle']:>4} {r['x'][0]:>8.4f} {r['x'][1]:>8.4f} {r['f']:>12.8f} "
              f"{r['shift']:>9.6f} {r['alpha1']:>+9.5f} {r['alpha2']:>+9.5f} {r['n_f']:>4}")
    print(f"θ* = ({pw['x'][0]:.6f}, {pw['x'][1]:.6f})  f* = {pw['f']:.10f}  "
          f"циклів = {pw['cycles']}  N_f = {pw['n_f']}")

    # ---------- порівняння ----------
    print("\nЗведена таблиця:")
    print(f"{'Метод':<14} {'σs*':>9} {'σr*':>9} {'f(θ*)':>13} {'ітер':>5} {'N_f':>5}")
    print(f"{'Нелдер—Мід':<14} {nm['x'][0]:>9.6f} {nm['x'][1]:>9.6f} {nm['f']:>13.10f} {nm['iterations']:>5} {nm['n_f']:>5}")
    print(f"{'Пауелл':<14} {pw['x'][0]:>9.6f} {pw['x'][1]:>9.6f} {pw['f']:>13.10f} {pw['cycles']:>5} {pw['n_f']:>5}")
    print(f"Пауелл: у {nm['iterations'] / pw['cycles']:.1f} раза менше циклів, "
          f"але у {pw['n_f'] / nm['n_f']:.1f} раза більше звернень")

    # ---------- відновлення ----------
    theta = nm["x"] if nm["f"] <= pw["f"] else pw["x"]
    restored = bilateral_filter(noisy, theta[0], theta[1])
    save_png(restored, "outputs/lab3_restored.png")
    psnr_lab3 = psnr(restored, clean)
    print(f"\nθ = ({theta[0]:.6f}, {theta[1]:.6f})")
    print(f"MSE  = {mse(restored, clean):.8f}")
    print(f"PSNR = {psnr_lab3:.4f} дБ")
    p0 = psnr(noisy, clean)
    print(f"\n{'Етап':<34} {'PSNR, дБ':>9} {'Приріст':>8}")
    print(f"{'зіпсоване':<34} {p0:>9.4f} {'—':>8}")
    print(f"{'лаб. 2, гаусів':<34} {psnr_lab2:>9.4f} {psnr_lab2 - p0:>+8.2f}")
    print(f"{'лаб. 3, білатеральний':<34} {psnr_lab3:>9.4f} {psnr_lab3 - psnr_lab2:>+8.2f}")

    # ---------- лінії рівня ----------
    ss = np.linspace(0.3, 5.0, 40)
    sr = np.linspace(0.01, 1.0, 40)
    grid = np.array([[oracle.peek([a, b]) for a in ss] for b in sr])   # peek!
    plt.figure(figsize=(9, 6))
    levels = np.unique(np.quantile(grid, np.linspace(0.0, 0.6, 25)))
    plt.contourf(ss, sr, grid, levels=levels, alpha=0.75)
    plt.colorbar(label="f(σ_s, σ_r) = MSE")
    nm_path = np.array([r["x"] for r in nm["rows"]])
    pw_path = np.array([theta0] + [r["x"] for r in pw["rows"]])
    plt.plot(nm_path[:, 0], nm_path[:, 1], "o-", ms=3, lw=1, label="Нелдер—Мід")
    plt.plot(pw_path[:, 0], pw_path[:, 1], "s-", ms=5, lw=1.5, label="Пауелл")
    plt.scatter([nm["x"][0]], [nm["x"][1]], marker="*", s=220, zorder=5,
                color="red", label="оптимум")
    plt.xlabel("σ_s — просторовий масштаб")
    plt.ylabel("σ_r — фотометричний поріг")
    plt.title("Лабораторна 3: лінії рівня і траєкторії методів")
    plt.legend()
    plt.savefig("outputs/lab3_contours.png", dpi=150, bbox_inches="tight")

    # ================= ДОСЛІДИ =================
    # --- 1: розмір початкового трикутника
    print("\n===== Дослід 1: ребро симплекса =====")
    for label, s in (("s/3", SIMPLEX_EDGE / 3), ("s", SIMPLEX_EDGE), ("3s", SIMPLEX_EDGE * 3)):
        oracle.reset()
        r = nelder_mead(oracle, theta0, s)
        print(f"{label:>4} (s = {s:.4f}): θ* = ({r['x'][0]:.4f}, {r['x'][1]:.4f})  "
              f"f* = {r['f']:.8f}  ітер = {r['iterations']}  N_f = {r['n_f']}")

    # --- 2: ціна першого циклу Пауелла
    print("\n===== Дослід 2: розклад вартості першого циклу Пауелла =====")
    first = stats[:2]
    for i, s in enumerate(first, 1):
        tot = s["svenn"] + s["golden"] + s["check"]
        print(f"лінійний пошук {i}: Свенн = {s['svenn']}, золотий = {s['golden']}, "
              f"перевірка = {s['check']}  → {tot}")
    tot1 = 1 + sum(s["svenn"] + s["golden"] + s["check"] for s in first)
    print(f"разом за цикл 1: 1 (f у θ0) + {tot1 - 1} = {tot1}   (у таблиці Пауелла: {pw['rows'][0]['n_f']})")
    avg = (pw["n_f"] - 1) / len(stats)
    print(f"усього лінійних пошуків: {len(stats)}, у середньому {avg:.1f} звернень на пошук")

    # --- 3: завелика точність зупинки Пауелла
    print("\n===== Дослід 3: tol Пауелла = 1e-7 =====")
    oracle.reset()
    pw7 = powell(oracle, theta0, tol=1e-7, max_cycles=50)
    print(f"tol=1e-4: циклів = {pw['cycles']}, N_f = {pw['n_f']}, f* = {pw['f']:.10f}")
    print(f"tol=1e-7: циклів = {pw7['cycles']}, N_f = {pw7['n_f']}, f* = {pw7['f']:.10f}, "
          f"θ* = ({pw7['x'][0]:.6f}, {pw7['x'][1]:.6f})")

    # --- 4: одна ручка проти двох (σr зафіксовано)
    print("\n===== Дослід 4: σr зафіксовано =====")
    sr_fix = float(theta[1])
    o_1d = Oracle(lambda s: oracle.peek([s, sr_fix]))
    br1 = svenn(o_1d, theta0[0], 0.3, 0.3, 5.0)
    o_1d.reset()
    g1 = golden_section(o_1d, br1["a"], br1["b"], 1e-4)
    print(f"σr = {sr_fix:.6f} (фіксоване), σs* = {g1['x']:.6f}, f = {g1['f']:.10f}, N_f = {g1['n_f']}")
    print(f"двовимірний оптимум:      f = {nm['f']:.10f}")
    print(f"різниця: {g1['f'] - nm['f']:.2e}")
    print(f"гаусів фільтр (лаб. 2):   f = {mse(gaussian_blur(noisy, sigma_star), clean):.10f}")


if __name__ == "__main__":
    main()