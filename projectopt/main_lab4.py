import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, X0, DELTA0, EPS, LAMBDA0, SIZE
from src.image import clean_image, add_noise, save_png
from src.oracle import Oracle
from src.metrics import mse, psnr
from src.lab1_svenn import svenn
from src.lab2_search import golden_section
from src.operators import (grad_xy, divergence, energy, energy_grad, tikhonov_exact)
from src.lab4_gradient import gradient_descent
from src.lab4_tv import tv_nesterov

MAX_ITER, TOL = 200, 1e-6


def main():
    os.makedirs("outputs", exist_ok=True)
    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    rng = np.random.default_rng(0)

    # ================= 4.4. дві обов'язкові перевірки =================
    u = rng.normal(size=(9, 7)); px = rng.normal(size=(9, 7)); py = rng.normal(size=(9, 7))
    gx, gy = grad_xy(u)
    left = float(np.sum(gx * px + gy * py))
    right = -float(np.sum(u * divergence(px, py)))
    print(f"перевірка 1, нев'язка спряженості: {abs(left - right):.2e}")

    I_t = rng.random((16, 16)); n_t = rng.random((16, 16)); V = rng.normal(size=(16, 16))
    t = 1e-6
    numeric = (energy(I_t + t * V, n_t, 0.3) - energy(I_t - t * V, n_t, 0.3)) / (2 * t)
    analytic = float(np.sum(energy_grad(I_t, n_t, 0.3) * V))
    print(f"перевірка 2, відносна різниця: {abs(numeric - analytic) / abs(analytic):.2e}")

    # ================= 4.6. підбір λ (дворівнева оптимізація) =================
    def g_of_lambda(lam):
        """Зовнішня ціль: MSE точного розв'язку внутрішньої задачі проти чистого."""
        return mse(np.clip(tikhonov_exact(noisy, lam), 0.0, 1.0), clean)

    outer = Oracle(g_of_lambda)
    br = svenn(outer, LAMBDA0, 0.2, 1e-4, 50.0)
    print(f"\nСвенн по λ: [{br['a']:.6f}, {br['b']:.6f}]   {br['n_f']} звернень")
    outer.reset()
    gs = golden_section(outer, br["a"], br["b"], 1e-3)
    lam = gs["x"]
    print(f"золотий переріз: λ* = {lam:.6f}   {gs['iterations']} ітерацій, "
          f"{gs['n_f']} звернень (разом зі Свенном {br['n_f'] + gs['n_f']})")
    print(f"g(λ*) = {g_of_lambda(lam):.8f}")

    L = 1.0 + 8.0 * lam
    mu = 1.0
    kappa = L / mu
    h = 1.0 / L
    beta_hb = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    beta_nag = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    print(f"\nL = {L:.6f}   κ = {kappa:.6f}   h = 1/L = {h:.6f}   межа стійкості 2/L = {2 / L:.6f}")
    print(f"β кульки (оптимальне) = {beta_hb:.6f}   β Нестерова (сильно опуклий) = {beta_nag:.6f}")

    # ================= 4.7. еталонний розв'язок =================
    I_star = tikhonov_exact(noisy, lam)
    E_star = energy(I_star, noisy, lam)
    print(f"\n‖∇E(I*)‖ = {np.linalg.norm(energy_grad(I_star, noisy, lam)):.2e}")
    print(f"E* = {E_star:.8f}     E(I0) = {energy(noisy, noisy, lam):.8f}")

    # ================= 4.8. шість методів =================
    runs = [
        ("Спуск, h=1/L",             dict(kind="gd")),
        ("Спуск, Армійо",            dict(kind="armijo")),
        (f"Кулька, β={beta_hb:.4f}", dict(kind="hb", beta=beta_hb)),
        ("Кулька, β=0,9",            dict(kind="hb", beta=0.9)),
        ("Нестеров, опуклий",        dict(kind="nag")),
        ("Нестеров, сильно опукл.",  dict(kind="nag", beta=beta_nag)),
    ]
    results = {}
    print(f"\n{'Метод':<26} {'ітер':>5} {'N_f':>5} {'N∇f':>5} {'разом':>6} {'E(I_N)':>14} {'‖∇E‖':>9} {'збігся':>7}")
    for name, kw in runs:
        r = gradient_descent(noisy, lam, L, max_iter=MAX_ITER, tol=TOL, x_ref=I_star, **kw)
        results[name] = r
        print(f"{name:<26} {r['iterations']:>5} {r['n_f']:>5} {r['n_grad']:>5} "
              f"{r['n_f'] + r['n_grad']:>6} {r['E']:>14.8f} {r['grad']:>9.1e} "
              f"{'так' if r['converged'] else 'НІ':>7}")
    print(f"E* (точний) = {E_star:.8f}")

    print("\nТаблиця градієнтного спуску:")
    print(f"{'k':>3} {'E(Ik)':>14} {'‖∇E‖':>11} {'h':>9} {'‖Ik−I*‖':>11} {'E−E*':>11}")
    rows = results["Спуск, h=1/L"]["rows"]
    for r in rows[:7] + rows[-1:]:
        print(f"{r['k']:>3} {r['E']:>14.8f} {r['grad']:>11.4e} {r['h']:>9.6f} "
              f"{r['dist']:>11.4e} {r['E'] - E_star:>11.4e}")

    # ---------- графіки збіжності ----------
    for key, fname, ylabel, title in (
        ("E", "lab4_convergence_f.png", "E(I_k) − E*", "Лабораторна 4: збіжність за значенням функції"),
        ("dist", "lab4_convergence_dist.png", "‖I_k − I*‖", "Лабораторна 4: збіжність за відстанню до I*"),
    ):
        plt.figure(figsize=(9, 5))
        for name, res in results.items():
            if key == "E":
                ys = [max(r["E"] - E_star, 1e-16) for r in res["rows"]]
            else:
                ys = [max(r["dist"], 1e-16) for r in res["rows"]]
            plt.semilogy(range(len(ys)), ys, label=name)
        plt.xlabel("номер ітерації k")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"outputs/{fname}", dpi=150, bbox_inches="tight")
        plt.close()

    # ================= 4.11. несподіванка =================
    best = results["Нестеров, сильно опукл."]["I"]
    restored = np.clip(best, 0.0, 1.0)
    save_png(restored, "outputs/lab4_restored.png")
    p_noisy = psnr(noisy, clean)
    p_lab2 = 21.0409          # результат лабораторної 2 (гаусів фільтр)
    p_lab3 = 23.0712          # результат лабораторної 3 (білатеральний фільтр)
    p_lab4 = psnr(restored, clean)
    print("\nPSNR попіксельного відновлення = %.4f дБ" % p_lab4)
    print(f"{'Етап':<40} {'PSNR, дБ':>9}")
    print(f"{'зіпсоване зображення':<40} {p_noisy:>9.4f}")
    print(f"{'лабораторна 2, гаусів фільтр':<40} {p_lab2:>9.4f}")
    print(f"{'лабораторна 3, білатеральний фільтр':<40} {p_lab3:>9.4f}")
    print(f"{'лабораторна 4, попіксельне відновлення':<40} {p_lab4:>9.4f}")

    # ---------- додаткове завдання: повна варіація ----------
    eps_tv = 5e-3
    lam_tv = max(0.02, 0.15 * lam)
    tv = tv_nesterov(noisy, lam_tv, eps_tv, max_iter=1000)
    restored_tv = np.clip(tv["I"], 0.0, 1.0)
    save_png(restored_tv, "outputs/lab4_restored_tv.png")
    p_tv = psnr(restored_tv, clean)
    print(f"\n[додатково] TV: λ_TV = {lam_tv:.6f}, L = {tv['L']:.2f}, β = {tv['beta']:.4f}, "
          f"ітерацій = {tv['iterations']}, ‖∇E‖ = {tv['grad']:.1e}")
    print(f"[додатково] PSNR (TV) = {p_tv:.4f} дБ   приріст до Тихонова: {p_tv - p_lab4:+.2f} дБ")

    # ================= 4.12. досліди =================
    # --- 1: велике число обумовленості
    print("\n===== Дослід 1: λ = 6 (κ = 49), ліміт 3000 ітерацій =====")
    lam6 = 6.0
    L6 = 1 + 8 * lam6
    b_hb6 = ((np.sqrt(L6) - 1) / (np.sqrt(L6) + 1)) ** 2
    b_n6 = (np.sqrt(L6) - 1) / (np.sqrt(L6) + 1)
    base = {"Градієнтний спуск": results["Спуск, h=1/L"]["iterations"],
            "Важка кулька": results[f"Кулька, β={beta_hb:.4f}"]["iterations"],
            "Нестеров, сильно опуклий": results["Нестеров, сильно опукл."]["iterations"]}
    for name, kw in (("Градієнтний спуск", dict(kind="gd")),
                     ("Важка кулька", dict(kind="hb", beta=b_hb6)),
                     ("Нестеров, сильно опуклий", dict(kind="nag", beta=b_n6))):
        r = gradient_descent(noisy, lam6, L6, max_iter=3000, tol=TOL, **kw)
        print(f"{name:<26} κ=49: {r['iterations']:>4} ітер. (збігся: {r['converged']})   "
              f"κ={kappa:.1f}: {base[name]:>3} ітер.")

    # --- 2: розбіжність при завеликому кроці
    print("\n===== Дослід 2: h = 2.1/L, 50 ітерацій =====")
    r = gradient_descent(noisy, lam, L, kind="gd", step=2.1 / L, max_iter=50, tol=TOL)
    print("k :", [0, 10, 20, 30, 40])
    print("E :", [round(r["rows"][k]["E"], 2) for k in (0, 10, 20, 30, 40)])

if __name__ == "__main__":
    main()