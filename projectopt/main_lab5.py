import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.image import clean_image, add_noise, save_png
from src.metrics import psnr
from src.operators import energy, energy_grad, tikhonov_exact
from src.lab5_constrained import (project, violation, projected_gradient,
                                  penalty_method)

LAM = 0.570909          # λ* з лабораторної 4 (варіант 20)


def main():
    os.makedirs("outputs", exist_ok=True)
    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    lam = LAM
    L = 1.0 + 8.0 * lam

    # ================= 5.1. сире спостереження =================
    out = int(np.sum((raw < 0) | (raw > 1)))
    print("сире спостереження:")
    print(f"  поза [0, 1]:  {out} пікселів із {raw.size} ({100 * out / raw.size:.2f} %)")
    print(f"  найменше значення: {raw.min():.4f}      найбільше: {raw.max():.4f}")

    unconstrained = tikhonov_exact(raw, lam)            # безумовний розв'язок на raw
    v_u = violation(unconstrained)
    print("\nбезумовний розв'язок:")
    print(f"  поза Ω:  {v_u['count']} пікселів ({100 * v_u['count'] / raw.size:.2f} %)")
    print(f"  максимальне порушення: {v_u['max']:.6f}")
    print(f"  E = {energy(unconstrained, raw, lam):.8f}")

    edges = np.linspace(raw.min(), raw.max(), 121)       # однакові стовпчики для обох
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(raw.ravel(), bins=120, color="tab:green", alpha=0.45,
            label=f"сире спостереження (поза [0, 1]: {out})")
    ax.hist(unconstrained.ravel(), bins=120, histtype="step", lw=1.6,
            color="tab:purple",
            label=f"безумовний розв'язок (поза [0, 1]: {v_u['count']})")
    ax.axvline(0.0, color="black", ls=":", lw=1.2, label="межі допустимої області")
    ax.axvline(1.0, color="black", ls=":", lw=1.2)
    ax.set_xlabel("яскравість пікселя")
    ax.set_ylabel("кількість пікселів")
    ax.set_title("Лабораторна 5: розподіл яскравостей")
    ax.legend()
    fig.savefig("outputs/lab5_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= 5.3–5.5. проєкція градієнта =================
    res_pg = projected_gradient(raw, lam, L)
    print("\n===== Проєкція градієнта =====")
    print(f"{'k':>3} {'E(Ik)':>14} {'‖Gh‖':>11} {'активних':>9} {'порушення':>10}")
    for r in res_pg["rows"][:6] + res_pg["rows"][-1:]:
        print(f"{r['k']:>3} {r['E']:>14.8f} {r['G']:>11.4e} {r['active']:>9} {r['violation']:>10.1f}")
    print(f"ітерацій = {res_pg['iterations']}   N_f = {res_pg['n_f']}   N∇f = {res_pg['n_grad']}   "
          f"разом {res_pg['n_f'] + res_pg['n_grad']} звернень")
    v_pg = violation(res_pg["I"])
    print(f"E = {res_pg['E']:.8f}    порушення = {v_pg['max']}   поза Ω: {v_pg['count']}")
    print("порушення на кожній ітерації точно нуль:", all(r["violation"] == 0.0 for r in res_pg["rows"]))

    # FISTA
    res_fi = projected_gradient(raw, lam, L, accelerated=True)
    print(f"\nFISTA: ітерацій = {res_fi['iterations']}   N_f = {res_fi['n_f']}   "
          f"N∇f = {res_fi['n_grad']}   разом {res_fi['n_f'] + res_fi['n_grad']}   E = {res_fi['E']:.8f}")

    # ================= 5.6–5.7. штрафи =================
    res_pen = penalty_method(raw, lam, L)
    print("\n===== Метод штрафів =====")
    print(f"{'s':>2} {'r':>10} {'E(Is)':>14} {'P(Is)':>10} {'r·P':>10} {'порушення':>10} {'поза Ω':>7}")
    for r in res_pen["rows"]:
        print(f"{r['s']:>2} {r['r']:>10.0f} {r['E']:>14.8f} {r['P']:>10.3e} "
              f"{r['rP']:>10.3e} {r['violation']:>10.2e} {r['outside']:>7}")
    v_pen = violation(res_pen["I"])
    print(f"N_f = {res_pen['n_f']}   N∇f = {res_pen['n_grad']}   разом {res_pen['n_f'] + res_pen['n_grad']}")
    print(f"E = {res_pen['E']:.8f}   порушення = {v_pen['max']:.2e}   поза Ω: {v_pen['count']}")

    # ================= 5.8. порівняння =================
    print("\nЗведена таблиця:")
    print(f"{'Метод':<24} {'E(I*)':>13} {'порушення':>10} {'поза Ω':>7} {'допуст.':>8} {'ітер':>6} {'N_f':>4} {'N∇f':>5} {'разом':>6}")
    for name, r, vv, it in (("Проєкція градієнта", res_pg, v_pg, str(res_pg["iterations"])),
                            ("FISTA", res_fi, violation(res_fi["I"]), str(res_fi["iterations"])),
                            ("Квадратичний штраф", res_pen, v_pen, f"{res_pen['outer']} зовн.")):
        print(f"{name:<24} {r['E']:>13.8f} {vv['max']:>10.2e} {vv['count']:>7} "
              f"{'так' if vv['count'] == 0 else 'ні':>8} {it:>6} {r['n_f']:>4} {r['n_grad']:>5} "
              f"{r['n_f'] + r['n_grad']:>6}")
    print(f"штраф дорожчий у {(res_pen['n_f'] + res_pen['n_grad']) / (res_pg['n_f'] + res_pg['n_grad']):.1f} раза")

    # графік і зображення
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(len(res_pg["rows"])), [r["E"] for r in res_pg["rows"]],
                "o-", ms=3, lw=1.3, markevery=5, label="Проєкція градієнта")
    ax.semilogy(range(len(res_fi["rows"])), [r["E"] for r in res_fi["rows"]],
                "s--", ms=3, lw=1.3, markevery=5, label="FISTA")
    ax.axhline(res_pg["E"], color="gray", ls=":", lw=1, label=f"E* = {res_pg['E']:.4f}")
    ax.set_xlabel("k — номер ітерації")
    ax.set_ylabel("E(I_k) (логарифмічна шкала)")
    ax.set_title("Лабораторна 5: збіжність проєкційних методів")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig("outputs/lab5_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_png(res_pg["I"], "outputs/lab5_restored.png")
    p5 = psnr(res_pg["I"], clean)
    print(f"\nPSNR з обмеженнями = {p5:.4f} дБ")
    print(f"PSNR безумовного (обрізаного) на raw = {psnr(np.clip(unconstrained, 0, 1), clean):.4f} дБ")
    print(f"PSNR лаб. 4 = 20.8954 дБ")

    # ================= 5.9. досліди =================
    # --- 1: неправильний критерій
    print("\n===== Дослід 1: критерій зупинки =====")
    wrong = projected_gradient(raw, lam, L, criterion="grad", max_iter=200)
    print(f"за ‖Gh‖ ≤ 1e-6:  ітерацій = {res_pg['iterations']:>3}, збігся: {res_pg['converged']}, "
          f"‖Gh‖ = {res_pg['G']:.2e}, ‖∇E‖ = {res_pg['grad']:.2e}")
    print(f"за ‖∇E‖ ≤ 1e-6:  ітерацій = {wrong['iterations']:>3}, збігся: {wrong['converged']}, "
          f"‖Gh‖ = {wrong['G']:.2e}, ‖∇E‖ = {wrong['grad']:.2e}")

    # --- 2: множник зростання
    print("\n===== Дослід 2: множник зростання штрафу =====")
    print(f"{'β':>4} {'зовн.':>6} {'N∇f':>6} {'разом':>6} {'порушення':>11} {'поза Ω':>7} {'E':>14}")
    for gth in (2.0, 10.0, 100.0):
        rp = penalty_method(raw, lam, L, growth=gth, outer=60)
        vv = violation(rp["I"])
        print(f"{gth:>4.0f} {rp['outer']:>6} {rp['n_grad']:>6} {rp['n_f'] + rp['n_grad']:>6} "
              f"{vv['max']:>11.2e} {vv['count']:>7} {rp['E']:>14.8f}")

    # --- 3: ціна обумовленості
    print("\n===== Дослід 3: ціна обумовленості =====")
    print(f"{'s':>2} {'r':>10} {'L(Φr)=L+2r':>14} {'крок 1/L(Φr)':>14} {'внутр. ітер.':>13}")
    for r in res_pen["rows"]:
        print(f"{r['s']:>2} {r['r']:>10.0f} {r['L_r']:>14.4f} {r['h']:>14.3e} {r['inner_grads']:>13}")

    # --- 4: карта активних обмежень
    I_star = res_pg["I"]
    at0 = I_star <= 1e-12
    at1 = I_star >= 1.0 - 1e-12
    cmap = np.full(I_star.shape, 0.5)
    cmap[at0] = 0.0
    cmap[at1] = 1.0
    print("\n===== Дослід 4: активні обмеження =====")
    print(f"на нулі (чорні): {int(at0.sum())}   на одиниці (білі): {int(at1.sum())}   "
          f"вільні (сірі): {int(I_star.size - at0.sum() - at1.sum())}")
    in_dark = int(np.sum(at0 & (clean < 0.3)))
    in_bright = int(np.sum(at1 & (clean > 0.8)))
    print(f"із них чорні на темних ділянках чистого (clean < 0.3): {in_dark} з {int(at0.sum())}")
    print(f"білі на яскравих ділянках чистого (clean > 0.8): {in_bright} з {int(at1.sum())}")
    plt.figure(figsize=(5, 5))
    plt.imshow(cmap, cmap="gray", vmin=0, vmax=1)
    plt.title("Активні обмеження: чорний = 0, білий = 1, сірий = вільні")
    plt.axis("off")
    plt.savefig("outputs/lab5_active_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ================= підсумок практикуму =================
    print("\n===== Уся дорога в одній таблиці (варіант 20) =====")
    print(f"{'Етап':<8} {'Метод':<24} {'Невідомих':>10} {'Звернень':>9} {'PSNR, дБ':>9}")
    print(f"{'Вхід':<8} {'—':<24} {'—':>10} {'—':>9} {psnr(noisy, clean):>9.4f}")
    print(f"{'Лаб. 1':<8} {'Свенн':<24} {1:>10} {5:>9} {'—':>9}")
    print(f"{'Лаб. 2':<8} {'золотий переріз':<24} {1:>10} {22:>9} {21.0409:>9.4f}")
    print(f"{'Лаб. 3':<8} {'Нелдер—Мід':<24} {2:>10} {143:>9} {23.0712:>9.4f}")
    print(f"{'Лаб. 4':<8} {'Нестеров':<24} {16384:>10} {56:>9} {20.8954:>9.4f}")
    print(f"{'Лаб. 5':<8} {'проєкція градієнта':<24} {16384:>10} "
          f"{res_pg['n_f'] + res_pg['n_grad']:>9} {p5:>9.4f}")


if __name__ == "__main__":
    main()