import numpy as np

from src.variant import SEED, NOISE_SIGMA, SIZE
from src.rng import Rng
from src.image import clean_image, add_noise, save_png, load_png, show_ascii
from src.metrics import mse, psnr
from src.oracle import Oracle


def print_patch(img, fmt="{:.4f}"):
    for i in range(60, 65):
        print("  ".join(fmt.format(img[i][j]) for j in range(60, 65)))


def main():
    print("seed =", SEED)

    r = Rng(437327)
    print("RNG (437327):", [round(r.normal(), 6) for _ in range(6)])
    r = Rng(SEED)
    print(f"RNG ({SEED}):", [round(r.normal(), 6) for _ in range(6)])

    clean = clean_image(SIZE)
    print("\nчисте 5x5:"); print_patch(clean)
    show_ascii(clean)

    ref = load_png("00_clean.png", SIZE)
    print("\nmax|clean - 00_clean.png| =", np.abs(clean - ref).max())

    save_png(clean, "outputs/00_clean.png")

    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    save_png(noisy, "outputs/00_noisy.png")
    print("\nзіпсоване 5x5:"); print_patch(noisy)
    print("\nшум 5x5:");       print_patch(noisy - clean, "{:+.4f}")
    out = int(np.sum((raw < 0) | (raw > 1)))
    print(f"\nпікселів поза [0,1]: {out} із {raw.size} ({100*out/raw.size:.2f} %)")
    print(f"raw: min = {raw.min():.4f}  max = {raw.max():.4f}")

    print(f"\nMSE(clean, clean)  = {mse(clean, clean)}")
    print(f"MSE(чорне, біле)   = {mse(np.zeros((SIZE, SIZE)), np.ones((SIZE, SIZE)))}")
    print(f"MSE(noisy, clean)  = {mse(noisy, clean):.8f}")
    print(f"PSNR(noisy, clean) = {psnr(noisy, clean):.4f} дБ")

    o = Oracle(lambda x: (x - 2) ** 2)
    print("\nOracle:", o.value(0.0), o.value(3.0), o.peek(5.0), o.n_f)


if __name__ == "__main__":
    main()