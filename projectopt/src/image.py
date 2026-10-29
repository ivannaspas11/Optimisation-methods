import math
import numpy as np
from PIL import Image as PILImage

from src.rng import Rng


def clean_image(size=128):
    img = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            u = j / (size - 1)          # від СТОВПЦЯ
            v = i / (size - 1)          # від РЯДКА

            val = 0.25 + 0.35 * (0.6 * u + 0.4 * v)
            val += 0.08 * math.cos(2 * math.pi * math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2))

            if (u - 0.32) ** 2 + (v - 0.34) ** 2 <= 0.16 ** 2:
                val = 0.92
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                val = 0.10
            if 0.60 < v < 0.90 and abs(u - 0.30) < 0.9 * (v - 0.60):
                val = 0.68
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                cell = max(3, size // 12)
                val = 0.85 if ((j // cell) + (i // cell)) % 2 == 0 else 0.20

            img[i][j] = min(max(val, 0.0), 1.0)

    for idx in range(4):
        thickness = max(2, (4 - idx) * max(1, size // 64))
        row = int(0.04 * size) + idx * max(3, size // 22)
        if row + thickness < size:
            img[row:row + thickness, int(0.05 * size):int(0.95 * size)] = 0.98

    return img


def show_ascii(img, step=3):
    chars = " .:-=+*#%@"
    for i in range(0, img.shape[0], 2 * step):
        line = "".join(chars[min(9, int(img[i][j] * 10))]
                       for j in range(0, img.shape[1], step))
        print("  " + line)


def save_png(img, path):
    data = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    PILImage.fromarray(data, mode="L").save(path)


def load_png(path, size=128):
    with PILImage.open(path) as handle:
        gray = handle.convert("L").resize((size, size), PILImage.BILINEAR)
    return np.asarray(gray, dtype=float) / 255.0


def add_noise(clean, sigma, seed):
    rng = Rng(seed)                          # ОДИН раз, до циклів
    h, w = clean.shape
    raw = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            raw[i][j] = clean[i][j] + sigma * rng.normal()
    noisy = np.clip(raw, 0.0, 1.0)
    return raw, noisy