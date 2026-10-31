import numpy as np
import math


def gaussian_kernel(sigma):
    """Нормоване гаусове ядро; радіус r = floor(4*sigma + 0.5)."""
    r = int(4.0 * sigma + 0.5)
    t = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-(t * t) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel, r


def gaussian_blur(img, sigma):
    """Два проходи: рядки, потім стовпці. Краї — найближчий піксель."""
    if sigma <= 1e-6:
        return img.copy()

    kernel, r = gaussian_kernel(sigma)
    h, w = img.shape

    # прохід 1: по горизонталі
    tmp = np.zeros_like(img)
    for k in range(-r, r + 1):
        cols = np.clip(np.arange(w) + k, 0, w - 1)
        tmp += kernel[k + r] * img[:, cols]

    # прохід 2: по вертикалі (читаємо з tmp, пишемо в out)
    out = np.zeros_like(img)
    for k in range(-r, r + 1):
        rows = np.clip(np.arange(h) + k, 0, h - 1)
        out += kernel[k + r] * tmp[rows, :]

    return out


def bilateral_filter(img, sigma_spatial, sigma_range, pad_mode="reflect"):
    """Білатеральний фільтр: усереднює лише схожих за яскравістю сусідів."""
    sigma_spatial = max(float(sigma_spatial), 1e-3)
    sigma_range = max(float(sigma_range), 1e-3)
    r = int(min(max(round(2.0 * sigma_spatial), 1), 5))

    h, w = img.shape
    padded = np.pad(img, r, mode=pad_mode)        # дзеркальні краї
    num = np.zeros_like(img)
    den = np.zeros_like(img)

    for dy in range(-r, r + 1):                   # цикл лише по вікну
        for dx in range(-r, r + 1):
            shifted = padded[r + dy:r + dy + h, r + dx:r + dx + w]
            w_s = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial ** 2))
            w_r = np.exp(-((shifted - img) ** 2) / (2.0 * sigma_range ** 2))
            weight = w_s * w_r
            num += weight * shifted
            den += weight

    return num / den