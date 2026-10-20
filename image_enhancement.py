"""
Tugas 4 - Image Enhancement
Menggunakan imageio, numpy, scipy.ndimage, dan matplotlib untuk histogram.
Jalankan:
  python image_enhancement.py
  python image_enhancement.py gambar_lain.png
"""
import os
import sys

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def histogram_equalization(gray):
    """Ekualisasi histogram sederhana untuk citra grayscale 8-bit."""
    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return gray.copy()
    cdf_min = nonzero[0]
    total = gray.size
    if total == cdf_min:
        return gray.copy()
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]


def save_histogram_comparison(original, enhanced, output_path):
    """Simpan visualisasi citra dan histogram secara rapi dalam satu figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("white")

    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Citra Awal", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced, cmap="gray")
    axes[0, 1].set_title("Citra Hasil", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    hist_original = np.bincount(original.ravel(), minlength=256)
    axes[1, 0].plot(hist_original, color="#1f1f1f", linewidth=1.5)
    axes[1, 0].set_title("Histogram Awal", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_xlabel("Intensitas")
    axes[1, 0].set_ylabel("Jumlah Piksel")
    axes[1, 0].grid(True, alpha=0.2)

    hist_enhanced = np.bincount(enhanced.ravel(), minlength=256)
    axes[1, 1].plot(hist_enhanced, color="#1f1f1f", linewidth=1.5)
    axes[1, 1].set_title("Histogram Hasil", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_xlabel("Intensitas")
    axes[1, 1].set_ylabel("Jumlah Piksel")
    axes[1, 1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "citra_awal.png"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    image = iio.imread(input_path)

    if image.ndim == 3:
        image = image[..., :3].mean(axis=2).astype(np.uint8)
    gray = image.astype(np.uint8)

    denoised = ndimage.gaussian_filter(gray.astype(float), sigma=0.8)
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    enhanced = histogram_equalization(denoised)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_main = f"hasil_{base_name}.png"
    output_hist = f"hasil_{base_name}_histogram.png"

    iio.imwrite(output_main, enhanced)
    save_histogram_comparison(gray, enhanced, output_hist)

    print("Proses selesai.")
    print(f"File sumber: {input_path}")
    print(f"File hasil utama: {output_main}")
    print(f"File hasil komparasi: {output_hist}")
    print(f"Ukuran citra: {gray.shape[1]} x {gray.shape[0]}")
    print(f"Rentang intensitas awal: {gray.min()} - {gray.max()}")
    print(f"Rentang intensitas hasil: {enhanced.min()} - {enhanced.max()}")


if __name__ == "__main__":
    main()
