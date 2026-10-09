# Бинаризация с адаптивным порогом (gaussian)
import os
import math
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
image_path = "/Users/arina/Desktop/university/computer-vision-labor/image1.png"
out_dir = os.path.join(here, "images")
os.makedirs(out_dir, exist_ok=True)

ksize = 31
C = 10
max_value = 255
crop_box = (290, 290, 160, 110)

# для сохранения
def save(name, img):
    cv2.imwrite(os.path.join(out_dir, name), img)

def adaptive_opencv(gray, ksize=ksize, c=C):
    return cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)

# Ядро Гаусса
def gaussian_kernel(ksize, sigma=0.0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = [i - (ksize - 1) / 2.0 for i in range(ksize)]
    k = [math.exp(-(xi ** 2) / (2.0 * sigma ** 2)) for xi in x]
    total = sum(k)
    return [v / total for v in k]

# Собственная реализация на Python
def adaptive_python(gray, ksize=ksize, c=C, sigma=0.0):
    h, w = gray.shape
    r = ksize // 2
    img = gray.tolist()
    kern = gaussian_kernel(ksize, sigma)
    res = [[0] * w for _ in range(h)]

    for y in range(h):
        for x in range(w):
            acc = 0.0
            for j in range(ksize):
                row = img[min(max(y + j - r, 0), h - 1)]
                for i in range(ksize):
                    acc += kern[j] * kern[i] * row[min(max(x + i - r, 0), w - 1)]

            thr = math.floor(acc + 0.5) - math.ceil(c)
            res[y][x] = max_value if img[y][x] > thr else 0

    return np.array(res, dtype=np.uint8)

# time
def timeit(fn, *args, repeat=5, **kw):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t)
    return best

# % отличающихся пикселей
def mismatch(a, b):
    return 100.0 * np.count_nonzero(a != b) / a.size

def main():
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError(image_path)

    print(f"\nГотовая реализация OpenCV (ksize={ksize}, C={C})")
    lib_res = adaptive_opencv(gray)
    save("03_lib_gaussian.png", lib_res)

    x, y, w, h = crop_box
    crop = gray[y:y + h, x:x + w]

    if crop.size == 0:
        raise ValueError("Выбран пустой фрагмент изображения")

    print(f"\nМоя реализация (чистый Python) на фрагменте {w}x{h}")
    py_res = adaptive_python(crop, 15, C)
    lib_crop = adaptive_opencv(crop, 15, C)

    save("crop_gray.png", crop)
    save("crop_my_python_gaussian.png", py_res)
    save("crop_opencv_gaussian.png", lib_crop)

    print(f"  расхождение с OpenCV: {mismatch(py_res, lib_crop):.4f} %")

    print("\nВлияние параметров (отдельные картинки)")
    for k in [7, 15, 31, 101]:
        save(f"ksize_{k}.png", adaptive_opencv(gray, k, C))

    for cc in [0, 5, 10, 25]:
        save(f"C_{cc}.png", adaptive_opencv(gray, ksize, cc))

    print(f"\nСкорость: чистый Python и OpenCV, фрагмент {w}x{h}")
    print(f"{'ksize':>5} | {'Python, мс':>10} | {'OpenCV, мс':>10} | Python/OpenCV")

    ksizes = [3, 7, 15]
    py_t, lib_t = [], []

    for k in ksizes:
        t_py = timeit(adaptive_python, crop, k, C, repeat=2) * 1e3
        t_lib = timeit(adaptive_opencv, crop, k, C) * 1e3

        py_t.append(t_py)
        lib_t.append(t_lib)

        print(f"{k:>5} | {t_py:10.3f} | {t_lib:10.3f} | {t_py / t_lib:8.1f}x")

    # график времени
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(ksizes, lib_t, marker="o", label="OpenCV")
    ax.plot(ksizes, py_t, marker="s", label="Моя (Python)")

    ax.set(
        xlabel="Размер окна ksize",
        ylabel="Время, мс",
        title="Время обработки vs размер окна"
    )

    ax.grid(alpha=.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_benchmark.png"), dpi=130)
    plt.close(fig)

    print("\n  сохранено: images/07_benchmark.png")


if __name__ == "__main__":
    main()