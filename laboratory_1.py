# Бинаризация с адаптивным порогом (gaussian)
import os
import sys
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
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()                                

# разделимая свертка
def adaptive_numpy(gray, ksize=ksize, c=C, sigma=0.0):
    r = ksize // 2
    k = gaussian_kernel(ksize, sigma)
    h, w = gray.shape
    p = np.pad(gray.astype(np.float64), r, mode="edge")   
    tmp = np.zeros((h + 2 * r, w))
    for i in range(ksize):                                
        tmp += k[i] * p[:, i:i + w]
    local = np.zeros((h, w))
    for i in range(ksize):                                
        local += k[i] * tmp[i:i + h, :]
    thr = np.floor(local + 0.5) - math.ceil(c)            
    return np.where(gray > thr, max_value, 0).astype(np.uint8)

def adaptive_python(gray, ksize=ksize, c=C, sigma=0.0):
    h, w = gray.shape
    r = ksize // 2
    img = gray.tolist()
    kern = gaussian_kernel(ksize, sigma).tolist()
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

    print("\nГлобальные пороги (для сравнения)")
    _, glob127 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    otsu_t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  порог Оцу = {otsu_t:.0f}")
    save("global_127.png", glob127)
    save("global_otsu.png", otsu)
 
    print(f"\nГотовая реализация OpenCV (ksize={ksize}, C={C})")
    lib_res = adaptive_opencv(gray)
    save("lib_gaussian.png", lib_res)
 
    print("\nМоя реализация (NumPy), сверка с OpenCV")
    my_res = adaptive_numpy(gray)
    save("numpy_gaussian.png", my_res)
    print(f"  расхождение: {mismatch(my_res, lib_res):.4f} %")
 
    x, y, w, h = crop_box
    crop = gray[y:y + h, x:x + w]
    print(f"\nМоя реализация (чистый Python) на фрагменте {w}x{h}")
    py_res = adaptive_python(crop, 15, C)
    lib_crop = adaptive_opencv(crop, 15, C)
    save("crop_gray.png", crop)
    save("crop_my_python_gaussian.png", py_res)
    print(f"  расхождение с OpenCV: {mismatch(py_res, lib_crop):.4f} %")
 
    print("\nВлияние параметров (отдельные картинки)")
    for k in [7, 15, 31, 101]:
        save(f"ksize_{k}.png", adaptive_opencv(gray, k, C))
    for cc in [0, 5, 10, 25]:
        save(f"C_{cc}.png", adaptive_opencv(gray, ksize, cc))
 
    print(f"\nСкорость: от размера изображения (ksize={ksize})")
    print(f"{'размер':>11} | {'OpenCV, мс':>10} | {'NumPy, мс':>10} | NumPy/OpenCV")
    mpix, lib_t, my_t = [], [], []
    for s in [0.25, 0.5, 1.0, 1.5]:
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t_lib = timeit(adaptive_opencv, g) * 1e3
        t_my = timeit(adaptive_numpy, g) * 1e3
        mpix.append(g.shape[0] * g.shape[1] / 1e6)
        lib_t.append(t_lib)
        my_t.append(t_my)
        print(f"{g.shape[1]:>5}x{g.shape[0]:<5} | {t_lib:10.3f} | {t_my:10.3f} | {t_my / t_lib:8.1f}x")
 
    print("\nСкорость: от размера окна (полный кадр)")
    print(f"{'ksize':>5} | {'OpenCV, мс':>10} | {'NumPy, мс':>10}")
    ksizes = [3, 7, 15, 31, 61, 101]
    lib_k, my_k = [], []
    for k in ksizes:
        t_lib = timeit(adaptive_opencv, gray, k, C) * 1e3
        t_my = timeit(adaptive_numpy, gray, k, C) * 1e3
        lib_k.append(t_lib)
        my_k.append(t_my)
        print(f"{k:>5} | {t_lib:10.3f} | {t_my:10.3f}")
 
    print(f"\nСкорость: чистый Python, фрагмент {w}x{h}")
    print(f"{'ksize':>5} | {'Python, мс':>10} | {'NumPy, мс':>10} | {'OpenCV, мс':>10}")
    for k in (3, 7, 15):
        t_py = timeit(adaptive_python, crop, k, C, repeat=2) * 1e3
        t_my = timeit(adaptive_numpy, crop, k, C) * 1e3
        t_lib = timeit(adaptive_opencv, crop, k, C) * 1e3
        print(f"{k:>5} | {t_py:10.1f} | {t_my:10.3f} | {t_lib:10.3f}")
 
    # график времени
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(mpix, lib_t, marker="o", label="OpenCV")
    ax[0].plot(mpix, my_t, marker="s", label="Моя (NumPy)")
    ax[0].set(xlabel="Размер изображения, Мпикс", ylabel="Время, мс",
              title=f"Время vs размер изображения (ksize={ksize})")
    ax[1].plot(ksizes, lib_k, marker="o", label="OpenCV")
    ax[1].plot(ksizes, my_k, marker="s", label="Моя (NumPy)")
    ax[1].set(xlabel="Размер окна ksize", ylabel="Время, мс", title="Время vs размер окна")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_benchmark.png"), dpi=130)
    plt.close(fig) 
 
if __name__ == "__main__":
    main()
 