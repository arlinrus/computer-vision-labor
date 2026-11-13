import zlib

import cv2
import numpy as np
from PIL import Image, ImageOps

import config


def load_photo(name: str) -> np.ndarray:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = config.PHOTOS_DIR / f"{name}{ext}"
        if path.exists():
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # учесть поворот фото с телефона
            return np.array(img)
    raise FileNotFoundError(f"нет файла {name}.jpg/.png в {config.PHOTOS_DIR}")


def best_crop(img: np.ndarray, size: int) -> np.ndarray:
    assert min(img.shape[:2]) >= size, f"фото меньше {size} px по короткой стороне"
    k = 4                                                      
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(g, (g.shape[1] // k, g.shape[0] // k), interpolation=cv2.INTER_AREA)
    s, step = size // k, 32 // k
    best, by, bx = -1.0, 0, 0
    for y in range(0, small.shape[0] - s + 1, step):
        for x in range(0, small.shape[1] - s + 1, step):
            v = cv2.Laplacian(small[y:y + s, x:x + s], cv2.CV_32F).var()
            if v > best:
                best, by, bx = v, y * k, x * k
    return img[by:by + size, bx:bx + size]


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode())


def bicubic_degrade(hr: np.ndarray) -> np.ndarray: # простое уменьшение
    h, w = hr.shape[:2]
    return np.array(Image.fromarray(hr).resize((w // 4, h // 4), Image.BICUBIC))


def unknown_degrade(hr: np.ndarray, seed: int) -> np.ndarray: # размытие + уменьшение
    rng = np.random.default_rng(seed)
    sx, sy = rng.uniform(0.8, 3.0, size=2)                     
    angle = rng.uniform(0, np.pi)                             
    r = 10
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    u = xx * np.cos(angle) + yy * np.sin(angle)
    v = -xx * np.sin(angle) + yy * np.cos(angle)
    kernel = np.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))
    kernel /= kernel.sum()                                     
    img = cv2.filter2D(hr, -1, kernel)
    h, w = hr.shape[:2]
    return cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)


def realesrgan_degrade(hr: np.ndarray, seed: int) -> np.ndarray: # изначальная
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(hr, (0, 0), rng.uniform(1.0, 2.5))                         
    h, w = hr.shape[:2]
    img = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)            
    img = img.astype(np.float32) + rng.normal(0, rng.uniform(5, 15), img.shape)       
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(30, 60))                                                    
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def get_pair(name: str, degradation: str):
    hr = best_crop(load_photo(name), config.LR_SIZE * config.UPSCALE)   # эталон 1024x1024
    if degradation == "bicubic":
        lr = bicubic_degrade(hr)
    elif degradation == "unknown":
        lr = unknown_degrade(hr, seed_of(name))
    elif degradation == "realesrgan":
        lr = realesrgan_degrade(hr, seed_of(name))
    else:
        raise ValueError(degradation)
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)
