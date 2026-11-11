"""Подготовка данных: скачивание DIV2K, выравненные кропы HR/LR и
деградация в стиле Real-ESRGAN (деградация второго порядка)."""
import io
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f"[data] downloading {url}")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def _read_from_zip(zip_path: Path, suffix: str) -> np.ndarray:
    """Читает из архива файл, имя которого заканчивается на suffix (без распаковки всего архива)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(suffix)]
        if not names:
            raise FileNotFoundError(f"{suffix} not found in {zip_path}")
        with zf.open(names[0]) as f:
            return np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))


def load_hr(img_id: str) -> np.ndarray:
    z = _download(config.DIV2K_URLS["HR"], config.DATA_DIR / "DIV2K_valid_HR.zip")
    return _read_from_zip(z, f"/{img_id}.png")


def load_div2k_lr(img_id: str, track: str) -> np.ndarray:
    z = _download(config.DIV2K_URLS[track], config.DATA_DIR / f"DIV2K_valid_LR_{track}_X4.zip")
    return _read_from_zip(z, f"/{img_id}x4.png")


# --------------------------------------------------------------------------
# Real-ESRGAN-like degradation (Wang et al., 2021): blur -> resize -> noise -> JPEG, дважды.
# Упрощённая реализация на numpy/cv2 (без sinc-фильтра и plateau-ядер).
# --------------------------------------------------------------------------
def _random_aniso_gaussian_kernel(rng, ksize, sigma_range):
    sx, sy = rng.uniform(*sigma_range, size=2)
    theta = rng.uniform(0, np.pi)
    r = ksize // 2
    xx, yy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (k / k.sum()).astype(np.float32)


def _degrade_once(img, rng, blur_sigma, scale_range, noise_sigma, jpeg_range, out_hw=None):
    ksize = int(rng.choice([7, 9, 11, 13, 15, 17, 19, 21]))
    img = cv2.filter2D(img, -1, _random_aniso_gaussian_kernel(rng, ksize, blur_sigma))
    h, w = img.shape[:2]
    if out_hw is None:
        s = rng.uniform(*scale_range)
        out_hw = (max(8, int(h * s)), max(8, int(w * s)))
    interp = int(rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    img = cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=interp)
    if rng.random() < 0.6:   # гауссов шум (цветной или серый)
        sigma = rng.uniform(*noise_sigma) / 255.0
        if rng.random() < 0.4:
            n = rng.normal(0, sigma, img.shape[:2])[..., None]
        else:
            n = rng.normal(0, sigma, img.shape)
        img = img + n
    else:                    # пуассоновский шум
        scale = 10 ** rng.uniform(np.log10(0.05), np.log10(3.0))
        vals = 255.0 * scale
        img = rng.poisson(np.clip(img, 0, 1) * vals) / vals
    img = np.clip(img, 0, 1).astype(np.float32)
    q = int(rng.uniform(*jpeg_range))
    ok, buf = cv2.imencode(".jpg", (img[..., ::-1] * 255).round().astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return img


def realesrgan_degrade(hr: np.ndarray, scale: int = 4, seed: int = 0) -> np.ndarray:
    """hr: uint8 HxWx3 -> LR uint8 (H/scale)x(W/scale)x3."""
    rng = np.random.default_rng(seed)
    h, w = hr.shape[:2]
    img = hr.astype(np.float32) / 255.0
    img = _degrade_once(img, rng, (0.2, 3.0), (0.15, 1.5), (1, 30), (30, 95))
    img = _degrade_once(img, rng, (0.2, 1.5), (0.3, 1.2), (1, 25), (30, 95),
                        out_hw=(h // scale, w // scale))
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


# --------------------------------------------------------------------------
def get_pair(img_id: str, degradation: str, lr_crop: int = config.LR_CROP, scale: int = config.UPSCALE):
    """Возвращает выравненные (LR, HR) uint8 кропы: LR lr_crop x lr_crop, HR в scale раз больше.
    Кроп берётся из центра изображения."""
    hr = load_hr(img_id)
    H, W = hr.shape[:2]
    hl, wl = H // scale, W // scale
    y0 = (hl - lr_crop) // 2
    x0 = (wl - lr_crop) // 2
    hr_crop = hr[y0 * scale:(y0 + lr_crop) * scale, x0 * scale:(x0 + lr_crop) * scale]

    if degradation in ("bicubic", "unknown"):
        lr = load_div2k_lr(img_id, degradation)
        lr_crop_img = lr[y0:y0 + lr_crop, x0:x0 + lr_crop]
    elif degradation == "realesrgan":
        lr_crop_img = realesrgan_degrade(hr_crop, scale, seed=int(img_id))
    else:
        raise ValueError(degradation)
    assert lr_crop_img.shape[:2] == (lr_crop, lr_crop), lr_crop_img.shape
    return np.ascontiguousarray(lr_crop_img), np.ascontiguousarray(hr_crop)
