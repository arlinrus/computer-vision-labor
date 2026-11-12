import io
import urllib.request
import zipfile

import cv2
import numpy as np
from PIL import Image

import config

URLS = {                                     
    "hr": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",              # эталоны
    "bicubic": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",  # уменьшены x4 бикубикой
    "unknown": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip",  # x4 с неизвестным размытием
}


def read_from_zip(kind: str, name_end: str) -> np.ndarray:
    path = config.DATA_DIR / f"{kind}.zip"
    if not path.exists():
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("скачиваю", URLS[kind])
        urllib.request.urlretrieve(URLS[kind], path)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(name_end))
        return np.array(Image.open(io.BytesIO(z.read(name))).convert("RGB"))


def realesrgan_degrade(hr: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(hr, (0, 0), rng.uniform(1.0, 2.5))                         # размытие
    h, w = hr.shape[:2]
    img = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)             # уменьшение в 4 раза
    img = img.astype(np.float32) + rng.normal(0, rng.uniform(5, 15), img.shape)       # гауссов шум
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(30, 60))                                                     # качество JPEG 30..60
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def get_pair(img_id: str, degradation: str):
    s, k = config.LR_SIZE, config.UPSCALE
    hr_full = read_from_zip("hr", f"/{img_id}.png")
    y0 = (hr_full.shape[0] // k - s) // 2                       # координаты центра в масштабе LR
    x0 = (hr_full.shape[1] // k - s) // 2
    hr = hr_full[y0 * k:(y0 + s) * k, x0 * k:(x0 + s) * k]      # эталон 1024x1024

    if degradation in ("bicubic", "unknown"):
        lr_full = read_from_zip(degradation, f"/{img_id}x4.png")
        lr = lr_full[y0:y0 + s, x0:x0 + s]                      # тот же кусок из готовой плохой версии
    elif degradation == "realesrgan":
        lr = realesrgan_degrade(hr, seed=int(img_id))           # портим сами
    else:
        raise ValueError(degradation)
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)
