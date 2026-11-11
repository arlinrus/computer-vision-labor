"""Маски SR-inpainting.

Маска задаётся бинарным массивом (H_lr x W_lr) в разрешении ВХОДНОГО изображения:
1 — область, которую нужно улучшить моделью, 0 — область, которую нужно оставить
без улучшения (простой бикубический апскейл входа).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def make_mask(kind: str, h: int, w: int, seed: int = 0) -> np.ndarray:
    if kind == "ones":
        return np.ones((h, w), dtype=np.uint8)
    if kind == "zeros":
        return np.zeros((h, w), dtype=np.uint8)
    if kind == "combined":
        # объединение эллипса и прямоугольника: есть и выпуклые, и прямые границы,
        # покрытие ~25-40% площади
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h * rng.uniform(0.3, 0.45), w * rng.uniform(0.3, 0.45)
        ry, rx = h * rng.uniform(0.18, 0.25), w * rng.uniform(0.2, 0.28)
        ell = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        y0, x0 = int(h * rng.uniform(0.5, 0.6)), int(w * rng.uniform(0.45, 0.6))
        rect = (yy >= y0) & (yy < y0 + int(h * 0.35)) & (xx >= x0) & (xx < x0 + int(w * 0.35))
        return (ell | rect).astype(np.uint8)
    raise ValueError(f"unknown mask kind {kind}")


def load_mask(path: str | Path, h: int, w: int) -> np.ndarray:
    """Маска из .npy или изображения (>127 -> 1). Должна совпадать по размеру со входом."""
    path = Path(path)
    m = np.load(path) if path.suffix == ".npy" else np.array(Image.open(path).convert("L")) > 127
    m = (np.asarray(m) > 0).astype(np.uint8)
    if m.shape != (h, w):
        raise ValueError(f"mask shape {m.shape} != input shape {(h, w)}")
    return m


def upscale_mask(mask_lr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Маска LR -> HR (nearest), остаётся бинарной."""
    return np.array(Image.fromarray(mask_lr * 255).resize((out_w, out_h), Image.NEAREST)) > 127


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def latent_mask(mask_hr: torch.Tensor, scale_factor: int = 8, dilate: int = 1) -> torch.Tensor:
    """HR-маска [1,1,H,W] -> мягкая латентная маска [1,1,H/8,W/8] в [0,1].

    area-интерполяция даёт долю покрытия латентного пикселя; dilate расширяет
    маску на несколько латентных пикселей, чтобы шов генерировался моделью
    (VAE-латент имеет рецептивное поле больше 8 px). Затем лёгкое сглаживание.
    """
    m = F.interpolate(mask_hr.float(), scale_factor=1 / scale_factor, mode="area")
    if dilate > 0:
        m = F.max_pool2d(m, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
    m = _gaussian_blur(m, 0.5 * max(dilate, 1)).clamp(0, 1)
    m[m > 0.999] = 1.0
    return m


def pixel_mask(mask_hr: torch.Tensor, feather_px: float) -> torch.Tensor:
    """Мягкая маска для финальной склейки в пикселях (размытие границы)."""
    return _gaussian_blur(mask_hr.float(), feather_px).clamp(0, 1)
