"""Метрики качества: PSNR (Y-канал), LPIPS (full-reference), CLIP-IQA (no-reference).

Для частичных масок метрики дополнительно считаются отдельно внутри и вне маски:
PSNR — по пикселям маски, LPIPS/CLIP-IQA — по ограничивающему прямоугольнику маски.
"""
import numpy as np
import torch


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> Y (ITU-R BT.601), как в стандартной SR-оценке."""
    x = img.astype(np.float64)
    return 16.0 + (65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]) / 255.0


def psnr(pred: np.ndarray, gt: np.ndarray, region: np.ndarray | None = None) -> float:
    d = (rgb_to_y(pred) - rgb_to_y(gt)) ** 2
    if region is not None:
        if region.sum() == 0:
            return float("nan")
        d = d[region.astype(bool)]
    mse = d.mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def bbox(region: np.ndarray, min_size: int = 64):
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if y1 - y0 < min_size or x1 - x0 < min_size:
        return None
    return y0, y1, x0, x1


class Evaluator:
    def __init__(self, device: str = "cuda"):
        import pyiqa
        self.device = device
        self.lpips = pyiqa.create_metric("lpips", device=device)
        self.clipiqa = pyiqa.create_metric("clipiqa", device=device)

    def _t(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray, mask_hr: np.ndarray) -> dict:
        res = {
            "psnr": psnr(pred, gt),
            "lpips": float(self.lpips(self._t(pred), self._t(gt))),
            "clipiqa": float(self.clipiqa(self._t(pred))),
            "psnr_in": psnr(pred, gt, mask_hr),
            "psnr_out": psnr(pred, gt, 1 - mask_hr),
            "lpips_in": float("nan"),
            "clipiqa_in": float("nan"),
        }
        b = bbox(mask_hr)
        if b is not None:
            y0, y1, x0, x1 = b
            p, g = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
            res["lpips_in"] = float(self.lpips(self._t(p), self._t(g)))
            res["clipiqa_in"] = float(self.clipiqa(self._t(p)))
        return res
