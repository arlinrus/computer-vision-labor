import numpy as np
import torch


def psnr(pred: np.ndarray, gt: np.ndarray, region: np.ndarray | None = None) -> float:
    def y(img):                                             # RGB -> яркость Y
        x = img.astype(np.float64)
        return 16 + (65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]) / 255
    d = (y(pred) - y(gt)) ** 2
    if region is not None:
        if region.sum() == 0:
            return float("nan")                             # пустая область — считать нечего
        d = d[region.astype(bool)]
    return float(10 * np.log10(255 ** 2 / d.mean()))


class Metrics:
    def __init__(self):
        import pyiqa
        self.lpips = pyiqa.create_metric("lpips", device="cuda")      # готовые реализации
        self.clipiqa = pyiqa.create_metric("clipiqa", device="cuda")

    @staticmethod
    def _t(img):                                            # numpy uint8 -> тензор [1,3,H,W] 0..1
        return torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()

    @torch.no_grad()
    def __call__(self, pred: np.ndarray, gt: np.ndarray, mask_hr: np.ndarray) -> dict:
        return {
            "psnr": psnr(pred, gt),                                    # вся картинка
            "lpips": float(self.lpips(self._t(pred), self._t(gt))),
            "clipiqa": float(self.clipiqa(self._t(pred))),
            "psnr_in": psnr(pred, gt, mask_hr),                        # только внутри маски
            "psnr_out": psnr(pred, gt, 1 - mask_hr),                   # только вне маски
        }
