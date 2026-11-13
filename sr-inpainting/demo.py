import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from data import get_pair
from masks import make_mask
from model import load_model
from sampling import masked_sr

p = argparse.ArgumentParser()
p.add_argument("--images", nargs="+", default=config.IMAGE_IDS)
p.add_argument("--degradation", default="realesrgan", choices=config.DEGRADATIONS)
args = p.parse_args()

pipe = load_model()
out_dir = config.RESULTS_DIR / "demo"
out_dir.mkdir(parents=True, exist_ok=True)
kinds = ["ones", "combined", "half"]

for name in args.images:
    lr, hr = get_pair(name, args.degradation)                 # плохая 256 и эталон 1024
    bic = Image.fromarray(lr).resize((hr.shape[1], hr.shape[0]), Image.BICUBIC)

    panels = [("бикубика", np.asarray(bic), None)]
    for kind in kinds:
        mask = make_mask(kind, *lr.shape[:2])                  # маска 256x256
        sr = masked_sr(pipe, Image.fromarray(lr), mask)        # <- наш алгоритм
        sr.save(out_dir / f"{name}_{args.degradation}_{kind}.png")
        mask_hr = np.kron(mask, np.ones((config.UPSCALE, config.UPSCALE), np.uint8))
        panels.append((f"маска {kind}", np.asarray(sr), mask_hr))
        print(name, args.degradation, kind, "готово")
    panels.append(("эталон", hr, None))

    fig, ax = plt.subplots(2, len(panels), figsize=(4 * len(panels), 8))
    c = slice(384, 640)                                        # центр 256x256 для увеличения
    for j, (title, im, m) in enumerate(panels):
        ax[0, j].imshow(im)
        ax[1, j].imshow(im[c, c])
        if m is not None and 0 < m.mean() < 1:                 # граница маски жёлтым
            ax[0, j].contour(m, levels=[0.5], colors="yellow", linewidths=1)
            ax[1, j].contour(m[c, c], levels=[0.5], colors="yellow", linewidths=1)
        ax[0, j].set_title(title)
        ax[1, j].set_title("увеличенный центр")
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(f"{name}, деградация {args.degradation}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_{args.degradation}_grid.png", dpi=90)
    plt.close(fig)

print("готово ->", out_dir)
