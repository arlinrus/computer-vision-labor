import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from masks import make_mask
from model import load_model
from sampling import masked_sr

p = argparse.ArgumentParser()
p.add_argument("--input", nargs="+", default=sorted(glob.glob(str(config.ROOT / "photos" / "*.png"))))
p.add_argument("--masks", nargs="+", default=["ones", "zeros", "combined", "half"])
args = p.parse_args()

pipe = load_model()
import pyiqa
clipiqa = pyiqa.create_metric("clipiqa", device="cuda")      # оценка качества без эталона


def score(img: Image.Image) -> float:
    t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
    return float(clipiqa(t))


out_dir = config.RESULTS_DIR / "photos"
out_dir.mkdir(parents=True, exist_ok=True)
rows = []

for path in args.input:
    name = Path(path).stem
    lr = Image.open(path).convert("RGB")                       # фото 256x256
    bic = lr.resize((lr.width * config.UPSCALE, lr.height * config.UPSCALE), Image.BICUBIC)
    rows.append({"photo": name, "mask": "bicubic", "clipiqa": score(bic)})

    results, masks_hr = {}, {}
    for kind in args.masks:
        mask = make_mask(kind, lr.height, lr.width)            # маска 256x256
        sr = masked_sr(pipe, lr, mask)                         # <- наш алгоритм
        sr.save(out_dir / f"{name}_{kind}.png")
        results[kind] = sr
        masks_hr[kind] = np.kron(mask, np.ones((config.UPSCALE, config.UPSCALE), np.uint8))
        rows.append({"photo": name, "mask": kind, "clipiqa": score(sr)})
        print(name, kind, "CLIP-IQA =", round(rows[-1]["clipiqa"], 4))

    # сравнение: бикубика и результаты с масками (zeros не рисуем — он совпадает с бикубикой)
    panels = [("бикубика", bic, None)] + [(f"маска {k}", results[k], masks_hr[k])
                                          for k in ["ones", "combined", "half"] if k in results]
    fig, ax = plt.subplots(2, len(panels), figsize=(4 * len(panels), 8))
    c = slice(384, 640)                                        # центр 256x256 для увеличения
    for j, (title, im, m) in enumerate(panels):
        arr = np.asarray(im)
        ax[0, j].imshow(arr)
        ax[1, j].imshow(arr[c, c])
        if m is not None and 0 < m.mean() < 1:                 # граница маски жёлтым
            ax[0, j].contour(m, levels=[0.5], colors="yellow", linewidths=1)
            ax[1, j].contour(m[c, c], levels=[0.5], colors="yellow", linewidths=1)
        q = [r["clipiqa"] for r in rows if r["photo"] == name and r["mask"] == (title.split()[-1] if m is not None else "bicubic")][0]
        ax[0, j].set_title(f"{title}\nCLIP-IQA {q:.3f}")
        ax[1, j].set_title("увеличенный центр")
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_grid.png", dpi=90)
    plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(out_dir / "clipiqa.csv", index=False)
print(df.pivot(index="photo", columns="mask", values="clipiqa").round(4))
print("готово ->", out_dir)
