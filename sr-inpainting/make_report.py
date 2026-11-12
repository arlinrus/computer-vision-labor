import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config

config.REPORT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(config.RESULTS_DIR / "metrics.csv", dtype={"image": str})

table = df.groupby(["degradation", "mask"])[["psnr", "lpips", "clipiqa", "psnr_in", "psnr_out"]].mean().round(4)
(config.REPORT_DIR / "results_table.md").write_text(table.reset_index().to_markdown(index=False))
print(table)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, title) in zip(axes, [("psnr", "PSNR ↑"), ("lpips", "LPIPS ↓"), ("clipiqa", "CLIP-IQA ↑")]):
    table[col].unstack("mask")[config.MASKS].plot.bar(ax=ax, rot=0)
    ax.set_title(title); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(config.REPORT_DIR / "metrics.png", dpi=120)

for deg in df.degradation.unique():
    img_id = sorted(df[df.degradation == deg].image.unique())[0]
    d = config.RESULTS_DIR / deg / img_id
    lr = Image.open(d / "lr.png")
    panels = [("бикубика", lr.resize((1024, 1024), Image.BICUBIC)),
              ("маска (белое = улучшать)", Image.open(d / "mask_combined.png").resize((1024, 1024)))]
    panels += [(f"маска: {k}", Image.open(d / f"sr_{k}.png")) for k in config.MASKS]
    panels += [("эталон", Image.open(d / "hr.png"))]
    fig, axes = plt.subplots(2, len(panels), figsize=(3 * len(panels), 6))
    for j, (title, im) in enumerate(panels):
        im = np.asarray(im)
        axes[0, j].imshow(im, cmap="gray"); axes[0, j].set_title(title, fontsize=9)
        axes[1, j].imshow(im[384:640, 384:640], cmap="gray"); axes[1, j].set_title("зум (центр)", fontsize=8)
        for a in axes[:, j]: a.axis("off")
    fig.suptitle(f"DIV2K {img_id}, порча: {deg}")
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / f"visual_{deg}.png", dpi=100)

print("графики ->", config.REPORT_DIR)
