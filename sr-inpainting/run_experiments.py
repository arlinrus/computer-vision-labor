import argparse

import numpy as np
import pandas as pd
from PIL import Image

import config
from data import get_pair
from masks import make_mask
from metrics import Metrics
from model import load_model
from sampling import masked_sr

p = argparse.ArgumentParser()
p.add_argument("--images", nargs="+", default=config.IMAGE_IDS)
p.add_argument("--degradations", nargs="+", default=config.DEGRADATIONS)
args = p.parse_args()

pipe = load_model()
metrics = Metrics()
rows = []

for deg in args.degradations:
    for img_id in args.images:
        lr, hr = get_pair(img_id, deg)                          # плохая 256 и эталон 1024
        folder = config.RESULTS_DIR / deg / img_id
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(lr).save(folder / "lr.png")
        Image.fromarray(hr).save(folder / "hr.png")

        for kind in config.MASKS:
            mask = make_mask(kind, *lr.shape[:2])               # маска 256x256
            out = masked_sr(pipe, Image.fromarray(lr), mask)    # <- наш алгоритм
            out.save(folder / f"sr_{kind}.png")
            Image.fromarray(mask * 255).save(folder / f"mask_{kind}.png")

            mask_hr = np.kron(mask, np.ones((config.UPSCALE, config.UPSCALE), np.uint8))  # маска 1024x1024
            m = metrics(np.asarray(out), hr, mask_hr)
            rows.append({"degradation": deg, "image": img_id, "mask": kind, **m})
            print(deg, img_id, kind, {k: round(v, 4) for k, v in m.items()})

        pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)   # сохраняем после каждой картинки

print("готово ->", config.RESULTS_DIR / "metrics.csv")
