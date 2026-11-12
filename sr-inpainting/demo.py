import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import config
from masks import load_mask, make_mask
from model import load_model
from sampling import masked_sr

p = argparse.ArgumentParser()
p.add_argument("--input", nargs="+", required=True, help="одна или несколько картинок до 256x256")
p.add_argument("--mask", default="half", help="ones / zeros / combined / half или путь к .png")
args = p.parse_args()

pipe = load_model()
out_dir = config.RESULTS_DIR / "demo"
out_dir.mkdir(parents=True, exist_ok=True)

for path in args.input:
    lr = Image.open(path).convert("RGB")
    if args.mask.endswith(".png"):
        mask = load_mask(args.mask, lr.height, lr.width)
    else:
        mask = make_mask(args.mask, lr.height, lr.width)
    sr = masked_sr(pipe, lr, mask)
    bic = lr.resize(sr.size, Image.BICUBIC)

    name = Path(path).stem
    sr.save(out_dir / f"{name}_sr.png")
    side = Image.new("RGB", (sr.width * 2 + 10, sr.height), "white")   # бикубика слева, результат справа
    side.paste(bic, (0, 0)); side.paste(sr, (sr.width + 10, 0))
    side.save(out_dir / f"{name}_compare.png")
    print("сохранено:", out_dir / f"{name}_compare.png")
