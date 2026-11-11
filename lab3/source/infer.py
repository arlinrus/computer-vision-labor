"""SR-inpainting одного изображения по своей маске.

    python source/infer.py --input photo.png --mask mask.png --output out.png
    python source/infer.py --input photo.png --mask_kind combined --output out.png

Маска — бинарный .npy или .png того же размера, что и вход (1/белое = улучшать).
"""
import argparse

import numpy as np
from PIL import Image

import config
from masks import load_mask, make_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mask", help="путь к маске (.npy/.png)")
    p.add_argument("--mask_kind", default="ones", choices=["ones", "zeros", "combined"])
    p.add_argument("--upscale", type=int, default=config.UPSCALE)
    p.add_argument("--prompt", default=config.PROMPT)
    p.add_argument("--steps", type=int, default=config.NUM_STEPS)
    p.add_argument("--guidance", type=float, default=config.GUIDANCE_SCALE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--cpu_offload", action="store_true")
    a = p.parse_args()

    from masked_faithdiff import load_pipeline, masked_super_resolve

    img = Image.open(a.input).convert("RGB")
    mask = load_mask(a.mask, img.height, img.width) if a.mask else make_mask(a.mask_kind, img.height, img.width)
    pipe = load_pipeline(cpu_offload=a.cpu_offload)
    out, st = masked_super_resolve(pipe, img, mask, upscale=a.upscale, prompt=a.prompt,
                                   num_inference_steps=a.steps, guidance_scale=a.guidance, seed=a.seed)
    out.save(a.output)
    print(f"saved {a.output}  ({st})")


if __name__ == "__main__":
    main()
