"""Полный эксперимент: изображения x деградации x маски -> results/ и results/metrics.csv.

Пример:
    python source/run_experiments.py                       # все настройки из config.py
    python source/run_experiments.py --images 0801 0804 --degradations bicubic --cpu_offload
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

import config
from data import get_pair
from masks import make_mask, upscale_mask
from metrics import Evaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", default=config.IMAGE_IDS)
    p.add_argument("--degradations", nargs="+", default=config.DEGRADATIONS,
                   choices=["bicubic", "unknown", "realesrgan"])
    p.add_argument("--masks", nargs="+", default=config.MASKS, choices=["ones", "zeros", "combined"])
    p.add_argument("--steps", type=int, default=config.NUM_STEPS)
    p.add_argument("--guidance", type=float, default=config.GUIDANCE_SCALE)
    p.add_argument("--scheduler", default="unipc", choices=["unipc", "ddim"])
    p.add_argument("--feather_px", type=float, default=config.FEATHER_PX)
    p.add_argument("--latent_dilate", type=int, default=config.LATENT_DILATE)
    p.add_argument("--blend_end", type=float, default=config.BLEND_END)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--cpu_offload", action="store_true", help="для GPU с <16 ГБ памяти")
    p.add_argument("--out", default=str(config.RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    from pathlib import Path
    from masked_faithdiff import load_pipeline, masked_super_resolve

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_args.json").write_text(json.dumps(vars(args), indent=2))

    pipe = load_pipeline(cpu_offload=args.cpu_offload, scheduler=args.scheduler)
    evaluator = Evaluator("cuda")
    csv_path = out_root / "metrics.csv"
    rows = pd.read_csv(csv_path, dtype={"image": str}).to_dict("records") if csv_path.exists() else []
    done = {(r["image"], r["degradation"], r["mask"]) for r in rows}

    for deg in args.degradations:
        for img_id in args.images:
            lr, hr = get_pair(img_id, deg)
            d = out_root / deg / img_id
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(lr).save(d / "lr.png")
            Image.fromarray(hr).save(d / "hr.png")
            lr_pil = Image.fromarray(lr)

            for kind in args.masks:
                if (img_id, deg, kind) in done:
                    print(f"skip {deg}/{img_id}/{kind}")
                    continue
                m_lr = make_mask(kind, lr.shape[0], lr.shape[1], seed=int(img_id))
                np.save(d / f"mask_{kind}.npy", m_lr)
                Image.fromarray(m_lr * 255).save(d / f"mask_{kind}.png")

                sr, st = masked_super_resolve(
                    pipe, lr_pil, m_lr, num_inference_steps=args.steps,
                    guidance_scale=args.guidance, feather_px=args.feather_px,
                    latent_dilate=args.latent_dilate, blend_end=args.blend_end, seed=args.seed)
                sr.save(d / f"sr_{kind}.png")

                m_hr = upscale_mask(m_lr, hr.shape[0], hr.shape[1]).astype(np.uint8)
                met = evaluator(np.asarray(sr), hr, m_hr)
                row = {"image": img_id, "degradation": deg, "mask": kind, **met, **st}
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"{deg}/{img_id}/{kind}: " +
                      ", ".join(f"{k}={v:.4f}" for k, v in met.items()) + f", time={st['time_s']:.1f}s")

            # бикубический апскейл — опорная точка (baseline) для всех масок
            if (img_id, deg, "bicubic_baseline") not in done:
                bic = np.asarray(lr_pil.resize((hr.shape[1], hr.shape[0]), Image.BICUBIC))
                Image.fromarray(bic).save(d / "bicubic.png")
                m_hr = upscale_mask(make_mask("combined", *lr.shape[:2], seed=int(img_id)),
                                    hr.shape[0], hr.shape[1]).astype(np.uint8)
                met = evaluator(bic, hr, m_hr)
                rows.append({"image": img_id, "degradation": deg, "mask": "bicubic_baseline", **met})
                pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"done -> {csv_path}")


if __name__ == "__main__":
    main()
