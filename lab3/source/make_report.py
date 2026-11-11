"""Графики и таблицы для README из results/metrics.csv -> report/.

    python source/make_report.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config

MASK_ORDER = ["ones", "combined", "zeros"]
MASK_LABEL = {"ones": "маска = 1 (всё)", "combined": "комбинированная", "zeros": "маска = 0 (ничего)",
              "bicubic_baseline": "бикубика"}
METRIC_LABEL = {"psnr": "PSNR, дБ (↑)", "lpips": "LPIPS (↓)", "clipiqa": "CLIP-IQA (↑)"}


def bar_metrics(df, out):
    degs = [d for d in config.DEGRADATIONS if d in df.degradation.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(degs))
    w = 0.26
    for ax, met in zip(axes, ["psnr", "lpips", "clipiqa"]):
        for j, m in enumerate(MASK_ORDER):
            sub = df[df["mask"] == m].groupby("degradation")[met]
            mean = [sub.mean().get(d, np.nan) for d in degs]
            std = [sub.std().get(d, np.nan) for d in degs]
            ax.bar(x + (j - 1) * w, mean, w, yerr=std, capsize=3, label=MASK_LABEL[m])
        ax.set_xticks(x, degs)
        ax.set_title(METRIC_LABEL[met])
        ax.set_xlabel("деградация")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Метрики по всему изображению (среднее ± std по изображениям)")
    fig.tight_layout()
    fig.savefig(out / "metrics_full.png", dpi=130)
    plt.close(fig)


def bar_in_out(df, out):
    """Для комбинированной маски: внутри маски должно быть как у 'ones', вне — как у бикубики."""
    degs = [d for d in config.DEGRADATIONS if d in df.degradation.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    variants = [("ones", "in", "SR всего, область маски"),
                ("combined", "in", "комбинир., внутри маски"),
                ("bicubic_baseline", "in", "бикубика, область маски"),
                ("combined", "out", "комбинир., вне маски"),
                ("bicubic_baseline", "out", "бикубика, вне маски")]
    x = np.arange(len(degs))
    w = 0.16
    for ax, met in zip(axes, ["psnr", "lpips", "clipiqa"]):
        for j, (m, reg, lab) in enumerate(variants):
            col = f"{met}_{reg}"
            if col not in df.columns:
                continue
            sub = df[df["mask"] == m].groupby("degradation")[col].mean()
            vals = [sub.get(d, np.nan) for d in degs]
            if np.all(np.isnan(vals)):
                continue
            ax.bar(x + (j - 2) * w, vals, w, label=lab)
        ax.set_xticks(x, degs)
        ax.set_title(METRIC_LABEL[met] + " по областям")
        ax.set_xlabel("деградация")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Комбинированная маска: метрики внутри / вне маски")
    fig.tight_layout()
    fig.savefig(out / "metrics_in_out.png", dpi=130)
    plt.close(fig)


def _overlay(img, mask):
    o = img.astype(np.float32).copy()
    o[mask > 0] = 0.6 * o[mask > 0] + 0.4 * np.array([255, 0, 0])
    return o.astype(np.uint8)


def visual_grid(res, deg, img_id, out):
    d = res / deg / img_id
    if not (d / "sr_combined.png").exists():
        return
    hr = np.asarray(Image.open(d / "hr.png"))
    lr = Image.open(d / "lr.png")
    bic = np.asarray(lr.resize((hr.shape[1], hr.shape[0]), Image.BICUBIC))
    m = np.asarray(Image.open(d / "mask_combined.png").resize((hr.shape[1], hr.shape[0]), Image.NEAREST))
    panels = [("вход (бикубика x4)", bic), ("маска (красным)", _overlay(bic, m))]
    for k in MASK_ORDER:
        if (d / f"sr_{k}.png").exists():
            panels.append((f"SR, {MASK_LABEL[k]}", np.asarray(Image.open(d / f"sr_{k}.png"))))
    panels.append(("HR (эталон)", hr))

    # зум на участок с границей маски
    ys, xs = np.nonzero(np.abs(np.diff(m.astype(int), axis=1)) > 0)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (hr.shape[0] // 2, hr.shape[1] // 2)
    r = 128
    y0, x0 = np.clip(cy - r, 0, hr.shape[0] - 2 * r), np.clip(cx - r, 0, hr.shape[1] - 2 * r)

    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 6.6))
    for j, (title, im) in enumerate(panels):
        axes[0, j].imshow(im)
        axes[0, j].add_patch(plt.Rectangle((x0, y0), 2 * r, 2 * r, fill=False, ec="yellow", lw=1.5))
        axes[0, j].set_title(title, fontsize=9)
        axes[1, j].imshow(im[y0:y0 + 2 * r, x0:x0 + 2 * r])
        axes[1, j].set_title("зум (жёлтая рамка)", fontsize=8)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(f"DIV2K {img_id}, деградация: {deg}")
    fig.tight_layout()
    fig.savefig(out / f"visual_{deg}_{img_id}.png", dpi=110)
    plt.close(fig)


def table(df, out):
    cols = ["psnr", "lpips", "clipiqa", "psnr_in", "psnr_out", "lpips_in", "clipiqa_in", "time_s"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(["degradation", "mask"])[cols].mean().round(4).reset_index()
    md = g.to_markdown(index=False)
    (out / "results_table.md").write_text(md + "\n")
    print(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.RESULTS_DIR))
    p.add_argument("--out", default=str(config.REPORT_DIR))
    a = p.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "metrics.csv", dtype={"image": str})
    bar_metrics(df, out)
    bar_in_out(df, out)
    table(df, out)
    for deg in df.degradation.unique():
        for img_id in sorted(df[df.degradation == deg].image.unique())[:2]:
            visual_grid(res, deg, img_id, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
