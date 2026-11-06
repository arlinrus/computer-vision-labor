"""Количественная оценка трекера.

1. Размытые видео mona-lisa-blur*.avi — это mona-lisa.avi с добавленным смазом
   (те же кадры). Поэтому результат трекинга на чистом видео, где почти все кадры
   найдены детекцией с десятками инлаеров, используется как эталонная разметка
   (pseudo ground truth), и для размытых версий считается ошибка углов рамки.
2. Сравнение детекторов SIFT / ORB / AKAZE и вклада KLT и прогноза.
3. Графики по кадрам: режим работы, число инлаеров, ошибка углов.

Запуск:  python source/evaluate.py --videos data --out results
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker import track_video

MODES = ["detect", "flow", "predict", "lost"]
MODE_COLORS = {"detect": "#2ca02c", "flow": "#ff9f0e", "predict": "#1f77b4", "lost": "#d62728"}

# если какой то рамки нет, то можно использовать
def corner_error(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(a - b, axis=1).mean())


def iou(a, b, shape=(360, 640)):
    if a is None or b is None:
        return 0.0
    ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.int32(np.round(a))], 1)
    cv2.fillPoly(mb, [np.int32(np.round(b))], 1)
    u = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / u) if u else 0.0


def blurred_mask(clean_path, blur_path, thr=3.0):
    a, b = cv2.VideoCapture(clean_path), cv2.VideoCapture(blur_path)
    a.read(), b.read()
    mask, shifts, hann = [], [], None
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        m = np.abs(f1.astype(np.int16) - f2).mean() > thr
        mask.append(m)
        if m:
            g1 = np.float32(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY))
            g2 = np.float32(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
            if hann is None:
                hann = cv2.createHanningWindow(g1.shape[::-1], cv2.CV_32F)
            shifts.append(cv2.phaseCorrelate(g1, g2, hann)[0])
        else:
            shifts.append((0.0, 0.0))
    return np.array(mask), np.float32(shifts)


def summarize(res, gt=None, seconds=None, mask=None):
    n = len(res)
    s = {m: round(100 * sum(r.mode == m for r in res) / n, 1) for m in MODES}
    det = [r for r in res if r.mode == "detect"]
    s["median_inliers"] = float(np.median([r.n_inliers for r in det])) if det else 0
    s["median_reproj_px"] = round(float(np.nanmedian([r.reproj_err for r in det])), 2) if det else None
    if seconds:
        s["fps"] = round(n / seconds, 1)
    if gt is not None:
        err = np.array([corner_error(r.quad, g.quad) for r, g in zip(res, gt)])
        ious = np.array([iou(r.quad, g.quad) for r, g in zip(res, gt)])
        s["median_corner_err_px"] = round(float(np.nanmedian(err)), 2)
        s["p90_corner_err_px"] = round(float(np.nanpercentile(err, 90)), 2)
        s["mean_iou"] = round(float(ious.mean()), 3)
        s["success_iou>0.5"] = round(100 * float((ious > 0.5).mean()), 1)
        if mask is not None and mask[0].any():
            mask, shifts = mask
            m = mask[:len(res)]
            s["blur_frames"] = int(m.sum())
            s["blur_tracked_%"] = round(100 * float(np.mean([res[i].quad is not None
                                                             for i in np.where(m)[0]])), 1)
            s["blur_mean_iou"] = round(float(ious[m].mean()), 3)
            s["blur_median_corner_err_px"] = round(float(np.nanmedian(err[m])), 2) \
                if np.isfinite(err[m]).any() else None
            idx = np.where(m)[0]
            err_c = [corner_error(res[i].quad, None if gt[i].quad is None else gt[i].quad + shifts[i])
                     for i in idx]
            s["blur_content_shift_px"] = round(float(np.median(np.linalg.norm(shifts[idx], axis=1))), 1)
            s["blur_median_corner_err_shift_corrected_px"] = round(float(np.nanmedian(err_c)), 2) \
                if np.isfinite(err_c).any() else None
        return s, err, ious
    return s, None, None


def timed(path, **kw):
    t = time.perf_counter()
    r = track_video(path, verbose=False, **kw)
    return r, time.perf_counter() - t


def plot_timeline(ax, res, title):
    for i, r in enumerate(res):
        ax.axvspan(i, i + 1, color=MODE_COLORS[r.mode], lw=0)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, len(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--own", nargs="*", default=[], help="собственные видео для теста")
    ap.add_argument("--no-ablation", action="store_true",
                    help="не сравнивать конфигурации (быстрый прогон)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    V = lambda n: os.path.join(a.videos, n)
    names = ["mona-lisa.avi", "mona-lisa-blur.avi", "mona-lisa-blur-extra-credit.avi"]

    # --- 1. pseudo ground truth --------------------------------------------- #
    gt, _ = timed(V(names[0]))
    masks = {n: blurred_mask(V(names[0]), V(n)) for n in names[1:]}
    table, per_video = {}, {}
    for name in names:
        res, sec = timed(V(name))
        s, err, ious = summarize(res, None if name == names[0] else gt, sec, masks.get(name))
        table[name] = s
        per_video[name] = (res, err)
        print(name, s)

    # --- 2. ablation: детекторы и компоненты ------------------------------- #
    configs = {
        "SIFT (только детекция)": dict(detector="sift", use_flow=False, use_predict=False, use_roi=False),
        "SIFT + ROI": dict(detector="sift", use_flow=False, use_predict=False),
        "SIFT + ROI + KLT": dict(detector="sift", use_predict=False),
        "SIFT + ROI + KLT + прогноз": dict(detector="sift"),
        "ORB + ROI + KLT + прогноз": dict(detector="orb"),
        "AKAZE + ROI + KLT + прогноз": dict(detector="akaze"),
    }
    ablation = {}
    if a.no_ablation:
        configs = {}
        prev = os.path.join(a.out, "metrics.json")
        if os.path.exists(prev):  # сохраняем результаты прошлого полного прогона
            with open(prev, encoding="utf-8") as f:
                ablation = json.load(f).get("ablation", {})
    for cname, kw in configs.items():
        ablation[cname] = {}
        for name in names[1:]:
            res, sec = timed(V(name), **kw)
            s, _, _ = summarize(res, gt, sec, masks[name])
            ablation[cname][name] = s
            print(cname, name, s)

    # --- 3. собственные видео ---------------------------------------------- #
    own = {}
    for p in a.own:
        res, sec = timed(p)
        own[os.path.basename(p)] = summarize(res, None, sec)[0]
        per_video[os.path.basename(p)] = (res, None)
        print(p, own[os.path.basename(p)])

    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(main=table, ablation=ablation, own=own), f, ensure_ascii=False, indent=2)

    # --- графики ----------------------------------------------------------- #
    vids = list(per_video)
    fig, axes = plt.subplots(len(vids), 1, figsize=(11, 1.1 * len(vids) + 0.8), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, vids):
        plot_timeline(ax, per_video[v][0], v)
    axes[-1].set_xlabel("номер кадра")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODE_COLORS.values()]
    fig.suptitle("Режим работы трекера по кадрам")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, ["детекция", "KLT-поток", "прогноз (фаз. корреляция)", "потерян"],
               loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(os.path.join(a.out, "timeline.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for v in names:
        res = per_video[v][0]
        ax[0].plot([r.n_inliers if r.mode == "detect" else np.nan for r in res], lw=1, label=v)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("инлаеров RANSAC")
    ax[0].set_title("Число инлаеров при детекции")
    ax[0].legend(fontsize=8)
    for k, v in enumerate(names[1:]):
        ax[1].plot(per_video[v][1], lw=1, label=v)
        m = masks[v][0]
        for i in np.where(m)[0]:
            ax[1].axvspan(i, i + 1, color="0.85", lw=0,
                          label="кадры со смазом" if (k == 1 and i == np.where(m)[0][0]) else None)
    ax[1].set_ylabel("ошибка углов, px")
    ax[1].set_xlabel("номер кадра")
    ax[1].set_title("Отклонение рамки от разметки (трекинг по чистому видео)\n"
                    "в серых зонах смаз сдвигает изображение на ~85 px, поэтому ошибка там велика")
    ax[1].set_ylim(0, 40)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "per_frame.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
