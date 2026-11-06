"""Plot frame-level tracking diagnostics and compare four CSV logs.

Usage:
python3 plot_tracking.py results/mona.csv results/blur.csv results/extra.csv results/my.csv
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("Укажи CSV-файлы после имени скрипта")

out_dir = Path("images")
out_dir.mkdir(exist_ok=True)
summary = []

for path_text in sys.argv[1:]:
    path = Path(path_text)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"Пропуск пустого CSV: {path}")
        continue

    frames = np.array([int(r["frame"]) for r in rows])
    methods = [r["method"] for r in rows]
    inliers = np.array([int(r["inliers"]) for r in rows])
    counts = {m: methods.count(m) for m in ("SIFT", "FLOW", "LOST")}
    total = len(rows)
    summary.append((path.stem, *[100 * counts[m] / total for m in ("SIFT", "FLOW", "LOST")]))

    # Plot 1: tracking mode per frame (not a measure of correctness)
    codes = {"LOST": 0, "FLOW": 1, "SIFT": 2}
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.step(frames, [codes[m] for m in methods], where="mid")
    ax.set_yticks([0, 1, 2], ["LOST", "FLOW", "SIFT"])
    ax.set_xlabel("Номер кадра")
    ax.set_title(f"Способ отслеживания — {path.stem}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{path.stem}_modes.png", dpi=160)
    plt.close(fig)

    # Plot 2: SIFT and FLOW inliers are shown separately
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for m, title in (("SIFT", "SIFT + RANSAC"), ("FLOW", "Лукас–Канаде + RANSAC")):
        keep = np.array([value == m for value in methods])
        ax.scatter(frames[keep], inliers[keep], s=9, label=title)
    ax.set_xlabel("Номер кадра")
    ax.set_ylabel("Количество инлайеров")
    ax.set_title(f"Инлайеры RANSAC — {path.stem}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{path.stem}_inliers.png", dpi=160)
    plt.close(fig)

if not summary:
    raise SystemExit("Нет непустых CSV для построения графиков")

names = [r[0] for r in summary]
values = np.array([r[1:] for r in summary])
fig, ax = plt.subplots(figsize=(10, 4))
left = np.zeros(len(names))
for i, label in enumerate(("SIFT", "FLOW", "LOST")):
    ax.barh(names, values[:, i], left=left, label=label)
    left += values[:, i]
ax.set_xlabel("Доля обработанных кадров, %")
ax.set_xlim(0, 100)
ax.set_title("Результаты обработки четырёх видео")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "comparison.png", dpi=160)
plt.close(fig)

for name, sift, flow, lost in summary:
    print(f"{name}: SIFT={sift:.1f}%, FLOW={flow:.1f}%, LOST={lost:.1f}%")
print("Графики сохранены в", out_dir)
