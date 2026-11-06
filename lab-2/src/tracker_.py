# import cv2
# import numpy as np

# cap = cv2.VideoCapture("mona-lisa.avi")
# ok, reference = cap.read()
# print(ok, reference.shape) 
# cv2.imwrite("reference.png", reference)


# ## найти ключевые точки на эталоне
# gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
# sift = cv2.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)
# print(len(kp), des.shape)

# vis = cv2.drawKeypoints(reference, kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite("key_points.png", vis)


# # сопоставить точки и эталоны с другим кадром
# for _ in range(50):
#     cap.read()
# cap, frame = cap.read()

# gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# matcher = cv2.BFMatcher (cv2.NORM_L2) 
# pairs = matcher.knnMatch(des, des2, k=2)
# good = [p[0] for p in pairs if len(p)==2 and p[0].distance < 0.78 * p[1].distance] # дэвид лоу определяет 70-80 процентов
# print("good pairs are:", len(good))

# # гомография 
# src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)    # точки на эталоне
# dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)   # те же точки на кадре
# H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
# print("инлайеров:", int(mask.sum()), "из", len(good))

# h, w = gray.shape
# m = 20
# corners = np.float32([[-m, -m], [w + m, -m], [w + m, h + m], [-m, h + m]]).reshape(-1, 1, 2)
# quad = cv2.perspectiveTransform(corners, H)                                # где они на кадре

# out = frame.copy()
# cv2.polylines(out, [np.int32(quad)], True, (0, 255, 0), 3)
# cv2.imwrite("result2.png", out)


import csv
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "mona-lisa.avi"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "out.mp4"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "Mona Lisa"
USE_FLOW = True        # оптический поток как запасной вариант
SHOW_POINTS = True     # рисовать ключевые точки на кадре

BOX_COLOR = (0, 255, 0)       # цвет рамки (BGR): зелёный
POINT_COLOR = (0, 0, 255)     # цвет точек (BGR): красный
LABEL_COLOR = (0, 255, 0) # цвет подписи (BGR): розовый

# первый кадр видео 
cap = cv2.VideoCapture(VIDEO)
ok, reference = cap.read()

sift = cv2.SIFT_create(contrastThreshold=0.02) 
matcher = cv2.BFMatcher(cv2.NORM_L2)

gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

# эталон на нескольких масштабах (нужно, когда объект в кадре становится мелким)
scales = [1, 0.6, 0.4, 0.25, 0.15]
levels = []
for s in scales:
    img = gray if s == 1 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k_, d_ = sift.detectAndCompute(img, None)
    if d_ is not None and len(k_) >= 12:
        pts = np.float32([k.pt for k in k_]) / s      # координаты в системе полного эталона
        levels.append((pts, d_))
# print("уровней эталона:", len(levels))

def valid_quad(quad):
    if quad is None:
        return False
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return (np.isfinite(q).all()
            and cv2.isContourConvex(q)
            and cv2.contourArea(q) > 1.0)

# поиск объекта по ключевым точкам
def find_quad(gray_now):
    kp2, des2 = sift.detectAndCompute(gray_now, None)
    if des2 is None or len(kp2) < 12:
        return None, None

    best_quad, best_pts, best_n = None, None, 0
    for pts, d_ in levels:
        pairs = matcher.knnMatch(d_, des2, k=2)
        good = [p[0] for p in pairs
                if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 12:
            continue
        src = pts[[match.queryIdx for match in good]].reshape(-1, 1, 2)
        dst = np.float32([kp2[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            continue
        inl = mask.ravel() == 1
        n = int(inl.sum())
        if n >= 12 and n > best_n:
            quad = cv2.perspectiveTransform(corners, H)
            if not valid_quad(quad):
                continue
            best_n = n
            best_quad = quad
            best_pts = dst[inl]
    return best_quad, best_pts


# оптический поток 
def follow(gray_now, prev_gray, prev_quad):
    mask = np.zeros_like(prev_gray)  # mask refers to prev_quad / prev_gray
    cv2.fillConvexPoly(mask, np.int32(prev_quad), 255)
    p0 = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 5, mask=mask)
    if p0 is None or len(p0) < 8:
        return None, None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_now, p0, None)
    if p1 is None or st is None:
        return None, None
    ok_ = st.ravel() == 1
    if ok_.sum() < 8:
        return None, None
    H, inlier_mask = cv2.findHomography(p0[ok_], p1[ok_], cv2.RANSAC, 3.0)
    if H is None or inlier_mask is None:
        return None, None
    quad = cv2.perspectiveTransform(prev_quad, H)
    if not valid_quad(quad):
        return None, None
    inl = inlier_mask.ravel() == 1
    return quad, p1[ok_][inl]

# цикл по видео 
fps = cap.get(cv2.CAP_PROP_FPS)
if not np.isfinite(fps) or fps <= 0:
    fps = 30
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

csv_path = Path(OUTPUT).with_suffix(".csv")
prev_gray, prev_quad = None, None
found = flow = lost = 0

with csv_path.open("w", newline="", encoding="utf-8") as f:
    log = csv.DictWriter(f, fieldnames=["frame", "method", "inliers"])
    log.writeheader()

    frame_idx = 1  
    while True:
        ok, frame = cap.read()
        if not ok:                                         # видео закончилось
            break
        gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quad, pts = find_quad(gray_now)                    # 1) пробуем SIFT
        if quad is not None:
            found += 1
            method = "SIFT"
        elif USE_FLOW and prev_quad is not None:           # 2) не вышло: оптический поток
            quad, pts = follow(gray_now, prev_gray, prev_quad)
            if quad is not None:
                flow += 1
                method = "FLOW"
            else:
                method = "LOST"
        else:
            method = "LOST"

        if quad is None:
            lost += 1

        prev_gray, prev_quad = gray_now, quad              # запоминаем для следующего кадра

        log.writerow({
                    "frame": frame_idx,
                    "method": method,
                    "inliers": len(pts) if pts is not None else 0,
                })
        frame_idx += 1
        
        if quad is not None:
            cv2.polylines(frame, [np.int32(quad)], True, BOX_COLOR, 3)
            if SHOW_POINTS and pts is not None:
                for x, y in pts.reshape(-1, 2):
                    cv2.circle(frame, (int(x), int(y)), 1, POINT_COLOR, -1)
            x, y = quad[0, 0]
            cv2.putText(frame, LABEL, (int(x), max(int(y) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_COLOR, 2)
        writer.write(frame)

cap.release()
writer.release()
print("найдено детекцией:", found, "| потоком:", flow, "| потеряно:", lost)
        