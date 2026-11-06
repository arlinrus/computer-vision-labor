"""Трекинг плоского объекта на видео по ключевым точкам.

Первый кадр видео — эталонное изображение объекта. На каждом следующем кадре:
  1) детектируем ключевые точки и считаем дескрипторы (SIFT / ORB / AKAZE) —
     сначала в увеличенной окрестности прошлой рамки, затем во всём кадре;
  2) сопоставляем их с дескрипторами эталона, описанного на нескольких
     масштабах (kNN + тест Лоу);
  3) оцениваем гомографию эталон -> кадр робастным методом (MAGSAC++);
  4) проверяем гомографию на правдоподобие (выпуклость, площадь, число инлаеров);
  5) если детекция не удалась (мелкий объект, смаз) — переносим рамку с прошлого
     кадра оптическим потоком Лукаса–Канаде (KLT) по точкам внутри объекта;
  6) если не сработал и поток (сильный смаз) — сдвигаем рамку на глобальный
     сдвиг кадра, найденный фазовой корреляцией;
  7) рисуем четырёхугольник — проекцию углов эталона — и подпись.

Запуск:  python source/tracker.py video.avi -o out.mp4 -l "Mona Lisa"
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Детекторы
# --------------------------------------------------------------------------- #
def make_detector(name: str, n_features: int = 0):
    """Возвращает (детектор, норма для сопоставления дескрипторов)."""
    name = name.lower()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02), cv2.NORM_L2
    if name == "orb":
        return cv2.ORB_create(nfeatures=n_features or 3000, scaleFactor=1.2, nlevels=8,
                              fastThreshold=10), cv2.NORM_HAMMING
    if name == "akaze":
        # в OpenCV 5 AKAZE перенесён в модуль xfeatures2d (opencv-contrib-python)
        create = getattr(cv2, "AKAZE_create", None) or cv2.xfeatures2d.AKAZE_create
        return create(threshold=5e-4), cv2.NORM_HAMMING
    raise ValueError(f"Неизвестный детектор: {name}")


def make_matcher(norm: int):
    if norm == cv2.NORM_L2:
        # FLANN (KD-деревья) — быстрый приближённый поиск для float-дескрипторов
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    # Для бинарных дескрипторов — полный перебор по расстоянию Хэмминга
    return cv2.BFMatcher(norm)


# --------------------------------------------------------------------------- #
#  Геометрические проверки
# --------------------------------------------------------------------------- #
def quad_is_plausible(quad: np.ndarray, frame_shape, ref_area: float,
                      prev_quad: np.ndarray | None) -> bool:
    """Отбрасывает вырожденные гомографии: самопересечения, невыпуклость,
    слишком маленькую/большую площадь, резкие скачки относительно прошлого кадра."""
    q = quad.reshape(-1, 2).astype(np.float32)
    if not np.all(np.isfinite(q)):
        return False
    if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(q)
    h, w = frame_shape[:2]
    if area < 0.002 * h * w or area > 4.0 * ref_area:
        return False
    # противоположные стороны не должны отличаться по длине больше чем в 4 раза
    sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
    if sides.min() < 1e-3 or max(sides[0] / sides[2], sides[2] / sides[0],
                                 sides[1] / sides[3], sides[3] / sides[1]) > 4.0:
        return False
    if prev_quad is not None:
        prev_area = cv2.contourArea(prev_quad.reshape(-1, 2).astype(np.float32))
        if prev_area > 0 and not (0.33 < area / prev_area < 3.0):
            return False
    return True


# --------------------------------------------------------------------------- #
#  Трекер
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    quad: np.ndarray | None          # 4x2 углы объекта в кадре (или None)
    mode: str                        # 'detect' | 'flow' | 'lost'
    n_kp: int = 0                    # ключевых точек в кадре
    n_matches: int = 0               # сопоставлений после теста Лоу (лучший уровень)
    n_inliers: int = 0               # инлаеров RANSAC
    reproj_err: float = float("nan")  # средняя ошибка репроекции инлаеров, px
    level: int = -1                  # уровень пирамиды эталона, давший результат


@dataclass
class KeypointTracker:
    reference: np.ndarray                      # эталон (BGR)
    detector_name: str = "sift"
    n_features: int = 0                        # 0 = без ограничения (для SIFT)
    ref_scales: tuple = (1.0, 0.6, 0.4, 0.25, 0.15)  # пирамида эталона
    ratio: float = 0.8                         # порог теста Лоу
    ransac_thresh: float = 4.0                 # px
    min_inliers: int = 12
    use_flow: bool = True                      # KLT-трекинг между детекциями
    use_roi: bool = True                       # искать сначала в окрестности прошлой рамки
    roi_size: int = 320                        # до какого размера увеличивать объект в окне, px
    max_upscale: float = 3.0
    max_flow_frames: int = 10_000                # сколько кадров подряд можно вести только потоком
    use_predict: bool = True                   # прогноз по фазовой корреляции при смазе
    max_predict_frames: int = 30
    smooth: float = 0.0                        # экспоненциальное сглаживание углов (0 = выкл)

    # внутреннее состояние
    _prev_gray: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_quad: np.ndarray | None = field(default=None, init=False, repr=False)
    _flow_pts: np.ndarray | None = field(default=None, init=False, repr=False)
    _flow_run: int = field(default=0, init=False, repr=False)
    _predict_run: int = field(default=0, init=False, repr=False)
    _hann: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.det, self.norm = make_detector(self.detector_name, self.n_features)
        ref_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        h, w = ref_gray.shape
        self.ref_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        self.ref_area = float(w * h)
        # Эталон описывается на нескольких масштабах; у каждого уровня свой
        # индекс. Если смешать все уровни в один индекс, тест Лоу начинает
        # отбрасывать верные пары (у точки находится «двойник» с соседнего уровня).
        self.levels = []
        for sc in self.ref_scales:
            img = ref_gray if sc == 1 else cv2.resize(ref_gray, None, fx=sc, fy=sc,
                                                      interpolation=cv2.INTER_AREA)
            kp, des = self.det.detectAndCompute(img, None)
            if des is None or len(kp) < self.min_inliers:
                continue
            m = make_matcher(self.norm)
            m.add([des])
            m.train()
            self.levels.append((np.float32([k.pt for k in kp]) / sc, m))
        if not self.levels:
            raise RuntimeError("На эталоне слишком мало ключевых точек — "
                               "выберите более текстурированный объект")

    # -- детекция по ключевым точкам ---------------------------------------- #
    def _match(self, gray: np.ndarray, kp, des, to_frame) -> tuple[FrameResult, np.ndarray | None]:
        """Сопоставляет точки кадра (или его фрагмента) с каждым уровнем эталона
        и выбирает гомографию с наибольшим числом инлаеров.
        to_frame — функция перевода координат фрагмента в координаты кадра."""
        best = FrameResult(None, "lost", n_kp=len(kp) if kp else 0)
        best_pts = None
        if des is None or len(kp) < self.min_inliers:
            return best, None
        frame_pts = to_frame(np.float32([k.pt for k in kp]))
        for li, (ref_pts, matcher) in enumerate(self.levels):
            knn = matcher.knnMatch(des, k=2)
            good = [p[0] for p in knn if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
            best.n_matches = max(best.n_matches, len(good))
            if len(good) < self.min_inliers:
                continue
            src = ref_pts[[m.trainIdx for m in good]]
            dst = frame_pts[[m.queryIdx for m in good]]
            H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, self.ransac_thresh,
                                         maxIters=5000, confidence=0.999)
            if H is None:
                continue
            inl = mask.ravel().astype(bool)
            n_inl = int(inl.sum())
            if n_inl < self.min_inliers or n_inl <= best.n_inliers:
                continue
            quad = cv2.perspectiveTransform(self.ref_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            if not quad_is_plausible(quad, gray.shape, self.ref_area, self._prev_quad):
                continue
            proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
            best = FrameResult(quad, "detect", len(kp), best.n_matches, n_inl,
                               float(np.linalg.norm(proj - dst[inl], axis=1).mean()), li)
            best_pts = dst[inl]
        return best, best_pts

    def _detect_roi(self, gray: np.ndarray) -> tuple[FrameResult, np.ndarray | None]:
        """Поиск в окрестности прошлого положения. Фрагмент увеличивается так,
        чтобы объект занимал ~roi_size пикселей: у мелкого объекта появляется
        больше ключевых точек, а время работы меньше, чем по всему кадру."""
        q = self._prev_quad
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        H_, W_ = gray.shape
        x0, x1 = int(max(0, cx - bw)), int(min(W_, cx + bw))      # окно в 2 раза больше рамки
        y0, y1 = int(max(0, cy - bh)), int(min(H_, cy + bh))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return FrameResult(None, "lost"), None
        k = float(np.clip(self.roi_size / max(bw, bh, 1), 1.0, self.max_upscale))
        crop = gray[y0:y1, x0:x1]
        if k > 1:
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        kp, des = self.det.detectAndCompute(crop, None)
        return self._match(gray, kp, des, lambda p: p / k + np.float32([x0, y0]))

    def _detect(self, gray: np.ndarray) -> FrameResult:
        res, pts = (FrameResult(None, "lost"), None)
        if self._prev_quad is not None and self.use_roi:
            res, pts = self._detect_roi(gray)
        if res.quad is None:
            kp, des = self.det.detectAndCompute(gray, None)
            full, fpts = self._match(gray, kp, des, lambda p: p)
            if full.quad is not None or res.n_kp == 0:
                res, pts = full, fpts
        if res.quad is not None:
            self._flow_pts = pts.reshape(-1, 1, 2)
        return res

    # -- оптический поток Лукаса–Канаде ------------------------------------- #
    def _replenish(self, gray: np.ndarray, quad: np.ndarray, pts: np.ndarray | None):
        """Добавляет углы Ши–Томаси внутри рамки, если точек для потока мало."""
        if pts is not None and len(pts) >= 60:
            return pts
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(np.round(quad)), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))   # без точек на самой границе
        new = cv2.goodFeaturesToTrack(gray, 150, 0.01, 5, mask=mask)
        if new is None:
            return pts
        new = new.astype(np.float32)
        return new if pts is None else np.vstack([pts, new])

    def _flow(self, gray: np.ndarray) -> FrameResult:
        if (self._prev_gray is None or self._prev_quad is None
                or self._flow_run >= self.max_flow_frames):
            return FrameResult(None, "lost")
        p0 = self._replenish(self._prev_gray, self._prev_quad, self._flow_pts)
        if p0 is None or len(p0) < 8:
            return FrameResult(None, "lost")
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **lk)
        # прямой-обратный контроль: возвращаем точки назад и отбрасываем «уплывшие»
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        if ok.sum() < 8:
            return FrameResult(None, "lost")
        H, mask = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
        if H is None:
            return FrameResult(None, "lost")
        quad = cv2.perspectiveTransform(self._prev_quad.reshape(-1, 1, 2), H).reshape(-1, 2)
        if not quad_is_plausible(quad, gray.shape, self.ref_area, self._prev_quad):
            return FrameResult(None, "lost")
        inl = mask.ravel().astype(bool)
        if inl.sum() < 8:
            return FrameResult(None, "lost")
        self._flow_pts = p1[ok][inl].reshape(-1, 1, 2)
        return FrameResult(quad, "flow", n_inliers=int(inl.sum()))

    # -- прогноз по глобальному сдвигу кадра (сильный смаз) ------------------ #
    def _predict(self, gray: np.ndarray) -> FrameResult:
        """При сильном смазе нет ни ключевых точек, ни углов для KLT, но
        низкочастотная картинка сохраняется. Фазовая корреляция двух соседних
        кадров даёт глобальный сдвиг, на который и переносится рамка."""
        if (self._prev_gray is None or self._prev_quad is None
                or self._predict_run >= self.max_predict_frames):
            return FrameResult(None, "lost")
        k = 0.5
        a = cv2.resize(self._prev_gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        b = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        if self._hann is None or self._hann.shape != a.shape:
            self._hann = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(a), np.float32(b), self._hann)
        if resp < 0.05:
            return FrameResult(None, "lost")
        quad = self._prev_quad + np.float32([dx, dy]) / k
        return FrameResult(quad, "predict")

    # -- обработка кадра ---------------------------------------------------- #
    def process(self, frame: np.ndarray) -> FrameResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = self._detect(gray)
        if res.quad is not None:
            self._flow_run = self._predict_run = 0
        elif self.use_flow:
            fr = self._flow(gray)
            if fr.quad is not None:
                self._flow_run += 1
                self._predict_run = 0
                fr.n_kp, fr.n_matches = res.n_kp, res.n_matches
                res = fr
        if res.quad is None and self.use_predict:
            pr = self._predict(gray)
            if pr.quad is not None:
                self._predict_run += 1
                pr.n_kp, pr.n_matches = res.n_kp, res.n_matches
                res = pr
        if res.quad is not None and self.smooth > 0 and self._prev_quad is not None:
            res.quad = self.smooth * self._prev_quad + (1 - self.smooth) * res.quad
        if res.mode in ("lost", "predict"):
            self._flow_pts = None   # после прогноза точки для KLT набираются заново
        self._prev_quad = res.quad
        self._prev_gray = gray
        return res


# --------------------------------------------------------------------------- #
#  Отрисовка
# --------------------------------------------------------------------------- #
COLORS = {"detect": (0, 220, 0), "flow": (0, 200, 255), "predict": (255, 120, 0)}


def draw(frame: np.ndarray, res: FrameResult, label: str, show_info: bool = True) -> np.ndarray:
    out = frame.copy()
    if res.quad is not None:
        q = np.int32(np.round(res.quad))
        color = COLORS[res.mode]
        cv2.polylines(out, [q], True, color, 3, cv2.LINE_AA)
        # подпись — над верхней точкой рамки
        top = q[np.argmin(q[:, 1])]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = int(np.clip(top[0] - tw // 2, 0, out.shape[1] - tw - 1))
        y = int(np.clip(top[1] - 10, th + 4, out.shape[0] - 4))
        cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + 4), color, -1)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if show_info:
        info = f"{res.mode}  kp={res.n_kp}  matches={res.n_matches}  inliers={res.n_inliers}"
        cv2.putText(out, info, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, info, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
#  Обработка видео целиком
# --------------------------------------------------------------------------- #
def load_reference(cap: cv2.VideoCapture, margin: float = 0.0) -> np.ndarray:
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр")
    if margin > 0:  # опционально обрезаем края, если объект не занимает весь кадр
        h, w = first.shape[:2]
        dy, dx = int(h * margin), int(w * margin)
        first = first[dy:h - dy, dx:w - dx]
    return first


def track_video(path: str, out_path: str | None = None, label: str = "Object",
                detector: str = "sift", use_flow: bool = True, margin: float = 0.0,
                show_info: bool = True, verbose: bool = True, **kw) -> list[FrameResult]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ref = load_reference(cap, margin)
    tracker = KeypointTracker(ref, detector_name=detector, use_flow=use_flow, **kw)
    writer = None
    results, t0 = [], time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = tracker.process(frame)
        results.append(res)
        if out_path:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(draw(frame, res, label, show_info))
    dt = time.perf_counter() - t0
    cap.release()
    if writer:
        writer.release()
    if verbose:
        n = len(results)
        cnt = {m: sum(r.mode == m for r in results) for m in ("detect", "flow", "predict", "lost")}
        print(f"{path}: кадров={n}, " + ", ".join(f"{k}={v}" for k, v in cnt.items())
              + f", {n / dt:.1f} кадр/с")
    return results


def main():
    ap = argparse.ArgumentParser(description="Трекинг объекта по ключевым точкам")
    ap.add_argument("video", help="входное видео (первый кадр — эталон объекта)")
    ap.add_argument("-o", "--output", help="куда сохранить видео с рамкой (.mp4)")
    ap.add_argument("-l", "--label", default="Object", help="подпись к рамке")
    ap.add_argument("-d", "--detector", default="sift", choices=["sift", "orb", "akaze"])
    ap.add_argument("--no-flow", action="store_true", help="отключить KLT-резерв")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="доля кадра, обрезаемая с каждого края эталона")
    ap.add_argument("--ratio", type=float, default=0.8, help="порог теста Лоу")
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="сглаживание углов рамки между кадрами, 0..1")
    ap.add_argument("--no-info", action="store_true", help="не выводить служебную строку")
    a = ap.parse_args()
    track_video(a.video, a.output, a.label, a.detector, not a.no_flow, a.margin,
                show_info=not a.no_info, ratio=a.ratio, min_inliers=a.min_inliers,
                smooth=a.smooth)


if __name__ == "__main__":
    main()
