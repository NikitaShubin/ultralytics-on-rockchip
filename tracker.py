"""Мультитрекер объектов (Фаза 1): Kalman XYAH + двухпроходная ассоциация + CMC.

Чистые numpy + cv2 + scipy ( Hungarian ), без torch/ultralytics (прод-образ).
Реализация по описанию алгоритмов ByteTrack (arXiv:2110.06864) и
Bot-SORT (arXiv:2206.14651) — не copy-paste чужого кода.

Состав:
  - KalmanFilterXYAH на numpy (состояние: cx,cy,aspect,h + скорости);
  - два прохода ассоциации: (1) уверенные детекции → треки,
    (2) слабые детекции → оставшиеся треки (ретенция ID при перекрытиях
    и пропусках детекции); новые треки — только из уверенных;
  - класс-консистентность: трек матчится только с детекциями того же class_id;
  - CMC (компенсация движения камеры, опция): SparseOptFlow cv2 + RANSAC
    на пониженном разрешении → аффинное смещение треков до предсказания;
  - хук под Фазу 2 (ReID): поле `Tracker.embedder` пока None.

API: `Tracker.update(objects, frame)` — in-place дописывает `track_id` в dict'ы
детекций (остальное не меняет). По умолчанию выключен (env TRACKER=off) —
деплой работает как раньше.
"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

# Пороги в стиле ByteTrack: новые треки — только из достоверных детекций,
# слабые — второй проход для удержания ID.
TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
IOU_GATE_HIGH = 0.2
IOU_GATE_LOW = 0.4
# Кадров жизни трека без детекции (при ~10 fps это ~3 с).
MAX_AGE = 30


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU пар боксов: (N,4) и (M,4) в xyxy → (N,M)."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-7, None)


def _xyxy_to_xyah(m: np.ndarray) -> np.ndarray:
    """xyxy → (cx, cy, aspect w/h, h)."""
    w = max(1e-6, float(m[2] - m[0]))
    h = max(1e-6, float(m[3] - m[1]))
    return np.array([(m[0] + m[2]) / 2.0, (m[1] + m[3]) / 2.0, w / h, h])


def _xyah_to_xyxy(mean: np.ndarray) -> np.ndarray:
    cx, cy, a, h = mean[:4]
    w = max(1e-3, a * h)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class _KalmanXYAH:
    """Kalman xyah (state 8: cx,cy,a,h + скорости; измерение 4)."""

    _STD_W = 1.0e-2
    _STD_V = 1.0e-4

    def __init__(self) -> None:
        self._move = np.eye(8)
        self._move[:4, 4:] = np.eye(4)  # dt = 1 кадр
        self._meas = np.eye(4, 8)

    def initiate(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = np.r_[[2.0 * self._STD_W * m[3], 2.0 * self._STD_W * m[3],
                     1e-2, 2.0 * self._STD_W * m[3]],
                    [10.0 * self._STD_V * m[3], 10.0 * self._STD_V * m[3],
                     1e-5, 10.0 * self._STD_V * m[3]]]
        return np.r_[m, np.zeros(4)], np.diag(std ** 2)

    def _predict_noise(self, m: np.ndarray) -> np.ndarray:
        return np.r_[[self._STD_W * m[3], self._STD_W * m[3], 1e-2, self._STD_W * m[3]],
                     [self._STD_V * m[3], self._STD_V * m[3], 1e-5, self._STD_V * m[3]]]

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> None:  # in-place
        mean[:] = self._move @ mean
        cov[:, :] = self._move @ cov @ self._move.T + np.diag(self._predict_noise(mean) ** 2)

    def update(self, mean: np.ndarray, cov: np.ndarray,
               m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sens = np.array([self._STD_W * m[3], self._STD_W * m[3], 1e-1,
                         self._STD_W * m[3]])
        pm = self._meas @ mean
        pc = self._meas @ cov @ self._meas.T + np.diag(sens ** 2)
        kg = np.linalg.solve(pc, self._meas @ cov).T
        return mean + kg @ (m - pm), cov - kg @ pc @ kg.T


class _Track:
    """Один трек: Kalman-состояние + метаданные класса."""

    __slots__ = ('tid', 'mean', 'cov', 'cls', 'score', 'hits', 'age')

    def __init__(self, tid: int, mean: np.ndarray, cov: np.ndarray,
                 cls: int, score: float) -> None:
        self.tid = tid
        self.mean = mean
        self.cov = cov
        self.cls = cls
        self.score = score
        self.hits = 1
        self.age = 0


class Tracker:
    """Ассоциатор детекций по кадрам.

    Args:
        cmc: 'sof' — компенсация движения камеры (LK-поток + RANSAC),
             'off' — отключена (статическая камера).
        cmc_size: ширина рабочего изображения CMC (px): скорость ~ 1/размер².
    """

    def __init__(self, cmc: str = 'sof', cmc_size: int = 320) -> None:
        self._kf = _KalmanXYAH()
        self._tracks: list[_Track] = []
        self._next_id = 0
        self._cmc = cmc != 'off'
        self._cmc_size = cmc_size
        self._prev_gray: np.ndarray | None = None
        self._cmc_missing = 0  # подряд кадров без оценённого сдвига
        # Хук Фазы 2 (ReID): callable(frame, boxes_xyxy) → (N, D) эмбеддингов.
        self.embedder: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None

    def reset(self) -> None:
        """Полный сброс (смена модели / источника / зацикливание файла)."""
        self._tracks = []
        self._prev_gray = None
        self._next_id = 0
        self._cmc_missing = 0

    # ------------------------------------------------------------------ API
    def update(self, objects: list[dict], frame: np.ndarray) -> None:
        """Ассоциировать детекции кадра; дописывает `track_id` в объекты.

        objects — результат pipeline.detect (bbox, score, class_id);
        кадр — BGR np.ndarray текущего разрешения (для CMC).
        """
        dets, det_idx = self._parse(objects)
        cur_gray = self._cmc_gray(frame)

        if not dets and not self._tracks:
            self._prev_gray = cur_gray
            return

        # CMC до предсказания: аффинный сдвиг треков (движение камеры).
        if self._tracks and self._prev_gray is not None and cur_gray is not None:
            try:
                self._apply_cmc(self.compute_cmc(self._prev_gray, cur_gray))
            except Exception:
                log.debug('CMC: оценка сдвига не удалась', exc_info=True)
        self._prev_gray = cur_gray

        for t in self._tracks:
            self._kf.predict(t.mean, t.cov)
        pred = (np.array([_xyah_to_xyxy(t.mean) for t in self._tracks])
                if self._tracks else np.zeros((0, 4)))

        pairs = self._run_association(pred, dets)
        updated: list[_Track] = []
        for ti, k in pairs:
            t = self._tracks[ti]
            t.mean, t.cov = self._kf.update(t.mean, t.cov,
                                            _xyxy_to_xyah(np.asarray(dets[k][:4])))
            t.score, t.hits, t.age = float(dets[k][4]), t.hits + 1, 0
            objects[det_idx[k]]['track_id'] = t.tid
            updated.append(t)

        # Новые треки — только из оставшихся уверенных детекций.
        used_dets = {k for _, k in pairs}
        for k, d in enumerate(dets):
            if k in used_dets or d[4] < TRACK_HIGH_THRESH:
                continue
            mean, cov = self._kf.initiate(_xyxy_to_xyah(np.asarray(d[:4])))
            t = _Track(self._next_id, mean, cov, int(d[5]), float(d[4]))
            self._next_id += 1
            self._tracks.append(t)
            objects[det_idx[k]]['track_id'] = t.tid
            updated.append(t)

        for t in self._tracks:
            if t not in updated:
                t.age += 1
        self._tracks = [t for t in self._tracks if t.age <= MAX_AGE]

    # ------------------------------------------------------------- внутреннее
    @staticmethod
    def _parse(objects: list[dict]) -> tuple[list[list[float]], list[int]]:
        """Извлечь детекции, пригодные для трекинга (bbox + score + class_id).

        OBB-объекты (angle, bbox в cx,cy,w,h) приводятся к AABB xyxy.
        """
        dets, det_idx = [], []
        for i, obj in enumerate(objects):
            bb = obj.get('bbox')
            if bb is None or len(bb) < 4:
                continue
            bb = np.asarray(bb, dtype=np.float64)
            if obj.get('angle') is not None:  # OBB: cx,cy,w,h → xyxy
                cx, cy, w_, h_ = bb[:4]
                bb = np.array([cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2])
            dets.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]),
                         float(obj.get('score', 0.0)), int(obj.get('class_id', -1))])
            det_idx.append(i)
        return dets, det_idx

    def _run_association(self, pred: np.ndarray, dets: list[list[float]]
                         ) -> list[tuple[int, int]]:
        """Два прохода ассоциации: высокий conf, затем слабые → свободные.

        Возвращает пары (индекс трека, индекс детекции).
        """
        if not self._tracks or not dets:
            return []
        dets_arr = np.asarray(dets, dtype=np.float64)
        track_cls = np.array([t.cls for t in self._tracks])
        matched: dict[int, int] = {}
        used_tracks: set[int] = set()

        high = np.where(dets_arr[:, 4] >= TRACK_HIGH_THRESH)[0] if dets else np.zeros(0, int)
        for pair in self._associate(pred, track_cls, dets_arr[high], IOU_GATE_HIGH):
            ti, hi = pair
            if ti not in used_tracks:
                matched[ti] = int(high[hi])
                used_tracks.add(ti)

        low = [int(i) for i in np.where((dets_arr[:, 4] >= TRACK_LOW_THRESH)
                                        & (dets_arr[:, 4] < TRACK_HIGH_THRESH))[0]] \
            if dets else []
        if low:
            free = [i for i in range(len(self._tracks)) if i not in used_tracks]
            pred_f = pred[free] if free else np.zeros((0, 4))
            cls_f = track_cls[free] if free else np.zeros(0, dtype=int)
            for ft, li in self._associate(
                    pred_f, cls_f, dets_arr[low], IOU_GATE_LOW):
                matched[free[ft]] = low[li]
        return list(matched.items())

    def _associate(self, pred_boxes: np.ndarray, track_cls: np.ndarray,
                   dets: np.ndarray, iou_gate: float) -> list[tuple[int, int]]:
        """Hungarian по классам: [(track, det_subidx)], гейт по IoU."""
        res: list[tuple[int, int]] = []
        if pred_boxes.shape[0] == 0 or dets.shape[0] == 0:
            return res
        iou = _iou_matrix(pred_boxes, dets[:, :4])
        det_cls = dets[:, 5].astype(int)
        for cls in np.unique(np.concatenate([track_cls, det_cls])):
            t_ids = np.where(track_cls == cls)[0]
            d_ids = np.where(det_cls == cls)[0]
            if not t_ids.size or not d_ids.size:
                continue
            sub = iou[np.ix_(t_ids, d_ids)]
            ok = sub >= iou_gate
            cost = np.where(ok, 1.0 - sub, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < 1e5:
                    res.append((int(t_ids[r]), int(d_ids[c])))
        return res

    # ------------------------------------------------------------------- CMC
    def _cmc_gray(self, frame: np.ndarray | None) -> np.ndarray | None:
        """Серый кадр для CMC, даунскейл до ширины cmc_size."""
        if frame is None or not self._cmc or self._cmc_size <= 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = self._cmc_size / max(1, w)
        return cv2.resize(gray, (self._cmc_size, max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_LINEAR)

    def compute_cmc(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray | None:
        """Аффинное смещение камеры prev→cur: LK-поток + RANSAC (2×3, None)."""
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8)
        if pts is None or pts.shape[0] < 10:
            self._cmc_missing += 1
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
        good = st.reshape(-1) > 0
        if good.sum() < 10:
            self._cmc_missing += 1
            return None
        A, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good],
                                           method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        self._cmc_missing = 0 if A is not None else self._cmc_missing + 1
        return A

    def _apply_cmc(self, A: np.ndarray | None) -> None:
        if A is None or not self._tracks:
            return
        s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
        for t in self._tracks:
            t.mean[:2] = A @ np.r_[t.mean[:2], 1.0]
            t.mean[3] *= s


__all__ = ['Tracker']
