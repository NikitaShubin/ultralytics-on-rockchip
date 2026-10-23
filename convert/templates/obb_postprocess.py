import numpy as np
import math

INPUT_SIZE = {{INPUT_SIZE}}
STRIDES = {{STRIDES}}
GRID_SIZES = {{GRID_SIZES}}
REG_MAX = {{REG_MAX}}
BIN_COUNT = {{BIN_COUNT}}
CONF_THRESH = {{CONF_THRESH}}
IOU_THRESH = {{IOU_THRESH}}
MAX_DET = 300
MAX_NMS = 30000
MAX_WH = 7680
NUM_CLASSES = {{NUM_CLASSES}}
COCO_CLASSES = {{COCO_CLASSES}}
EVAL_PRECISE = False

def dfl_nodfl(position):
    return position

def dfl(position):
    n, c, h, w = position.shape
    mc = c // BIN_COUNT
    y = position.reshape(n, BIN_COUNT, mc, h, w)
    y_exp = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
    weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * weights).sum(axis=2)

_dfl = dfl_nodfl if {{NO_DFL}} else dfl


def dist2rbox_np(pred_dist, angle, grid_x, grid_y, stride):
    """Convert distance predictions + angle to rotated box in pixel coords."""
    l, t, r, b = pred_dist[:, 0], pred_dist[:, 1], pred_dist[:, 2], pred_dist[:, 3]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    xf = (r - l) / 2
    yf = (b - t) / 2
    cx = grid_x + xf * cos_a - yf * sin_a
    cy = grid_y + xf * sin_a + yf * cos_a
    cx = cx * stride
    cy = cy * stride
    w = (l + r) * stride
    h = (b + t) * stride
    return cx, cy, w, h


def process_head(box_dist, cls_scores, angle_raw, stride, grid_h, grid_w):
    """RK-opt/RKNN per-stride decode: raw DFL + raw angle → cx,cy,w,h,angle."""
    dbox = _dfl(box_dist)
    dbox = np.nan_to_num(dbox, nan=0.0, posinf=1e6, neginf=-1e6)
    col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
    row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
    grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w)) + 0.5
    grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w)) + 0.5

    angle_raw_clip = np.clip(angle_raw, -80, 80)
    ang = (1 / (1 + np.exp(-angle_raw_clip)) - 0.25) * math.pi if {{ANGLE_SIGMOID}} else angle_raw_clip

    boxes = dbox.transpose(0, 2, 3, 1).reshape(-1, 4)
    ang_f = ang.transpose(0, 2, 3, 1).reshape(-1)
    gx = grid_x.transpose(0, 2, 3, 1).reshape(-1)
    gy = grid_y.transpose(0, 2, 3, 1).reshape(-1)

    scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, NUM_CLASSES)

    cx, cy, bw, bh = dist2rbox_np(boxes, ang_f, gx, gy, stride)

    return np.column_stack([cx, cy, bw, bh]), ang_f, scores


def _covariance_matrix_np(boxes, floor=0.01):
    """(N,5) xywhr → (a, b, c): порт ultralytics _get_covariance_matrix."""
    w2 = boxes[:, 2] ** 2 / 12 + floor
    h2 = boxes[:, 3] ** 2 / 12 + floor
    c = boxes[:, 4]
    cos, sin = np.cos(c), np.sin(c)
    a = w2 * cos ** 2 + h2 * sin ** 2
    b = w2 * sin ** 2 + h2 * cos ** 2
    cc = (w2 - h2) * cos * sin
    return a, b, cc


def batch_probiou_np(obb1, obb2, eps=1e-7):
    """Вероятностный IoU повернутых боксов (N,5)x(M,5) → (N,M).

    Порт ultralytics.utils.metrics.batch_probiou на numpy: прод-образ
    percept не содержит ни torch, ни ultralytics (это осознанное решение —
    рантайм зависит только от rknnlite + numpy).
    """
    obb1 = np.asarray(obb1, dtype=np.float64)
    obb2 = np.asarray(obb2, dtype=np.float64)
    x1 = obb1[:, 0][:, None]
    y1 = obb1[:, 1][:, None]
    x2 = obb2[:, 0][None, :]
    y2 = obb2[:, 1][None, :]
    a1, b1, c1 = (v[:, None] for v in _covariance_matrix_np(obb1))
    a2, b2, c2 = (v[None, :] for v in _covariance_matrix_np(obb2))
    den = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / den) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / den) * 0.5
    det1 = np.clip(a1 * b1 - c1 ** 2, 0, None)
    det2 = np.clip(a2 * b2 - c2 ** 2, 0, None)
    num = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2
    t3 = np.log(num / (4 * np.sqrt(det1 * det2) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def _get_probiou():
    """Выбор реализации batch_probiou.

    EVAL_PRECISE (eval-режим, ultralytics доступен в convert-контейнере) —
    torch-версия (байт-эквивалент ultralytics .val()); в проде (EVAL_PRECISE
    не выставляется, torch/ultralytics не установлены) — numpy-порт.
    """
    if EVAL_PRECISE:
        from ultralytics.utils.metrics import batch_probiou as fn

        def wrapper(a, b):
            return fn(a, b).numpy()
        return wrapper
    return batch_probiou_np


def _rotated_nms(boxes_xywhr, scores, iou_thresh, max_det, classes=None, block=4096):
    """Блочный greedy NMS для obb — байт-эквивалент полной batch_probiou
    (fast_nms: triu + sum(0)<=0), память O(block^2) вместо O(N^2).

    Класс-offset (MAX_WH) исключает пересечения между классами, поэтому при
    заданном classes NMS считается per-class: пар в ~C раз меньше (на 30000
    кандидатов fp16 = минус OOM и ~15x ускорение). Результат идентичен
    глобальному greedy.

    Возвращает индексы выбранных боксов в исходной нумерации boxes_xywhr.
    """
    bp = _get_probiou()
    n = scores.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if classes is None:
        groups = [np.arange(n)]
    else:
        groups = [np.where(classes == cid)[0] for cid in np.unique(classes)]
    keep = np.empty(0, dtype=np.int64)
    for g in groups:
        ng = len(g)
        if ng == 0:
            continue
        b = np.ascontiguousarray(boxes_xywhr[g], dtype=np.float32)
        sg = scores[g]
        sorted_idx = np.argsort(-sg, kind='stable')
        b = b[sorted_idx]
        nb = (ng + block - 1) // block
        max_iou = np.zeros(ng, dtype=np.float32)
        for jb in range(nb):
            j0, j1 = jb * block, min(ng, (jb + 1) * block)
            bj = b[j0:j1]
            for ib in range(jb):
                i0, i1 = ib * block, min(ng, (ib + 1) * block)
                iou = bp(b[i0:i1], bj)
                max_iou[j0:j1] = np.maximum(max_iou[j0:j1], iou.max(axis=0))
            iou_jj = bp(bj, bj)
            if j1 - j0 > 1:
                max_iou[j0:j1] = np.maximum(max_iou[j0:j1],
                                            np.triu(iou_jj, k=1).max(axis=0))
        kl = np.where(max_iou < iou_thresh)[0]
        keep = np.concatenate([keep, g[sorted_idx[kl]]])
    if len(keep) > max_det:
        keep = keep[np.argsort(-scores[keep])[:max_det]]
    return keep


def postprocess(outputs, scale, pad, orig_shape):
    """Postprocess matching ultralytics val non_max_suppression for obb.

    Args:
        outputs: std ONNX — один tensor (1, 4+nc+1, total) decoded cxcywh+cls+angle;
                 RK-opt/RKNN — per-stride [box_dist, cls_sigmoid, angle_raw].
        scale: letterbox scale
        pad: (pad_w, pad_h) total padding
        orig_shape: (orig_h, orig_w)
    Returns:
        list of dicts with bbox [cx,cy,w,h], score, class_id, angle
    """
    single_output = (len(outputs) == 1 and outputs[0].ndim == 3
                     and outputs[0].shape[1] == 4 + NUM_CLASSES + 1)
    if single_output:
        # std ONNX: уже decoded (cx,cy,w,h + sigmoid cls + angle)
        pred = outputs[0][0].astype(np.float32)  # (4+nc+1, total)
        nc = pred.shape[0] - 4 - 1
        xc = pred[4:4 + nc].max(axis=0) > CONF_THRESH
        pred = pred.T[xc]
        if len(pred) == 0:
            return []
        box = pred[:, :4]           # cx, cy, w, h
        cls = pred[:, 4:4 + nc]     # (N, nc)
        angle = pred[:, 4 + nc:]    # (N, 1)
    else:
        # RK-opt/RKNN: per-stride raw
        parts = []
        for i in range(len(STRIDES)):
            boxxy, ang, scores = process_head(
                outputs[i * 3], outputs[i * 3 + 1], outputs[i * 3 + 2],
                STRIDES[i], GRID_SIZES[i], GRID_SIZES[i])
            parts.append((boxxy, ang, scores))
        box = np.concatenate([p[0] for p in parts])
        angle = np.concatenate([p[1] for p in parts])[:, None]
        cls = np.concatenate([p[2] for p in parts])
        xc = cls.max(axis=1) > CONF_THRESH
        box, angle, cls = box[xc], angle[xc], cls[xc]
        if len(box) == 0:
            return []

    # multi_label=True — expand each box by each class above conf_thres
    i, j = np.where(cls > CONF_THRESH)
    if len(i) == 0:
        return []

    # (M, 7): [cx, cy, w, h, confidence, class_id, angle]
    x = np.column_stack([
        box[i],                          # (M, 4) — repeated boxes
        cls[i, j, None],                 # (M, 1) — per-class confidence
        j[:, None].astype(np.float32),   # (M, 1) — class IDs
        angle[i],                        # (M, 1) — angle per box
    ])

    # sort + max_nms cap
    if x.shape[0] > MAX_NMS:
        filt = x[:, 4].argsort()[::-1][:MAX_NMS]
        x = x[filt]

    # rotated NMS — блочный greedy (эквивалент ultralytics fast_nms с полной
    # batch_probiou-матрицей), память O(BLOCK^2) вместо O(N^2) → нет OOM на
    # conf=0.001 (fp16/int8 дают ~30000 кандидатов = 3.6GB полной матрицы).
    # Реализация на numpy: в прод-образе нет torch/ultralytics.
    c = x[:, 5:6] * MAX_WH
    boxes_xywhr = np.column_stack([
        x[:, :2] + c,  # cx+class_offset, cy+class_offset
        x[:, 2:4],     # w, h
        x[:, -1:],     # angle
    ])
    scores = x[:, 4]
    keep = _rotated_nms(boxes_xywhr, scores, IOU_THRESH, MAX_DET,
                        classes=x[:, 5])
    x = x[keep]

    if len(x) == 0:
        return []

    # reverse letterbox on cx,cy,w,h (xywh=True mode)
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    x[:, 0] = (x[:, 0] - pad_left) / gain  # cx
    x[:, 1] = (x[:, 1] - pad_top) / gain   # cy
    x[:, 2] = x[:, 2] / gain               # w
    x[:, 3] = x[:, 3] / gain               # h

    results = []
    for row in x:
        cx, cy, w, h = row[:4]
        conf, cls_id, ang = row[4], int(row[5]), float(row[6])
        results.append({
            "bbox": [float(cx), float(cy), float(w), float(h)],
            "score": float(conf),
            "class_id": cls_id,
            "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
            "angle": ang,
        })
    return results


def visualize(frame, objects):
    import cv2
    h, w = frame.shape[:2]
    s = max(h, w) / 1000
    thick = max(1, int(s * 3))
    fs = max(0.3, s * 0.5)
    fthick = max(1, int(s * 1.5))
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]
    for i, obj in enumerate(objects):
        cx, cy, w, h = obj["bbox"]
        angle = obj.get("angle", 0)
        color = colors[i % len(colors)]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = w / 2, h / 2
        corners = np.array([
            [cx + cos_a * (-dx) - sin_a * (-dy), cy + sin_a * (-dx) + cos_a * (-dy)],
            [cx + cos_a * (dx) - sin_a * (-dy), cy + sin_a * (dx) + cos_a * (-dy)],
            [cx + cos_a * (dx) - sin_a * (dy), cy + sin_a * (dx) + cos_a * (dy)],
            [cx + cos_a * (-dx) - sin_a * (dy), cy + sin_a * (-dx) + cos_a * (dy)],
        ], dtype=np.int32)
        cv2.polylines(frame, [corners], True, color, thick)
        label = f"{obj['class_name']}: {obj['score']:.2f}"
        cv2.putText(frame, label, (int(cx) - 20, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, fthick)
    return frame
