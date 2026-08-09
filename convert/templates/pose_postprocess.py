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
NUM_KPTS = {{NUM_KPTS}}
COCO_CLASSES = {{COCO_CLASSES}}
EVAL_PRECISE = False
{{NO_DFL}}

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

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]
SKELETON_COLORS = [
    (255, 0, 0), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255),
    (85, 0, 255), (170, 0, 255),
]
KP_COLORS = [
    (255, 0, 0), (255, 0, 0), (255, 85, 0), (255, 170, 0),
    (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (255, 0, 0), (0, 255, 170),
    (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255),
]


def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def process_head(box_dist, cls_scores, kpt_raw, stride, grid_h, grid_w):
    """RK-opt/RKNN per-stride decode: raw DFL + raw kpts → xyxy boxes + sigmoid cls + decoded kpts."""
    col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
    row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
    grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w))
    grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w))
    grid = np.concatenate([grid_x, grid_y], axis=1)

    dbox = _dfl(box_dist)
    dbox = np.nan_to_num(dbox, nan=0.0, posinf=1e6, neginf=-1e6)
    l, t, r, b = dbox[:, 0:1], dbox[:, 1:2], dbox[:, 2:3], dbox[:, 3:4]
    x1 = (grid[:, 0:1] + 0.5 - l) * stride
    y1 = (grid[:, 1:2] + 0.5 - t) * stride
    x2 = (grid[:, 0:1] + 0.5 + r) * stride
    y2 = (grid[:, 1:2] + 0.5 + b) * stride
    boxes = np.concatenate([x1, y1, x2, y2], axis=1)
    boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
    scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, NUM_CLASSES)

    nk = NUM_KPTS
    kpts = kpt_raw.reshape(1, nk, 3, grid_h, grid_w).transpose(0, 3, 4, 1, 2).reshape(-1, nk, 3)
    gx = (grid_x.flatten() + 0.5)[:, None]
    gy = (grid_y.flatten() + 0.5)[:, None]
    if {{NO_DFL}}:
        kpts[:, :, 0] = (kpts[:, :, 0] + gx) * stride
        kpts[:, :, 1] = (kpts[:, :, 1] + gy) * stride
    else:
        kpts[:, :, 0] = (kpts[:, :, 0] * 2.0 + gx - 0.5) * stride
        kpts[:, :, 1] = (kpts[:, :, 1] * 2.0 + gy - 0.5) * stride
    kpts[:, :, 2] = 1.0 / (1.0 + np.exp(-kpts[:, :, 2]))

    return boxes, scores, kpts


def nms(boxes, scores, iou_threshold):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        if inter.sum() == 0:
            order = rest
            continue
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def postprocess(outputs, scale, pad, orig_shape):
    """Postprocess matching ultralytics val non_max_suppression for pose.

    Args:
        outputs: std ONNX — один tensor (1, 4+nc+3*nk, total) decoded xywh+cls+kpts;
                 RK-opt/RKNN — per-stride [box_dist, cls_sigmoid, kpt_raw].
        scale: letterbox scale
        pad: (pad_w, pad_h) total padding
        orig_shape: (orig_h, orig_w)
    Returns:
        list of dicts with bbox, score, class_id, class_name, keypoints
    """
    orig_h, orig_w = orig_shape
    nk = NUM_KPTS
    nk_flat = nk * 3

    single_output = (len(outputs) == 1 and outputs[0].ndim == 3
                     and outputs[0].shape[1] == 4 + NUM_CLASSES + nk_flat)
    if single_output:
        # std ONNX: уже decoded xywh + sigmoid cls + kpts (x,y px, vis sigmoid)
        pred = outputs[0][0].astype(np.float32)  # (4+nc+nk_flat, total)
        nc = pred.shape[0] - 4 - nk_flat
        xc = pred[4:4 + nc].max(axis=0) > CONF_THRESH
        pred = pred.T[xc]
        if len(pred) == 0:
            return []
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        box = pred[:, :4]
        cls = pred[:, 4:4 + nc]
        kpts = pred[:, 4 + nc:]
        conf = cls.max(axis=1)
        cls_id = cls.argmax(axis=1)
    else:
        # RK-opt/RKNN: per-stride raw
        all_boxes, all_scores, all_kpts = [], [], []
        for idx in range(len(STRIDES)):
            boxes, scores, kpts = process_head(
                outputs[idx * 3], outputs[idx * 3 + 1], outputs[idx * 3 + 2],
                STRIDES[idx], GRID_SIZES[idx], GRID_SIZES[idx])
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kpts.append(kpts)
        box = np.concatenate(all_boxes)
        cls = np.concatenate(all_scores)
        kpts = np.concatenate(all_kpts)
        conf = cls.max(axis=1)
        cls_id = cls.argmax(axis=1)
        kpts = kpts.reshape(-1, nk_flat)

    filt = conf > CONF_THRESH
    if not filt.any():
        return []
    box = box[filt]
    conf = conf[filt]
    cls_id = cls_id[filt]
    kpts = kpts[filt]

    # (M, 5+nk_flat): [x1, y1, x2, y2, confidence, class_id, kpts...]
    x = np.column_stack([
        box, conf.astype(np.float32),
        cls_id.astype(np.float32),
        kpts,
    ])

    if x.shape[0] > MAX_NMS:
        filt = x[:, 4].argsort()[::-1][:MAX_NMS]
        x = x[filt]

    # class-aware NMS in letterboxed space
    c = x[:, 5:6] * MAX_WH
    boxes_nms = x[:, :4] + c
    scores = x[:, 4]
    keep = nms(boxes_nms, scores, IOU_THRESH)
    keep = keep[:MAX_DET]
    x = x[keep]

    if len(x) == 0:
        return []

    # reverse letterbox
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    x[:, 0] = (x[:, 0] - pad_left) / gain
    x[:, 1] = (x[:, 1] - pad_top) / gain
    x[:, 2] = (x[:, 2] - pad_left) / gain
    x[:, 3] = (x[:, 3] - pad_top) / gain
    if not EVAL_PRECISE:
        x[:, [0, 2]] = np.clip(x[:, [0, 2]], 0, orig_w)
        x[:, [1, 3]] = np.clip(x[:, [1, 3]], 0, orig_h)

    # scale keypoints (same transform as boxes); visibility unchanged
    for i in range(nk_flat):
        if i % 3 == 0:
            x[:, 6 + i] = (x[:, 6 + i] - pad_left) / gain
        elif i % 3 == 1:
            x[:, 6 + i] = (x[:, 6 + i] - pad_top) / gain

    results = []
    for row in x:
        c = int(row[5])
        kpt_vals = row[6:6 + nk_flat]
        keypoints = kpt_vals.reshape(nk, 3).tolist()
        if EVAL_PRECISE:
            bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        else:
            bbox = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
        results.append({
            "bbox": bbox,
            "score": float(row[4]),
            "class_id": c,
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
            "keypoints": keypoints,
        })
    return results


def visualize(frame, objects):
    h, w = frame.shape[:2]
    s = max(h, w) / 1000
    thick = max(1, int(s * 3))
    rad = max(2, int(s * 4))
    fs = max(0.3, s * 0.5)
    fthick = max(1, int(s * 1.5))

    import cv2
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        score = obj["score"]
        label = f"{obj['class_name']}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thick)

        kpts = obj.get("keypoints", [])
        if kpts:
            kpts_arr = np.array(kpts, dtype=np.int32)
            vis = kpts_arr[:, 2] > 0.5
            for (i, j), color in zip(COCO_SKELETON, SKELETON_COLORS):
                if vis[i] and vis[j]:
                    cv2.line(frame, kpts_arr[i, :2], kpts_arr[j, :2], color, thick)
            for i, (x, y, v) in enumerate(kpts_arr):
                if v > 0.5:
                    cv2.circle(frame, (x, y), rad, KP_COLORS[i % len(KP_COLORS)], -1)
                    cv2.circle(frame, (x, y), rad, (255, 255, 255), max(1, thick // 2))

        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), fthick)

    return frame
