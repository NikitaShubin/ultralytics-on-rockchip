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

    # rotated NMS — fast_nms (TorchNMS) + class offset, batch_probiou
    from ultralytics.utils.metrics import batch_probiou
    import torch

    c = x[:, 5:6] * MAX_WH
    boxes_xywhr = torch.from_numpy(np.column_stack([
        x[:, :2] + c,  # cx+class_offset, cy+class_offset
        x[:, 2:4],     # w, h
        x[:, -1:],     # angle
    ]))
    scores = torch.from_numpy(x[:, 4])

    sorted_idx = torch.argsort(scores, descending=True)
    boxes_sorted = boxes_xywhr[sorted_idx]
    ious = batch_probiou(boxes_sorted, boxes_sorted)
    ious_np = ious.numpy()
    n = ious_np.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    for j in range(1, n):
        if np.any(ious_np[:j, j] >= IOU_THRESH):
            keep_mask[j] = False

    sorted_indices = sorted_idx.numpy()
    keep_sorted = np.where(keep_mask)[0][:MAX_DET]
    keep = sorted_indices[keep_sorted]
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
            "class_name": str(cls_id),
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
