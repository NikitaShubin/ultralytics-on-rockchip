import numpy as np
import cv2
import math

INPUT_SIZE = {{INPUT_SIZE}}
STRIDES = {{STRIDES}}
GRID_SIZES = {{GRID_SIZES}}
REG_MAX = {{REG_MAX}}
BIN_COUNT = {{BIN_COUNT}}
CONF_THRESH = {{CONF_THRESH}}
IOU_THRESH = {{IOU_THRESH}}
MIN_WH = {{MIN_WH}}
NUM_CLASSES = {{NUM_CLASSES}}
COCO_CLASSES = {{COCO_CLASSES}}
{{NO_DFL}}

_OFFSETS = [gs * gs for gs in GRID_SIZES]
_TOTAL_GRID = sum(_OFFSETS)
# OBB output format: 20 channels = 4(box) + NUM_CLASSES(cls) + 1(angle)
_OBB_NUM_CLS_CH = {{NUM_CLASSES}}

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


def _split_single_output(feat):
    """Split single (1, 20, total_grid) output into per-stride box/cls/angle."""
    n, c, total = feat.shape
    off = 0
    splits = []
    for i, g in enumerate(GRID_SIZES):
        sz = g * g
        f = feat[:, :, off:off+sz].reshape(n, c, g, g)
        box = f[:, 0:4, :, :]
        cls_raw = f[:, 4:4+_OBB_NUM_CLS_CH, :, :]
        angle_raw = f[:, 4+_OBB_NUM_CLS_CH:4+_OBB_NUM_CLS_CH+1, :, :]
        splits.append((box, cls_raw, angle_raw))
        off += sz
    return splits


def dist2rbox_np(pred_dist, angle, grid_x, grid_y, stride):
    """Convert distance predictions + angle to rotated box in pixel coords.
    Returns [cx, cy, w, h] in image coordinates.
    """
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
    gh, gw = grid_h, grid_w

    dbox = _dfl(box_dist)
    dbox = np.nan_to_num(dbox, nan=0.0, posinf=1e6, neginf=-1e6)
    col = np.arange(gw, dtype=np.float32).reshape(1, 1, 1, gw)
    row = np.arange(gh, dtype=np.float32).reshape(1, 1, gh, 1)
    grid_x = np.broadcast_to(col, (1, 1, gh, gw)) + 0.5
    grid_y = np.broadcast_to(row, (1, 1, gh, gw)) + 0.5

    angle_raw_clip = np.clip(angle_raw, -80, 80)
    ang = (1 / (1 + np.exp(-angle_raw_clip)) - 0.25) * math.pi

    boxes = dbox.transpose(0, 2, 3, 1).reshape(-1, 4)
    ang_f = ang.transpose(0, 2, 3, 1).reshape(-1)
    gx = grid_x.transpose(0, 2, 3, 1).reshape(-1)
    gy = grid_y.transpose(0, 2, 3, 1).reshape(-1)

    scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, NUM_CLASSES)

    cx, cy, bw, bh = dist2rbox_np(boxes, ang_f, gx, gy, stride)

    swap = bh > bw
    if np.any(swap):
        bw[swap], bh[swap] = bh[swap], bw[swap]
        ang_f[swap] -= np.pi / 2
        under = swap & (ang_f < -np.pi / 4)
        ang_f[under] += np.pi

    return cx, cy, bw, bh, ang_f, scores


def nms(cx, cy, bw, bh, scores, iou_thresh):
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    areas = bw * bh
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep)


def dedup(boxes, scores, class_ids):
    if len(boxes) < 2:
        return np.arange(len(boxes))
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    groups = []
    assigned = set()
    for i in range(len(boxes)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(boxes)):
            if j in assigned:
                continue
            if class_ids[j] != class_ids[i]:
                continue
            w_r = min(widths[i], widths[j]) / max(widths[i], widths[j])
            h_r = min(heights[i], heights[j]) / max(heights[i], heights[j])
            center_dist = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            same_center = center_dist < 50
            same_size = w_r > 0.75 and h_r > 0.75
            if same_size or (same_center and center_dist < max(widths[i], widths[j]) * 0.75):
                group.append(j)
                assigned.add(j)
        groups.append(group)
    keep = []
    for group in groups:
        best = max(group, key=lambda i: scores[i])
        keep.append(best)
    return np.array(keep)


def postprocess(outputs, scale, pad, orig_shape):
    orig_h, orig_w = orig_shape
    all_cx, all_cy, all_bw, all_bh = [], [], [], []
    all_scores, all_cls, all_angle = [], [], []

    has_single_output = len(outputs) == 1 and outputs[0].shape[1] == 4 + _OBB_NUM_CLS_CH + 1

    if has_single_output:
        feat = outputs[0]
        splits = _split_single_output(feat)
    else:
        splits = [(outputs[i * 3], outputs[i * 3 + 1], outputs[i * 3 + 2]) for i in range(len(STRIDES))]

    for idx in range(len(STRIDES)):
        box_dist, cls_raw, angle_raw = splits[idx]
        stride = STRIDES[idx]
        gh = gw = GRID_SIZES[idx]

        cls_scores = 1 / (1 + np.exp(-cls_raw)) if has_single_output else cls_raw

        cx, cy, bw, bh, ang, scores = process_head(box_dist, cls_scores, angle_raw, stride, gh, gw)

        max_cls = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        mask = max_cls > CONF_THRESH
        if not mask.any():
            continue
        all_cx.append(cx[mask])
        all_cy.append(cy[mask])
        all_bw.append(bw[mask])
        all_bh.append(bh[mask])
        all_angle.append(ang[mask])
        all_scores.append(max_cls[mask])
        all_cls.append(cls_ids[mask])

    if not all_cx:
        return []

    cx = np.concatenate(all_cx)
    cy = np.concatenate(all_cy)
    bw = np.concatenate(all_bw)
    bh = np.concatenate(all_bh)
    angle = np.concatenate(all_angle)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)

    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    cx = (cx - pad_left) / scale
    cy = (cy - pad_top) / scale
    bw = bw / scale
    bh = bh / scale
    cx = np.clip(cx, 0, orig_w)
    cy = np.clip(cy, 0, orig_h)

    valid = (bw > 0) & (bh > 0)
    cx, cy, bw, bh = cx[valid], cy[valid], bw[valid], bh[valid]
    angle, scores, cls_ids = angle[valid], scores[valid], cls_ids[valid]
    if not len(cx):
        return []

    valid = np.maximum(bw, bh) >= MIN_WH
    cx, cy, bw, bh = cx[valid], cy[valid], bw[valid], bh[valid]
    angle, scores, cls_ids = angle[valid], scores[valid], cls_ids[valid]
    if not len(cx):
        return []

    keep = nms(cx, cy, bw, bh, scores, IOU_THRESH)
    cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
    angle, scores, cls_ids = angle[keep], scores[keep], cls_ids[keep]
    boxes = np.stack([cx, cy, bw, bh], axis=1)
    keep = dedup(boxes, scores, cls_ids)
    cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
    angle, scores, cls_ids = angle[keep], scores[keep], cls_ids[keep]

    results = []
    for i in range(len(cx)):
        c = int(cls_ids[i])
        results.append({
            "bbox": [int(cx[i]), int(cy[i]), int(bw[i]), int(bh[i])],
            "angle": float(angle[i]),
            "score": float(scores[i]),
            "class_id": c,
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
        })
    return results


def _rotated_rect(cx, cy, w, h, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    wx = w / 2 * cos_a
    wy = w / 2 * sin_a
    hx = -h / 2 * sin_a
    hy = h / 2 * cos_a
    pts = np.array([
        [cx - wx - hx, cy - wy - hy],
        [cx + wx - hx, cy + wy - hy],
        [cx + wx + hx, cy + wy + hy],
        [cx - wx + hx, cy - wy + hy],
    ], dtype=np.int32)
    return pts


def visualize(frame, objects):
    h, w = frame.shape[:2]
    s = max(h, w) / 1000
    thick = max(1, int(s * 3))
    rad = max(2, int(s * 4))
    fs = max(0.3, s * 0.5)
    fthick = max(1, int(s * 1.5))
    for obj in objects:
        cx, cy, bw, bh = obj["bbox"]
        angle = obj["angle"]
        score = obj["score"]
        label = f"{obj['class_name']}: {score:.2f}"

        pts = _rotated_rect(cx, cy, bw, bh, angle)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=thick)
        cv2.circle(frame, (int(cx), int(cy)), rad, (0, 255, 0), -1)
        cv2.putText(frame, label, (int(cx), int(cy) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), fthick)
    return frame
