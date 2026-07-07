import numpy as np
import cv2

INPUT_SIZE = 640
STRIDES = [8, 16, 32]
GRID_SIZES = [80, 40, 20]
REG_MAX = 16
BIN_COUNT = 4
CONF_THRESH = 0.25
IOU_THRESH = 0.5
MIN_WH = 20
NUM_CLASSES = 1
NUM_KPTS = 17
COCO_CLASSES = ["person"]

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


def dfl(position):
    n, c, h, w = position.shape
    mc = c // BIN_COUNT
    y = position.reshape(n, BIN_COUNT, mc, h, w)
    y_exp = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
    weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * weights).sum(axis=2)


def process_head(box_dist, cls_scores, kpt_raw, stride, grid_h, grid_w):
    col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
    row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
    grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w))
    grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w))
    grid = np.concatenate([grid_x, grid_y], axis=1)

    dbox = dfl(box_dist)
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
    kpts[:, :, 0] = (kpts[:, :, 0] * 2.0 + grid_x.flatten()) * stride
    kpts[:, :, 1] = (kpts[:, :, 1] * 2.0 + grid_y.flatten()) * stride
    kpts[:, :, 2] = 1.0 / (1.0 + np.exp(-kpts[:, :, 2]))

    return boxes, scores, kpts


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
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
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
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
    all_boxes, all_scores, all_cls, all_kpts = [], [], [], []

    for idx in range(3):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        kpt_raw = outputs[idx * 3 + 2]
        stride = STRIDES[idx]
        gh = gw = GRID_SIZES[idx]
        boxes, scores, kpts = process_head(box_dist, cls_scores, kpt_raw, stride, gh, gw)
        max_cls = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        mask = max_cls > CONF_THRESH
        if not mask.any():
            continue
        all_boxes.append(boxes[mask])
        all_scores.append(max_cls[mask])
        all_cls.append(cls_ids[mask])
        all_kpts.append(kpts[mask])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)
    kpts = np.concatenate(all_kpts)

    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    kpts[:, :, 0] = (kpts[:, :, 0] - pad_left) / scale
    kpts[:, :, 1] = (kpts[:, :, 1] - pad_top) / scale

    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, cls_ids, kpts = boxes[valid], scores[valid], cls_ids[valid], kpts[valid]
    if not len(boxes):
        return []

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    valid = (bw >= MIN_WH) & (bh >= MIN_WH)
    boxes, scores, cls_ids, kpts = boxes[valid], scores[valid], cls_ids[valid], kpts[valid]
    if not len(boxes):
        return []

    keep = nms(boxes, scores, IOU_THRESH)
    boxes, scores, cls_ids, kpts = boxes[keep], scores[keep], cls_ids[keep], kpts[keep]
    keep = dedup(boxes, scores, cls_ids)
    boxes, scores, cls_ids, kpts = boxes[keep], scores[keep], cls_ids[keep], kpts[keep]

    results = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int).tolist()
        c = int(cls_ids[i])
        kp = kpts[i].astype(int).tolist()
        results.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(scores[i]),
            "class_id": c,
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
            "keypoints": kp,
        })
    return results


def visualize(frame, objects):
    h, w = frame.shape[:2]
    s = max(h, w) / 1000
    thick = max(1, int(s * 3))
    rad = max(2, int(s * 4))
    fs = max(0.3, s * 0.5)
    fthick = max(1, int(s * 1.5))

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
