import numpy as np
import cv2

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


def dfl(position):
    n, c, h, w = position.shape
    mc = c // BIN_COUNT
    y = position.reshape(n, BIN_COUNT, mc, h, w)
    y_exp = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
    weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * weights).sum(axis=2)


def process_head(box_dist, cls_scores, stride, grid_h, grid_w):
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
    return boxes, scores


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
    """Decode RKNN outputs into detection objects.

    Args:
        outputs: list of numpy arrays from RKNN inference (9 tensors: 3 heads × [box, cls, sm])
        scale: float, resize scale from preprocess
        pad: tuple (total_pad_w, total_pad_h)
        orig_shape: tuple (orig_h, orig_w)

    Returns:
        list of dicts: [{"bbox": [x1,y1,x2,y2], "score": float, "class_id": int, "class_name": str}, ...]
    """
    orig_h, orig_w = orig_shape
    all_boxes, all_scores, all_cls = [], [], []

    for idx in range(3):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        stride = STRIDES[idx]
        gh = gw = GRID_SIZES[idx]
        boxes, scores = process_head(box_dist, cls_scores, stride, gh, gw)
        max_cls = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        mask = max_cls > CONF_THRESH
        if not mask.any():
            continue
        all_boxes.append(boxes[mask])
        all_scores.append(max_cls[mask])
        all_cls.append(cls_ids[mask])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)

    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if not len(boxes):
        return []

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    valid = (bw >= MIN_WH) & (bh >= MIN_WH)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if not len(boxes):
        return []

    keep = nms(boxes, scores, IOU_THRESH)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    keep = dedup(boxes, scores, cls_ids)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

    results = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int).tolist()
        c = int(cls_ids[i])
        results.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(scores[i]),
            "class_id": c,
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
        })
    return results


def visualize(frame, objects):
    """Draw detection results on frame.

    Args:
        frame: numpy array (H, W, 3) BGR
        objects: list of dicts from postprocess()

    Returns:
        numpy array with drawn detections
    """
    h, w = frame.shape[:2]
    s = max(h, w) / 1000
    thick = max(1, int(s * 3))
    fs = max(0.3, s * 0.5)
    fthick = max(1, int(s * 1.5))
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        score = obj["score"]
        label = f"{obj['class_name']}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thick)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), fthick)
    return frame
