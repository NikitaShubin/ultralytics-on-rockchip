import numpy as np
import cv2

INPUT_SIZE = {{INPUT_SIZE}}
STRIDES = {{STRIDES}}
GRID_SIZES = {{GRID_SIZES}}
REG_MAX = {{REG_MAX}}
BIN_COUNT = {{BIN_COUNT}}
CONF_THRESH = {{CONF_THRESH}}
IOU_THRESH = {{IOU_THRESH}}
MIN_WH = 1
MAX_DET = 300
MAX_NMS = 30000
MAX_WH = 7680
DEDUP_ENABLED = False
EVAL_PRECISE = False
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


def xywh2xyxy(x):
    """Center-based xywh → corner xyxy. Matches ultralytics.ops.xywh2xyxy."""
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def process_head(box_dist, cls_scores, stride, grid_h, grid_w):
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
    return boxes, scores


def nms(boxes, scores, iou_threshold):
    """Greedy NMS — exact copy of ultralytics TorchNMS (numpy version).

    boxes: (N, 4) xyxy in model coords
    scores: (N,) confidence
    returns: (K,) indices of kept boxes
    """
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


def _decode_per_stride(outputs):
    """RK-opt/RKNN формат: per-stride [box_dist, cls_sigmoid, sm]."""
    all_boxes, all_scores = [], []
    for idx in range(len(STRIDES)):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        boxes, scores = process_head(box_dist, cls_scores, STRIDES[idx], GRID_SIZES[idx], GRID_SIZES[idx])
        all_boxes.append(boxes)
        all_scores.append(scores)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    return np.concatenate([boxes, scores], axis=1)  # (N, 4+nc) xyxy + sigmoid cls


def postprocess(outputs, scale, pad, orig_shape):
    """Postprocess matching ultralytics non_max_suppression() exactly.

    Args:
        outputs: std ONNX — один tensor (1, 4+nc, total_grid) decoded xywh+sigmoid;
                 RK-opt/RKNN — per-stride [box_dist, cls_sigmoid, sm].
        scale: letterbox scale factor
        pad: (pad_w, pad_h) total padding
        orig_shape: (orig_h, orig_w)
    Returns:
        list of dicts with bbox, score, class_id, class_name
    """
    orig_h, orig_w = orig_shape
    single_output = (len(outputs) == 1 and outputs[0].ndim == 3
                     and outputs[0].shape[1] == 4 + NUM_CLASSES)
    if single_output:
        pred = outputs[0][0].astype(np.float32)  # (4+nc, total)
        pred[:4] = xywh2xyxy(pred[:4].T).T
        pred = pred.T                            # (total, 4+nc)
    else:
        pred = _decode_per_stride(outputs)

    # Step 1: confidence filter (max class score > CONF_THRESH)
    xc = pred[:, 4:4 + NUM_CLASSES].max(axis=1) > CONF_THRESH
    pred = pred[xc]
    if len(pred) == 0:
        return []

    box = pred[:, :4]
    cls = pred[:, 4:4 + NUM_CLASSES]

    # Step 2: multi_label=True — expand each box by each class above conf
    i, j = np.where(cls > CONF_THRESH)
    if len(i) == 0:
        return []

    # (M, 6): [x1, y1, x2, y2, confidence, class_id]
    x = np.column_stack([
        box[i],                      # (M, 4) — repeated boxes
        cls[i, j, None],             # (M, 1) — per-class confidence
        j[:, None].astype(np.float32),  # (M, 1) — class IDs
    ])

    # Step 3: sort by confidence, cap at max_nms
    if x.shape[0] > MAX_NMS:
        filt = x[:, 4].argsort()[::-1][:MAX_NMS]
        x = x[filt]

    # Step 4: class-aware NMS — offset boxes by class_id * max_wh
    c = x[:, 5:6] * MAX_WH
    boxes_nms = x[:, :4] + c
    keep = nms(boxes_nms, x[:, 4], IOU_THRESH)
    keep = keep[:MAX_DET]
    x = x[keep]

    # Step 5: reverse letterbox — model coords → original image coords
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    x[:, 0] = (x[:, 0] - pad_left) / gain
    x[:, 1] = (x[:, 1] - pad_top) / gain
    x[:, 2] = (x[:, 2] - pad_left) / gain
    x[:, 3] = (x[:, 3] - pad_top) / gain

    # Step 6: clip to image bounds (skip in EVAL_PRECISE — ultralytics doesn't clip)
    if not EVAL_PRECISE:
        x[:, [0, 2]] = np.clip(x[:, [0, 2]], 0, orig_w)
        x[:, [1, 3]] = np.clip(x[:, [1, 3]], 0, orig_h)

    # Step 7: format output
    results = []
    for row in x:
        x1, y1, x2, y2, conf, cls_id = row
        c = int(cls_id)
        if EVAL_PRECISE:
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(conf),
                "class_id": c,
                "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
            })
        else:
            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(conf),
                "class_id": c,
                "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
            })
    return results


def visualize(frame, objects):
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
        x1, y1, x2, y2 = obj["bbox"]
        color = colors[i % len(colors)]
        score = obj["score"]
        label = f"{obj['class_name']}: {score:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, fthick)
    return frame
