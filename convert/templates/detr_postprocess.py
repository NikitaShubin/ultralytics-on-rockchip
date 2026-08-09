import numpy as np
import cv2

INPUT_SIZE = {{INPUT_SIZE}}
CONF_THRESH = {{CONF_THRESH}}
NUM_CLASSES = {{NUM_CLASSES}}
COCO_CLASSES = {{COCO_CLASSES}}


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return np.array([], dtype=int)
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


def postprocess(outputs, scale, pad, orig_shape):
    """Decode RT-DETR RKNN outputs into detection objects.

    Args:
        outputs: list with one tensor [1, 300, 6] = [cx, cy, w, h, prob, cls]
        scale: float, resize scale from preprocess
        pad: tuple (total_pad_w, total_pad_h)
        orig_shape: tuple (orig_h, orig_w)

    Returns:
        list of dicts: [{"bbox": [x1,y1,x2,y2], "score": float, "class_id": int, "class_name": str}, ...]
    """
    orig_h, orig_w = orig_shape
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)

    if not outputs or len(outputs) == 0:
        return []

    pred = outputs[0].squeeze(0)
    if pred.ndim != 2 or pred.shape[1] != 6:
        return []

    mask = pred[:, 4] > CONF_THRESH
    if not mask.any():
        return []
    pred = pred[mask]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    scores = pred[:, 4]
    cls_ids = pred[:, 5].astype(int)

    half_w = w / 2.0
    half_h = h / 2.0
    x1 = (cx - half_w) * INPUT_SIZE
    y1 = (cy - half_h) * INPUT_SIZE
    x2 = (cx + half_w) * INPUT_SIZE
    y2 = (cy + half_h) * INPUT_SIZE

    x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_top) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_top) / scale, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid.any():
        return []
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]

    keep = nms(boxes, scores, 0.5)
    if len(keep) == 0:
        return []
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
    """Draw detection results on frame."""
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
