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
NM = {{NM}}
COCO_CLASSES = {{COCO_CLASSES}}
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


def reconstruct_masks(proto, coeffs, boxes, orig_shape, scale, pad):
    """Reconstruct masks from prototype masks and per-detection coefficients.

    Args:
        proto: (NM, ph, pw) prototype masks from model
        coeffs: (N, NM) mask coefficients per detection
        boxes: (N, 4) boxes in original image coords [x1,y1,x2,y2]
        orig_shape: (orig_h, orig_w)
        scale: resize scale
        pad: (total_pad_w, total_pad_h)

    Returns:
        list of (bh, bw) uint8 binary masks cropped to each box
    """
    orig_h, orig_w = orig_shape
    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    nm, ph, pw = proto.shape
    proto_flat = proto.reshape(nm, ph * pw)
    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=1e3, neginf=-1e3)
    raw_masks = 1.0 / (1.0 + np.exp(-coeffs @ proto_flat))
    raw_masks = raw_masks.reshape(-1, ph, pw)

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    box_masks = []
    for i in range(len(boxes)):
        x1_i, y1_i, x2_i, y2_i = boxes[i].astype(int)
        if x2_i <= x1_i or y2_i <= y1_i:
            box_masks.append(np.zeros((1, 1), dtype=np.uint8))
            continue

        mask_proto = raw_masks[i]
        mask_canvas = cv2.resize(mask_proto, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)
        mask_unpadded = mask_canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        mask_full = cv2.resize(mask_unpadded, (orig_w, orig_h), cv2.INTER_LINEAR)

        bh = y2_i - y1_i
        bw = x2_i - x1_i
        mask_crop = mask_full[y1_i:y2_i, x1_i:x2_i]
        mask_bin = (mask_crop > 0.5).astype(np.uint8) * 255
        box_masks.append(mask_bin)

    return box_masks


def postprocess(outputs, scale, pad, orig_shape):
    """Decode RKNN outputs into segmentation objects.

    Args:
        outputs: 10 tensors: [box, cls, coeff] x 3 heads + proto
        scale: float resize scale
        pad: (total_pad_w, total_pad_h)
        orig_shape: (orig_h, orig_w)

    Returns:
        list of dicts with "bbox", "score", "class_id", "class_name", "mask"
    """
    orig_h, orig_w = orig_shape
    all_boxes, all_scores, all_cls = [], [], []
    all_coeffs = []

    proto = outputs[-1][0]

    for idx in range(3):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        coeff = outputs[idx * 3 + 2]
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
        coeff_flat = coeff[0].reshape(NM, gh * gw).T
        all_coeffs.append(coeff_flat[mask])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)
    coeffs = np.concatenate(all_coeffs)

    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, cls_ids, coeffs = boxes[valid], scores[valid], cls_ids[valid], coeffs[valid]
    if not len(boxes):
        return []

    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    valid = (bw >= MIN_WH) & (bh >= MIN_WH)
    boxes, scores, cls_ids, coeffs = boxes[valid], scores[valid], cls_ids[valid], coeffs[valid]
    if not len(boxes):
        return []

    keep = nms(boxes, scores, IOU_THRESH)
    boxes, scores, cls_ids, coeffs = boxes[keep], scores[keep], cls_ids[keep], coeffs[keep]
    keep = dedup(boxes, scores, cls_ids)
    boxes, scores, cls_ids, coeffs = boxes[keep], scores[keep], cls_ids[keep], coeffs[keep]

    box_masks = reconstruct_masks(proto, coeffs, boxes, orig_shape, scale, pad)

    results = []
    for i in range(len(boxes)):
        x1_i, y1_i, x2_i, y2_i = boxes[i].astype(int).tolist()
        c = int(cls_ids[i])
        results.append({
            "bbox": [x1_i, y1_i, x2_i, y2_i],
            "score": float(scores[i]),
            "class_id": c,
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
            "mask": box_masks[i],
        })
    return results


def visualize(frame, objects):
    """Draw segmentation masks and boxes on frame.

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
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]
    overlay = frame.copy()
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = colors[i % len(colors)]
        mask_bin = obj["mask"]
        if mask_bin.size > 1:
            overlay[y1:y2, x1:x2][mask_bin > 0] = color
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = colors[i % len(colors)]
        mask_bin = obj["mask"]
        score = obj["score"]
        label = f"{obj['class_name']}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, fthick)
        if mask_bin.size > 1:
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt[:, 0, 0] += x1
                cnt[:, 0, 1] += y1
                cv2.drawContours(frame, [cnt], -1, color, thick)
    return frame
