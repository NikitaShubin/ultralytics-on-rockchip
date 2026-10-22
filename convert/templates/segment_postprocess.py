import numpy as np
import cv2

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
NM = {{NM}}
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
    """Greedy NMS — exact copy of ultralytics TorchNMS (numpy version)."""
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


def process_mask_160(protos, masks_in, bboxes_letter, imgsz=640):
    """Маски на разрешении прототипов (160x160) в letterboxed-пространстве.

    Ровно как ultralytics ops.process_mask(..., upsample=False):
      masks = (coeffs @ proto.view(32, -1)).view(n, 160, 160);
      binarize > 0 (= sigmoid > 0.5 без сигмоида); crop_mask по letterboxed-bbox.
    """
    c, mh, mw = protos.shape
    if masks_in.shape[0] == 0:
        return np.zeros((0, mh, mw), dtype=np.uint8)
    masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
    masks = (masks > 0.0).astype(np.uint8)
    if len(bboxes_letter):
        bx = np.asarray(bboxes_letter, dtype=np.float32) * (mw / imgsz)
        xx = np.arange(mw, dtype=np.float32)[None, None, :]
        yy = np.arange(mh, dtype=np.float32)[None, :, None]
        crop = (
            (xx >= bx[:, 0][:, None, None])
            & (xx < bx[:, 2][:, None, None])
            & (yy >= bx[:, 1][:, None, None])
            & (yy < bx[:, 3][:, None, None])
        ).astype(np.uint8)
        masks = masks * crop
    return masks


def _decode_per_stride(outputs):
    """RK-opt/RKNN формат: per-stride [box_dist, cls_sigmoid, coeff] x len(STRIDES) + proto."""
    all_boxes, all_scores, all_coeffs = [], [], []
    for idx in range(len(STRIDES)):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        coeff = outputs[idx * 3 + 2]
        boxes, scores = process_head(box_dist, cls_scores, STRIDES[idx], GRID_SIZES[idx], GRID_SIZES[idx])
        nm = coeff.shape[1]
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_coeffs.append(coeff[0].reshape(nm, -1).T)  # (gh*gw, nm)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    coeffs = np.concatenate(all_coeffs)
    return boxes, scores, coeffs


def postprocess(outputs, scale, pad, orig_shape):
    """Postprocess matching ultralytics non_max_suppression() + process_mask.

    Args:
        outputs: std ONNX — [output0 (1, 4+nc+NM, total), proto (1, NM, 160, 160)];
                 RK-opt/RKNN — per-stride [box_dist, cls_sigmoid, coeff] x len(STRIDES)
                               + proto в конце (последний выход).
        scale: letterbox scale factor
        pad: (pad_w, pad_h) total padding
        orig_shape: (orig_h, orig_w)
    Returns:
        list of dicts with bbox, score, class_id, class_name, masks_160
        (и full-res "segmentation" когда EVAL_PRECISE=False).
    """
    orig_h, orig_w = orig_shape
    single_output = (len(outputs) == 2)
    if single_output:
        pred = outputs[0][0].astype(np.float32)  # (4+nc+NM, total)
        nc = pred.shape[0] - 4 - NM
        proto = outputs[1]
        if proto.ndim == 4:
            proto = proto[0]
        pred[:4] = xywh2xyxy(pred[:4].T).T
        pred = pred.T                            # (total, 4+nc+NM)
        box = pred[:, :4]
        cls = pred[:, 4:4 + nc]
        coeffs = pred[:, 4 + nc:]
    else:
        proto = outputs[-1]
        if proto.ndim == 4:
            proto = proto[0]
        box, cls, coeffs = _decode_per_stride(outputs)

    # Step 1: confidence filter
    xc = cls.max(axis=1) > CONF_THRESH
    box = box[xc]
    cls = cls[xc]
    coeffs = coeffs[xc]
    if len(box) == 0:
        return []

    # Step 2: multi_label=True — expand each box by each class above conf
    i, j = np.where(cls > CONF_THRESH)
    if len(i) == 0:
        return []

    # (M, 6+NM): [x1, y1, x2, y2, confidence, class_id, coeffs...]
    x = np.column_stack([
        box[i],
        cls[i, j, None],
        j[:, None].astype(np.float32),
        coeffs[i],
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
    if len(x) == 0:
        return []

    # Step 5: masks at prototype resolution (160x160), letterboxed space
    bboxes_letter = x[:, :4].copy()
    masks_160 = process_mask_160(proto, x[:, 6:], bboxes_letter)

    # Step 6: reverse letterbox
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    x[:, 0] = (x[:, 0] - pad_left) / gain
    x[:, 1] = (x[:, 1] - pad_top) / gain
    x[:, 2] = (x[:, 2] - pad_left) / gain
    x[:, 3] = (x[:, 3] - pad_top) / gain

    # Step 7: clip to image bounds (skip in EVAL_PRECISE — ultralytics doesn't clip;
    # masks_160 считаются по letterboxed-bbox до reverse letterbox)
    if not EVAL_PRECISE:
        x[:, [0, 2]] = np.clip(x[:, [0, 2]], 0, orig_w)
        x[:, [1, 3]] = np.clip(x[:, [1, 3]], 0, orig_h)

    # Step 8: format output
    if EVAL_PRECISE:
        results = []
        for i in range(len(x)):
            c = int(x[i, 5])
            results.append({
                "bbox": [float(x[i, 0]), float(x[i, 1]), float(x[i, 2]), float(x[i, 3])],
                "score": float(x[i, 4]),
                "class_id": c,
                "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
                "masks_160": masks_160[i],
            })
        return results

    # Full-res masks: masks_160 (160x160, letterboxed space) → input_size →
    # crop by padding → scale to original frame, so the mask matches the bbox
    # (which is already in original coordinates).
    results = []
    inp = int(INPUT_SIZE)
    # Размер изображения внутри letterbox-пространства inp x inp
    crop_w = min(inp, int(round(orig_w * gain)))
    crop_h = min(inp, int(round(orig_h * gain)))
    for i in range(len(x)):
        x1, y1, x2, y2 = x[i, :4].astype(int)
        c = int(x[i, 5])
        mask_crop = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            m = cv2.resize(masks_160[i].astype(np.uint8), (inp, inp),
                           interpolation=cv2.INTER_NEAREST)
            # убрать padding letterbox: изображение занимает crop_w x crop_h
            m = m[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w]
            if m.size > 0:
                mask_crop = cv2.resize(m, (orig_w, orig_h),
                                       interpolation=cv2.INTER_NEAREST) * 255
        results.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": float(x[i, 4]),
            "class_id": c,
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
            "segmentation": mask_crop,
            "masks_160": masks_160[i],
        })
    return results


def visualize(frame, objects):
    """Draw segmentation masks and boxes on frame."""
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
        mask_bin = obj.get("segmentation")
        if isinstance(mask_bin, np.ndarray) and mask_bin.size > 1:
            mask_bool = mask_bin > 127
            overlay[mask_bool] = color
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = colors[i % len(colors)]
        score = obj["score"]
        label = f"{obj['class_name']}: {score:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, fthick)
    return frame
