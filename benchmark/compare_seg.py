#!/usr/bin/env python3
"""Compare seg NMS boxes only (skip mask reconstruction for speed).

Same as compare_nms.py but for seg: output0 (1,116,8400), nc=80, extra=32.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import cv2, numpy as np

INPUT_SIZE = 640


def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_w, pad_h = new_size - new_w, new_size - new_h
    pad_left = round(pad_w / 2 - 0.1)
    pad_top = round(pad_h / 2 - 0.1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT, value=color)
    return canvas, scale, (pad_w, pad_h)


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None
    orig_shape = img.shape[:2]
    canvas, scale, pad = letterbox(img, INPUT_SIZE)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
    return tensor, scale, pad, orig_shape


def run_ultralytics_nms(raw_output, conf, iou, max_det, nc=80,
                        scale=1.0, pad=(0, 0), orig_shape=(640, 640)):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    pred = torch.from_numpy(raw_output.astype(np.float32))
    outputs = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, nc=nc,
                                  multi_label=True, agnostic=False, max_det=max_det,
                                  end2end=False, rotated=False)
    if not outputs or len(outputs[0]) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    result = outputs[0].cpu().numpy().astype(np.float32)
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    ratio_pad = ((scale,), (pad_left, pad_top))
    boxes_tensor = torch.from_numpy(result[:, :4])
    scale_boxes((INPUT_SIZE, INPUT_SIZE), boxes_tensor, orig_shape, ratio_pad)
    result[:, :4] = boxes_tensor.numpy()
    return result[:, :6]


def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_numpy(boxes, scores, iou_thresh):
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
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def run_our_nms(raw_output, conf, iou, max_det, nc=80, extra=32,
                scale=1.0, pad=(0, 0), orig_shape=(640, 640)):
    """Our NMS for seg: box + cls + mask_coeffs, matching ultralytics exactly."""
    pred = raw_output[0] if raw_output.ndim == 3 else raw_output  # (116, 8400)
    xc = pred[4:4 + nc].max(axis=0) > conf
    pred = pred.T  # (8400, 116)
    pred[:, :4] = xywh2xyxy(pred[:, :4])
    pred = pred[xc]
    if len(pred) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    box = pred[:, :4]
    cls = pred[:, 4:4 + nc]
    i_idx, j_idx = np.where(cls > conf)
    if len(i_idx) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x = np.column_stack([
        box[i_idx], cls[i_idx, j_idx, None],
        j_idx[:, None].astype(np.float32),
    ])

    if x.shape[0] > 30000:
        filt = x[:, 4].argsort()[::-1][:30000]
        x = x[filt]

    c = x[:, 5:6] * 7680
    keep = nms_numpy(x[:, :4] + c, x[:, 4], iou)
    keep = keep[:max_det]
    x = x[keep]

    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    x[:, 0] = (x[:, 0] - pad_left) / scale
    x[:, 1] = (x[:, 1] - pad_top) / scale
    x[:, 2] = (x[:, 2] - pad_left) / scale
    x[:, 3] = (x[:, 3] - pad_top) / scale
    x[:, [0, 2]] = np.clip(x[:, [0, 2]], 0, orig_shape[1])
    x[:, [1, 3]] = np.clip(x[:, [1, 3]], 0, orig_shape[0])
    return x[:, :6]


def compute_iou_matrix(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
    xa = np.maximum(a[:, 0, None], b[None, :, 0])
    ya = np.maximum(a[:, 1, None], b[None, :, 1])
    xb = np.minimum(a[:, 2, None], b[None, :, 2])
    yb = np.minimum(a[:, 3, None], b[None, :, 3])
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-10)


def greedy_match(iou_matrix, iou_thresh=0.5):
    if iou_matrix.size == 0:
        return [], [], 0.0
    matched_a, matched_b = [], []
    used_b = set()
    flat_idx = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat_idx:
        a = idx // iou_matrix.shape[1]
        b = idx % iou_matrix.shape[1]
        if iou_matrix[a, b] < iou_thresh:
            break
        if a in matched_a or b in used_b:
            continue
        matched_a.append(a)
        matched_b.append(b)
        used_b.add(b)
    avg_iou = float(iou_matrix[matched_a, matched_b].mean()) if matched_a else 0.0
    return matched_a, matched_b, avg_iou


def load_coco_images(data_dir, limit):
    from pycocotools.coco import COCO
    for candidate in [
        str(Path(data_dir).parent / 'annotations' / 'instances_val2017.json'),
        str(Path(data_dir) / 'annotations' / 'instances_val2017.json'),
    ]:
        if Path(candidate).exists():
            coco = COCO(candidate)
            img_ids = sorted(coco.getImgIds())[:limit]
            return [{'file_name': coco.loadImgs(i)[0]['file_name']} for i in img_ids]
    raise FileNotFoundError('instances_val2017.json not found')


def find_img_path(file_name, data_dir):
    data_path = Path(data_dir)
    for c in [data_path / 'images' / 'val2017' / file_name,
              data_path / 'images' / 'val' / file_name,
              data_path / file_name]:
        if c.exists():
            return str(c)
    return str(data_path / file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-name', default='yolo11n-seg_std')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--max-det', type=int, default=300)
    args = parser.parse_args()

    import onnxruntime as ort
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    images = load_coco_images(args.data, args.limit)
    sys.stderr.write(f'Seg NMS compare: {len(images)} images\n')

    total_ref = total_test = total_matched = 0
    all_ious = []
    t_start = time.time()

    for idx, img_info in enumerate(images):
        img_path = find_img_path(img_info['file_name'], args.data)
        tensor, scale, pad, orig_shape = preprocess(img_path)
        if tensor is None:
            continue
        raw = session.run(None, {input_name: tensor / 255.0})[0]

        ref = run_ultralytics_nms(raw, args.conf, args.iou, args.max_det, nc=80,
                                  scale=scale, pad=pad, orig_shape=orig_shape)
        test = run_our_nms(raw[0], args.conf, args.iou, args.max_det, nc=80, extra=32,
                           scale=scale, pad=pad, orig_shape=orig_shape)

        total_ref += len(ref)
        total_test += len(test)
        iou_matrix = compute_iou_matrix(ref[:, :4], test[:, :4])
        matched_a, matched_b, avg_iou = greedy_match(iou_matrix, iou_thresh=0.5)
        total_matched += len(matched_a)
        all_ious.append(avg_iou)

        if idx % 10 == 0:
            sys.stderr.write(f'  [{idx+1}/{len(images)}] ref={len(ref)} test={len(test)} '
                           f'matched={len(matched_a)} iou={avg_iou:.4f}\n')

    elapsed = time.time() - t_start
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    perfect = sum(1 for i in all_ious if i == 1.0)

    result = {
        'model': args.model_name, 'task': 'seg_nms_only',
        'total_images': len(images),
        'total_ref': total_ref, 'total_test': total_test, 'total_matched': total_matched,
        'mean_box_iou': round(mean_iou, 6), 'perfect_matches': perfect,
        'elapsed_s': round(elapsed, 1),
    }
    sys.stderr.write(f'\n=== Seg NMS compare ===\n')
    sys.stderr.write(f'Ref: {total_ref}, Test: {total_test}, Matched: {total_matched}\n')
    sys.stderr.write(f'Mean IoU: {mean_iou:.6f}, Perfect: {perfect}/{len(images)}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
