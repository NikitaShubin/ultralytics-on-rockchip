#!/usr/bin/env python3
"""Изолированное сравнение постпроцессов для obb: один ONNX → два NMS.

Запуск:
    docker run --rm -v ... --entrypoint python percept-converter:latest \
        /app/benchmark/compare_obb.py \
        --model /app/cache/yolo11n-obb_std/model.onnx \
        --data /app/datasets/dotav1 \
        --limit 100
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

INPUT_SIZE = 640


def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    pad_left = round(pad_w / 2 - 0.1)
    pad_top = round(pad_h / 2 - 0.1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=color,
    )
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


def run_ultralytics_nms(raw_output, conf, iou, max_det, nc=15,
                        scale=1.0, pad=(0, 0), orig_shape=(640, 640)):
    """Ultralytics NMS for obb (rotated)."""
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    pred = torch.from_numpy(raw_output.astype(np.float32))
    outputs = non_max_suppression(
        pred, conf_thres=conf, iou_thres=iou, nc=nc,
        multi_label=False, agnostic=False, max_det=max_det,
        end2end=False, rotated=True,
    )
    if not outputs or len(outputs[0]) == 0:
        return np.zeros((0, 7))

    result = outputs[0].cpu().numpy().astype(np.float32)
    # result: (N, 7) = [cx, cy, w, h, conf, cls, angle]

    # scale_boxes with xywh=True
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    result[:, 0] = (result[:, 0] - pad_left) / gain
    result[:, 1] = (result[:, 1] - pad_top) / gain
    result[:, 2] = result[:, 2] / gain
    result[:, 3] = result[:, 3] / gain

    return result  # [cx, cy, w, h, conf, cls, angle]


def load_our_postprocess(model_dir, model_name, conf, iou):
    post_path = Path(model_dir) / f'{model_name}_postprocess.py'
    spec = importlib.util.spec_from_file_location('postprocess_mod', str(post_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['postprocess_mod'] = mod
    spec.loader.exec_module(mod)
    mod.CONF_THRESH = conf
    mod.IOU_THRESH = iou
    return mod


def run_our_postprocess(mod, raw_output, scale, pad, orig_shape):
    results = mod.postprocess([raw_output], scale, pad, orig_shape)
    if not results:
        return np.zeros((0, 7))
    arr = np.zeros((len(results), 7), dtype=np.float32)
    for i, d in enumerate(results):
        cx, cy, w, h = d['bbox']
        arr[i] = [cx, cy, w, h, d['score'], d['class_id'], d['angle']]
    return arr


def obb_to_xyxy(cx, cy, w, h, angle):
    """Convert rotated box to axis-aligned bounding box (xyxy)."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = w / 2, h / 2
    corners_x = np.array([-dx, dx, dx, -dx])
    corners_y = np.array([-dy, -dy, dy, dy])
    rx = corners_x * cos_a - corners_y * sin_a + cx
    ry = corners_x * sin_a + corners_y * cos_a + cy
    return np.array([rx.min(), ry.min(), rx.max(), ry.max()])


def compute_obb_iou_matrix(ref, test):
    """Compute IoU matrix using axis-aligned bbox of rotated boxes."""
    n, m = len(ref), len(test)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    # Convert both to xyxy (axis-aligned bbox)
    ref_xyxy = np.array([obb_to_xyxy(*r) for r in ref[:, :5]])
    test_xyxy = np.array([obb_to_xyxy(*t) for t in test[:, :5]])

    xa = np.maximum(ref_xyxy[:, 0, None], test_xyxy[None, :, 0])
    ya = np.maximum(ref_xyxy[:, 1, None], test_xyxy[None, :, 1])
    xb = np.minimum(ref_xyxy[:, 2, None], test_xyxy[None, :, 2])
    yb = np.minimum(ref_xyxy[:, 3, None], test_xyxy[None, :, 3])
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area_a = (ref_xyxy[:, 2] - ref_xyxy[:, 0]) * (ref_xyxy[:, 3] - ref_xyxy[:, 1])
    area_b = (test_xyxy[:, 2] - test_xyxy[:, 0]) * (test_xyxy[:, 3] - test_xyxy[:, 1])
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


def load_dota_images(data_dir, limit):
    data_path = Path(data_dir)
    img_dir = data_path / 'images' / 'val'
    images = sorted(str(f) for f in img_dir.glob('*.jpg'))[:limit]
    return [{'file_name': Path(p).name, 'path': p} for p in images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-name', default='yolo11n-obb_std')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--max-det', type=int, default=300)
    args = parser.parse_args()

    import onnxruntime as ort
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    model_dir = str(Path(args.model).parent)
    our_mod = load_our_postprocess(model_dir, args.model_name, args.conf, args.iou)

    images = load_dota_images(args.data, args.limit)
    sys.stderr.write(f'OBB compare: {len(images)} images, nc=15, extra=1 (angle)\n')

    total_ref = 0
    total_test = 0
    total_matched = 0
    all_ious = []
    all_angle_diffs = []
    t_start = time.time()

    for idx, img_info in enumerate(images):
        img_path = img_info['path']
        tensor, scale, pad, orig_shape = preprocess(img_path)
        if tensor is None:
            continue

        raw = session.run(None, {input_name: tensor / 255.0})[0]

        ref = run_ultralytics_nms(
            raw, args.conf, args.iou, args.max_det, nc=15,
            scale=scale, pad=pad, orig_shape=orig_shape,
        )
        test = run_our_postprocess(our_mod, raw[0], scale, pad, orig_shape)

        n_ref = len(ref)
        n_test = len(test)
        total_ref += n_ref
        total_test += n_test

        # Matching using axis-aligned bbox IoU
        iou_matrix = compute_obb_iou_matrix(ref, test)
        matched_a, matched_b, avg_iou = greedy_match(iou_matrix, iou_thresh=0.5)
        total_matched += len(matched_a)
        all_ious.append(avg_iou)

        # Angle difference for matched pairs
        for a, b in zip(matched_a, matched_b):
            diff = abs(ref[a, 6] - test[b, 6])
            # Normalize to [0, pi]
            diff = min(diff, np.pi - diff)
            all_angle_diffs.append(diff)

        if idx % 10 == 0:
            sys.stderr.write(f'  [{idx+1}/{len(images)}] ref={n_ref} test={n_test} '
                           f'matched={len(matched_a)} iou={avg_iou:.4f}\n')

    elapsed = time.time() - t_start
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_angle_diff = float(np.mean(all_angle_diffs)) if all_angle_diffs else 0.0
    perfect = sum(1 for i in all_ious if i == 1.0)

    result = {
        'model': args.model_name,
        'total_images': len(images),
        'total_ref': total_ref,
        'total_test': total_test,
        'total_matched': total_matched,
        'mean_iou': round(mean_iou, 6),
        'mean_angle_diff_rad': round(mean_angle_diff, 4),
        'mean_angle_diff_deg': round(np.degrees(mean_angle_diff), 2),
        'perfect_matches': perfect,
        'elapsed_s': round(elapsed, 1),
    }
    sys.stderr.write(f'\n=== OBB NMS compare ===\n')
    sys.stderr.write(f'Ref: {total_ref}, Test: {total_test}\n')
    sys.stderr.write(f'Matched: {total_matched}, Mean IoU: {mean_iou:.6f}\n')
    sys.stderr.write(f'Mean angle diff: {np.degrees(mean_angle_diff):.2f}°\n')
    sys.stderr.write(f'Perfect: {perfect}/{len(images)}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
