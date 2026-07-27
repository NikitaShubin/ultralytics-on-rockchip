#!/usr/bin/env python3
"""Изолированное сравнение постпроцессов для pose: один ONNX → два NMS.

Запуск:
    docker run --rm -v ... --entrypoint python percept-converter:latest \
        /app/benchmark/compare_pose.py \
        --model /app/cache/yolo11n-pose_std/model.onnx \
        --data /app/datasets/coco_val2017 \
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


def run_ultralytics_nms(raw_output, conf, iou, max_det, nc=1,
                        scale=1.0, pad=(0, 0), orig_shape=(640, 640)):
    """Ultralytics NMS for pose."""
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    pred = torch.from_numpy(raw_output.astype(np.float32))
    outputs = non_max_suppression(
        pred, conf_thres=conf, iou_thres=iou, nc=nc,
        multi_label=False, agnostic=False, max_det=max_det,
        end2end=False, rotated=False,
    )
    if not outputs or len(outputs[0]) == 0:
        return np.zeros((0, 6)), np.zeros((0, 17, 3))

    result = outputs[0].cpu().numpy().astype(np.float32)
    # result: (N, 57) = [x1, y1, x2, y2, conf, cls, kpt×51]

    # scale_boxes
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    ratio_pad = ((scale,), (pad_left, pad_top))
    img1_shape = (INPUT_SIZE, INPUT_SIZE)
    boxes_tensor = torch.from_numpy(result[:, :4])
    scale_boxes(img1_shape, boxes_tensor, orig_shape, ratio_pad)
    result[:, :4] = boxes_tensor.numpy()

    # Scale keypoints
    kpts = result[:, 6:].reshape(-1, 17, 3).copy()
    # scale_coords for kpts: same transform as boxes
    for k in range(17):
        kpts[:, k, 0] = (kpts[:, k, 0] - pad_left) / scale
        kpts[:, k, 1] = (kpts[:, k, 1] - pad_top) / scale

    boxes = result[:, :6]  # xyxy, conf, cls
    return boxes, kpts


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
        return np.zeros((0, 6)), np.zeros((0, 17, 3))
    xyxy = np.array([d['bbox'] for d in results], dtype=np.float32)
    conf = np.array([d['score'] for d in results], dtype=np.float32)
    cls = np.array([d['class_id'] for d in results], dtype=np.float32)
    boxes = np.column_stack([xyxy, conf, cls])

    kpts = np.zeros((len(results), 17, 3), dtype=np.float32)
    for i, d in enumerate(results):
        if 'keypoints' in d:
            kpts[i] = np.array(d['keypoints'], dtype=np.float32)
    return boxes, kpts


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
    ann_file = None
    for candidate in [
        str(Path(data_dir).parent / 'annotations' / 'instances_val2017.json'),
        str(Path(data_dir) / 'annotations' / 'instances_val2017.json'),
    ]:
        if Path(candidate).exists():
            ann_file = candidate
            break
    if not ann_file:
        raise FileNotFoundError('instances_val2017.json not found')
    coco = COCO(ann_file)
    img_ids = sorted(coco.getImgIds())[:limit]
    images = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        images.append({'file_name': info['file_name']})
    return images


def find_img_path(file_name, data_dir):
    data_path = Path(data_dir)
    for candidate in [
        data_path / 'images' / 'val2017' / file_name,
        data_path / 'images' / 'val' / file_name,
        data_path / file_name,
    ]:
        if candidate.exists():
            return str(candidate)
    return str(data_path / file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-name', default='yolo11n-pose_std')
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

    images = load_coco_images(args.data, args.limit)
    sys.stderr.write(f'Pose compare: {len(images)} images, nc=1, extra=51 (17×3)\n')

    total_ref = 0
    total_test = 0
    total_matched = 0
    all_ious = []
    all_kpt_dists = []
    t_start = time.time()

    for idx, img_info in enumerate(images):
        img_path = find_img_path(img_info['file_name'], args.data)
        tensor, scale, pad, orig_shape = preprocess(img_path)
        if tensor is None:
            continue

        raw = session.run(None, {input_name: tensor / 255.0})[0]

        # Ultralytics reference
        ref_boxes, ref_kpts = run_ultralytics_nms(
            raw, args.conf, args.iou, args.max_det, nc=1,
            scale=scale, pad=pad, orig_shape=orig_shape,
        )

        # Our postprocess
        test_boxes, test_kpts = run_our_postprocess(
            our_mod, raw, scale, pad, orig_shape,
        )

        n_ref = len(ref_boxes)
        n_test = len(test_boxes)
        total_ref += n_ref
        total_test += n_test

        # Box matching
        iou_matrix = compute_iou_matrix(ref_boxes[:, :4], test_boxes[:, :4])
        matched_a, matched_b, avg_iou = greedy_match(iou_matrix, iou_thresh=0.5)
        total_matched += len(matched_a)
        all_ious.append(avg_iou)

        # Keypoint distance for matched pairs
        for a, b in zip(matched_a, matched_b):
            dist = np.sqrt(((ref_kpts[a] - test_kpts[b])[:, :2] ** 2).sum(axis=1)).mean()
            all_kpt_dists.append(dist)

        if idx % 10 == 0:
            sys.stderr.write(f'  [{idx+1}/{len(images)}] ref={n_ref} test={n_test} '
                           f'matched={len(matched_a)} iou={avg_iou:.4f}\n')

    elapsed = time.time() - t_start
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_kpt_dist = float(np.mean(all_kpt_dists)) if all_kpt_dists else 0.0
    perfect = sum(1 for i in all_ious if i == 1.0)

    result = {
        'model': args.model_name,
        'total_images': len(images),
        'total_ref': total_ref,
        'total_test': total_test,
        'total_matched': total_matched,
        'mean_box_iou': round(mean_iou, 6),
        'mean_kpt_dist_px': round(mean_kpt_dist, 2),
        'perfect_matches': perfect,
        'elapsed_s': round(elapsed, 1),
    }
    sys.stderr.write(f'\n=== Pose NMS compare ===\n')
    sys.stderr.write(f'Ref boxes: {total_ref}, Test boxes: {total_test}\n')
    sys.stderr.write(f'Matched: {total_matched}, Mean IoU: {mean_iou:.6f}\n')
    sys.stderr.write(f'Mean keypoint dist: {mean_kpt_dist:.2f} px\n')
    sys.stderr.write(f'Perfect: {perfect}/{len(images)}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
