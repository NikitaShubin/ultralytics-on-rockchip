#!/usr/bin/env python3
"""Полное сравнение pipeline: ultralytics .predict() vs наш Pipeline.detect().

Каждый путь делает СВОЙ прогон ONNX (свои letterbox + normalize + inference).
Это тестирует полный пайплайн, включая препроцесс.

Запуск:
    python /app/benchmark/compare_full.py \
        --model /app/cache/yolo11n_std/model.onnx \
        --data /app/datasets/coco_val2017 \
        --model-name yolo11n_std \
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


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    n, m = len(boxes_a), len(boxes_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
    xa = np.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    ya = np.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    xb = np.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    yb = np.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
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
    for c in [
        str(Path(data_dir).parent / 'annotations' / 'instances_val2017.json'),
        str(Path(data_dir) / 'annotations' / 'instances_val2017.json'),
    ]:
        if Path(c).exists():
            ann_file = c
            break
    if not ann_file:
        raise FileNotFoundError('instances_val2017.json not found')
    coco = COCO(ann_file)
    img_ids = sorted(coco.getImgIds())[:limit]
    return [{'id': i, 'file_name': coco.loadImgs(i)[0]['file_name'],
             'width': coco.loadImgs(i)[0]['width'],
             'height': coco.loadImgs(i)[0]['height']} for i in img_ids]


def find_img_path(file_name, data_dir):
    dp = Path(data_dir)
    for c in [dp / 'images' / 'val2017' / file_name,
              dp / 'images' / 'val' / file_name,
              dp / file_name]:
        if c.exists():
            return str(c)
    return str(dp / file_name)


def main():
    parser = argparse.ArgumentParser(description='Полное сравнение: ultralytics .predict() vs Pipeline.detect()')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--save-divergent', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    app_dir = str(Path(__file__).resolve().parent.parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # ultralytics model
    sys.stderr.write(f'[FULL] Загрузка ultralytics: {args.model}\n')
    from ultralytics import YOLO
    ul_model = YOLO(args.model)

    # Pipeline
    sys.stderr.write(f'[FULL] Загрузка Pipeline\n')
    from pipeline import Pipeline
    mn_file = Path(args.model).parent / 'model_name.txt'
    model_name = args.model_name or (mn_file.read_text().strip() if mn_file.exists() else Path(args.model).stem)
    pipeline = Pipeline(model_dir=str(Path(args.model).parent), model_name=model_name, backend='cpu')

    # eval mode
    post_mod = sys.modules.get('postprocess_mod')
    if post_mod is not None:
        post_mod.CONF_THRESH = args.conf
        post_mod.IOU_THRESH = args.iou

    images = load_coco_images(args.data, args.limit)
    sys.stderr.write(f'[FULL] Images: {len(images)}\n')

    results = []
    total_ref = total_test = total_matched = 0
    all_ious = []
    t_start = time.time()

    for idx, img_info in enumerate(images):
        img_path = find_img_path(img_info['file_name'], args.data)

        # Path A: ultralytics .predict()
        r = ul_model.predict(source=img_path, conf=args.conf, iou=args.iou,
                             max_det=args.max_det, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy_ref = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        else:
            xyxy_ref = np.zeros((0, 4), dtype=np.float32)

        # Path B: Pipeline.detect()
        frame = cv2.imread(img_path)
        objects, _ = pipeline.detect(frame) if frame is not None else ([], {})
        xyxy_test = np.array([d['bbox'] for d in objects], dtype=np.float32) if objects else np.zeros((0, 4), dtype=np.float32)

        n_ref, n_test = len(xyxy_ref), len(xyxy_test)
        total_ref += n_ref
        total_test += n_test

        iou_matrix = compute_iou_matrix(xyxy_ref, xyxy_test)
        _, _, avg_iou = greedy_match(iou_matrix)
        total_matched += min(n_ref, n_test)
        all_ious.append(avg_iou)

        result = {
            'file_name': img_info['file_name'],
            'count_ref': n_ref, 'count_test': n_test,
            'avg_iou': round(avg_iou, 6),
        }

        if args.verbose or n_ref != n_test or avg_iou < 0.99:
            sys.stderr.write(
                f'  [{idx+1}/{len(images)}] {img_info["file_name"]}  '
                f'ref={n_ref} test={n_test} iou={avg_iou:.4f}\n')

        results.append(result)
        if (idx + 1) % 10 == 0:
            sys.stderr.write(f'[FULL] {idx+1}/{len(images)} ({time.time()-t_start:.1f}s)\n')

    elapsed = time.time() - t_start
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    divergent = [r for r in results if r['avg_iou'] < 0.99]

    summary = {
        'model': model_name, 'mode': 'full_pipeline',
        'params': {'conf': args.conf, 'iou': args.iou, 'max_det': args.max_det},
        'total_images': len(images),
        'total_ref': total_ref, 'total_test': total_test,
        'mean_iou': round(mean_iou, 6),
        'min_iou': round(float(np.min(all_ious)), 6),
        'divergent_count': len(divergent),
        'elapsed_s': round(elapsed, 1),
        'images': results,
    }

    sys.stderr.write(f'\n=== Полное сравнение pipeline ===\n')
    sys.stderr.write(f'ultralytics .predict():  {total_ref} детекций\n')
    sys.stderr.write(f'Pipeline.detect():       {total_test} детекций\n')
    sys.stderr.write(f'Средний IoU:             {mean_iou:.6f}\n')
    sys.stderr.write(f'Min IoU:                 {float(np.min(all_ious)):.6f}\n')
    sys.stderr.write(f'Divergent (< 0.99):      {len(divergent)}/{len(images)}\n')

    if args.save_divergent:
        with open(args.save_divergent, 'w') as f:
            json.dump({'model': model_name, 'mode': 'full_pipeline',
                        'divergent_count': len(divergent), 'images': divergent}, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        sys.stderr.write(f'Error: {e}\n{traceback.format_exc()}\n')
        sys.exit(1)
