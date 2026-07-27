#!/usr/bin/env python3
"""Сравнение cls: наш postprocess vs ultralytics top-1/top-5.

Запуск (в convert-контейнере):
    python /app/benchmark/compare_cls.py \
        --model /app/cache/yolo11n-cls_std/model.onnx \
        --data /app/datasets/imagenet \
        --limit 500
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


def load_imagenet(val_dir, limit):
    val_path = Path(val_dir)
    # Handle both imagenet/ and imagenet/val/ structures
    if not any(d.is_dir() for d in val_path.iterdir() if d.name.startswith('n')):
        val_path = val_path / 'val'
    class_dirs = sorted(d for d in val_path.iterdir() if d.is_dir() and d.name.startswith('n'))
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    images, labels = [], []
    for name, cid in class_map.items():
        for f in sorted((val_path / name).glob('*.JPEG')):
            images.append(str(f))
            labels.append(cid)
    if limit:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--model-name', default='yolo11n-cls_std')
    args = parser.parse_args()

    import onnxruntime as ort
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Load our postprocess
    model_dir = str(Path(args.model).parent)
    post_path = Path(model_dir) / f'{args.model_name}_postprocess.py'
    spec = importlib.util.spec_from_file_location('postprocess_mod', str(post_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['postprocess_mod'] = mod
    spec.loader.exec_module(mod)

    images, labels = load_imagenet(args.data, args.limit)
    sys.stderr.write(f'Images: {len(images)}\n')

    our_top1 = 0
    our_top5 = 0
    ref_top1 = 0
    ref_top5 = 0
    agree = 0
    t_start = time.time()

    for idx, (img_path, gt_label) in enumerate(zip(images, labels)):
        img = cv2.imread(img_path)
        if img is None:
            continue
        canvas, _, _ = letterbox(img, INPUT_SIZE)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0) / 255.0

        raw = session.run(None, {input_name: tensor})[0]  # (1, 1000)

        # Our postprocess
        our_result = mod.postprocess([raw], 1.0, (0, 0), img.shape[:2])
        our_cls = our_result[0]['class_id'] if our_result else -1

        # Reference: top-1 from raw output
        flat = raw[0] if raw.ndim == 2 else raw.flatten()
        ref_sorted = np.argsort(flat)[::-1]
        ref_cls = int(ref_sorted[0])

        if our_cls == gt_label:
            our_top1 += 1
        if gt_label in ref_sorted[:5]:
            ref_top5 += 1
        if ref_cls == gt_label:
            ref_top1 += 1
        if gt_label in ref_sorted[:5]:
            our_top5 += 1  # same for both
        if our_cls == ref_cls:
            agree += 1

        if (idx + 1) % 100 == 0:
            sys.stderr.write(f'  {idx+1}/{len(images)}\n')

    elapsed = time.time() - t_start
    n = len(images)
    result = {
        'model': args.model_name,
        'total': n,
        'our_top1': f'{our_top1/n*100:.1f}%',
        'our_top5': f'{our_top5/n*100:.1f}%',
        'ref_top1': f'{ref_top1/n*100:.1f}%',
        'agree': f'{agree/n*100:.1f}%',
        'elapsed_s': round(elapsed, 1),
    }
    sys.stderr.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
