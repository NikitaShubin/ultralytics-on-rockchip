#!/usr/bin/env python3
"""Изолированное сравнение постпроцессов: один ONNX выход → два NMS.

Один прогон ONNX, сырой выход (1,84,8400) идёт в:
  A) ultralytics.utils.nms.non_max_suppression() — эталон
  B) наш postprocess_std.py — тест

Запуск:
    python /app/benchmark/compare_nms.py \
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

INPUT_SIZE = 640


# ---------------------------------------------------------------------------
# Preprocessing (одинаковый для обоих путей)
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, new_size: int = 640,
              color: tuple = (114, 114, 114)):
    """Letterbox resize — точная копия из percept/io + ultralytics."""
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    pad_left = round(pad_w / 2 - 0.1)
    pad_top = round(pad_h / 2 - 0.1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized,
        pad_top, pad_h - pad_top,
        pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=color,
    )
    return canvas, scale, (pad_w, pad_h)


def preprocess(img_path: str, input_size: int = 640):
    """Letterbox + normalize. Возвращает (tensor, scale, pad, orig_shape)."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None
    orig_shape = img.shape[:2]  # (H, W)
    canvas, scale, pad = letterbox(img, input_size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)  # (1,3,640,640)
    return tensor, scale, pad, orig_shape


# ---------------------------------------------------------------------------
# Ultralytics NMS (эталон)
# ---------------------------------------------------------------------------

def run_ultralytics_nms(raw_output: np.ndarray, conf: float, iou: float,
                        max_det: int, nc: int = 0,
                        scale: float = 1.0, pad: tuple = (0, 0),
                        orig_shape: tuple = (640, 640)):
    """Вызов ultralytics NMS + scale_boxes. Возвращает (N,6) numpy в original coords."""
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    pred = torch.from_numpy(raw_output.astype(np.float32))
    # non_max_suppression ожидает (batch, channels, num_boxes) = (1, 84, 8400)
    outputs = non_max_suppression(
        pred,
        conf_thres=conf,
        iou_thres=iou,
        nc=nc,
        multi_label=True,
        agnostic=False,
        max_det=max_det,
        end2end=False,
    )
    # outputs — list of (N, 6): [x1, y1, x2, y2, conf, cls] в model coords
    if not outputs or len(outputs[0]) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    result = outputs[0].cpu().numpy().astype(np.float32)

    # scale_boxes: model coords → original image coords
    # ratio_pad = ((gain,), (pad_x, pad_y)) — pad_x/pad_y = left/top offsets ONLY
    img1_shape = (INPUT_SIZE, INPUT_SIZE)  # (H, W) model
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    ratio_pad = ((scale,), (pad_left, pad_top))
    boxes_tensor = torch.from_numpy(result[:, :4])
    scale_boxes(img1_shape, boxes_tensor, orig_shape, ratio_pad)
    result[:, :4] = boxes_tensor.numpy()
    return result


# ---------------------------------------------------------------------------
# Наш postprocess (тест)
# ---------------------------------------------------------------------------

def load_our_postprocess(model_dir: str, model_name: str, conf: float, iou: float):
    """Загрузить наш postprocess модуль и установить пороги."""
    post_path = Path(model_dir) / f'{model_name}_postprocess.py'
    if not post_path.exists():
        post_path = Path(model_dir) / 'postprocess.py'
    if not post_path.exists():
        raise FileNotFoundError(f'postprocess не найден: {post_path}')

    spec = importlib.util.spec_from_file_location('postprocess_mod', str(post_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['postprocess_mod'] = mod
    spec.loader.exec_module(mod)

    # Установка порогов
    mod.CONF_THRESH = conf
    mod.IOU_THRESH = iou
    if hasattr(mod, 'dedup'):
        mod.dedup = lambda boxes, scores, class_ids: np.arange(len(boxes))

    return mod


def run_our_postprocess(mod, raw_output, scale, pad, orig_shape):
    """Запуск нашего postprocess. Возвращает (N,6) numpy: [x1,y1,x2,y2,conf,cls]."""
    outputs = [raw_output]  # postprocess ожидает list
    results = mod.postprocess(outputs, scale, pad, orig_shape)
    if not results:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = np.array([d['bbox'] for d in results], dtype=np.float32)
    conf = np.array([d['score'] for d in results], dtype=np.float32)
    cls = np.array([d['class_id'] for d in results], dtype=np.float32)
    return np.column_stack([xyxy, conf, cls])


# ---------------------------------------------------------------------------
# Сравнение
# ---------------------------------------------------------------------------

def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU матрица (N,4) x (M,4) xyxy."""
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


def greedy_match(iou_matrix: np.ndarray, iou_thresh: float = 0.5):
    """Жадный matching по IoU."""
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_coco_images(data_dir: str, limit: int) -> list[dict]:
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
        raise FileNotFoundError(f'instances_val2017.json не найден')
    coco = COCO(ann_file)
    img_ids = sorted(coco.getImgIds())[:limit]
    images = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        images.append({
            'id': img_id,
            'file_name': info['file_name'],
            'width': info['width'],
            'height': info['height'],
        })
    return images


def find_img_path(file_name: str, data_dir: str) -> str:
    data_path = Path(data_dir)
    for candidate in [
        data_path / 'images' / 'val2017' / file_name,
        data_path / 'images' / 'val' / file_name,
        data_path / file_name,
    ]:
        if candidate.exists():
            return str(candidate)
    return str(data_path / file_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Изолированное сравнение NMS: один ONNX → два постпроцесса')
    parser.add_argument('--model', required=True, help='Путь к ONNX модели')
    parser.add_argument('--data', required=True, help='Каталог COCO val2017')
    parser.add_argument('--model-name', required=True, help='Имя модели для постпроцесса')
    parser.add_argument('--limit', type=int, default=100, help='Количество изображений')
    parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=300, help='Max detections per image')
    parser.add_argument('--divergent-file', type=str, default='', help='Только divergent images')
    parser.add_argument('--save-divergent', type=str, default='', help='Сохранить divergent images')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')
    args = parser.parse_args()

    # Загрузка ONNX (одна сессия)
    sys.stderr.write(f'[NMS-COMPARE] Загрузка ONNX: {args.model}\n')
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Загрузка нашего postprocess
    model_dir = str(Path(args.model).parent)
    sys.stderr.write(f'[NMS-COMPARE] Загрузка postprocess: {args.model_name}\n')
    our_mod = load_our_postprocess(model_dir, args.model_name, args.conf, args.iou)

    # Загрузка изображений
    if args.divergent_file:
        with open(args.divergent_file) as f:
            div_data = json.load(f)
        images = [{'file_name': img['file_name']} for img in div_data['images']]
        sys.stderr.write(f'[NMS-COMPARE] Divergent images: {len(images)}\n')
    else:
        images = load_coco_images(args.data, args.limit)
        sys.stderr.write(f'[NMS-COMPARE] COCO images: {len(images)}\n')

    # Сравнение
    results = []
    total_ref = 0
    total_test = 0
    total_matched = 0
    all_ious = []
    t_start = time.time()

    for idx, img_info in enumerate(images):
        file_name = img_info['file_name']
        img_path = find_img_path(file_name, args.data)

        # 1. Препроцесс (один раз)
        tensor, scale, pad, orig_shape = preprocess(img_path)
        if tensor is None:
            results.append({'file_name': file_name, 'count_ref': 0, 'count_test': 0,
                            'count_matched': 0, 'avg_iou': 0.0})
            continue

        # 2. ONNX inference (один раз)
        raw = session.run(None, {input_name: tensor / 255.0})[0]  # (1, 84, 8400)

        # 3a. Ultralytics NMS + scale_boxes (эталон)
        ref = run_ultralytics_nms(raw, args.conf, args.iou, args.max_det,
                                  scale=scale, pad=pad, orig_shape=orig_shape)
        xyxy_ref = ref[:, :4]
        conf_ref = ref[:, 4]
        cls_ref = ref[:, 5]

        # 3b. Наш postprocess (тест)
        test = run_our_postprocess(our_mod, raw, scale, pad, orig_shape)
        xyxy_test = test[:, :4]
        conf_test = test[:, 4]
        cls_test = test[:, 5]

        n_ref = len(xyxy_ref)
        n_test = len(xyxy_test)
        total_ref += n_ref
        total_test += n_test

        # 4. Matching
        iou_matrix = compute_iou_matrix(xyxy_ref, xyxy_test)
        matched_a, matched_b, avg_iou = greedy_match(iou_matrix, iou_thresh=0.5)

        total_matched += len(matched_a)
        all_ious.append(avg_iou)

        result = {
            'file_name': file_name,
            'count_ref': n_ref,
            'count_test': n_test,
            'count_matched': len(matched_a),
            'avg_iou': round(avg_iou, 6),
        }

        if args.verbose or (n_ref != n_test) or (avg_iou < 0.99):
            sys.stderr.write(
                f'  [{idx+1}/{len(images)}] {file_name}  '
                f'ref={n_ref} test={n_test} matched={len(matched_a)} iou={avg_iou:.4f}\n'
            )

        results.append(result)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            sys.stderr.write(f'[NMS-COMPARE] {idx+1}/{len(images)} ({elapsed:.1f}s)\n')

    elapsed = time.time() - t_start

    # Итоги
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    min_iou = float(np.min(all_ious)) if all_ious else 0.0
    perfect = sum(1 for r in results if r['count_ref'] == r['count_test'] and r['avg_iou'] == 1.0)
    divergent = [r for r in results if r['avg_iou'] < 0.99]

    summary = {
        'model': args.model_name,
        'mode': 'isolated_nms',
        'params': {'conf': args.conf, 'iou': args.iou, 'max_det': args.max_det},
        'total_images': len(images),
        'total_ref': total_ref,
        'total_test': total_test,
        'total_matched': total_matched,
        'mean_iou': round(mean_iou, 6),
        'min_iou': round(min_iou, 6),
        'perfect_matches': perfect,
        'divergent_count': len(divergent),
        'elapsed_s': round(elapsed, 1),
        'images': results,
    }

    sys.stderr.write(f'\n=== Изолированное сравнение NMS ===\n')
    sys.stderr.write(f'Один ONNX прогон → два постпроцесса\n')
    sys.stderr.write(f'Изображений: {len(images)}\n')
    sys.stderr.write(f'Эталон (ultralytics NMS): {total_ref} детекций\n')
    sys.stderr.write(f'Тест (наш postprocess):   {total_test} детекций\n')
    sys.stderr.write(f'Совпавшие пары:           {total_matched}\n')
    sys.stderr.write(f'Средний IoU:              {mean_iou:.6f}\n')
    sys.stderr.write(f'Мин. IoU:                 {min_iou:.6f}\n')
    sys.stderr.write(f'Абсолютные совпадения:    {perfect}/{len(images)}\n')
    sys.stderr.write(f'Divergent (< 0.99):       {len(divergent)}/{len(images)}\n')

    if divergent:
        sys.stderr.write(f'\nТоп-10 divergent:\n')
        for r in sorted(divergent, key=lambda x: x['avg_iou'])[:10]:
            sys.stderr.write(
                f"  {r['file_name']}  ref={r['count_ref']} test={r['count_test']} "
                f"matched={r['count_matched']} iou={r['avg_iou']:.6f}\n"
            )

    if args.save_divergent:
        div_output = {
            'model': summary['model'],
            'mode': summary['mode'],
            'params': summary['params'],
            'divergent_count': len(divergent),
            'images': divergent,
        }
        with open(args.save_divergent, 'w') as f:
            json.dump(div_output, f, indent=2)
        sys.stderr.write(f'\nСохранено divergent: {args.save_divergent}\n')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        sys.stderr.write(f'Ошибка: {e}\n{traceback.format_exc()}\n')
        sys.exit(1)
