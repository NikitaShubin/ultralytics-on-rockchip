#!/usr/bin/env python3
"""Eval скрипт: проверка пайплайна целиком (изображение → bboxes).

Запускается внутри percept-контейнера:
    docker exec percept python /app/benchmark/eval.py detect /app/datasets/coco8
    docker exec percept python /app/benchmark/eval.py detect /app/datasets/coco_val2017

Использует Pipeline.detect() — тот же пайплайн что и в продакшене.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_coco(ann_file: str):
    """Загрузить COCO аннотации."""
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    cat_ids = sorted(coco.getCatIds())
    cat_map = {c: i for i, c in enumerate(cat_ids)}

    images = {}
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        images[img_id] = {
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height'],
        }

    ann_by_img = {}
    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns(ann_id)[0]
        img_id = ann['image_id']
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append({
            'category_id': cat_map[ann['category_id']],
            'bbox': ann['bbox'],
            'area': ann.get('area', 0),
            'iscrowd': ann.get('iscrowd', 0),
            'keypoints': ann.get('keypoints', []),
            'num_keypoints': ann.get('num_keypoints', 0),
            'segmentation': ann.get('segmentation', []),
        })

    return images, ann_by_img, cat_ids


def load_yolo_as_coco(data_dir: str):
    """Загрузить YOLO-формат аннотаций как COCO."""
    data_path = Path(data_dir)
    img_dir = data_path / 'images' / 'val'
    label_dir = data_path / 'labels' / 'val'

    if not img_dir.exists():
        for s in ((data_path / 'images').iterdir() if (data_path / 'images').exists() else []):
            if s.is_dir():
                img_dir = s
                break
    if not label_dir.exists():
        for s in ((data_path / 'labels').iterdir() if (data_path / 'labels').exists() else []):
            if s.is_dir():
                label_dir = s
                break

    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f'YOLO structure not found: img={img_dir} label={label_dir}')

    images = {}
    ann_by_img = {}
    all_cat_ids = set()
    ann_id = 1

    img_files = sorted(f for f in img_dir.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))

    for img_id, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        h, w = (img.shape[:2] if img is not None else (0, 0))
        images[img_id] = {'file_name': img_path.name, 'width': w, 'height': h}

        label_file = label_dir / (img_path.stem + '.txt')
        anns = []
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    all_cat_ids.add(cls_id)
                    abs_w, abs_h = bw * w, bh * h
                    anns.append({
                        'id': ann_id, 'image_id': img_id, 'category_id': cls_id,
                        'bbox': [xc * w - abs_w / 2, yc * h - abs_h / 2, abs_w, abs_h],
                        'area': abs_w * abs_h, 'iscrowd': 0,
                    })
                    ann_id += 1
        if anns:
            ann_by_img[img_id] = anns

    cat_ids = sorted(all_cat_ids)
    coco_ann = {
        'images': [{'id': k, 'file_name': v['file_name'], 'width': v['width'], 'height': v['height']}
                    for k, v in images.items()],
        'annotations': [a for anns in ann_by_img.values() for a in anns],
        'categories': [{'id': c, 'name': str(c)} for c in cat_ids],
    }
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='coco_')
    json.dump(coco_ann, tmp)
    tmp.close()
    return images, ann_by_img, cat_ids, tmp.name


def load_imagenet(val_dir: str):
    """Загрузить ImageNet val."""
    val_path = Path(val_dir)
    class_dirs = sorted(d for d in val_path.iterdir() if d.is_dir())
    class_map = {d.name: i for i, d in enumerate(class_dirs)}
    images, labels = [], []
    for name, cid in class_map.items():
        for f in sorted((val_path / name).glob('*.JPEG')):
            images.append(str(f))
            labels.append(cid)
    return images, labels


def load_dota(label_dir: str, img_dir: str):
    """Загрузить DOTA аннотации."""
    images = sorted(str(f) for f in Path(img_dir).glob('*.jpg'))
    labels = {}
    for lbl in Path(label_dir).glob('*.txt'):
        polygons = []
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                polygons.append({
                    'class_id': int(parts[0]),
                    'polygon': [float(x) for x in parts[1:9]],
                })
        labels[lbl.stem] = polygons
    return images, labels


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def find_img_path(img_info: dict, data_dir: str) -> str:
    """Найти путь к изображению по data_dir структуре."""
    data_path = Path(data_dir)
    for candidate in [
        data_path / 'images' / 'val2017',
        data_path / 'images' / 'val',
        data_path if data_path.name in ('val2017', 'val') and data_path.parent.name == 'images' else None,
        data_path,
    ]:
        if candidate is not None:
            p = candidate / img_info['file_name']
            if p.exists():
                return str(p)
    return str(data_path / img_info['file_name'])


# ---------------------------------------------------------------------------
# COCO format conversion
# ---------------------------------------------------------------------------

def detections_to_coco(detections: list[dict], image_id: int, cat_ids: list[int]) -> list[dict]:
    """Конвертировать detections в COCO формат."""
    results = []
    for det in detections:
        cat_id = det.get('class_id', 0)
        coco_cat_id = cat_ids[cat_id] if cat_id < len(cat_ids) else cat_id

        entry = {
            'image_id': image_id,
            'category_id': coco_cat_id,
            'score': float(det.get('score', 1.0)),
        }

        if 'bbox' in det:
            xyxy = det['bbox']
            if isinstance(xyxy, (list, np.ndarray)) and len(xyxy) == 4:
                x1, y1, x2, y2 = [float(x) for x in xyxy]
                entry['bbox'] = [x1, y1, x2 - x1, y2 - y1]

        if 'segmentation' in det and det['segmentation'] is not None:
            mask = det['segmentation']
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                import pycocotools.mask as mask_util
                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf-8')
                entry['segmentation'] = rle

        if 'keypoints' in det:
            kpts = det.get('keypoints', [])
            flat = []
            for kp in kpts:
                if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                    flat.extend([float(kp[0]), float(kp[1]), int(kp[2])])
                elif isinstance(kp, (list, tuple)) and len(kp) == 2:
                    flat.extend([float(kp[0]), float(kp[1]), 2])
            entry['keypoints'] = flat
            entry['num_keypoints'] = sum(1 for i in range(2, len(flat), 3) if flat[i] > 0)

        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_coco_map(coco_results: list[dict], ann_file: str, task: str = 'detect') -> dict:
    """Вычислить mAP через pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)
    if not coco_results:
        return {'map50': 0.0, 'map50_95': 0.0, 'ar100': 0.0}

    coco_dt = coco_gt.loadRes(coco_results)
    eval_type = {'segment': 'segm', 'pose': 'keypoints'}.get(task, 'bbox')

    coco_eval = COCOeval(coco_gt, coco_dt, eval_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        'map50': float(stats[1]),
        'map50_95': float(stats[0]),
        'ar100': float(stats[8]),
    }


def compute_imagenet_accuracy(predictions: list[int], ground_truth: list[int]) -> dict:
    """Вычислить top1 accuracy для ImageNet."""
    if not predictions:
        return {'top1': 0.0}
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return {'top1': correct / len(predictions)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(task: str, data_dir: str, model_dir: str = '/app/model',
             backend: str = '', model_name: str = '') -> dict:
    """Eval через Pipeline — тот же пайплайн что в продакшене."""
    import sys as _sys
    # Добавляем /app в путь чтобы найти pipeline.py и percept/
    app_dir = str(Path(__file__).resolve().parent.parent)
    if app_dir not in _sys.path:
        _sys.path.insert(0, app_dir)

    from pipeline import Pipeline

    t_start = time.time()

    # Определяем backend из model/ если не задан
    model_path = Path(model_dir)
    if not backend:
        if (model_path / 'model.rknn').exists():
            backend = 'npu'
        elif (model_path / 'model.onnx').exists():
            backend = 'cpu'
        else:
            backend = 'npu'

    # Определяем model_name из model_name.txt если не задан
    if not model_name:
        mn_file = model_path / 'model_name.txt'
        if mn_file.exists():
            model_name = mn_file.read_text().strip()
        else:
            model_name = 'model'

    sys.stderr.write(f'[EVAL] Модель: {model_name} (backend={backend})\n')
    pipeline = Pipeline(model_dir=str(model_path), model_name=model_name, backend=backend)

    # Eval mode: минимальный порог чтобы pycocotools сам выбрал оптимальный
    import sys as _sys2
    post_mod = _sys2.modules.get('postprocess_mod')
    if post_mod is not None:
        post_mod.CONF_THRESH = 0.001
        post_mod.IOU_THRESH = 0.7
        if hasattr(post_mod, 'dedup'):
            post_mod.dedup = lambda boxes, scores, class_ids: np.arange(len(boxes))
        sys.stderr.write(f'[EVAL] eval_mode: CONF_THRESH=0.001, IOU_THRESH=0.7\n')

    # Загрузка датасета
    ann_file = None
    tmp_ann_file = None
    images = {}
    ann_by_img = {}
    cat_ids = []

    if task == 'classify':
        img_paths, gt_labels = load_imagenet(data_dir)
        sys.stderr.write(f'[EVAL] ImageNet: {len(img_paths)} изображений\n')
    elif task in ('detect', 'segment', 'pose'):
        for candidate in [
            str(Path(data_dir).parent / 'annotations' / 'instances_val2017.json'),
            str(Path(data_dir) / 'annotations' / 'instances_val2017.json'),
        ]:
            if Path(candidate).exists():
                ann_file = candidate
                break

        if ann_file:
            images, ann_by_img, cat_ids = load_coco(ann_file)
            sys.stderr.write(f'[EVAL] COCO: {len(images)} изображений\n')
        else:
            try:
                images, ann_by_img, cat_ids, tmp_ann_file = load_yolo_as_coco(data_dir)
                ann_file = tmp_ann_file
                sys.stderr.write(f'[EVAL] YOLO→COCO: {len(images)} изображений\n')
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f'Аннотации не найдены в {data_dir}.\n{e}'
                )
    elif task == 'obb':
        dota_images, dota_labels = load_dota(
            str(Path(data_dir) / 'labels' / 'val'),
            str(Path(data_dir) / 'images' / 'val'),
        )
        sys.stderr.write(f'[EVAL] DOTA: {len(dota_images)} изображений\n')
    elif task == 'semantic':
        raise NotImplementedError('Semantic segmentation eval пока не реализован')

    # Инференс через Pipeline
    coco_results = []
    cls_predictions = []
    total = 0
    errors = 0

    if task == 'classify':
        for img_path, gt_label in zip(img_paths, gt_labels):
            frame = cv2.imread(img_path)
            if frame is None:
                errors += 1
                continue
            objects, _ = pipeline.detect(frame)
            cls_predictions.append(objects[0]['class_id'] if objects else -1)
            total += 1
            if total % 500 == 0:
                sys.stderr.write(f'[EVAL] {total}/{len(img_paths)}\n')
        metrics = compute_imagenet_accuracy(cls_predictions, gt_labels)
        metrics['total'] = total
        metrics['errors'] = errors

    elif task in ('detect', 'segment', 'pose'):
        for img_id, img_info in images.items():
            img_path = find_img_path(img_info, data_dir)
            frame = cv2.imread(img_path)
            if frame is None:
                errors += 1
                continue
            objects, _ = pipeline.detect(frame)
            coco_results.extend(detections_to_coco(objects, img_id, cat_ids))
            total += 1
            if total % 500 == 0:
                sys.stderr.write(f'[EVAL] {total}/{len(images)}\n')

        sys.stderr.write(f'[EVAL] mAP ({len(coco_results)} pred)...\n')
        metrics = compute_coco_map(coco_results, ann_file, task)
        metrics['total'] = total
        metrics['errors'] = errors
        metrics['predictions'] = len(coco_results)

    elif task == 'obb':
        coco_ann = {'images': [], 'annotations': [], 'categories': []}
        img_id_map = {}
        for i, img_path in enumerate(dota_images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = Path(img_path).stem
            coco_ann['images'].append({'id': i, 'file_name': Path(img_path).name, 'width': w, 'height': h})
            img_id_map[stem] = i
            for poly in dota_labels.get(stem, []):
                xs, ys = poly['polygon'][0::2], poly['polygon'][1::2]
                x1, y1 = min(xs) * w, min(ys) * h
                x2, y2 = max(xs) * w, max(ys) * h
                coco_ann['annotations'].append({
                    'id': len(coco_ann['annotations']), 'image_id': i,
                    'category_id': poly['class_id'],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0,
                })
        for i in range(15):
            coco_ann['categories'].append({'id': i, 'name': str(i)})

        for img_path in dota_images:
            stem = Path(img_path).stem
            if stem not in img_id_map:
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                errors += 1
                continue
            objects, _ = pipeline.detect(frame)
            coco_results.extend(detections_to_coco(objects, img_id_map[stem], cat_ids))
            total += 1

        import tempfile
        ann_file = tempfile.mktemp(suffix='.json')
        with open(ann_file, 'w') as f:
            json.dump(coco_ann, f)
        metrics = compute_coco_map(coco_results, ann_file, 'detect')
        metrics['total'] = total
        metrics['errors'] = errors
        metrics['predictions'] = len(coco_results)

    elapsed = time.time() - t_start
    metrics['elapsed_s'] = round(elapsed, 1)
    metrics['model'] = model_name
    metrics['task'] = task

    sys.stderr.write(f'[EVAL] Готово за {elapsed:.0f}с: {json.dumps(metrics)}\n')
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Использование: eval.py <task> <data_dir> [--model-dir DIR] [--backend npu|cpu] [--model-name NAME]'}))
        sys.exit(1)

    task = sys.argv[1]
    data_dir = sys.argv[2]

    model_dir = '/app/model'
    backend = ''
    model_name = ''
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == '--model-dir' and i + 1 < len(args):
            model_dir = args[i + 1]
            i += 2
        elif args[i] == '--backend' and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        elif args[i] == '--model-name' and i + 1 < len(args):
            model_name = args[i + 1]
            i += 2
        else:
            i += 1

    result = evaluate(task, data_dir, model_dir=model_dir, backend=backend, model_name=model_name)
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        result = {
            'model': '?',
            'task': sys.argv[1] if len(sys.argv) > 1 else '?',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        print(json.dumps(result))
        sys.exit(1)
