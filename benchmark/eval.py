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

def load_coco(ann_file: str, task: str = 'detect'):
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
        # Skip non-keypoint annotations for pose eval
        if task == 'pose' and ann.get('num_keypoints', 0) == 0:
            continue
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
    # Handle both imagenet/ and imagenet/val/ structures
    if not any(d.is_dir() and d.name.startswith('n') for d in val_path.iterdir()):
        val_path = val_path / 'val'
    class_dirs = sorted(d for d in val_path.iterdir() if d.is_dir() and d.name.startswith('n'))
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
                    v = int(kp[2]) if kp[2] >= 1 else (2 if kp[2] > 0 else 0)
                    flat.extend([float(kp[0]), float(kp[1]), v])
                elif isinstance(kp, (list, tuple)) and len(kp) == 2:
                    flat.extend([float(kp[0]), float(kp[1]), 2])
            entry['keypoints'] = flat
            entry['num_keypoints'] = sum(1 for i in range(2, len(flat), 3) if flat[i] > 0)

        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_coco_map(coco_results: list[dict], ann_file: str, task: str = 'detect',
                      eval_img_ids: list = None) -> dict:
    """Вычислить mAP через pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import json as _json

    coco_gt = COCO(ann_file)
    if not coco_results:
        return {'map50': 0.0, 'map50_95': 0.0, 'ar100': 0.0}

    coco_dt = coco_gt.loadRes(coco_results)
    eval_type = {'segment': 'segm', 'pose': 'keypoints'}.get(task, 'bbox')

    if task == 'pose':
        import tempfile
        with open(ann_file) as f:
            coco_data = _json.load(f)
        # Filter GT to only keep annotations with keypoints
        coco_data['annotations'] = [
            a for a in coco_data['annotations']
            if a.get('num_keypoints', 0) > 0
        ]
        # Filter GT to only evaluated images
        if eval_img_ids is not None:
            eval_set = set(eval_img_ids)
            coco_data['images'] = [
                img for img in coco_data['images'] if img['id'] in eval_set
            ]
            coco_data['annotations'] = [
                a for a in coco_data['annotations'] if a['image_id'] in eval_set
            ]
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='kp_')
        _json.dump(coco_data, tmp)
        tmp.close()
        coco_gt = COCO(tmp.name)
    elif eval_img_ids is not None:
        pass  # handled below after coco_eval is created

    coco_eval = COCOeval(coco_gt, coco_dt, eval_type)
    if eval_img_ids is not None:
        coco_eval.params.imgIds = eval_img_ids
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


def compute_obb_map(gt_boxes_xywha: list, pred_boxes_xywha: list,
                    gt_classes: list, pred_classes: list,
                    pred_scores: list, iou_threshold: float = 0.5,
                    num_classes: int = 15) -> dict:
    """Вычислить mAP для OBB через rotated IoU (batch_probiou).

    Args:
        gt_boxes_xywha: list of np.array [cx, cy, w, h, angle] per image
        pred_boxes_xywha: list of np.array [cx, cy, w, h, angle] per image
        gt_classes: list of np.array [class_id] per image
        pred_classes: list of np.array [class_id] per image
        pred_scores: list of np.array [score] per image
    """
    from ultralytics.utils.metrics import batch_probiou
    import torch

    ap_per_class = []
    for cls_id in range(num_classes):
        all_tp = []
        all_fp = []
        all_scores = []

        for img_idx in range(len(gt_boxes_xywha)):
            gt_cls = gt_classes[img_idx]
            pred_cls = pred_classes[img_idx]
            gt_mask = gt_cls == cls_id
            pred_mask = pred_cls == cls_id

            gt_boxes = gt_boxes_xywha[img_idx][gt_mask] if gt_mask.any() else np.empty((0, 5))
            pred_boxes = pred_boxes_xywha[img_idx][pred_mask] if pred_mask.any() else np.empty((0, 5))
            scores = pred_scores[img_idx][pred_mask] if pred_mask.any() else np.empty(0)

            n_gt = len(gt_boxes)
            if len(pred_boxes) == 0:
                continue

            matched = np.zeros(n_gt, dtype=bool)

            order = scores.argsort()[::-1]
            for idx in order:
                all_scores.append(scores[idx])
                if n_gt == 0:
                    all_tp.append(0)
                    all_fp.append(1)
                    continue

                pred_t = torch.tensor(pred_boxes[idx:idx+1], dtype=torch.float32)
                gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
                ious = batch_probiou(pred_t, gt_t).numpy().ravel()

                best_iou = ious.max() if len(ious) > 0 else 0
                best_idx = ious.argmax() if len(ious) > 0 else -1

                if best_iou >= iou_threshold and not matched[best_idx]:
                    all_tp.append(1)
                    all_fp.append(0)
                    matched[best_idx] = True
                else:
                    all_tp.append(0)
                    all_fp.append(1)

        if not all_scores:
            ap_per_class.append(0.0)
            continue

        scores_arr = np.array(all_scores)
        tp_arr = np.array(all_tp)
        fp_arr = np.array(all_fp)

        order = scores_arr.argsort()[::-1]
        tp_cum = np.cumsum(tp_arr[order])
        fp_cum = np.cumsum(fp_arr[order])

        n_gt_total = sum(len(gt_boxes_xywha[i][gt_classes[i] == cls_id])
                         for i in range(len(gt_boxes_xywha)))
        if n_gt_total == 0:
            ap_per_class.append(0.0)
            continue

        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / n_gt_total

        # VOC AP: area under precision-recall curve (11-point)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = precision[recall >= t]
            ap += prec_at_recall.max() / 11 if len(prec_at_recall) > 0 else 0
        ap_per_class.append(ap)

    ap_arr = np.array(ap_per_class)
    map50 = float(ap_arr.mean()) if len(ap_arr) > 0 else 0.0
    return {'map50': map50, 'map50_95': map50, 'ar100': map50}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(task: str, data_dir: str, model_dir: str = '/app/model',
             backend: str = '', model_name: str = '', limit: int = 0) -> dict:
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
        ann_name = 'person_keypoints_val2017.json' if task == 'pose' else 'instances_val2017.json'
        for candidate in [
            str(Path(data_dir).parent / 'annotations' / ann_name),
            str(Path(data_dir) / 'annotations' / ann_name),
        ]:
            if Path(candidate).exists():
                ann_file = candidate
                break

        if ann_file:
            images, ann_by_img, cat_ids = load_coco(ann_file, task)
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
            if limit and total >= limit:
                break
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
        eval_img_ids = []
        for img_id, img_info in images.items():
            if limit and total >= limit:
                break
            img_path = find_img_path(img_info, data_dir)
            frame = cv2.imread(img_path)
            if frame is None:
                errors += 1
                continue
            objects, _ = pipeline.detect(frame)
            coco_results.extend(detections_to_coco(objects, img_id, cat_ids))
            eval_img_ids.append(img_id)
            total += 1
            if total % 500 == 0:
                sys.stderr.write(f'[EVAL] {total}/{len(images)}\n')

        sys.stderr.write(f'[EVAL] mAP ({len(coco_results)} pred)...\n')
        metrics = compute_coco_map(coco_results, ann_file, task, eval_img_ids=eval_img_ids)
        metrics['total'] = total
        metrics['errors'] = errors
        metrics['predictions'] = len(coco_results)

    elif task == 'obb':
        def poly_to_xywha(polygon, img_w, img_h):
            """DOTA polygon [x1,y1,...,x4,y4] (normalized) → [cx, cy, w, h, angle_rad]."""
            pts = np.array([[polygon[i]*img_w, polygon[i+1]*img_h] for i in range(0, 8, 2)], dtype=np.float32)
            rect = cv2.minAreaRect(pts)
            (cx, cy), (w, h), angle_deg = rect
            angle_rad = angle_deg * np.pi / 180.0
            return np.array([cx, cy, w, h, angle_rad], dtype=np.float32)

        gt_boxes_per_img = []
        gt_classes_per_img = []
        pred_boxes_per_img = []
        pred_classes_per_img = []
        pred_scores_per_img = []

        for img_path in dota_images:
            if limit and total >= limit:
                break
            img = cv2.imread(img_path)
            if img is None:
                errors += 1
                continue
            h, w = img.shape[:2]
            stem = Path(img_path).stem

            polys = dota_labels.get(stem, [])
            if polys:
                boxes = np.array([poly_to_xywha(p['polygon'], w, h) for p in polys])
                classes = np.array([p['class_id'] for p in polys])
            else:
                boxes = np.empty((0, 5), dtype=np.float32)
                classes = np.array([], dtype=np.int64)
            gt_boxes_per_img.append(boxes)
            gt_classes_per_img.append(classes)

            objects, _ = pipeline.detect(img)
            if objects:
                bboxes = np.array([obj['bbox'] for obj in objects])
                scores = np.array([obj['score'] for obj in objects])
                cl = np.array([obj['class_id'] for obj in objects])
                angles = np.array([obj.get('angle', 0.0) for obj in objects])
                boxes_xywha = np.column_stack([bboxes, angles]).astype(np.float32)
            else:
                boxes_xywha = np.empty((0, 5), dtype=np.float32)
                scores = np.array([], dtype=np.float32)
                cl = np.array([], dtype=np.int64)
            pred_boxes_per_img.append(boxes_xywha)
            pred_classes_per_img.append(cl)
            pred_scores_per_img.append(scores)
            total += 1

        sys.stderr.write(f'[EVAL] OBB: {total} images, computing rotated mAP...\n')
        metrics = compute_obb_map(
            gt_boxes_per_img, pred_boxes_per_img,
            gt_classes_per_img, pred_classes_per_img, pred_scores_per_img,
            num_classes=15,
        )
        metrics['total'] = total
        metrics['errors'] = errors
        metrics['predictions'] = sum(len(p) for p in pred_boxes_per_img)

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
    limit = 0
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
        elif args[i] == '--limit' and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1

    result = evaluate(task, data_dir, model_dir=model_dir, backend=backend, model_name=model_name, limit=limit)
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
