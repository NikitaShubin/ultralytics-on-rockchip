#!/usr/bin/env python3
"""Оценка accuracy модели (yolo val).

Запускается внутри контейнера convert.
Использование: python3 eval_bench.py <model_name> <task> <model_path> [data_dir]

Для classify: top1, top5 accuracy.
Для detect/seg/pose/obb: mAP@0.5, mAP@0.5:0.95.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')

from ultralytics import YOLO


def _make_eval_yaml(task: str, data_dir: str) -> str:
    """Сгенерировать YAML для eval на лету."""
    data_path = Path(data_dir)

    if task == 'obb':
        lines = [
            f"path: {data_dir}",
            "train: images/val",
            "val: images/val",
            "nc: 15",
            "names:",
            "  0: plane", "  1: ship", "  2: storage-tank",
            "  3: baseball-diamond", "  4: tennis-court",
            "  5: basketball-court", "  6: ground-track-field",
            "  7: soccer-ball-field", "  8: roundabout",
            "  9: harbor", "  10: swimming-pool",
            "  11: helicopter", "  12: vehicle",
            "  13: helicopter", "  14: container-crane",
        ]
    elif task == 'semantic':
        lines = [
            f"path: {data_dir}",
            "train: images/val",
            "val: images/val",
            "nc: 19",
            "names:",
            "  0: road", "  1: sidewalk", "  2: building",
            "  3: wall", "  4: fence", "  5: pole",
            "  6: traffic light", "  7: traffic sign",
            "  8: vegetation", "  9: terrain",
            "  10: sky", "  11: person", "  12: rider",
            "  13: car", "  14: truck", "  15: bus",
            "  16: train", "  17: motorcycle", "  18: bicycle",
        ]
    else:
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush",
        ]
        lines = [
            f"path: {data_dir}",
            "train: images/val2017",
            "val: images/val2017",
            "nc: 80",
            "names:",
        ]
        for i, name in enumerate(coco_classes):
            lines.append(f"  {i}: {name}")
        lines.extend([
            "annotations:",
            "  train: annotations/instances_val2017.json",
            "  val: annotations/instances_val2017.json",
        ])

    yaml_path = Path(f"/tmp/eval_{task}.yaml")
    yaml_path.write_text("\n".join(lines) + "\n")
    return str(yaml_path)


def _ensure_yolo_labels(data_dir: str, task: str) -> None:
    """Конвертировать COCO annotations → YOLO labels если labels/ не существует."""
    data_path = Path(data_dir)

    # data_dir может быть images/val2017 — подняться к корню датасета
    if data_path.name in ('val2017', 'val', 'train') and data_path.parent.name == 'images':
        root = data_path.parent.parent
    else:
        root = data_path

    labels_dir = root / 'labels' / 'val2017'
    ann_file = root / 'annotations' / 'instances_val2017.json'

    if labels_dir.exists() and any(labels_dir.glob('*.txt')):
        sys.stderr.write(f'[EVAL] Labels already exist: {labels_dir}\n')
        sys.stderr.flush()
        return

    if not ann_file.exists():
        sys.stderr.write(f'[EVAL] No COCO annotations found: {ann_file}\n')
        sys.stderr.flush()
        return

    sys.stderr.write(f'[EVAL] Converting COCO annotations → YOLO labels...\n')
    sys.stderr.flush()

    import json as json_mod

    with open(ann_file) as f:
        coco = json_mod.load(f)

    from ultralytics.data.converter import coco91_to_coco80_class
    coco91to80 = coco91_to_coco80_class()
    cat_map = {cat['id']: coco91to80[cat['id'] - 1] if 1 <= cat['id'] <= 90 else 0 for cat in coco['categories']}

    # Image id → size
    img_info = {}
    for img in coco['images']:
        img_info[img['id']] = (img['width'], img['height'])

    # Collect annotations per image
    ann_by_img = {}
    for ann in coco['annotations']:
        if 'bbox' not in ann:
            continue
        img_id = ann['image_id']
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for img in coco['images']:
        img_id = img['id']
        w, h = img_info[img_id]
        fname = Path(img['file_name']).stem + '.txt'
        lines = []
        for ann in ann_by_img.get(img_id, []):
            cls_id = cat_map.get(ann['category_id'], 0)
            bbox = ann['bbox']  # [x, y, w, h] absolute
            x_center = (bbox[0] + bbox[2] / 2) / w
            y_center = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h
            lines.append(f'{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}')
        (labels_dir / fname).write_text('\n'.join(lines) + '\n')
        converted += 1

    sys.stderr.write(f'[EVAL] Converted {converted} labels → {labels_dir}\n')
    sys.stderr.flush()


def _resolve_data(data_dir: str, task: str) -> str:
    """Определить data: если это YAML — вернуть, если директория — сгенерировать YAML."""
    p = Path(data_dir)
    if p.suffix in ('.yaml', '.yml') and p.exists():
        return str(p)
    # Если data_dir указывает на images subdir — подняться к корню датасета
    if p.name in ('val2017', 'val', 'train') and p.parent.name == 'images':
        root = str(p.parent.parent)
    else:
        root = data_dir
    return _make_eval_yaml(task, root)


def main() -> dict:
    model_name = sys.argv[1]
    task = sys.argv[2]
    model_path = sys.argv[3]
    data_dir = sys.argv[4] if len(sys.argv) > 4 else None

    sys.stderr.write(json.dumps({'phase': 'eval_start', 'model': model_name, 'task': task, 'path': model_path, 'data': data_dir}) + '\n')
    sys.stderr.flush()

    t_start = time.time()

    if task == 'classify':
        data_path = data_dir or '/app/datasets/imagenet'
    else:
        if data_dir:
            _ensure_yolo_labels(data_dir, task)
        data_path = _resolve_data(data_dir, task) if data_dir else None

    yolo_task = 'classify' if task == 'classify' else 'detect' if task == 'detect' else 'segment' if task in ('segment', 'semantic') else task

    sys.stderr.write(f'[EVAL] Loading model: {model_path} (task={yolo_task})\n')
    sys.stderr.flush()

    model = YOLO(model_path)

    val_kwargs = {'imgsz': 640, 'verbose': False}
    if data_path:
        val_kwargs['data'] = data_path

    sys.stderr.write(f'[EVAL] Running yolo val with data={data_path}\n')
    sys.stderr.flush()

    results = model.val(task=yolo_task, **val_kwargs)

    total_elapsed = time.time() - t_start
    result = {}

    if task == 'classify':
        result['top1'] = float(results.top1) if hasattr(results, 'top1') and results.top1 is not None else None
        result['top5'] = float(results.top5) if hasattr(results, 'top5') and results.top5 is not None else None
    else:
        result['map50'] = float(results.box.map50) if hasattr(results, 'box') and results.box.map50 is not None else None
        result['map50_95'] = float(results.box.map) if hasattr(results, 'box') and results.box.map is not None else None
        if hasattr(results, 'box') and results.box.mp is not None:
            result['precision'] = float(results.box.mp)
            result['recall'] = float(results.box.mr)

    result['elapsed'] = round(total_elapsed, 1)
    result['model'] = model_name
    result['task'] = task

    metrics_summary = {k: v for k, v in result.items() if k not in ("elapsed", "model", "task") and v is not None}
    sys.stderr.write(f'[EVAL] Done in {total_elapsed:.0f}s: {json.dumps(metrics_summary)}\n')
    sys.stderr.flush()

    print(json.dumps(result))
    return result


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        result = {
            'model': sys.argv[1] if len(sys.argv) > 1 else '?',
            'task': sys.argv[2] if len(sys.argv) > 2 else '?',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        print(json.dumps(result))
        sys.exit(1)
