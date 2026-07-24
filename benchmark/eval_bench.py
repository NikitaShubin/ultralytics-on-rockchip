#!/usr/bin/env python3
"""Оценка accuracy модели (yolo val).

Запускается внутри контейнера converter.
Использование: python3 eval_bench.py <model_name> <task> <model_path> [data_dir]

Для classify: top1, top5 accuracy.
Для detect/seg/pose/obb: mAP@0.5, mAP@0.5:0.95.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/app/converter')
import converter  # noqa: E402


def main() -> dict:
    """Основная функция: eval модели."""
    model_name = sys.argv[1]
    task = sys.argv[2]
    model_path = sys.argv[3]
    data_dir = sys.argv[4] if len(sys.argv) > 4 else None

    sys.stderr.write(json.dumps({'phase': 'eval_start', 'model': model_name, 'task': task}) + '\n')

    t_start = time.time()

    if task == 'classify':
        result = _eval_cls(model_name, model_path, data_dir)
    elif task in ('detect', 'segment', 'pose', 'obb', 'semantic'):
        result = _eval_map(model_name, task, model_path, data_dir)
    else:
        result = {'error': f'задача {task!r} не поддерживается для eval'}

    total_elapsed = time.time() - t_start
    result['elapsed'] = round(total_elapsed, 1)
    result['model'] = model_name
    result['task'] = task

    sys.stderr.write(json.dumps({'phase': 'eval_done', 'elapsed': round(total_elapsed, 1)}) + '\n')
    print(json.dumps(result))
    return result


def _eval_cls(model_name: str, model_path: str, data_dir: str | None) -> dict:
    """Оценка classification модели."""
    if data_dir is None:
        data_dir = '/app/imagenet'

    cmd = f'yolo val task=classify model={model_path} data={data_dir}'
    sys.stderr.write(f'[EVAL] {cmd}\n')

    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    sys.stderr.write(f'[EVAL] yolo val classify завершён за {elapsed:.0f}с\n')

    result = {}
    stdout = proc.stdout
    stderr = proc.stderr
    sys.stderr.write(f'[EVAL] stdout ({len(stdout)} байт)\n')
    sys.stderr.write(f'[EVAL] stderr ({len(stderr)} байт)\n')

    for line in stdout.splitlines():
        line_s = line.strip()
        if 'top1' in line_s.lower() or 'accuracy' in line_s.lower():
            parts = line_s.split()
            for i, p in enumerate(parts):
                p_lower = p.lower()
                if 'top1' in p_lower or p_lower == 'top-1':
                    try:
                        result['top1'] = float(parts[i + 1].rstrip('%'))
                    except (ValueError, IndexError):
                        pass
                elif 'top5' in p_lower or p_lower == 'top-5':
                    try:
                        result['top5'] = float(parts[i + 1].rstrip('%'))
                    except (ValueError, IndexError):
                        pass
        if '"top1_acc"' in line_s or '"top5_acc"' in line_s:
            try:
                data = json.loads(line_s)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if 'top1' in k.lower():
                            result['top1'] = v
                        if 'top5' in k.lower():
                            result['top5'] = v
            except (json.JSONDecodeError, TypeError):
                pass

    return result


def _eval_map(model_name: str, task: str, model_path: str, data_dir: str | None) -> dict:
    """Оценка detect/segment/pose/obb модели (mAP)."""
    if task == 'semantic':
        task_yolo = 'segment'
    else:
        task_yolo = task

    if data_dir is None:
        data_yaml = str(converter.COCO_VAL_YAML)
    else:
        data_yaml = _make_eval_yaml(task, data_dir)

    cmd = f'yolo val task={task_yolo} model={model_path} data={data_yaml}'
    sys.stderr.write(f'[EVAL] {cmd}\n')

    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    sys.stderr.write(f'[EVAL] yolo val завершён за {elapsed:.0f}с\n')

    result = {}
    stdout = proc.stdout
    stderr = proc.stderr
    sys.stderr.write(f'[EVAL] stdout ({len(stdout)} байт)\n')
    sys.stderr.write(f'[EVAL] stderr ({len(stderr)} байт)\n')

    for line in stdout.splitlines():
        line_s = line.strip()
        if line_s.startswith('all') and len(line_s.split()) >= 6:
            parts = line_s.split()
            try:
                result['precision'] = float(parts[2]) if len(parts) > 2 else None
                result['recall'] = float(parts[3]) if len(parts) > 3 else None
                result['map50'] = float(parts[4]) if len(parts) > 4 else None
                result['map50_95'] = float(parts[5]) if len(parts) > 5 else None
            except (ValueError, IndexError):
                pass
        if '"map50"' in line_s or '"metrics/mAP50(B)"' in line_s:
            try:
                data = json.loads(line_s)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if 'map50' in k.lower():
                            result['map50'] = v
                        if 'map50' in k.lower() and '95' in k.lower():
                            result['map50_95'] = v
                        if k == 'precision' or 'precision' in k.lower():
                            result['precision'] = v
                        if k == 'recall' or 'recall' in k.lower():
                            result['recall'] = v
            except (json.JSONDecodeError, TypeError):
                pass

    return result


def _make_eval_yaml(task: str, data_dir: str) -> str:
    """Сгенерировать YAML для eval на лету."""
    data_path = Path(data_dir)

    if task == 'obb':
        n = len(list((data_path / 'images' / 'val').glob('*.png'))) if (data_path / 'images' / 'val').exists() else 0
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
