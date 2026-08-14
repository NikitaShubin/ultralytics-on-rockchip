#!/usr/bin/env python
"""Генерация YOLO labels (detect/pose/seg) из COCO JSON.

Работает внутри convert-контейнера (numpy+cv2+ultralytics) — на хосте numpy нет.
Структура результата повторяет ../percept/datasets:

  coco_val2017/images/val2017/*.jpg     (уже есть, общий источник)
  coco_val2017/labels/val2017/*.txt     detect — 5 колонок (cls cx cy w h)
  coco_pose/images/val2017/*.jpg        hardlink на coco_val2017/images
  coco_pose/labels/val2017/*.txt        pose — 56 колонок (cls cx cy w h + 17*(x y v))
  coco_seg/images/val2017/*.jpg         hardlink на coco_val2017/images
  coco_seg/labels/val2017/*.txt         seg — cls + полигон

Для каждого изображения из JSON создаётся файл (пустой `\\n`, если аннотаций нет).

Пример запуска:
  docker compose -f convert/docker-compose.yml run --rm --entrypoint python \
    convert /app/convert/setup_coco_labels.py --all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fallback-копии из ultralytics (на случай запуска вне контейнера/без ультралайтс)
# ---------------------------------------------------------------------------

def _coco91_to_coco80_class():
    return [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, None, 24, 25, None, None, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None,
        61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, None, 73, 74, 75,
        76, 77, 78, 79, None,
    ]


def _min_index(arr1, arr2):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return (int(x) for x in __import__("numpy").unravel_index(
        __import__("numpy").argmin(dis, axis=None), dis.shape))


def _merge_multi_segment(segments):
    import numpy as np

    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    for i in range(1, len(segments)):
        idx1, idx2 = _min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def _load_helpers():
    """Вернуть (coco91_to_coco80, merge_multi_segment). Предпочитаем ultralytics."""
    try:
        from ultralytics.data.converter import coco91_to_coco80_class, merge_multi_segment
        return coco91_to_coco80_class(), merge_multi_segment
    except Exception:
        return _coco91_to_coco80_class(), _merge_multi_segment


# ---------------------------------------------------------------------------
# Генерация labels
# ---------------------------------------------------------------------------

def _img_lines_detect(anns, img_w, img_h, coco80):
    """Строки detect-labels для списка аннотаций изображения.

    ВНИМАНИЕ: iscrowd НЕ фильтруем — eval.py (compute_inmemory_box_map →
    _box_tp_per_image) включает их в GT. Для segment iscrowd фильтруется
    (см. _img_lines_seg), поэтому в percept seg-labels без iscrowd.
    """
    lines = []
    for ann in anns:
        box = [float(x) for x in ann["bbox"]]  # tl x, tl y, w, h
        cx = (box[0] + box[2] / 2) / img_w
        cy = (box[1] + box[3] / 2) / img_h
        bw = box[2] / img_w
        bh = box[3] / img_h
        if bw <= 0 or bh <= 0:
            continue
        cls80 = coco80[ann["category_id"] - 1]
        if cls80 is None:
            continue
        lines.append([cls80, cx, cy, bw, bh])
    return lines


def _img_lines_pose(anns, img_w, img_h):
    """Строки pose-labels (cls cx cy w h + 17*(x y v)) для аннотаций изображения.

    ВНИМАНИЕ: iscrowd НЕ фильтруем — percept включает их в pose-labels
    (отличий от detect/seg: там convert_coco-фильтр работает).
    """
    lines = []
    for ann in anns:
        box = [float(x) for x in ann["bbox"]]
        cx = (box[0] + box[2] / 2) / img_w
        cy = (box[1] + box[3] / 2) / img_h
        bw = box[2] / img_w
        bh = box[3] / img_h
        if bw <= 0 or bh <= 0:
            continue
        kpts = ann.get("keypoints")
        if kpts is None:
            continue
        row = [0, cx, cy, bw, bh]
        kpts = [k / d for k, d in zip(kpts, [img_w, img_h, 1] * 17)]
        for i in range(2, len(kpts), 3):
            kpts[i] = int(round(kpts[i]))  # visibility 0/1/2 — как в percept
        row.extend(kpts)
        lines.append(row)
    return lines


def _img_lines_seg(anns, img_w, img_h, coco80, merge_multi_segment):
    """Строки seg-labels (cls + полигон) для аннотаций изображения."""
    import numpy as np

    lines = []
    for ann in anns:
        if ann.get("iscrowd", False):
            continue
        cls80 = coco80[ann["category_id"] - 1]
        if cls80 is None:
            continue
        seg = ann.get("segmentation")
        if seg is None or len(seg) == 0:
            continue
        if len(seg) > 1:
            s = merge_multi_segment(seg)
            s = (np.concatenate(s, axis=0) / np.array([img_w, img_h])).reshape(-1).tolist()
        else:
            s = [j for i in seg for j in i]
            s = (np.array(s).reshape(-1, 2) / np.array([img_w, img_h])).reshape(-1).tolist()
        lines.append([cls80, *s])
    return lines


def _fmt(lines):
    """Записать строки в файл: полная точность float64 (str == repr)."""
    out = []
    for line in lines:
        out.append(" ".join(str(x) for x in line))
    return "\n".join(out) + "\n"


def _write_labels(label_dir, anns_by_img, images, task, merge_multi_segment, coco80):
    label_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        anns = anns_by_img.get(img["id"], [])
        h, w = img["height"], img["width"]
        if task == "detect":
            lines = _img_lines_detect(anns, w, h, coco80)
        elif task == "pose":
            lines = _img_lines_pose(anns, w, h)
        else:
            lines = _img_lines_seg(anns, w, h, coco80, merge_multi_segment)
        (label_dir / (img["file_name"] or f"{img['id']:012d}.jpg")).with_suffix(".txt").write_text(_fmt(lines), encoding="utf-8")


def _hardlink_images(src_img_dir, dst_img_dir):
    """Hardlink всех изображений из src в dst (идемпотентно)."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_img_dir.iterdir()):
        if not src.is_file():
            continue
        dst = dst_img_dir / src.name
        if not dst.exists():
            os.link(src, dst)


def ensure_coco_labels(task, data_dir="/app/datasets", force=False):
    """Сгенерировать yolo-labels для задачи из COCO JSON.

    task: "detect" | "segment" | "seg" | "pose"
    Возвращает корень датасета (coco_val2017, coco_pose или coco_seg).
    """
    task = "seg" if task == "segment" else task
    data_dir = Path(data_dir)
    coco_dir = data_dir / "coco_val2017"
    ann_dir = coco_dir / "annotations"

    json_name = "person_keypoints_val2017.json" if task == "pose" else "instances_val2017.json"
    ann_file = ann_dir / json_name
    if not ann_file.exists():
        raise RuntimeError(f"Аннотации не найдены: {ann_file}")

    # Имя дерева результата
    if task == "detect":
        root = coco_dir
        labels_dir = coco_dir / "labels" / "val2017"
        link_images = False
    elif task == "pose":
        root = data_dir / "coco_pose"
        labels_dir = root / "labels" / "val2017"
        link_images = True
    else:
        root = data_dir / "coco_seg"
        labels_dir = root / "labels" / "val2017"
        link_images = True

    with open(ann_file, encoding="utf-8") as f:
        data = json.load(f)

    images = sorted(data["images"], key=lambda x: x["id"])
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Идемпотентность: все файлы уже есть
    if not force and labels_dir.is_dir():
        have = sum(1 for p in labels_dir.glob("*.txt"))
        if have >= len(images):
            print(f"[LABELS] {task}: уже готово ({have} файлов), skip")
            if link_images:
                _hardlink_images(coco_dir / "images" / "val2017", root / "images" / "val2017")
            return root

    coco80, merge_multi_segment = _load_helpers()

    print(f"[LABELS] {task}: {len(images)} изображений → {labels_dir}")
    _write_labels(labels_dir, anns_by_img, images, task, merge_multi_segment, coco80)

    if link_images:
        src_img = coco_dir / "images" / "val2017"
        dst_img = root / "images" / "val2017"
        if src_img.is_dir():
            _hardlink_images(src_img, dst_img)
        else:
            raise RuntimeError(f"Нет изображений для линковки: {src_img}")

    n_files = sum(1 for p in labels_dir.glob("*.txt"))
    print(f"[LABELS] {task}: готово, {n_files} файлов в {labels_dir}")
    return root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Генерация YOLO labels из COCO JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Все задачи: detect + pose + seg")
    group.add_argument("--detect", action="store_true", help="Только detect-labels")
    group.add_argument("--pose", action="store_true", help="Только pose-labels")
    group.add_argument("--seg", action="store_true", help="Только seg-labels")
    parser.add_argument("--data-dir", default="/app/datasets", help="Корень датасетов")
    parser.add_argument("--force", action="store_true", help="Перегенерировать даже если готово")
    args = parser.parse_args()

    tasks = []
    if args.all or not any((args.detect, args.pose, args.seg)):
        tasks = ["detect", "pose", "seg"]
    if args.detect:
        tasks = ["detect"]
    if args.pose:
        tasks = ["pose"]
    if args.seg:
        tasks = ["seg"]

    for task in tasks:
        ensure_coco_labels(task, data_dir=args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
