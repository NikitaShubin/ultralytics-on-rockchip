#!/usr/bin/env python3
"""Eval скрипт: проверка пайплайна целиком (изображение → bboxes).

Запускается внутри convert-контейнера (нужен ultralytics для метрик):
    docker compose -f convert/docker-compose.yml run --rm --entrypoint python \
        convert /app/benchmark/eval.py detect /app/datasets/coco_val2017

Использует Pipeline.detect() — тот же пайплайн что и в продакшене.
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import os
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
                    ann = {
                        'id': ann_id, 'image_id': img_id, 'category_id': cls_id,
                        'bbox': [xc * w - abs_w / 2, yc * h - abs_h / 2, abs_w, abs_h],
                        'area': abs_w * abs_h, 'iscrowd': 0,
                    }
                    # YOLO pose: extra 3*17=51 values (17 keypoints x,y,v)
                    if len(parts) >= 56:
                        flat_kpts = []
                        n_kpts = (len(parts) - 5) // 3
                        for k in range(n_kpts):
                            kx = float(parts[5 + k*3]) * w
                            ky = float(parts[5 + k*3 + 1]) * h
                            kv = int(float(parts[5 + k*3 + 2]))
                            flat_kpts.extend([kx, ky, kv])
                        ann['keypoints'] = flat_kpts
                        ann['num_keypoints'] = sum(1 for i in range(2, len(flat_kpts), 3) if flat_kpts[i] > 0)
                    anns.append(ann)
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


def xyxyxyxy2xywhr_np(x):
    """Реплика ultralytics.utils.ops.xyxyxyxy2xywhr для одного (8,) массива."""
    pts = np.asarray(x, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    theta = angle_deg / 180.0 * np.pi
    if w < h:
        w, h = h, w
        theta += np.pi / 2
    while theta >= 3 * np.pi / 4:
        theta -= np.pi
    while theta < -np.pi / 4:
        theta += np.pi
    return np.array([cx, cy, w, h, theta], dtype=np.float32)


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


def load_cityscapes(img_dir: str, mask_dir: str, nc: int = 19,
                    label_mapping: dict | None = None) -> list[tuple[str, str]]:
    """Загрузить пары (изображение, маска) для семантической сегментации.

    Структура как в cityscapes8.yaml: images/val + masks/val (PNG, те же stem).
    Маски при необходимости ремапятся label_mapping (исходные id → trainId,
    'ignore_label'/-1 → 255), иначе используются как есть (должны быть trainId 0..nc-1).
    """
    imgs = sorted(str(p) for p in Path(img_dir).glob('*.*') if p.suffix.lower() in ('.jpg', '.png'))
    pairs = []
    for img in imgs:
        mask = str(Path(mask_dir) / f'{Path(img).stem}.png')
        if Path(mask).exists():
            pairs.append((img, mask))
    return pairs


CITYSCAPES_LABEL_MAPPING = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
}


def apply_cityscapes_label_mapping(mask: np.ndarray) -> np.ndarray:
    """Ремап cityscapes labelIds → trainId (0..18, 255 ignore), как cityscapes8.yaml."""
    out = np.full(mask.shape, 255, dtype=np.int64)
    for src, dst in CITYSCAPES_LABEL_MAPPING.items():
        out[mask == src] = dst
    return out


def letterbox_semantic_gt(mask: np.ndarray, scale: float, pad: tuple, isz: int) -> np.ndarray:
    """Реплика ultralytics LetterBox.apply_semantic для GT-маски.

    Маска NEAREST-ресайзится к new_unpad и паддится 255 (ignore), как в
    ultralytics/data/augment.py (cv2.copyMakeBorder BORDER_CONSTANT value=255).
    pad = (pad_w, pad_h) — суммарный паддинг; new_unpad = isz - pad, чтобы точно
    совпасть с пространством pred (percept letterbox scale), pad_left/top = round(pad/2 - 0.1).
    """
    h, w = mask.shape[:2]
    new_w = isz - pad[0]
    new_h = isz - pad[1]
    if (new_w, new_h) != (w, h):
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    mask = cv2.copyMakeBorder(
        mask, pad_top, pad[1] - pad_top, pad_left, pad[0] - pad_left,
        cv2.BORDER_CONSTANT, value=255,
    )
    return mask


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
    # Fallback: любая поддиректория images/ (мини-датасеты coco128/coco8-pose
    # хранят изображения в images/train2017)
    imgs_root = data_path / 'images'
    if imgs_root.exists():
        for hit in imgs_root.rglob(img_info['file_name']):
            return str(hit)
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


def _match_predictions_greedy(pred_cls, true_cls, iou, iouv):
    """Реплика ultralytics BaseValidator.match_predictions (greedy one-to-one).

    pred_cls: (D,), true_cls: (L,), iou: (L, D), iouv: (10,) порогов.
    Возвращает (D, len(iouv)) bool.
    """
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    if len(pred_cls) == 0 or len(true_cls) == 0:
        return correct
    correct_class = (true_cls[:, None] == pred_cls[None, :])  # (L, D)
    iou = np.asarray(iou, dtype=np.float64) * correct_class
    for i, thr in enumerate(iouv):
        matches = np.argwhere(iou >= thr)
        if matches.shape[0] == 0:
            continue
        if matches.shape[0] > 1:
            vals = iou[matches[:, 0], matches[:, 1]]
            matches = matches[np.argsort(-vals, kind='stable')]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1], i] = True
    return correct


def _ap_summary(ap_arr):
    """map50 / map50_95 из матрицы AP [nc, 10]."""
    if ap_arr.shape[0] == 0:
        return 0.0, 0.0
    return float(ap_arr[:, 0].mean()), float(ap_arr.mean())


def _min_index(arr1, arr2):
    """Копия ultralytics.data.converter.min_index (кратчайшая дистанция между двумя наборами точек)."""
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Копия ultralytics.data.converter.merge_multi_segment (слияние полигонов в один).

    Нужна, чтобы GT-маска в eval.py строилась ТО ЖЕ, что и в yolo seg-лейблах,
    сгенерированных convert_coco(use_segments=True).
    """
    s = []
    segs = [np.asarray(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segs))]
    for i in range(1, len(segs)):
        idx1, idx2 = _min_index(segs[i - 1], segs[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)
    for k in range(2):
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segs[i] = segs[i][::-1, :]
                segs[i] = np.roll(segs[i], -idx[0], axis=0)
                segs[i] = np.concatenate([segs[i], segs[i][:1]])
                if i in {0, len(idx_list) - 1}:
                    s.append(segs[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segs[i][idx[0]:idx[1] + 1])
        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segs[i][nidx:])
    return s


def compute_inmemory_box_map(images, ann_by_img, preds_by_img, scales_pads,
                             eval_img_ids, imgsz=640):
    """Реплика ultralytics in-memory DetMetrics (.val() без save_json).

    GT и preds — в letterboxed-пространстве (scale+pad из pipeline-метадаты,
    как в Format(normalize=True)+_prepare_batch). box_iou + match_predictions +
    ap_per_class — байт-в-байт совпадает с ultralytics. Накопление через
    _box_tp_per_image + _ap_finalize (инкрементально, без хранения preds).
    """
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    for img_id in eval_img_ids:
        sx, sy, pad_left, pad_top = scales_pads[img_id]
        tp, conf, pred_cls, gt_cls = _box_tp_per_image(
            ann_by_img.get(img_id, []), preds_by_img.get(img_id, []),
            sx, sy, pad_left, pad_top)
        all_tp.append(tp)
        all_conf.append(conf)
        all_pred_cls.append(pred_cls)
        all_target_cls.append(gt_cls)
    return _ap_finalize(all_tp, all_conf, all_pred_cls, all_target_cls)


def compute_inmemory_pose_map(images, ann_by_img, preds_by_img, scales_pads,
                              eval_img_ids, imgsz=640):
    """Реплика ultralytics in-memory PoseMetrics (.val() без save_json).

    Все GT-персоны (включая num_keypoints=0) — таргеты. GT boxes/kpts маппятся
    как orig_px*scale+pad (letterboxed-пространство, как Format+_prepare_batch),
    preds — в том же letterboxed-пространстве. Метрика pose — по kpt_iou (OKS).
    Совпадает с ultralytics байт-в-байт. Накопление через _pose_tp_per_image.
    """
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    for img_id in eval_img_ids:
        sx, sy, pad_left, pad_top = scales_pads[img_id]
        tp_p, conf, pred_cls, gt_cls = _pose_tp_per_image(
            ann_by_img.get(img_id, []), preds_by_img.get(img_id, []),
            sx, sy, pad_left, pad_top)
        all_tp.append(tp_p)
        all_conf.append(conf)
        all_pred_cls.append(pred_cls)
        all_target_cls.append(gt_cls)
    return _ap_finalize(all_tp, all_conf, all_pred_cls, all_target_cls,
                        prefix='inmem_pose')


def _seg_tp_per_image(anns, preds, sx, sy, pad_left, pad_top, imgsz=640):
    """Вклад одного изображения в in-memory SegmentMetrics.

    Возвращает (tp_m, conf, pred_cls, gt_cls) — маленькие массивы, которые
    можно накапливать между кадрами. Pred-маски используются только внутри
    вызова и выбрасываются (как в инкрементальном ultralytics .val()), а не
    хранятся в preds_by_img — это устраняет OOM на 5000 imgs (~3.5GB масок).

    GT маски: mask_overlap=True (hyp default) → polygons2masks_overlap((640,640),
    letterboxed_segments, ratio=4) даёт (160,160) int-слой со значениями 1..N
    (1 = инстанс с МАКСИМАЛЬНОЙ площадью, N = с минимальной; маленькие рисуются
    поверх больших через np.maximum). Валидатор декодирует per-instance маски
    через masks = (overlap == i+1) и сортирует cls по площади (desc), ровно как
    Format + _prepare_batch. Метрика: mask_iou + match_predictions + ap_per_class.
    """
    from ultralytics.data.utils import polygons2masks, polygons2masks_overlap

    n = len(anns)
    gt_cls = np.zeros(n, dtype=np.float32)
    gt_masks = np.zeros((n, imgsz // 4, imgsz // 4), dtype=np.float32)
    if n:
        segs = []
        cls_list = []
        for i, a in enumerate(anns):
            cls_list.append(a['category_id'])
            seg = a.get('segmentation', [])
            if isinstance(seg, dict) or not seg:
                segs.append(None)
                continue
            if len(seg) == 1:
                pts = np.asarray(seg[0], dtype=np.float32).reshape(-1, 2)
            else:
                pts = np.concatenate(merge_multi_segment(seg), axis=0).astype(np.float32)
            pts[:, 0] = pts[:, 0] * sx + pad_left
            pts[:, 1] = pts[:, 1] * sy + pad_top
            segs.append(pts.reshape(-1))
        gt_cls = np.array(cls_list, dtype=np.float32)
        if all(s is not None for s in segs):
            overlap_img, order = polygons2masks_overlap(
                (imgsz, imgsz), segs, downsample_ratio=4)
            gt_masks = np.stack([(overlap_img == (i + 1)) for i in range(n)], 0).astype(np.float32)
            gt_cls = gt_cls[order]
        else:
            for i, s in enumerate(segs):
                if s is not None:
                    gt_masks[i] = polygons2masks((imgsz, imgsz), [s], color=1,
                                                 downsample_ratio=4)[0]

    m = len(preds)
    conf = np.array([p['score'] for p in preds], dtype=np.float32)
    pred_cls = np.array([p['class_id'] for p in preds], dtype=np.float32)
    if n == 0 or m == 0:
        return np.zeros((m, 10), dtype=bool), conf, pred_cls, gt_cls

    pred_masks = np.array([p['masks_160'] for p in preds], dtype=np.float32)
    iouv = np.linspace(0.5, 0.95, 10)
    gt_flat = gt_masks.reshape(n, -1)
    pd_flat = pred_masks.reshape(m, -1)
    inter = gt_flat @ pd_flat.T
    iou = inter / (gt_flat.sum(axis=1)[:, None] + pd_flat.sum(axis=1)[None, :] - inter + 1e-7)
    tp_m = _match_predictions_greedy(pred_cls, gt_cls, iou, iouv)
    return tp_m, conf, pred_cls, gt_cls


def _ap_finalize(all_tp, all_conf, all_pred_cls, all_target_cls, prefix='inmem'):
    """ap_per_class по накопленным между кадрами маленьким массивам.

    Универсальный финализатор для detect/pose/segment/obb: принимает списки
    per-image (tp, conf, pred_cls, gt_cls) и возвращает {prefix_map50,
    prefix_map50_95}. Используется и интерливным runner'ом, и evaluate().
    """
    from ultralytics.utils.metrics import ap_per_class

    target_cls = np.concatenate(all_target_cls) if all_target_cls else np.empty(0, dtype=np.float32)
    conf = np.concatenate(all_conf) if all_conf else np.empty(0)
    pred_cls = np.concatenate(all_pred_cls) if all_pred_cls else np.empty(0, dtype=np.float32)
    tp = np.concatenate(all_tp, axis=0) if all_tp else np.empty((0, 10), dtype=bool)

    if tp.shape[0] == 0:
        return {f'{prefix}_map50': 0.0, f'{prefix}_map50_95': 0.0}
    m50, m5095 = _ap_summary(ap_per_class(tp, conf, pred_cls, target_cls)[5])
    return {
        f'{prefix}_map50': round(m50, 4),
        f'{prefix}_map50_95': round(m5095, 4),
    }


def _seg_finalize(all_tp_m, all_conf, all_pred_cls, all_target_cls):
    """Обёртка над _ap_finalize с префиксом inmem_seg (для совместимости)."""
    return _ap_finalize(all_tp_m, all_conf, all_pred_cls, all_target_cls, prefix='inmem_seg')


def _box_tp_per_image(anns, preds, sx, sy, pad_left, pad_top):
    """Вклад одного изображения в in-memory DetMetrics (box).

    Возвращает (tp, conf, pred_cls, gt_cls) — маленькие массивы для
    инкрементального накопления между кадрами. GT boxes и preds — в
    letterboxed-пространстве (orig_px*scale+pad), как в evaluate().
    """
    import torch
    from ultralytics.utils.metrics import box_iou

    iouv = np.linspace(0.5, 0.95, 10)
    n = len(anns)
    gt_cls = np.zeros(n, dtype=np.float32)
    gt_boxes = np.zeros((n, 4), dtype=np.float32)
    for i, a in enumerate(anns):
        gt_cls[i] = a['category_id']
        bx, by, bw, bh = a['bbox']
        gt_boxes[i] = [
            bx * sx + pad_left,
            by * sy + pad_top,
            (bx + bw) * sx + pad_left,
            (by + bh) * sy + pad_top,
        ]

    m = len(preds)
    conf = np.array([p['score'] for p in preds], dtype=np.float32)
    pred_cls = np.array([p['class_id'] for p in preds], dtype=np.float32)
    if m == 0 or n == 0:
        return np.zeros((m, 10), dtype=bool), conf, pred_cls, gt_cls

    pred_boxes = np.array([p['bbox_xyxy_letter'] for p in preds], dtype=np.float32)
    iou_b = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(pred_boxes)).numpy()
    tp = _match_predictions_greedy(pred_cls, gt_cls, iou_b, iouv)
    return tp, conf, pred_cls, gt_cls


def _pose_tp_per_image(anns, preds, sx, sy, pad_left, pad_top):
    """Вклад одного изображения в in-memory PoseMetrics (OKS по kpts).

    Возвращает (tp_p, conf, pred_cls, gt_cls). Все GT-персоны (включая
    num_keypoints=0) — таргеты; GT boxes/kpts в letterboxed-пространстве.
    """
    import torch
    from ultralytics.utils.metrics import OKS_SIGMA, kpt_iou

    iouv = np.linspace(0.5, 0.95, 10)
    n = len(anns)
    gt_cls = np.zeros(n, dtype=np.float32)
    gt_boxes = np.zeros((n, 4), dtype=np.float32)
    gt_kpts = np.zeros((n, 17, 3), dtype=np.float32)
    for i, a in enumerate(anns):
        gt_cls[i] = a['category_id']
        bx, by, bw, bh = a['bbox']
        gt_boxes[i] = [
            bx * sx + pad_left,
            by * sy + pad_top,
            (bx + bw) * sx + pad_left,
            (by + bh) * sy + pad_top,
        ]
        kp = a['keypoints']
        for k in range(17):
            gt_kpts[i, k, 0] = kp[k * 3] * sx + pad_left
            gt_kpts[i, k, 1] = kp[k * 3 + 1] * sy + pad_top
            gt_kpts[i, k, 2] = kp[k * 3 + 2]

    m = len(preds)
    conf = np.array([p['score'] for p in preds], dtype=np.float32)
    pred_cls = np.array([p['class_id'] for p in preds], dtype=np.float32)
    if m == 0 or n == 0:
        return np.zeros((m, 10), dtype=bool), conf, pred_cls, gt_cls

    pred_kpts = np.array([p['kpts_letter'] for p in preds], dtype=np.float32)
    area = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])) * 0.53
    oks = kpt_iou(
        torch.from_numpy(gt_kpts), torch.from_numpy(pred_kpts),
        area=torch.from_numpy(area), sigma=OKS_SIGMA,
    ).numpy()
    tp_p = _match_predictions_greedy(pred_cls, gt_cls, oks, iouv)
    return tp_p, conf, pred_cls, gt_cls


def _obb_tp_per_image(gt_boxes_letter, gt_cls, pred_boxes_letter, pred_cls, pred_scores):
    """Вклад одного изображения в in-memory OBBMetrics.

    Возвращает (tp, conf, pred_cls, gt_cls). Все массивы в letterboxed-
    пространстве ([cx, cy, w, h, theta]); batch_probiou + greedy match.
    """
    import torch
    from ultralytics.utils.metrics import batch_probiou

    iouv = np.linspace(0.5, 0.95, 10)
    gt_c = np.asarray(gt_cls, dtype=np.float32)
    pr_c = np.asarray(pred_cls, dtype=np.float32)
    pr_s = np.asarray(pred_scores, dtype=np.float32)
    n, m = len(gt_boxes_letter), len(pred_boxes_letter)
    if m == 0 or n == 0:
        return np.zeros((m, 10), dtype=bool), pr_s, pr_c, gt_c
    iou = batch_probiou(torch.from_numpy(np.asarray(gt_boxes_letter, dtype=np.float32)),
                        torch.from_numpy(np.asarray(pred_boxes_letter, dtype=np.float32))).numpy()
    tp = _match_predictions_greedy(pr_c, gt_c, iou, iouv)
    return tp, pr_s, pr_c, gt_c


def compute_inmemory_obb_map(gt_boxes_per_img, pred_boxes_per_img,
                             gt_classes_per_img, pred_classes_per_img,
                             pred_scores_per_img, num_classes=15):
    """Реплика ultralytics in-memory OBBMetrics (.val() без save_json).

    GT и preds — [cx, cy, w, h, theta] в letterboxed-пространстве.
    batch_probiou + match_predictions + ap_per_class — байт-в-байт с ultralytics.
    Накопление через _obb_tp_per_image (инкрементально).
    """
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    for gt_b, gt_c, pr_b, pr_c, pr_s in zip(
            gt_boxes_per_img, gt_classes_per_img,
            pred_boxes_per_img, pred_classes_per_img, pred_scores_per_img):
        tp, conf, pred_cls, gt_cls = _obb_tp_per_image(gt_b, gt_c, pr_b, pr_c, pr_s)
        all_tp.append(tp)
        all_conf.append(conf)
        all_pred_cls.append(pred_cls)
        all_target_cls.append(gt_cls)
    return _ap_finalize(all_tp, all_conf, all_pred_cls, all_target_cls)


def compute_inmemory_sem_map(pred_maps: list[np.ndarray], gt_maps: list[np.ndarray],
                             num_classes: int = 19) -> dict:
    """Реплика ultralytics SemanticMetrics (mIoU + pixel accuracy).

    pred/gt — карты классов [H, W] в letterboxed-пространстве (pad GT = 255).
    Confusion matrix [gt, pred]; пиксели с gt==255 или pred вне [0, nc) — вне подсчёта.
    mIoU усредняется ТОЛЬКО по классам, присутствующим в GT (row_sum > 0),
    pixel accuracy = diag.sum() / matrix.sum() — байт-в-байт с
    ultralytics/utils/metrics.py SemanticMetrics.update_stats/process.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for pred, gt in zip(pred_maps, gt_maps):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        valid = (gt != 255) & (pred >= 0) & (pred < num_classes) & (gt >= 0) & (gt < num_classes)
        np.add.at(matrix, (gt[valid].astype(np.int64), pred[valid].astype(np.int64)), 1)
    return sem_metrics_from_matrix(matrix, num_classes)


def sem_metrics_from_matrix(matrix: np.ndarray, num_classes: int = 19) -> dict:
    """Метрики семантики из confusion matrix [gt, pred] (см. compute_inmemory_sem_map)."""
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    inter = np.diag(matrix)
    union = row_sum + col_sum - inter
    iou = np.where(union > 0, inter / union, 0.0)
    present = row_sum > 0
    miou = float(iou[present].mean()) if present.any() else 0.0
    pix_acc = float(inter.sum() / (matrix.sum() + 1e-10))
    return {
        'inmem_miou': round(miou, 4),
        'inmem_pixacc': round(pix_acc, 4),
        'inmem_per_class_iou': {i: round(float(v), 4) for i, v in enumerate(iou)},
    }


def compute_imagenet_accuracy(predictions: list[int], ground_truth: list[int]) -> dict:
    """Вычислить top1 accuracy для ImageNet."""
    if not predictions:
        return {'top1': 0.0}
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return {'top1': correct / len(predictions)}


def _obb_ap_at_iou(gt_boxes_xywha, pred_boxes_xywha, gt_classes, pred_classes,
                    pred_scores, iou_threshold, num_classes):
    """Compute per-class AP at a single IoU threshold using COCO all-points AP."""
    from ultralytics.utils.metrics import batch_probiou
    import torch

    ap_per_class = []
    ar_per_class = []

    for cls_id in range(num_classes):
        all_tp = []
        all_fp = []
        all_scores = []
        n_gt_cls = 0

        for img_idx in range(len(gt_boxes_xywha)):
            gt_cls = gt_classes[img_idx]
            pred_cls = pred_classes[img_idx]
            gt_mask = gt_cls == cls_id
            pred_mask = pred_cls == cls_id

            gt_boxes = gt_boxes_xywha[img_idx][gt_mask] if gt_mask.any() else np.empty((0, 5))
            pred_boxes = pred_boxes_xywha[img_idx][pred_mask] if pred_mask.any() else np.empty((0, 5))
            scores = pred_scores[img_idx][pred_mask] if pred_mask.any() else np.empty(0)

            n_gt = len(gt_boxes)
            n_gt_cls += n_gt
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

        if not all_scores or n_gt_cls == 0:
            ap_per_class.append(0.0)
            ar_per_class.append(0.0)
            continue

        scores_arr = np.array(all_scores)
        tp_arr = np.array(all_tp)
        fp_arr = np.array(all_fp)

        order = scores_arr.argsort()[::-1]
        tp_cum = np.cumsum(tp_arr[order])
        fp_cum = np.cumsum(fp_arr[order])

        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / n_gt_cls

        # COCO all-points AP
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[x + 1] - mrec[x]) * mpre[x + 1])
        ap_per_class.append(ap)

        # AR: recall at max 100 predictions per image (approximate)
        ar_per_class.append(min(1.0, recall[-1]) if len(recall) > 0 else 0.0)

    return ap_per_class, ar_per_class


def compute_obb_map(gt_boxes_xywha: list, pred_boxes_xywha: list,
                    gt_classes: list, pred_classes: list,
                    pred_scores: list, num_classes: int = 15) -> dict:
    """Compute OBB mAP matching ultralytics COCO-style evaluation.

    - map50: AP at IoU=0.5
    - map50_95: mean AP at IoU=0.5:0.05:0.95
    - ar100: mean max recall across classes
    """
    # Find which classes actually appear in GT (match pycocotools behavior)
    classes_with_gt = set()
    for i in range(len(gt_classes)):
        classes_with_gt.update(gt_classes[i].tolist())
    if not classes_with_gt:
        return {'map50': 0.0, 'map50_95': 0.0, 'ar100': 0.0}

    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    all_ap = []
    all_ar = []
    for iou_thresh in iou_thresholds:
        ap_list, ar_list = _obb_ap_at_iou(
            gt_boxes_xywha, pred_boxes_xywha,
            gt_classes, pred_classes, pred_scores,
            iou_thresh, num_classes,
        )
        all_ap.append([ap_list[c] for c in sorted(classes_with_gt)])
        all_ar.append([ar_list[c] for c in sorted(classes_with_gt)])

    ap_arr = np.array(all_ap)
    ar_arr = np.array(all_ar)

    map50 = float(ap_arr[0].mean()) if ap_arr.shape[1] > 0 else 0.0
    map50_95 = float(ap_arr.mean()) if ap_arr.size > 0 else 0.0
    ar100 = float(ar_arr[0].mean()) if ar_arr.shape[1] > 0 else 0.0

    return {'map50': round(map50, 4), 'map50_95': round(map50_95, 4), 'ar100': round(ar100, 4)}


# ---------------------------------------------------------------------------
# Parallel NMS (eval-only): infer идёт последовательно на NPU, но NMS кадров —
# независимые задачи и выполняется в пуле потоков по числу производительных ядер.
# torchvision.ops.nms освобождает GIL → потоки дают реальный параллелизм (~3.4x).
# ---------------------------------------------------------------------------

def _big_core_count() -> int:
    """Число ядер CPU для пула параллельного NMS.

    В eval (conf=0.001) постпроцесс — узкое место, NPU почти простаивает
    (>95% idle). Замер probe_parallel (2026-08-12): N=4 даёт 3.0–3.5x,
    N=8 добавляет ~9% на detect и ~0% на segment, но равномернее грузит
    A55. По решению пользователя используем ВСЕ ядра — CPU должен быть
    загружен максимально (троттлинга нет: tmax≤66°C при N=8).
    """
    try:
        cpus = os.sched_getaffinity(0)
        if cpus:
            return max(1, len(cpus))
    except Exception:
        pass
    return os.cpu_count() or 4


def _detect_pre_nms(pp, outputs):
    """Шаги 1-3 постпроцесса detect до NMS: decode → conf-filter → multi_label expand → top-MAX_NMS.

    Возвращает (M, 6) [x1,y1,x2,y2,conf,cls] или None. Быстро — без NMS.
    """
    single_output = (len(outputs) == 1 and outputs[0].ndim == 3
                     and outputs[0].shape[1] == 4 + pp.NUM_CLASSES)
    if single_output:
        pred = outputs[0][0].astype(np.float32)
        pred[:4] = pp.xywh2xyxy(pred[:4].T).T
        pred = pred.T
    else:
        pred = pp._decode_per_stride(outputs)
    xc = pred[:, 4:4 + pp.NUM_CLASSES].max(axis=1) > pp.CONF_THRESH
    pred = pred[xc]
    if len(pred) == 0:
        return None
    box = pred[:, :4]
    cls = pred[:, 4:4 + pp.NUM_CLASSES]
    i, j = np.where(cls > pp.CONF_THRESH)
    if len(i) == 0:
        return None
    x = np.column_stack([box[i], cls[i, j, None], j[:, None].astype(np.float32)])
    if x.shape[0] > pp.MAX_NMS:
        filt = x[:, 4].argsort()[::-1][:pp.MAX_NMS]
        x = x[filt]
    return x


def _detect_nms_finish(pp, x, scale, pad, orig_shape, fast_nms):
    """Шаги 4-7 постпроцесса detect: class-aware NMS → reverse letterbox → формат."""
    c = x[:, 5:6] * pp.MAX_WH
    boxes_nms = x[:, :4] + c
    keep = fast_nms(boxes_nms, x[:, 4], pp.IOU_THRESH)
    keep = keep[:pp.MAX_DET]
    x = x[keep]
    orig_h, orig_w = orig_shape
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    x[:, 0] = (x[:, 0] - pad_left) / gain
    x[:, 1] = (x[:, 1] - pad_top) / gain
    x[:, 2] = (x[:, 2] - pad_left) / gain
    x[:, 3] = (x[:, 3] - pad_top) / gain
    if not pp.EVAL_PRECISE:
        x[:, [0, 2]] = np.clip(x[:, [0, 2]], 0, orig_w)
        x[:, [1, 3]] = np.clip(x[:, [1, 3]], 0, orig_h)
    results = []
    for row in x:
        x1, y1, x2, y2, conf, cls_id = row
        c = int(cls_id)
        results.append({
            "bbox": ([float(x1), float(y1), float(x2), float(y2)] if pp.EVAL_PRECISE
                     else [int(x1), int(y1), int(x2), int(y2)]),
            "score": float(conf),
            "class_id": c,
            "class_name": pp.COCO_CLASSES[c] if c < len(pp.COCO_CLASSES) else str(c),
        })
    return results


def _seg_pre_nms(pp, outputs):
    """Шаги 1-3 сегментного постпроцесса до NMS: decode → conf-filter → multi-label expand → cap.

    Возвращает (x, proto) или None. Точная копия шагов
    segment_postprocess.py:151-194 (без NMS/masks — они в _seg_nms_finish).
    """
    orig_h, orig_w = (640, 640)  # не используется до NMS
    single_output = (len(outputs) == 2)
    if single_output:
        pred = outputs[0][0].astype(np.float32)  # (4+nc+NM, total)
        nc = pred.shape[0] - 4 - pp.NM
        proto = outputs[1]
        if proto.ndim == 4:
            proto = proto[0]
        pred[:4] = pp.xywh2xyxy(pred[:4].T).T
        pred = pred.T                            # (total, 4+nc+NM)
        box = pred[:, :4]
        cls = pred[:, 4:4 + nc]
        coeffs = pred[:, 4 + nc:]
    else:
        proto = outputs[-1]
        if proto.ndim == 4:
            proto = proto[0]
        box, cls, coeffs = pp._decode_per_stride(outputs)
    xc = cls.max(axis=1) > pp.CONF_THRESH
    box = box[xc]
    cls = cls[xc]
    coeffs = coeffs[xc]
    if len(box) == 0:
        return None
    i, j = np.where(cls > pp.CONF_THRESH)
    if len(i) == 0:
        return None
    x = np.column_stack([box[i], cls[i, j, None], j[:, None].astype(np.float32), coeffs[i]])
    if x.shape[0] > pp.MAX_NMS:
        filt = x[:, 4].argsort()[::-1][:pp.MAX_NMS]
        x = x[filt]
    return x, proto


def _seg_nms_finish(pp, x, proto, scale, pad, orig_shape, fast_nms):
    """Шаги 4-8 сегментного постпроцесса: class-aware NMS → masks_160 → формат."""
    c = x[:, 5:6] * pp.MAX_WH
    boxes_nms = x[:, :4] + c
    keep = fast_nms(boxes_nms, x[:, 4], pp.IOU_THRESH)
    keep = keep[:pp.MAX_DET]
    x = x[keep]
    if len(x) == 0:
        return []
    bboxes_letter = x[:, :4].copy()
    masks_160 = pp.process_mask_160(proto, x[:, 6:], bboxes_letter)
    orig_h, orig_w = orig_shape
    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    gain = scale
    x[:, 0] = (x[:, 0] - pad_left) / gain
    x[:, 1] = (x[:, 1] - pad_top) / gain
    x[:, 2] = (x[:, 2] - pad_left) / gain
    x[:, 3] = (x[:, 3] - pad_top) / gain
    if not pp.EVAL_PRECISE:
        x[:, [0, 2]] = np.clip(x[:, [0, 2]], 0, orig_w)
        x[:, [1, 3]] = np.clip(x[:, [1, 3]], 0, orig_h)
    results = []
    for i in range(len(x)):
        cid = int(x[i, 5])
        x1, y1, x2, y2 = x[i, :4]
        if not pp.EVAL_PRECISE:
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
            mask_crop = np.zeros((orig_h, orig_w), dtype=np.uint8)
            bx1, by1 = max(0, x1), max(0, y1)
            bx2, by2 = min(orig_w, x2), min(orig_h, y2)
            if bx2 > bx1 and by2 > by1:
                m = cv2.resize(masks_160[i].astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
                mask_crop[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2] * 255
            seg = mask_crop
        else:
            seg = masks_160[i]
        obj = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(x[i, 4]),
            "class_id": cid,
            "class_name": pp.COCO_CLASSES[cid] if cid < len(pp.COCO_CLASSES) else str(cid),
            "masks_160": masks_160[i],
        }
        if not pp.EVAL_PRECISE:
            obj["segmentation"] = seg
        results.append(obj)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(task: str, data_dir: str, model_dir: str = '/app/model',
             backend: str = '', model_name: str = '', limit: int = 0,
             rknn_suffix: str = '') -> dict:
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
    pipeline = Pipeline(model_dir=str(model_path), model_name=model_name, backend=backend,
                        rknn_suffix=rknn_suffix)

    # Мониторинг троттлинга: частоты NPU/CPU, температуры на время eval.
    # Диагностика — в результат добавляются npu_freq_avg, cpu_freq_ratio,
    # npu_temp_*, thermal_max_c, throttled (критерии — как в fps_bench/telemetry).
    from telemetry import TelemetrySampler
    sampler = TelemetrySampler(collect_cpu=True, interval=2.0)
    sampler.start()

    # Eval mode: минимальный порог чтобы pycocotools сам выбрал оптимальный
    import sys as _sys2
    post_mod = _sys2.modules.get('postprocess_mod')
    if post_mod is not None:
        post_mod.CONF_THRESH = 0.001
        post_mod.IOU_THRESH = 0.7
        post_mod.MIN_WH = 1
        post_mod.MAX_DET = 300
        post_mod.DEDUP_ENABLED = False
        if hasattr(post_mod, 'EVAL_PRECISE'):
            post_mod.EVAL_PRECISE = True
        sys.stderr.write(f'[EVAL] eval_mode: CONF_THRESH=0.001, IOU_THRESH=0.7 MIN_WH=1 MAX_DET=300 DEDUP=OFF\n')

        # Fast NMS: RKNN-квантование даёт шум в cls_scores → при conf=0.001
        # сотни тысяч кандидатов → numpy-цикл NMS = секунды на кадр.
        # Используем torchvision.ops.nms (C++, тот же greedy, что у col1/ultralytics),
        # если torch доступен; иначе fallback на numpy. Продакшен-рантайм не трогаем.
        if hasattr(post_mod, 'nms'):
            _orig_nms = post_mod.nms
            try:
                import torch
                import torchvision
                _tv_nms = torchvision.ops.nms
            except ImportError:
                _tv_nms = None
            if _tv_nms is not None:
                def _fast_nms(boxes, scores, iou_threshold):
                    try:
                        tb = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
                        ts = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
                        keep = _tv_nms(tb, ts, iou_threshold)
                        return keep.numpy().astype(np.int64)
                    except Exception:
                        return _orig_nms(boxes, scores, iou_threshold)
                post_mod.nms = _fast_nms
                sys.stderr.write('[EVAL] fast NMS: torchvision.ops.nms (fallback: numpy)\n')

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
        sem_pairs = load_cityscapes(
            str(Path(data_dir) / 'images' / 'val'),
            str(Path(data_dir) / 'masks' / 'val'),
        )
        sys.stderr.write(f'[EVAL] Cityscapes: {len(sem_pairs)} изображений\n')

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
        preds_by_img = {}
        scales_pads = {}
        seg_tp, seg_conf, seg_pred_cls, seg_gt_cls = [], [], [], []
        if task == 'detect':
            # Конвейер: letterbox+infer последовательно на NPU, NMS кадров —
            # параллельно в пуле по числу производительных ядер. NMS на 30000
            # кандидатов при conf=0.001 — единственное узкое место (1.35с/кадр),
            # распараллеливание даёт ~3.4x без ущерба метрикам (тот же greedy NMS).
            from percept.io import letterbox
            n_workers = _big_core_count()
            sys.stderr.write(f'[EVAL] parallel NMS: {n_workers} workers\n')
            if post_mod is not None and hasattr(post_mod, 'nms'):
                _orig_nms = post_mod.nms
                try:
                    import torch
                    import torchvision
                    torch.set_num_threads(1)
                    _tv_nms = torchvision.ops.nms
                except ImportError:
                    _tv_nms = None
                if _tv_nms is not None:
                    def _fast_nms(boxes, scores, iou_threshold):
                        tb = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
                        ts = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
                        return _tv_nms(tb, ts, iou_threshold).numpy().astype(np.int64)
                else:
                    _fast_nms = _orig_nms
            else:
                _fast_nms = None
            futures = {}
            meta_by_img = {}
            _pool = _cf.ThreadPoolExecutor(max_workers=n_workers) if _fast_nms is not None else None
            for img_id, img_info in images.items():
                if limit and total >= limit:
                    break
                img_path = find_img_path(img_info, data_dir)
                frame = cv2.imread(img_path)
                if frame is None:
                    errors += 1
                    continue
                h, w = frame.shape[:2]
                canvas, scale, pad = letterbox(frame, pipeline.input_size)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
                tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
                if pipeline._backend.backend == 'cpu':
                    tensor = tensor / 255.0
                outputs = pipeline._backend.infer(tensor)
                meta_by_img[img_id] = (scale, pad, (h, w))
                if _pool is not None:
                    x = _detect_pre_nms(post_mod, outputs)
                    if x is not None:
                        futures[img_id] = _pool.submit(
                            _detect_nms_finish, post_mod, x, scale, pad, (h, w), _fast_nms)
                    else:
                        futures[img_id] = None
                else:
                    objects = post_mod.postprocess(outputs, scale, pad, (h, w)) if post_mod else []
                    futures[img_id] = objects
                eval_img_ids.append(img_id)
                total += 1
                if total % 500 == 0:
                    sys.stderr.write(f'[EVAL] {total}/{len(images)}\n')
            if _pool is not None:
                _pool.shutdown()
                for img_id, fut in futures.items():
                    futures[img_id] = fut.result() if fut is not None else []
            for img_id in eval_img_ids:
                objects = futures[img_id]
                scale, pad, (h, w) = meta_by_img[img_id]
                coco_results.extend(detections_to_coco(objects, img_id, cat_ids))
                pad_left = round(pad[0] / 2 - 0.1)
                pad_top = round(pad[1] / 2 - 0.1)
                isz = pipeline.input_size
                sx = (isz - pad[0]) / w
                sy = (isz - pad[1]) / h
                scales_pads[img_id] = (sx, sy, pad_left, pad_top)
                preds_by_img[img_id] = []
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox']
                    preds_by_img[img_id].append({
                        'bbox_xyxy_letter': [
                            x1 * scale + pad_left, y1 * scale + pad_top,
                            x2 * scale + pad_left, y2 * scale + pad_top,
                        ],
                        'kpts_letter': None,
                        'score': obj['score'],
                        'class_id': obj['class_id'],
                    })
        else:
            # segment: NMS+masks при conf=0.001 — то же узкое место, что у detect
            # (NMS на 30000 кандидатов + process_mask_160) → распараллеливаем
            # финиш (NMS → masks → формат) в ThreadPool по числу больших ядер.
            # pose: обычный путь (кандидатов мало — NMS не узкое место).
            if task == 'segment' and post_mod is not None and hasattr(post_mod, 'nms'):
                _orig_nms = post_mod.nms
                try:
                    import torch
                    import torchvision
                    torch.set_num_threads(1)
                    _tv_nms = torchvision.ops.nms
                except ImportError:
                    _tv_nms = None
                if _tv_nms is not None:
                    def _fast_nms(boxes, scores, iou_threshold):
                        try:
                            tb = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
                            ts = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
                            keep = _tv_nms(tb, ts, iou_threshold)
                            return keep.numpy().astype(np.int64)
                        except Exception:
                            return _orig_nms(boxes, scores, iou_threshold)
                    post_mod.nms = _fast_nms
                    sys.stderr.write(f'[EVAL] segment parallel NMS: {_big_core_count()} workers\n')
                    n_workers = _big_core_count()
                    _pool = _cf.ThreadPoolExecutor(max_workers=n_workers)
                else:
                    _pool = None
            else:
                _pool = None

            futures = {}
            meta_by_img = {}
            for img_id, img_info in images.items():
                if limit and total >= limit:
                    break
                img_path = find_img_path(img_info, data_dir)
                frame = cv2.imread(img_path)
                if frame is None:
                    errors += 1
                    continue
                if _pool is not None:
                    h, w = frame.shape[:2]
                    from percept.io import letterbox
                    canvas, scale, pad = letterbox(frame, pipeline.input_size)
                    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
                    tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
                    if pipeline._backend.backend == 'cpu':
                        tensor = tensor / 255.0
                    outputs = pipeline._backend.infer(tensor)
                    meta_by_img[img_id] = (scale, pad, (h, w))
                    pre = _seg_pre_nms(post_mod, outputs)
                    if pre is not None:
                        x, proto = pre
                        futures[img_id] = _pool.submit(
                            _seg_nms_finish, post_mod, x, proto, scale, pad, (h, w), _fast_nms)
                    else:
                        futures[img_id] = None
                    eval_img_ids.append(img_id)
                    total += 1
                    if total % 500 == 0:
                        sys.stderr.write(f'[EVAL] {total}/{len(images)}\n')
                    continue
                objects, meta = pipeline.detect(frame)
                coco_results.extend(detections_to_coco(objects, img_id, cat_ids))
                eval_img_ids.append(img_id)
                pad = meta['pad']
                pad_left = round(pad[0] / 2 - 0.1)
                pad_top = round(pad[1] / 2 - 0.1)
                scale = meta['scale']
                isz = meta['input_size']
                sx = (isz - pad[0]) / frame.shape[1]
                sy = (isz - pad[1]) / frame.shape[0]
                scales_pads[img_id] = (sx, sy, pad_left, pad_top)
                preds_by_img[img_id] = []
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox']
                    kpts_letter = None
                    if task == 'pose':
                        kpts_letter = []
                        for k in (obj.get('keypoints') or []):
                            if len(k) >= 3:
                                kpts_letter.append([k[0] * scale + pad_left, k[1] * scale + pad_top, k[2]])
                            else:
                                kpts_letter.append([k[0] * scale + pad_left, k[1] * scale + pad_top, 0.0])
                    preds_by_img[img_id].append({
                        'bbox_xyxy_letter': [
                            x1 * scale + pad_left, y1 * scale + pad_top,
                            x2 * scale + pad_left, y2 * scale + pad_top,
                        ],
                        'kpts_letter': kpts_letter,
                        'score': obj['score'],
                        'class_id': obj['class_id'],
                    })
                if task == 'segment':
                    anns = [a for a in ann_by_img.get(img_id, []) if not a.get('iscrowd', 0)]
                    tp_m, conf, pred_cls, gt_cls = _seg_tp_per_image(
                        anns, objects, sx, sy, pad_left, pad_top)
                    seg_tp.append(tp_m)
                    seg_conf.append(conf)
                    seg_pred_cls.append(pred_cls)
                    seg_gt_cls.append(gt_cls)
                total += 1
                if total % 500 == 0:
                    sys.stderr.write(f'[EVAL] {total}/{len(images)}\n')

            if _pool is not None:
                _pool.shutdown()
                for img_id, fut in futures.items():
                    futures[img_id] = fut.result() if fut is not None else []
                for img_id in eval_img_ids:
                    objects = futures[img_id]
                    scale, pad, (h, w) = meta_by_img[img_id]
                    coco_results.extend(detections_to_coco(objects, img_id, cat_ids))
                    pad_left = round(pad[0] / 2 - 0.1)
                    pad_top = round(pad[1] / 2 - 0.1)
                    isz = pipeline.input_size
                    sx = (isz - pad[0]) / w
                    sy = (isz - pad[1]) / h
                    scales_pads[img_id] = (sx, sy, pad_left, pad_top)
                    preds_by_img[img_id] = []
                    for obj in objects:
                        x1, y1, x2, y2 = obj['bbox']
                        preds_by_img[img_id].append({
                            'bbox_xyxy_letter': [
                                x1 * scale + pad_left, y1 * scale + pad_top,
                                x2 * scale + pad_left, y2 * scale + pad_top,
                            ],
                            'kpts_letter': None,
                            'score': obj['score'],
                            'class_id': obj['class_id'],
                        })
                    anns = [a for a in ann_by_img.get(img_id, []) if not a.get('iscrowd', 0)]
                    tp_m, conf, pred_cls, gt_cls = _seg_tp_per_image(
                        anns, objects, sx, sy, pad_left, pad_top)
                    seg_tp.append(tp_m)
                    seg_conf.append(conf)
                    seg_pred_cls.append(pred_cls)
                    seg_gt_cls.append(gt_cls)

        sys.stderr.write(f'[EVAL] mAP ({len(coco_results)} pred)...\n')
        if os.environ.get('EVAL_DUMP_PREDS'):
            import pickle as _pkl
            _pkl.dump({
                'preds_by_img': preds_by_img, 'scales_pads': scales_pads,
                'eval_img_ids': eval_img_ids, 'images': images, 'ann_by_img': ann_by_img,
            }, open('/app/percept/benchmark/dump_preds.pkl', 'wb'))
            sys.stderr.write('[EVAL] dumped preds to benchmark/dump_preds.pkl\n')
        metrics = compute_coco_map(coco_results, ann_file, task, eval_img_ids=eval_img_ids)
        if task in ('detect', 'pose'):
            metrics.update(
                compute_inmemory_box_map(images, ann_by_img, preds_by_img, scales_pads, eval_img_ids)
            )
        if task == 'pose':
            metrics.update(
                compute_inmemory_pose_map(images, ann_by_img, preds_by_img, scales_pads, eval_img_ids)
            )
        if task == 'segment':
            metrics.update(_seg_finalize(seg_tp, seg_conf, seg_pred_cls, seg_gt_cls))
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
        gt_boxes_letter_per_img = []
        pred_boxes_letter_per_img = []

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

            objects, meta = pipeline.detect(img)
            scale = meta.get('scale', 1.0)
            pad = meta.get('pad', (0, 0))
            isz = meta.get('input_size', 640)
            pad_left = round(pad[0] / 2 - 0.1)
            pad_top = round(pad[1] / 2 - 0.1)
            sx = (isz - pad[0]) / w
            sy = (isz - pad[1]) / h

            if objects:
                bboxes = np.array([obj['bbox'] for obj in objects])
                scores = np.array([obj['score'] for obj in objects])
                cl = np.array([obj['class_id'] for obj in objects])
                angles = np.array([obj.get('angle', 0.0) for obj in objects])
                boxes_xywha = np.column_stack([bboxes, angles]).astype(np.float32)
                boxes_letter = boxes_xywha.copy()
                boxes_letter[:, 0] = boxes_letter[:, 0] * scale + pad_left
                boxes_letter[:, 1] = boxes_letter[:, 1] * scale + pad_top
                boxes_letter[:, 2] *= scale
                boxes_letter[:, 3] *= scale
            else:
                boxes_xywha = np.empty((0, 5), dtype=np.float32)
                boxes_letter = np.empty((0, 5), dtype=np.float32)
                scores = np.array([], dtype=np.float32)
                cl = np.array([], dtype=np.int64)
            pred_boxes_per_img.append(boxes_xywha)
            pred_classes_per_img.append(cl)
            pred_scores_per_img.append(scores)

            if polys:
                gt_letter = np.array([
                    xyxyxyxy2xywhr_np([(p['polygon'][i] * w) * sx + pad_left
                                       if i % 2 == 0 else (p['polygon'][i] * h) * sy + pad_top
                                       for i in range(8)])
                    for p in polys
                ], dtype=np.float32)
            else:
                gt_letter = np.empty((0, 5), dtype=np.float32)
            gt_boxes_letter_per_img.append(gt_letter)
            pred_boxes_letter_per_img.append(boxes_letter)
            total += 1

        sys.stderr.write(f'[EVAL] OBB: {total} images, computing rotated mAP...\n')
        metrics = compute_obb_map(
            gt_boxes_per_img, pred_boxes_per_img,
            gt_classes_per_img, pred_classes_per_img, pred_scores_per_img,
            num_classes=15,
        )
        metrics.update(
            compute_inmemory_obb_map(
                gt_boxes_letter_per_img, pred_boxes_letter_per_img,
                gt_classes_per_img, pred_classes_per_img, pred_scores_per_img,
                num_classes=15,
            )
        )
        metrics['total'] = total
        metrics['errors'] = errors
        metrics['predictions'] = sum(len(p) for p in pred_boxes_per_img)

    elif task == 'semantic':
        from percept.io import letterbox
        isz = pipeline.input_size
        num_cls = 19
        matrix = np.zeros((num_cls, num_cls), dtype=np.float64)
        pred_pixels = 0
        for img_path, mask_path in sem_pairs:
            if limit and total >= limit:
                break
            frame = cv2.imread(img_path)
            if frame is None:
                errors += 1
                continue
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                errors += 1
                continue
            if int(gt.max()) > 18:  # raw Cityscapes labelIds (7..33) → trainId
                gt = apply_cityscapes_label_mapping(gt)
            canvas, scale, pad = letterbox(frame, isz)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
            tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
            if pipeline.backend == 'cpu':
                tensor = tensor / 255.0
            outputs = pipeline._backend.infer(tensor)
            pred = outputs[0][0]
            if pred.ndim == 3:
                pred = pred.argmax(axis=0)
            pred = pred.astype(np.int64)
            gt_map = letterbox_semantic_gt(gt, scale, pad, isz).astype(np.int64)
            valid = (gt_map != 255) & (pred >= 0) & (pred < num_cls) & (gt_map >= 0) & (gt_map < num_cls)
            np.add.at(matrix, (gt_map[valid], pred[valid]), 1)
            pred_pixels += int(pred.size)
            total += 1
            if total % 100 == 0:
                sys.stderr.write(f'[EVAL] {total}/{len(sem_pairs)}\n')
        sys.stderr.write(f'[EVAL] semantic: {total} images, computing mIoU...\n')
        metrics = sem_metrics_from_matrix(matrix, num_cls)
        metrics['total'] = total
        metrics['errors'] = errors
        metrics['predictions'] = pred_pixels

    elapsed = time.time() - t_start
    metrics['elapsed_s'] = round(elapsed, 1)
    metrics['model'] = model_name
    metrics['task'] = task

    sampler.stop()
    metrics.update(sampler.summarize())

    sys.stderr.write(f'[EVAL] Готово за {elapsed:.0f}с: {json.dumps(metrics)}\n')
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Использование: eval.py <task> <data_dir> [--model-dir DIR] [--backend npu|cpu] [--model-name NAME] [--limit N] [--rknn-suffix SUFFIX]'}))
        sys.exit(1)

    task = sys.argv[1]
    data_dir = sys.argv[2]

    model_dir = '/app/model'
    backend = ''
    model_name = ''
    limit = 0
    rknn_suffix = ''
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
        elif args[i] == '--rknn-suffix' and i + 1 < len(args):
            rknn_suffix = args[i + 1]
            i += 2
        else:
            i += 1

    result = evaluate(task, data_dir, model_dir=model_dir, backend=backend,
                      model_name=model_name, limit=limit, rknn_suffix=rknn_suffix)
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
