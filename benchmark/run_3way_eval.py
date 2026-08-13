#!/usr/bin/env python3
"""Универсальный интерливный 3-way runner: ref | std | rko за один проход.

Термины (см. также README → «Сравнение постпроцессов (ref/std/rko)»):
  ref  = эталонный NMS из ultralytics (non_max_suppression) — то, что даёт .val()
  std  = наш штатный постпроцесс (продакшен-Pipeline)
  rko  = наш RK-opt постпроцесс (per-stride, для NPU)

Один цикл по кадрам, на каждом кадре:
  - letterbox + /255 один раз → один тензор (NCHW float32)
  - std_raw = std_backend.infer(tensor)   (std ONNX, cache/<model>_std/)
  - ref     = ultralytics non_max_suppression(std_raw) — эталонный NMS, letterboxed
  - std     = наш std постпроцесс(std_raw)  — orig, затем обратно в letterboxed
  - rko_raw = rko_backend.infer(tensor)   (RK-opt ONNX, cache/<model>/)
  - rko     = наш RK-opt постпроцесс(rko_raw)

Сравнение preds с допуском (float round-trip): ref-vs-std (наш постпроцесс vs
ultralytics NMS) и std-vs-rko (std vs RK-opt). Кадры с расхождением попадают в
diverged-список; --save-diverged копирует сами изображения в каталог.

Метрики всех трёх колонок считаются ОДНИМ и тем же in-memory кодом
(_box/_pose/_seg/_obb_tp_per_image + _ap_finalize) с общим GT — если preds
совпадают, метрики совпадают автоматически. Это строже прежнего подхода, где
ref = отдельный ultralytics .val().

Порядок preds не зависит от пространства: ref (nms) рождается сразу в
letterboxed, std/rko переводятся из orig обратно через x*scale+pad_left.

CLI:
  python run_3way_eval.py <model> [task] [--pt PATH] [--data-dir DIR]
                           [--limit N] [--save-diverged] [--diverged-dir DIR]
                           [--force]

  <model>  — pretrained имя (yolov8n-seg) или путь к кастомному .pt
  task     — detect|segment|pose|obb|classify|semantic (по умолчанию из .pt)
  --pt     — кастомный .pt (копируется в cache/<model>/, name = stem)
  --data-dir — каталог данных (по умолчанию эталон задачи)
  --limit  — ограничить число изображений (smoke)
  --save-diverged — копировать расходящиеся кадры в каталог
  --diverged-dir  — каталог для diverged-кадров (по умолчанию cache/diverged)
  --force  — перезапустить, игнорируя существующие json
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ['ULTRALYTICS_NO_UPDATE_CHECK'] = 'True'

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setup_datasets import ensure_full_dataset, ensure_mini_dataset, ensure_ultralytics_dir  # noqa: E402

ensure_ultralytics_dir()

RESULTS_DIR = ROOT / 'cache' / '3way_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _cache_dir(name):
    return ROOT / 'cache' / name


# ---------------------------------------------------------------------------
# Задача → эталонные данные (все 6 датасетов скачаны setup_datasets.py)
# ---------------------------------------------------------------------------

DATA = {
    'detect': {
        'eval_data': str(ROOT / 'datasets' / 'coco_val2017'),
        'ann_name': 'instances_val2017.json',
    },
    'segment': {
        'eval_data': str(ROOT / 'datasets' / 'coco_val2017'),
        'ann_name': 'instances_val2017.json',
    },
    'pose': {
        'eval_data': str(ROOT / 'datasets' / 'coco_val2017'),
        'ann_name': 'person_keypoints_val2017.json',
    },
    'obb': {
        'eval_data': str(ROOT / 'datasets' / 'dotav1'),
        'ann_name': '',
    },
    'classify': {
        'eval_data': str(ROOT / 'datasets' / 'imagenet'),
        'ann_name': '',
    },
    'semantic': {
        'eval_data': str(ROOT / 'datasets' / 'cityscapes'),
        'ann_name': '',
    },
}


# ---------------------------------------------------------------------------
# Разрешение модели
# ---------------------------------------------------------------------------

def resolve_imgsz(pt_path, std_model_dir):
    """Нативное рабочее разрешение модели: args.imgsz из .pt (или вход std ONNX)."""
    from ultralytics import YOLO
    if pt_path and Path(pt_path).exists():
        try:
            m = YOLO(pt_path)
            args = getattr(m, 'model', None) and getattr(m.model, 'args', None)
            sz = getattr(args, 'imgsz', None)
            if isinstance(sz, (list, tuple)):
                sz = sz[0] if sz else None
            if isinstance(sz, int) and sz > 0:
                return sz
        except Exception:
            pass
    onnx_path = Path(std_model_dir) / 'model.onnx'
    if onnx_path.exists():
        try:
            import onnxruntime as rt
            sess = rt.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            shape = sess.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[2], int):
                return shape[2]
        except Exception:
            pass
    return 640


# ---------------------------------------------------------------------------
# Автоподготовка кэша (реюз convert)
# ---------------------------------------------------------------------------

# Задачи, где встроенный мини-датасет даёт валидные GT для 3way-метрик
# (bbox/kpts из YOLO-разметки). Для остальных минидатасеты несовместимы
# (seg — нет полигонов, obb/cls/sem — другая структура) → используется полный.
MINI_SMOKE_TASKS = {'detect', 'pose'}


def _full_data_dir(task: str) -> Path:
    return ROOT / 'datasets' / {
        'detect': 'coco_val2017',
        'segment': 'coco_val2017',
        'pose': 'coco_val2017',
        'obb': 'dotav1',
        'classify': 'imagenet',
        'semantic': 'cityscapes',
    }[task]


def _dataset_present(full: Path) -> bool:
    """Полный датасет уже скачан (есть изображения)?"""
    for probe in (
        full / 'images' / 'val2017',
        full / 'images' / 'val',
        full / 'images',
    ):
        if probe.exists() and any(probe.glob('*.*')):
            return True
    return False


def resolve_eval_data(task: str, limit: int, override: str = '') -> str:
    """Выбрать датасет для eval/3way без лишних скачиваний.

    Правило:
      1. --data-dir задан явно → его;
      2. полный датасет задачи уже скачан → его (subset через --limit);
      3. smoke (limit>0) и задача поддерживает мини-датасет → встроенный
         мини-датасет (качается в datasets/, крошечный);
      4. иначе → полный датасет качается сам (код подтягивает необходимое).
    """
    if override:
        return override
    full = _full_data_dir(task)
    if _dataset_present(full):
        return str(full)
    if limit and task in MINI_SMOKE_TASKS:
        mini = ensure_mini_dataset(task)
        print(f'[DATA] полный датасет не скачан, smoke на мини-датасете: {mini}')
        return str(mini)
    print(f'[DATA] полный датасет не скачан, качаю: {full}')
    ensure_full_dataset(task)
    return str(full)


def ensure_model_pt(model, pt_arg):
    """Вернуть путь к .pt в cache/<model>/<model>.pt (скачивает pretrained или копирует кастомный)."""
    from convert.convert import resolve_pt
    cache_pt = _cache_dir(model) / f'{model}.pt'
    if cache_pt.exists():
        return str(cache_pt)
    if pt_arg:
        pt = Path(pt_arg)
        if not pt.exists():
            raise RuntimeError(f'--pt не найден: {pt}')
        if pt.resolve() != cache_pt.resolve():
            _cache_dir(model).mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(pt), str(cache_pt))
        return str(cache_pt)
    resolved = Path(resolve_pt(model))
    if not resolved.exists():
        raise RuntimeError(f'.pt не удалось получить для {model}')
    if resolved != cache_pt:
        _cache_dir(model).mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(resolved), str(cache_pt))
    return str(cache_pt)


def ensure_rko_cache(model, pt_path, task, force=False):
    """RK-opt ONNX + postprocess в cache/<model>/."""
    from convert.convert import _detect_model_info, _generate_postprocess, run_convert
    rko_dir = _cache_dir(model)
    onnx_path = rko_dir / f'{model}.onnx'
    pp_path = rko_dir / 'postprocess.py'
    if (onnx_path.exists() and pp_path.exists()) and not force:
        (rko_dir / 'model_name.txt').write_text(model + '\n')
        (rko_dir / 'quantize.txt').write_text('onnx\n')
        return
    run_convert(model, 'onnx')
    info = _detect_model_info(pt_path)
    info['task'] = task
    _generate_postprocess(info, rko_dir)
    (rko_dir / 'model_name.txt').write_text(model + '\n')
    (rko_dir / 'quantize.txt').write_text('onnx\n')


def ensure_std_cache(model, pt_path, force=False):
    """std ONNX + postprocess в cache/<model>_std/."""
    from benchmark.export_std_onnx import export_std_model
    std_dir = _cache_dir(f'{model}_std')
    if (std_dir / 'model.onnx').exists() and (std_dir / 'postprocess.py').exists() and not force:
        return
    export_std_model(model, Path(pt_path), std_dir)


def prepare_caches(model, task, pt_arg, force=False):
    """Подготовить все кэши (pt, RK-opt, std). Возвращает (pt_path, std_model_dir, rko_model_dir)."""
    pt_path = ensure_model_pt(model, pt_arg)
    ensure_rko_cache(model, pt_path, task, force)
    ensure_std_cache(model, pt_path, force)
    return pt_path, str(_cache_dir(f'{model}_std')), str(_cache_dir(model))


# ---------------------------------------------------------------------------
# Загрузка постпроцессов под уникальными именами (без глобального postprocess_mod)
# ---------------------------------------------------------------------------

def load_pp_module(module_name, path):
    """Загрузить postprocess.py как независимый модуль с уникальным именем."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Не удалось загрузить спецификацию постпроцесса: {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_eval_mode(mod):
    """Eval-режим постпроцесса (минимальные пороги, точные bbox, без clip)."""
    mod.CONF_THRESH = 0.001
    mod.IOU_THRESH = 0.7
    mod.MIN_WH = 1
    mod.MAX_DET = 300
    mod.DEDUP_ENABLED = False
    if hasattr(mod, 'EVAL_PRECISE'):
        mod.EVAL_PRECISE = True
    return mod


def load_pp(model_dir, model_name, module_name):
    """Найти postprocess.py в каталоге модели и загрузить как модуль."""
    pp_path = Path(model_dir) / f'{model_name}_postprocess.py'
    if not pp_path.exists():
        pp_path = Path(model_dir) / 'postprocess.py'
    if not pp_path.exists():
        raise FileNotFoundError(f'postprocess не найден: {pp_path}')
    return set_eval_mode(load_pp_module(module_name, str(pp_path)))


# ---------------------------------------------------------------------------
# Канонический формат preds (letterboxed-пространство)
#   {'score', 'class_id', 'bbox': [4 floats], 'extra': ...}
#   extra: pose -> 'kpts' (N,17,3); seg -> 'masks' (uint8 160,160); obb -> 'angle'
# ---------------------------------------------------------------------------

def _bbox_letter(x, scale, pad_left, pad_top, xywh=False):
    if xywh:
        cx, cy, w, h = x
        return [cx * scale + pad_left, cy * scale + pad_top, w * scale, h * scale]
    x1, y1, x2, y2 = x
    return [x1 * scale + pad_left, y1 * scale + pad_top,
            x2 * scale + pad_left, y2 * scale + pad_top]


def canon_ref(det, std_raw, task, pp, imgsz):
    """ref (ultralytics nms) — сразу в letterboxed-пространстве."""
    out = []
    for row in det:
        rec = {
            'score': float(row[4]),
            'class_id': int(row[5]),
            'bbox': [float(v) for v in row[:4]],
        }
        rec['bbox_xyxy_letter'] = rec['bbox']
        if task == 'pose':
            nk = pp.NUM_KPTS
            rec['kpts'] = row[6:6 + 3 * nk].reshape(nk, 3)
            rec['kpts_letter'] = rec['kpts']
        elif task == 'segment':
            proto = std_raw[1]
            if proto.ndim == 4:
                proto = proto[0]
            coeffs = row[6:6 + pp.NM][None, :]
            rec['masks_160'] = pp.process_mask_160(proto, coeffs, row[:4][None, :], imgsz)[0]
        elif task == 'obb':
            rec['angle'] = float(row[6])
        out.append(rec)
    return out


def canon_objs(objects, scale, pad_left, pad_top, task):
    """std/rko (наш постпроцесс) — orig → letterboxed."""
    out = []
    for obj in objects:
        rec = {
            'score': float(obj['score']),
            'class_id': int(obj['class_id']),
        }
        if task == 'obb':
            rec['bbox'] = _bbox_letter(obj['bbox'], scale, pad_left, pad_top, xywh=True)
            rec['angle'] = float(obj.get('angle', 0.0))
        else:
            rec['bbox'] = _bbox_letter(obj['bbox'], scale, pad_left, pad_top)
            rec['bbox_xyxy_letter'] = rec['bbox']
            if task == 'pose':
                kpts = obj.get('keypoints') or []
                rec['kpts'] = np.array([
                    [k[0] * scale + pad_left, k[1] * scale + pad_top, k[2] if len(k) >= 3 else 0.0]
                    for k in kpts
                ], dtype=np.float32)
                rec['kpts_letter'] = rec['kpts']
            elif task == 'segment':
                rec['masks_160'] = obj['masks_160']
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Сравнение preds (с допуском)
# ---------------------------------------------------------------------------

def _pred_sort_key(p):
    b = p['bbox']
    return (-p['score'], p['class_id'], b[0], b[1], b[2], b[3])


def compare_preds(preds_a, preds_b, task,
                  tol_score=1e-6, tol_bbox=1e-3, tol_angle=1e-4,
                  tol_kpts=1e-3, tol_mask=0.01):
    """Сравнить два canon-списка. Возвращает (ok, reason)."""
    a = sorted(preds_a, key=_pred_sort_key)
    b = sorted(preds_b, key=_pred_sort_key)
    if len(a) != len(b):
        return False, f'count {len(a)} vs {len(b)}'
    for x, y in zip(a, b):
        if x['class_id'] != y['class_id']:
            return False, f'cls {x["class_id"]} vs {y["class_id"]}'
        sd = abs(x['score'] - y['score'])
        if sd > tol_score:
            return False, f'score diff {sd:.2e}'
        bd = max(abs(xi - yi) for xi, yi in zip(x['bbox'], y['bbox']))
        if bd > tol_bbox:
            return False, f'bbox diff {bd:.2e}'
        if task == 'pose':
            kd = float(np.abs(x['kpts'] - y['kpts']).max())
            if kd > tol_kpts:
                return False, f'kpts diff {kd:.2e}'
        elif task == 'obb':
            ad = abs(x['angle'] - y['angle'])
            if ad > tol_angle:
                return False, f'angle diff {ad:.2e}'
        elif task == 'segment':
            mr = float((x['masks_160'] != y['masks_160']).mean())
            if mr > tol_mask:
                return False, f'mask diff ratio {mr:.4f}'
    return True, ''


# ---------------------------------------------------------------------------
# Интерливный проход
# ---------------------------------------------------------------------------

def _ann_file(data_dir, ann_name):
    for candidate in [
        str(Path(data_dir).parent / 'annotations' / ann_name),
        str(Path(data_dir) / 'annotations' / ann_name),
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def interleaved_detect_like(task, cfg, std_be, rko_be, std_pp, rko_pp, limit, state):
    """Один цикл по detect/segment/pose кадрам; метрики + diverged."""
    from benchmark.eval import (find_img_path, load_coco, load_yolo_as_coco,
                                _box_tp_per_image, _pose_tp_per_image,
                                _seg_tp_per_image)
    from percept.io import letterbox

    data_dir = cfg['eval_data']
    isz = std_be.input_size
    if rko_be.input_size != isz:
        raise RuntimeError(f'input_size не совпадает: std={isz} rko={rko_be.input_size}')

    ann_file = _ann_file(data_dir, cfg['ann_name'])
    if ann_file:
        images, ann_by_img, cat_ids = load_coco(ann_file, task)
    else:
        images, ann_by_img, cat_ids, _ = load_yolo_as_coco(data_dir)

    agg = {c: {'tp': [], 'conf': [], 'pred_cls': [], 'gt_cls': []}
           for c in ('ref', 'std', 'rko')}
    if task == 'pose':
        pose_agg = {c: {'tp': [], 'conf': [], 'pred_cls': [], 'gt_cls': []}
                    for c in ('ref', 'std', 'rko')}

    preds_count = {'ref': 0, 'std': 0, 'rko': 0}

    total = state['total']
    errors = state['errors']

    for img_id, img_info in images.items():
        if limit and total >= limit:
            break
        img_path = find_img_path(img_info, data_dir)
        frame = cv2.imread(img_path)
        if frame is None:
            errors += 1
            continue
        h, w = frame.shape[:2]

        canvas, scale, pad = letterbox(frame, isz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0) / 255.0

        std_raw = std_be.infer(tensor)
        rko_raw = rko_be.infer(tensor)

        det = std_pp._nms_det(std_raw)  # (N, 6+extra) letterboxed
        ref = canon_ref(det, std_raw, task, std_pp, isz)

        pad_left = round(pad[0] / 2 - 0.1)
        pad_top = round(pad[1] / 2 - 0.1)
        objs2 = std_pp.postprocess(std_raw, scale, pad, (h, w))
        std = canon_objs(objs2, scale, pad_left, pad_top, task)

        objs3 = rko_pp.postprocess(rko_raw, scale, pad, (h, w))
        rko = canon_objs(objs3, scale, pad_left, pad_top, task)

        ok_ref_std, reason_ref_std = compare_preds(ref, std, task)
        ok_std_rko, reason_std_rko = compare_preds(std, rko, task)
        if not ok_ref_std:
            state['diverged_ref_std'].append((img_path, reason_ref_std))
        if not ok_std_rko:
            state['diverged_std_rko'].append((img_path, reason_std_rko))

        sx = (isz - pad[0]) / w
        sy = (isz - pad[1]) / h
        anns = ann_by_img.get(img_id, [])
        if task == 'segment':
            anns = [a for a in anns if not a.get('iscrowd', 0)]

        for c, preds in (('ref', ref), ('std', std), ('rko', rko)):
            preds_count[c] += len(preds)
            if task == 'pose':
                tp, conf, pcls, gcls = _box_tp_per_image(anns, preds, sx, sy, pad_left, pad_top)
                agg[c]['tp'].append(tp); agg[c]['conf'].append(conf)
                agg[c]['pred_cls'].append(pcls); agg[c]['gt_cls'].append(gcls)
                tpp, confp, pclsp, gclsp = _pose_tp_per_image(anns, preds, sx, sy, pad_left, pad_top)
                pose_agg[c]['tp'].append(tpp); pose_agg[c]['conf'].append(confp)
                pose_agg[c]['pred_cls'].append(pclsp); pose_agg[c]['gt_cls'].append(gclsp)
            elif task == 'segment':
                tp, conf, pcls, gcls = _seg_tp_per_image(anns, preds, sx, sy, pad_left, pad_top)
                agg[c]['tp'].append(tp); agg[c]['conf'].append(conf)
                agg[c]['pred_cls'].append(pcls); agg[c]['gt_cls'].append(gcls)
            else:
                tp, conf, pcls, gcls = _box_tp_per_image(anns, preds, sx, sy, pad_left, pad_top)
                agg[c]['tp'].append(tp); agg[c]['conf'].append(conf)
                agg[c]['pred_cls'].append(pcls); agg[c]['gt_cls'].append(gcls)

        total += 1
        if total % 500 == 0:
            sys.stderr.write(f'[3way][{task}] {total}/{len(images)}\n')

    state['total'] = total
    state['errors'] = errors
    state['preds'] = preds_count

    from benchmark.eval import _ap_finalize
    cols = {}
    for c in ('ref', 'std', 'rko'):
        a = agg[c]
        cols[c] = _ap_finalize(a['tp'], a['conf'], a['pred_cls'], a['gt_cls'])
        if task == 'pose':
            p = pose_agg[c]
            cols[c].update(_ap_finalize(p['tp'], p['conf'], p['pred_cls'], p['gt_cls'],
                                        prefix='inmem_pose'))
    return cols


def interleaved_obb(task, cfg, std_be, rko_be, std_pp, rko_pp, limit, state):
    """Один цикл по obb (DOTA) кадрам; метрики + diverged."""
    from benchmark.eval import (load_dota, xyxyxyxy2xywhr_np,
                                _obb_tp_per_image, _ap_finalize)
    from percept.io import letterbox

    data_dir = cfg['eval_data']
    isz = std_be.input_size
    if rko_be.input_size != isz:
        raise RuntimeError(f'input_size не совпадает: std={isz} rko={rko_be.input_size}')

    dota_images, dota_labels = load_dota(
        str(Path(data_dir) / 'labels' / 'val'),
        str(Path(data_dir) / 'images' / 'val'),
    )

    agg = {c: {'tp': [], 'conf': [], 'pred_cls': [], 'gt_cls': []}
           for c in ('ref', 'std', 'rko')}
    preds_count = {'ref': 0, 'std': 0, 'rko': 0}
    total = state['total']
    errors = state['errors']

    for img_path in dota_images:
        if limit and total >= limit:
            break
        img = cv2.imread(img_path)
        if img is None:
            errors += 1
            continue
        h, w = img.shape[:2]
        stem = Path(img_path).stem

        canvas, scale, pad = letterbox(img, isz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0) / 255.0

        std_raw = std_be.infer(tensor)
        rko_raw = rko_be.infer(tensor)

        det = std_pp._nms_det(std_raw)  # (N, 7) [cx,cy,w,h,conf,cls,angle]
        ref = canon_ref(det, std_raw, 'obb', std_pp, isz)

        pad_left = round(pad[0] / 2 - 0.1)
        pad_top = round(pad[1] / 2 - 0.1)
        objs2 = std_pp.postprocess(std_raw, scale, pad, (h, w))
        std = canon_objs(objs2, scale, pad_left, pad_top, 'obb')
        objs3 = rko_pp.postprocess(rko_raw, scale, pad, (h, w))
        rko = canon_objs(objs3, scale, pad_left, pad_top, 'obb')

        ok_ref_std, reason_ref_std = compare_preds(ref, std, 'obb')
        ok_std_rko, reason_std_rko = compare_preds(std, rko, 'obb')
        if not ok_ref_std:
            state['diverged_ref_std'].append((img_path, reason_ref_std))
        if not ok_std_rko:
            state['diverged_std_rko'].append((img_path, reason_std_rko))

        sx = (isz - pad[0]) / w
        sy = (isz - pad[1]) / h
        polys = dota_labels.get(stem, [])
        if polys:
            gt_letter = np.array([
                xyxyxyxy2xywhr_np([
                    (p['polygon'][i] * w) * sx + pad_left if i % 2 == 0
                    else (p['polygon'][i] * h) * sy + pad_top
                    for i in range(8)
                ])
                for p in polys
            ], dtype=np.float32)
            gt_cls = np.array([p['class_id'] for p in polys], dtype=np.float32)
        else:
            gt_letter = np.empty((0, 5), dtype=np.float32)
            gt_cls = np.array([], dtype=np.float32)

        for c, preds in (('ref', ref), ('std', std), ('rko', rko)):
            preds_count[c] += len(preds)
            if preds:
                pr_letter = np.array([p['bbox'] + [p['angle']] for p in preds], dtype=np.float32)
                pr_cls = np.array([p['class_id'] for p in preds], dtype=np.float32)
                pr_s = np.array([p['score'] for p in preds], dtype=np.float32)
            else:
                pr_letter = np.empty((0, 5), dtype=np.float32)
                pr_cls = np.array([], dtype=np.float32)
                pr_s = np.array([], dtype=np.float32)
            tp, conf, pcls, gcls = _obb_tp_per_image(gt_letter, gt_cls, pr_letter, pr_cls, pr_s)
            agg[c]['tp'].append(tp); agg[c]['conf'].append(conf)
            agg[c]['pred_cls'].append(pcls); agg[c]['gt_cls'].append(gcls)

        total += 1
        if total % 100 == 0:
            sys.stderr.write(f'[3way][obb] {total}/{len(dota_images)}\n')

    state['total'] = total
    state['errors'] = errors
    state['preds'] = preds_count

    cols = {}
    for c in ('ref', 'std', 'rko'):
        a = agg[c]
        cols[c] = _ap_finalize(a['tp'], a['conf'], a['pred_cls'], a['gt_cls'])
    return cols


def interleaved_classify(cfg, std_be, rko_be, std_pp, rko_pp, limit, state):
    """classify: ref == std по построению (тот же std_raw); rko = RK-opt."""
    from benchmark.eval import load_imagenet, compute_imagenet_accuracy

    img_paths, gt_labels = load_imagenet(cfg['eval_data'])
    preds = {'std': [], 'rko': []}
    total = state['total']
    errors = state['errors']

    for img_path, gt_label in zip(img_paths, gt_labels):
        if limit and total >= limit:
            break
        frame = cv2.imread(img_path)
        if frame is None:
            errors += 1
            continue
        tensor, _, _ = std_pp.preprocess(frame)
        std_raw = std_be.infer(tensor)
        rko_raw = rko_be.infer(tensor)
        objs2 = std_pp.postprocess(std_raw, 1.0, (0, 0), frame.shape[:2])
        objs3 = rko_pp.postprocess(rko_raw, 1.0, (0, 0), frame.shape[:2])
        preds['std'].append(objs2[0]['class_id'] if objs2 else -1)
        preds['rko'].append(objs3[0]['class_id'] if objs3 else -1)
        total += 1

    state['total'] = total
    state['errors'] = errors
    acc_std = compute_imagenet_accuracy(preds['std'], gt_labels)
    acc_rko = compute_imagenet_accuracy(preds['rko'], gt_labels)
    return {'ref': acc_std, 'std': acc_std, 'rko': acc_rko}


def interleaved_semantic(cfg, std_be, rko_be, std_pp, rko_pp, limit, state):
    """semantic: ref == std по построению; rko = RK-opt; miou через confusion."""
    from benchmark.eval import (load_cityscapes, apply_cityscapes_label_mapping,
                                letterbox_semantic_gt, sem_metrics_from_matrix)
    from percept.io import letterbox

    isz = std_be.input_size
    if rko_be.input_size != isz:
        raise RuntimeError(f'input_size не совпадает: std={isz} rko={rko_be.input_size}')
    num_cls = 19

    sem_pairs = load_cityscapes(
        str(Path(cfg['eval_data']) / 'images' / 'val'),
        str(Path(cfg['eval_data']) / 'masks' / 'val'),
    )

    matrix = {'std': np.zeros((num_cls, num_cls), dtype=np.float64),
              'rko': np.zeros((num_cls, num_cls), dtype=np.float64)}
    total = state['total']
    errors = state['errors']

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
        if int(gt.max()) > 18:
            gt = apply_cityscapes_label_mapping(gt)

        canvas, scale, pad = letterbox(frame, isz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0) / 255.0

        std_raw = std_be.infer(tensor)
        rko_raw = rko_be.infer(tensor)

        # semantic: pred-карта из raw выходов в letterboxed-пространстве (как eval.py),
        # НЕ через постпроцесс (он возвращает orig-пространство и несовместим с gt_map).
        def _sem_map(raw):
            pm = np.asarray(raw[0][0])
            if pm.ndim == 3:
                pm = pm.argmax(axis=0)
            return pm.astype(np.int64)

        pm_std = _sem_map(std_raw)
        pm_rko = _sem_map(rko_raw)

        gt_map = letterbox_semantic_gt(gt, scale, pad, isz).astype(np.int64)
        for pm, key in ((pm_std, 'std'), (pm_rko, 'rko')):
            valid = (gt_map != 255) & (pm >= 0) & (pm < num_cls) & (gt_map >= 0) & (gt_map < num_cls)
            np.add.at(matrix[key], (gt_map[valid], pm[valid]), 1)

        total += 1

    state['total'] = total
    state['errors'] = errors
    m_std = sem_metrics_from_matrix(matrix['std'], num_cls)
    m_rko = sem_metrics_from_matrix(matrix['rko'], num_cls)
    return {'ref': m_std, 'std': m_std, 'rko': m_rko}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_nms_fn(std_pp, task):
    """Обёртка ultralytics non_max_suppression: полный NMS (без time_limit).

    Возвращает numpy (N, 6+extra) в letterboxed-пространстве (model coords).
    Прибивается к std_pp как _nms_det для единого вызова.
    """
    import torch
    from ultralytics.utils.nms import non_max_suppression

    nm = getattr(std_pp, 'NM', 0)
    nc = std_pp.NUM_CLASSES
    rotated = (task == 'obb')

    def _nms_det(std_raw):
        # .copy() — nms модифицирует pred инплейс (xywh2xyxy через вью transpose),
        # поэтому нельзя передавать буфер std_raw: postprocess(std_raw) после nms
        # получил бы уже xyxy и сломал бы декодирование.
        pred = torch.from_numpy(np.asarray(std_raw[0], dtype=np.float32).copy())
        out = non_max_suppression(
            pred,
            conf_thres=std_pp.CONF_THRESH,
            iou_thres=std_pp.IOU_THRESH,
            multi_label=True,
            max_det=std_pp.MAX_DET,
            rotated=rotated,
            max_time_img=1000.0,
            nc=nc,
        )
        det = out[0]
        if det is None or len(det) == 0:
            return np.zeros((0, 6 + nm), dtype=np.float32)
        return det.cpu().numpy().astype(np.float32)

    std_pp._nms_det = _nms_det
    return _nms_det


def _save_diverged(state, diverged_dir, force):
    """Сохранить diverged-списки; при --save-diverged — копировать изображения."""
    for key in ('diverged_ref_std', 'diverged_std_rko'):
        entries = state.get(key, [])
        if not entries:
            continue
        out = {'count': len(entries), 'items': [
            {'image': p, 'reason': r} for p, r in entries]}
        (diverged_dir / f'{key}.json').write_text(json.dumps(out, indent=2))
    if state.get('save_diverged') and diverged_dir:
        for key in ('diverged_ref_std', 'diverged_std_rko'):
            for img_path, _reason in state.get(key, []):
                stem = Path(img_path).stem
                dst = diverged_dir / key / f'{stem}.jpg'
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and not force:
                    continue
                shutil.copy2(img_path, str(dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', help='pretrained имя (yolov8n-seg) или путь к кастомному .pt')
    ap.add_argument('task', nargs='?', default='',
                    help='detect|segment|pose|obb|classify|semantic (по умолчанию из .pt)')
    ap.add_argument('--pt', default='', help='кастомный .pt')
    ap.add_argument('--data-dir', default='', help='каталог данных (по умолчанию эталон задачи)')
    ap.add_argument('--limit', type=int, default=0, help='ограничить число изображений (smoke)')
    ap.add_argument('--save-diverged', action='store_true',
                    help='копировать расходящиеся кадры в каталог diverged')
    ap.add_argument('--diverged-dir', default='', help='каталог для diverged-кадров')
    ap.add_argument('--force', action='store_true', help='перезапустить, игнорируя существующие json')
    args = ap.parse_args()

    model_arg = args.model
    custom_pt = args.pt or (model_arg if Path(model_arg).suffix in ('.pt', '.onnx', '.rknn') else '')
    if custom_pt and Path(custom_pt).exists():
        model = Path(custom_pt).stem
    else:
        model = Path(model_arg).stem if Path(model_arg).suffix else model_arg

    pt_path = ensure_model_pt(model, custom_pt)

    task = args.task
    if not task:
        from ultralytics import YOLO
        task = YOLO(pt_path).task
        task = 'segment' if task == 'seg' else task
    if task not in DATA:
        print(f'ERROR: неизвестная задача {task!r}; допустимо: {sorted(DATA)}', file=sys.stderr)
        sys.exit(1)

    std_model_dir, rko_model_dir = prepare_caches(model, task, '', args.force)[1:]

    cfg = dict(DATA[task])
    cfg['std_model_dir'] = std_model_dir
    cfg['rko_model_dir'] = rko_model_dir
    cfg['pt_path'] = pt_path
    cfg['eval_data'] = resolve_eval_data(task, args.limit, args.data_dir)
    cfg['imgsz'] = resolve_imgsz(pt_path, std_model_dir)

    task_key = f'{model}-{task}'
    result_file = RESULTS_DIR / f'{task_key}_interleaved.json'

    if result_file.exists() and not args.force:
        print(f'[{task_key}] SKIP (already done: {result_file})')
        return

    t_start = time.time()
    print(f'[{task_key}] imgsz={cfg["imgsz"]} eval_data={cfg["eval_data"]}', flush=True)

    # Бэкенды: ModelBackend напрямую (без Pipeline, чтобы не конфликтовал
    # глобальный postprocess_mod). Оба ONNX CPU.
    from percept.inference import ModelBackend
    std_be = ModelBackend(std_model_dir, 'model', backend='cpu')
    rko_be = ModelBackend(rko_model_dir, model, backend='cpu')

    std_pp = load_pp(std_model_dir, 'model', 'pp_std')
    rko_pp = load_pp(rko_model_dir, model, 'pp_rko')

    state = {
        'total': 0,
        'errors': 0,
        'diverged_ref_std': [],
        'diverged_std_rko': [],
        'save_diverged': args.save_diverged,
    }

    if task in ('detect', 'segment', 'pose'):
        _make_nms_fn(std_pp, task)
        cols = interleaved_detect_like(task, cfg, std_be, rko_be, std_pp, rko_pp,
                                       args.limit, state)
    elif task == 'obb':
        _make_nms_fn(std_pp, task)
        cols = interleaved_obb(task, cfg, std_be, rko_be, std_pp, rko_pp,
                               args.limit, state)
    elif task == 'classify':
        cols = interleaved_classify(cfg, std_be, rko_be, std_pp, rko_pp,
                                    args.limit, state)
    elif task == 'semantic':
        cols = interleaved_semantic(cfg, std_be, rko_be, std_pp, rko_pp,
                                    args.limit, state)

    diverged_dir = Path(args.diverged_dir) if args.diverged_dir else (
        ROOT / 'cache' / 'diverged' / task_key)
    diverged_dir.mkdir(parents=True, exist_ok=True)
    _save_diverged(state, diverged_dir, args.force)

    result = {
        'ref': cols['ref'],
        'std': cols['std'],
        'rko': cols['rko'],
        'total': state['total'],
        'errors': state['errors'],
        'preds': state.get('preds', {}),
        'diverged': {
            'ref_vs_std': len(state['diverged_ref_std']),
            'std_vs_rko': len(state['diverged_std_rko']),
        },
        'diverged_dir': str(diverged_dir),
        'elapsed_s': round(time.time() - t_start, 1),
        'model': model,
        'task': task,
        'imgsz': cfg['imgsz'],
        'equal': {
            'ref_vs_std': not state['diverged_ref_std'],
            'std_vs_rko': not state['diverged_std_rko'],
        },
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f'[{task_key}] DONE: {json.dumps(result)}')


if __name__ == '__main__':
    main()
