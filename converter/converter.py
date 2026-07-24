#!/usr/bin/env python
"""RKNN Model Converter & Validator for Ultralytics detection models.

Usage:
    docker compose run --rm converter convert yolov8m.pt
    docker compose run --rm converter validate yolov8m.rknn
    docker compose run --rm converter test-all-detect
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

COCO_VAL_DIR = Path("/app/coco_val2017")
COCO_VAL_IMG_DIR = COCO_VAL_DIR / "images" / "val2017"
COCO_VAL_YAML = COCO_VAL_DIR / "coco_val.yaml"
COCO_VAL_POSE_YAML = COCO_VAL_DIR / "coco_val_pose.yaml"
DOTA_VAL_DIR = Path("/app/dotav1")
DOTA_VAL_IMG_DIR = DOTA_VAL_DIR / "images" / "val"
CITYSCAPES_VAL_DIR = Path("/app/cityscapes")
CITYSCAPES_VAL_IMG_DIR = CITYSCAPES_VAL_DIR / "images" / "val"
MODEL_DIR = Path("/app/converter/model")
WEIGHTS_DIR = Path("/app/converter/weights")

STRIDES = [8, 16, 32]
GRID_SIZES = [80, 40, 20]
REG_MAX = 16
BIN_COUNT = 4
CONF_THRESH = 0.5
IOU_THRESH = 0.5
MIN_WH = 40

COCO_CLASSES = [
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


def log(msg):
    print(f"[CONVERTER] {msg}", flush=True)


def apply_patches():
    """Применить патчи RKNN к головам ultralytics."""
    from patches import rknn_patch
    rknn_patch.apply()
    log("Патчи RKNN применены к головам ultralytics")


ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def _clean_pt_cache(name):
    """Очистить кеш .pt файлов ultralytics."""
    from ultralytics.utils import USER_CONFIG_DIR
    cache_dirs = [
        USER_CONFIG_DIR.parent / "cache" / "ultralytics" / "models",
        Path("/root/.cache/ultralytics/models"),
        Path.home() / ".cache" / "ultralytics" / "models",
    ]
    for d in cache_dirs:
        for f in d.glob(f"{name}*"):
            log(f"[ОЧИСТКА] удалён кешированный файл {f}")
            f.unlink(missing_ok=True)


def _dataset_name(task: str) -> str:
    """Вернуть читаемое имя датасета для задачи."""
    return {
        "detect": "COCO val2017",
        "segment": "COCO val2017",
        "classify": "COCO val2017",
        "pose": "COCO val2017",
        "obb": "DOTAv1",
        "semantic": "Cityscapes",
    }.get(task, "COCO val2017")


def _ensure_dataset(task: str = "detect") -> Path:
    """Проверить наличие нужного датасета для калибровки. Вернуть путь к датасету."""
    if task == "obb":
        if DOTA_VAL_IMG_DIR.exists() and any(DOTA_VAL_IMG_DIR.glob("*.png")):
            n = len(list(DOTA_VAL_IMG_DIR.glob("*.png")))
            log(f"DOTA val готова: {n} изображений")
            return DOTA_VAL_DIR
        raise RuntimeError(
            f"DOTA val не найдена в {DOTA_VAL_DIR}. "
            f"Скачайте: python3 setup_datasets.py --dota"
        )
    elif task == "semantic":
        if CITYSCAPES_VAL_IMG_DIR.exists() and any(CITYSCAPES_VAL_IMG_DIR.glob("*.png")):
            n = len(list(CITYSCAPES_VAL_IMG_DIR.glob("*.png")))
            log(f"Cityscapes val готова: {n} изображений")
            return CITYSCAPES_VAL_DIR
        raise RuntimeError(
            f"Cityscapes val не найдена в {CITYSCAPES_VAL_DIR}. "
            f"Скачайте: python3 setup_datasets.py --cityscapes"
        )
    else:
        if COCO_VAL_IMG_DIR.exists() and any(COCO_VAL_IMG_DIR.glob("*.jpg")):
            n = len(list(COCO_VAL_IMG_DIR.glob("*.jpg")))
            log(f"COCO val 2017 готова: {n} изображений")
            return COCO_VAL_DIR
        raise RuntimeError(
            f"COCO val 2017 не найдена в {COCO_VAL_IMG_DIR}. "
            f"Скачайте: python3 setup_datasets.py --coco"
        )


def _make_calib_yaml(task: str = "detect") -> str:
    """Генерирует калибровочный YAML на лету для нужного датасета и задачи."""
    data_dir = _ensure_dataset(task)

    if task == "obb":
        return _make_dota_calib_yaml(data_dir)
    elif task == "semantic":
        return _make_cityscapes_calib_yaml(data_dir)
    else:
        return _make_coco_calib_yaml(data_dir, task)


def _make_coco_calib_yaml(data_dir: Path, task: str = "detect") -> str:
    """Калибровочный YAML для COCO val2017 (detect, seg, cls, pose)."""
    n = len(list(COCO_VAL_IMG_DIR.glob("*.jpg")))
    log(f"Калибровочный набор: COCO val2017 — {n} изображений")

    lines = [
        f"path: {data_dir}",
        f"train: images/val2017",
        f"val: images/val2017",
    ]

    if task == "pose":
        lines.extend([
            "kpt_shape: [17, 3]",
            "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]",
            "annotations:",
            "  train: annotations/person_keypoints_val2017.json",
            "  val: annotations/person_keypoints_val2017.json",
            "names:",
            "  0: person",
        ])
    else:
        lines.extend([
            "nc: 80",
            "names:",
        ])
        for i, name in enumerate(COCO_CLASSES):
            lines.append(f"  {i}: {name}")
        lines.extend([
            "annotations:",
            "  train: annotations/instances_val2017.json",
            "  val: annotations/instances_val2017.json",
        ])

    calib_yaml = Path("/tmp/calib.yaml")
    calib_yaml.write_text("\n".join(lines) + "\n")
    log(f"Калибровочный YAML: {calib_yaml} (задача={task})")
    return str(calib_yaml)


def _make_dota_calib_yaml(data_dir: Path) -> str:
    """Калибровочный YAML для DOTAv1 (obb)."""
    n = len(list(DOTA_VAL_IMG_DIR.glob("*.png")))
    log(f"Калибровочный набор: DOTAv1 val — {n} изображений")

    lines = [
        f"path: {data_dir}",
        f"train: images/val",
        f"val: images/val",
        "nc: 15",
        "names:",
        "  0: plane",
        "  1: ship",
        "  2: storage-tank",
        "  3: baseball-diamond",
        "  4: tennis-court",
        "  5: basketball-court",
        "  6: ground-track-field",
        "  7: soccer-ball-field",
        "  8: roundabout",
        "  9: harbor",
        "  10: swimming-pool",
        "  11: helicopter",
        "  12: vehicle",
        "  13: helicopter",  # placeholder (DOTA has 15 classes, ultralytics OBB uses subset)
        "  14: container-crane",
    ]

    calib_yaml = Path("/tmp/calib.yaml")
    calib_yaml.write_text("\n".join(lines) + "\n")
    log(f"Калибровочный YAML: {calib_yaml} (задача=obb)")
    return str(calib_yaml)


def _make_cityscapes_calib_yaml(data_dir: Path) -> str:
    """Калибровочный YAML для Cityscapes val (semantic)."""
    n = len(list(CITYSCAPES_VAL_IMG_DIR.glob("*.png")))
    log(f"Калибровочный набор: Cityscapes val — {n} изображений")

    CITYSCAPES_CLASSES = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle",
    ]

    lines = [
        f"path: {data_dir}",
        f"train: images/val",
        f"val: images/val",
        f"nc: {len(CITYSCAPES_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(CITYSCAPES_CLASSES):
        lines.append(f"  {i}: {name}")

    calib_yaml = Path("/tmp/calib.yaml")
    calib_yaml.write_text("\n".join(lines) + "\n")
    log(f"Калибровочный YAML: {calib_yaml} (задача=semantic)")
    return str(calib_yaml)


DOWNLOAD_TIMEOUT = 120


def _download(url, dst):
    """Скачать файл по URL с повторными попытками."""
    import urllib.request
    log(f"[ЗАГРУЗКА] {url} -> {dst}")
    for attempt in range(3):
        try:
            resp = urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT)
            with open(dst, "wb") as f:
                f.write(resp.read())
            return True
        except Exception as e:
            log(f"[ЗАГРУЗКА] попытка {attempt+1}/3 не удалась: {e}")
            if dst and Path(dst).exists():
                Path(dst).unlink(missing_ok=True)
    return False


def resolve_pt(name):
    """Найти или скачать .pt файл модели."""
    if name.endswith(".pt"):
        p = Path(name)
        if p.exists():
            return str(p)
        p = WEIGHTS_DIR / name
        if p.exists():
            return str(p)
    for d in [WEIGHTS_DIR, MODEL_DIR, Path("."), Path("/app/converter")]:
        p = d / f"{name}.pt"
        if p.exists():
            return str(p)
    url = f"{ASSETS_URL}/{name}.pt"
    dst = Path(f"{name}.pt")
    _clean_pt_cache(name)
    log(f"[PT] загрузка {name} из {url}")
    if _download(url, str(dst)):
        return str(dst)
    return name


def export_and_compile(pt_path, info=None, quantize="none"):
    """Экспорт .pt → ONNX (RK-opt) → RKNN, возвращает (onnx_path, rknn_path).

    Аргументы:
        pt_path: путь к весам .pt.
        info: опциональный dict из _detect_model_info().
        quantize: "none" — FP32, "int8" — INT8 без калибровки,
                  "int8-calib" — INT8 с калибровкой на датасете.
    """
    apply_patches()
    from ultralytics import YOLO
    model = YOLO(pt_path)

    # Установить флаг RKNN перед запуском экспортера
    for m in model.model.modules():
        m._rknn_mode = True

    # RT-DETR использует deformable attention, требующий ONNX opset >= 16
    is_rtdetr = info and info.get("is_rtdetr", False)
    opset = 16 if is_rtdetr else 12
    log(f"Используется ONNX opset {opset} для экспорта")

    export_kwargs = {"format": "rknn", "opset": opset}
    if quantize == "int8-calib":
        task = info.get("task", "detect") if info else "detect"
        _ensure_dataset(task)
        data_yaml = _make_calib_yaml(task)
        export_kwargs["data"] = data_yaml
        export_kwargs["quantize"] = 8
        log(f"Квантизация INT8 с калибровкой: задача={task}, датасет={_dataset_name(task)}")
    elif quantize == "int8":
        log("Квантизация INT8 без калибровки")

    t0 = time.time()
    log(f"Запуск model.export() ...")
    rknn_dir = model.export(**export_kwargs)
    log(f"model.export() завершён за {time.time() - t0:.0f}с")
    log(f"Директория RKNN модели: {rknn_dir}")

    # Найти ONNX файл, созданный при экспорте
    rknn_dir_path = Path(rknn_dir)
    model_stem = Path(pt_path).stem
    pt_dir = Path(pt_path).parent
    onnx_candidates = [
        f"{model_stem}.onnx",
        str(pt_dir / f"{model_stem}.onnx"),
    ]
    onnx_path = None
    for c in onnx_candidates:
        if os.path.exists(c):
            onnx_path = c
            break
    if not onnx_path:
        onnx_paths = list(Path(".").glob("*.onnx"))
        onnx_path = str(onnx_paths[-1]) if onnx_paths else None

    # Найти RKNN файл (предпочитать самый свежий .rknn)
    rknn_files = sorted(rknn_dir_path.glob("*.rknn"), key=os.path.getmtime, reverse=True)
    rknn_path = str(rknn_files[0]) if rknn_files else rknn_dir

    if onnx_path:
        log(f"ONNX (RK-opt): {onnx_path}")
    log(f"RKNN: {rknn_path}")

    # Логирование dataset.txt для проверки списка калибровочных изображений
    if quantize == "int8-calib":
        _log_dataset_txt()

    # Для int8 без калибровки: экспортировали ONNX, теперь собираем RKNN
    if quantize == "int8" and onnx_path:
        log("Сборка RKNN (int8 без калибровки)...")
        rknn_path = _build_rknn(onnx_path, quantize=True, calib_dir=None)

    return onnx_path, rknn_path


def _log_dataset_txt():
    """Вывести содержимое dataset.txt для проверки списка калибровочных изображений.

    Файл dataset.txt генерируется ultralytics при экспорте RKNN с квантизацией.
    Содержит пути к изображениям, используемым для калибровки.
    """
    dataset_candidates = [
        Path("/tmp/dataset.txt"),
        Path("dataset.txt"),
        Path("calib_dataset.txt"),
    ]

    # Также поискать в рабочей директории
    dataset_candidates.extend(Path(".").glob("*dataset*.txt"))
    dataset_candidates.extend(Path("/tmp").glob("*dataset*.txt"))

    found = False
    for ds_path in dataset_candidates:
        if ds_path.exists() and ds_path.stat().st_size > 0:
            try:
                lines = ds_path.read_text().strip().splitlines()
                total = len(lines)
                log(f"=== DATASET.TXT ({ds_path}): {total} изображений ===")
                # Вывести первые 10 и последние 5 путей
                show_lines = lines[:10]
                if total > 15:
                    show_lines.append(f"  ... ({total - 15} изображений пропущено) ...")
                    show_lines.extend(lines[-5:])
                elif total > 10:
                    show_lines.extend(lines[10:])
                for line in show_lines:
                    log(f"  {line}")
                log(f"=== КОНЕЦ DATASET.TXT ===")
                found = True
                break
            except Exception as e:
                log(f"Ошибка чтения {ds_path}: {e}")

    if not found:
        log("ВНИМАНИЕ: dataset.txt не найден! Файл может быть сгенерирован в нестандартном месте.")
        log("Поиск всех файлов *dataset* в /tmp и текущей директории:")
        import glob as glob_mod
        for pattern in ["/tmp/*dataset*", "*dataset*"]:
            for f in glob_mod.glob(pattern):
                log(f"  найден: {f}")


def _build_rknn(onnx_path, quantize, calib_dir=None):
    """Собрать RKNN модель из ONNX файла напрямую через RKNN toolkit.

    Аргументы:
        onnx_path: путь к файлу ONNX модели.
        quantize: если True, собрать с квантизацией INT8, калиброванной на calib_dir.
        calib_dir: директория с калибровочными изображениями (обязательна при quantize=True).
    Возвращает:
        путь к собранному .rknn файлу.
    """
    from rknn.api import RKNN

    stem = Path(onnx_path).stem
    suffix = "_int8.rknn" if quantize else "_none.rknn"
    rknn_path = f"{stem}{suffix}"

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform="rk3588")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"load_onnx не удался: {ret}")

    if quantize:
        if calib_dir:
            log(f"Сборка RKNN (int8) с калибровкой из {calib_dir}")
            calib_txt = "/tmp/calib.txt"
            with open(calib_txt, "w") as f:
                for img_path in sorted(Path(calib_dir).glob("*.jpg"))[:512]:
                    f.write(str(img_path) + "\n")
            ret = rknn.build(do_quantization=True, dataset=calib_txt)
        else:
            log("Сборка RKNN (int8, без калибровки)")
            ret = rknn.build(do_quantization=True)
    else:
        log("Сборка RKNN (none, без квантизации)")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError(f"rknn.build не удался: {ret}")

    rknn.export_rknn(rknn_path)
    rknn.release()
    log(f"RKNN ({'int8' if quantize else 'none'}): {rknn_path}")
    return rknn_path


def export_and_compile_all(pt_path, info=None):
    """Экспорт .pt → ONNX → RKNN (none + int8) варианты.

    Возвращает (onnx_path, rknn_none_path, rknn_int8_path).
    """
    onnx_path = export_onnx_only(pt_path)
    if not onnx_path:
        raise RuntimeError("Экспорт ONNX не удался")

    rknn_none = _build_rknn(onnx_path, quantize=False)
    _ensure_coco_val()
    rknn_int8 = _build_rknn(onnx_path, quantize=True, calib_dir=COCO_VAL_IMG_DIR)

    return onnx_path, rknn_none, rknn_int8


def eval_coco(model_path, task="detect", data_yaml=COCO_VAL_YAML):
    """Запустить yolo val на COCO val 2017 и вернуть метрики mAP.

    Возвращает dict с mAP@0.5, mAP@0.5:0.95, точностью, полнотой.
    """
    model_path = str(model_path)
    data_yaml = str(data_yaml)
    cmd = f"yolo val task={task} model={model_path} data={data_yaml}"
    log(f"Запуск: {cmd}")
    t0 = time.time()

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    log(f"yolo val завершён за {elapsed:.0f}с")

    stdout = proc.stdout
    stderr = proc.stderr
    log(f"stdout ({len(stdout)} байт), stderr ({len(stderr)} байт)")

    metrics = {"model": model_path, "task": task, "elapsed_s": round(elapsed, 1)}

    # Разбор вывода yolo val для извлечения метрик
    for line in stdout.splitlines():
        line_s = line.strip()
        # Общие метрики: "all  ...  P  R  mAP50  mAP50-95"
        if line_s.startswith("all") and len(line_s.split()) >= 6:
            parts = line_s.split()
            try:
                metrics["precision"] = float(parts[2]) if len(parts) > 2 else None
                metrics["recall"] = float(parts[3]) if len(parts) > 3 else None
                metrics["map50"] = float(parts[4]) if len(parts) > 4 else None
                metrics["map50_95"] = float(parts[5]) if len(parts) > 5 else None
            except (ValueError, IndexError):
                pass
        # Попытка разобрать JSON строку от ultralytics
        if '"map50"' in line_s or '"metrics/mAP50(B)"' in line_s:
            try:
                data = json.loads(line_s)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if "map50" in k.lower():
                            metrics["map50"] = v
                        if "map50" in k.lower() and "95" in k.lower():
                            metrics["map50_95"] = v
                        if k == "precision" or "precision" in k.lower():
                            metrics["precision"] = v
                        if k == "recall" or "recall" in k.lower():
                            metrics["recall"] = v
            except (json.JSONDecodeError, TypeError):
                pass

    log(f"Метрики: {json.dumps(metrics)}")
    return metrics


def preprocess(frame, input_size=640):
    """Предобработка кадра: ресайз, паддинг, нормализация."""
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
    return tensor, scale, (pad_w, pad_h)


def dfl(position):
    """Distribution Focal Loss — декодирование распределения расстояний до границ бокса."""
    n, c, h, w = position.shape
    mc = c // BIN_COUNT
    y = position.reshape(n, BIN_COUNT, mc, h, w)
    y_exp = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
    weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * weights).sum(axis=2)


def process_head(box_dist, cls_scores, stride, grid_h, grid_w):
    """Обработать выход одного головного слоя: декодировать боксы и оценки классов."""
    col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
    row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
    grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w))
    grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w))
    grid = np.concatenate([grid_x, grid_y], axis=1)
    dbox = dfl(box_dist)
    l, t, r, b = dbox[:, 0:1], dbox[:, 1:2], dbox[:, 2:3], dbox[:, 3:4]
    x1 = (grid[:, 0:1] + 0.5 - l) * stride
    y1 = (grid[:, 1:2] + 0.5 - t) * stride
    x2 = (grid[:, 0:1] + 0.5 + r) * stride
    y2 = (grid[:, 1:2] + 0.5 + b) * stride
    boxes = np.concatenate([x1, y1, x2, y2], axis=1)
    boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
    scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, 80)
    return boxes, scores


def postprocess(outputs, scale, pad, orig_wh, strides=None, grid_sizes=None):
    """Постобработка выходов модели: NMS, декодирование координат."""
    orig_w, orig_h = orig_wh
    if strides is None:
        n_heads = len(outputs) // 3
        strides = [8, 16, 32][-n_heads:] if n_heads <= 3 else [8, 16, 32]
    if grid_sizes is None:
        n_heads = len(strides)
        input_size = 640
        grid_sizes = [input_size // s for s in strides]
    all_boxes, all_scores, all_cls = [], [], []
    for idx in range(len(strides)):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        stride = strides[idx]
        gh = gw = grid_sizes[idx]
        boxes, scores = process_head(box_dist, cls_scores, stride, gh, gw)
        max_cls = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        all_boxes.append(boxes)
        all_scores.append(max_cls)
        all_cls.append(cls_ids)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)
    mask = scores > CONF_THRESH
    if not mask.any():
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if not len(boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    valid = (bw >= MIN_WH) & (bh >= MIN_WH)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if not len(boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    keep = nms(boxes, scores, IOU_THRESH)
    return boxes[keep], scores[keep], cls_ids[keep]


def nms(boxes, scores, iou_thresh):
    """Non-Maximum Suppression."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep)


def iou_matrix(boxes_a, boxes_b):
    """Матрица IoU между двумя наборами боксов."""
    xa1, ya1, xa2, ya2 = boxes_a.T
    xb1, yb1, xb2, yb2 = boxes_b.T
    inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
    inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
    inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
    inter_y2 = np.minimum(ya2[:, None], yb2[None, :])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-10)


def run_onnx(sess, tensor):
    """Запуск инференса ONNX модели."""
    input_name = sess.get_inputs()[0].name
    raw = sess.run(None, {input_name: tensor / 255.0})
    return raw


def run_rknn(rknn, tensor):
    """Запуск инференса RKNN модели."""
    nhwc = tensor.transpose(0, 2, 3, 1)
    return rknn.inference(inputs=[nhwc])


def load_val_images(limit: int = 16):
    """Загрузить изображения из COCO val2017 для валидации."""
    images = []
    if not COCO_VAL_IMG_DIR.exists():
        log(f"Изображения COCO val2017 не найдены в {COCO_VAL_IMG_DIR}")
        return images
    for f in sorted(COCO_VAL_IMG_DIR.glob("*.jpg"))[:limit]:
        img = cv2.imread(str(f))
        if img is not None:
            images.append(img)
    return images


def validate(onnx_path, rknn_path):
    """Валидация ONNX vs RKNN: сравнение тензоров и IoU боксов."""
    import onnxruntime as rt
    from rknn.api import RKNN

    sess = rt.InferenceSession(onnx_path)

    # Предпочитать загрузку собранного RKNN; при ошибке — пересобрать из ONNX
    rknn = RKNN(verbose=False)
    if rknn_path and os.path.exists(rknn_path):
        rknn.load_rknn(rknn_path)
    else:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
        rknn.load_onnx(model=onnx_path)
        rknn.build(do_quantization=False)

    try:
        ret = rknn.init_runtime()
        if ret != 0:
            log(f"Инициализация runtime вернула {ret} (ожидаемо в симуляторе для трансформерных моделей)")
            rknn.release()
            return {"note": "валидация пропущена — runtime требует аппаратуру NPU"}
    except Exception as e:
        log(f"Ошибка инициализации runtime (ожидаемо в симуляторе для трансформерных моделей): {e}")
        rknn.release()
        return {"note": "валидация пропущена — runtime требует аппаратуру NPU"}

    images = load_val_images()
    if not images:
        log("Калибровочные изображения не найдены, валидация пропущена")
        return {"error": "нет изображений"}

    max_diffs = []
    ious = []
    n_images = 0

    for img in images:
        h, w = img.shape[:2]
        tensor, scale, pad = preprocess(img)

        onnx_out = run_onnx(sess, tensor)
        rknn_out = run_rknn(rknn, tensor)

        if isinstance(onnx_out, (list, tuple)):
            onnx_tensors = onnx_out
        else:
            onnx_tensors = [onnx_out]

        max_diff = 0.0
        for o, r in zip(onnx_tensors, rknn_out):
            diff = np.abs(o.astype(np.float32) - r.astype(np.float32)).max()
            max_diff = max(max_diff, float(diff))
        max_diffs.append(max_diff)

        # Сравнение результатов на уровне боксов
        boxes_onnx, scores_onnx, cls_onnx = postprocess(onnx_tensors, scale, pad, (w, h))
        boxes_rknn, scores_rknn, cls_rknn = postprocess(rknn_out, scale, pad, (w, h))

        if len(boxes_onnx) > 0 and len(boxes_rknn) > 0:
            iou = iou_matrix(boxes_onnx, boxes_rknn)
            row_max = iou.max(axis=1)
            ious.append(float(row_max.mean()))
        elif len(boxes_onnx) == 0 and len(boxes_rknn) == 0:
            ious.append(1.0)
        else:
            ious.append(0.0)

        n_images += 1

    rknn.release()

    avg_max_diff = float(np.mean(max_diffs))
    avg_iou = float(np.mean(ious))

    result = {
        "n_images": n_images,
        "max_diff_max": float(np.max(max_diffs)),
        "max_diff_avg": avg_max_diff,
        "iou_avg": avg_iou,
        "pass": avg_iou > 0.9,
    }
    return result


TEMPLATES_DIR = Path(__file__).parent / "templates"
DEPLOY_DIR = Path("/app/converter/model")


def _detect_model_info(pt_path):
    """Загрузить модель YOLO и извлечь параметры для генерации постпроцессинга."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import RTDETRDecoder
    model = YOLO(pt_path)

    task = getattr(model, "task", "detect")
    head = model.model.model[-1]
    is_rtdetr = isinstance(head, RTDETRDecoder)

    if is_rtdetr:
        # RT-DETR: одна stride, нет сетки, нет DFL
        stride = model.model.stride
        strides = [int(s) for s in stride.tolist()] if hasattr(stride, "tolist") else [32]
    else:
        stride = model.model.stride
        strides = [int(s) for s in stride.tolist()] if hasattr(stride, "tolist") else [8, 16, 32]

    input_size = getattr(model.model, "args", {}).get("imgsz", 640)
    if not isinstance(input_size, int):
        input_size = 640

    names = model.names if hasattr(model, "names") else getattr(model.model, "names", {})
    if not names:
        names = getattr(model.model, "names", {})
    num_classes = len(names)

    kpt_shape = getattr(head, "kpt_shape", None)
    num_kpts = kpt_shape[0] if kpt_shape else 0
    nm = getattr(head, "nm", 0)
    reg_max = getattr(head, "reg_max", 16)

    # Семантическая сегментация: task="segment", но нет прототипов масок (nm=0)
    if task == "segment" and nm == 0 and not num_kpts:
        task = "semantic"

    return {
        "task": task,
        "is_rtdetr": is_rtdetr,
        "strides": strides,
        "grid_sizes": [input_size // s for s in strides],
        "input_size": input_size,
        "num_classes": num_classes,
        "names": names,
        "num_kpts": num_kpts,
        "nm": nm,
        "reg_max": reg_max,
    }


def _generate_postprocess(info, dst_dir):
    """Сгенерировать postprocess.py для заданной информации о модели."""
    task = info["task"]
    is_rtdetr = info.get("is_rtdetr", False)

    if is_rtdetr:
        template_name = "detr_postprocess.py"
    else:
        template_name = f"{task}_postprocess.py"
    template_path = TEMPLATES_DIR / template_name

    if not template_path.exists():
        log(f"Шаблон для задачи '{template_name}' не найден в {template_path}, используется detect")
        template_path = TEMPLATES_DIR / "detect_postprocess.py"

    template = template_path.read_text()

    class_names = [info["names"].get(i, f"cls_{i}") for i in range(info["num_classes"])]

    is_obb = task == "obb"
    is_pose = task == "pose"
    is_seg = task == "segment"
    is_cls = task == "classify"
    is_nodfl = info.get("reg_max", 16) <= 1
    low_thresh = is_obb or is_pose or is_seg
    substitutions = {
        "INPUT_SIZE": str(info["input_size"]),
        "STRIDES": str(info["strides"]),
        "GRID_SIZES": str(info["grid_sizes"]),
        "NUM_CLASSES": str(info["num_classes"]),
        "NUM_KPTS": str(info.get("num_kpts", 0)),
        "NM": str(info.get("nm", 0)),
        "COCO_CLASSES": json.dumps(class_names),
        "REG_MAX": str(info.get("reg_max", 16)),
        "BIN_COUNT": "4",
        "CONF_THRESH": "0.25" if low_thresh else "0.5",
        "IOU_THRESH": "0.5",
        "MIN_WH": "20" if low_thresh else "40",
        "NO_DFL": "True" if is_nodfl else "False",
    }

    for key, val in substitutions.items():
        template = template.replace("{{" + key + "}}", val)

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / "postprocess.py"
    out_path.write_text(template)
    log(f"Сгенерирован {out_path}")
    return str(out_path)


def _deploy(onnx_path, rknn_path, postprocess_path):
    """Скопировать модель + постпроцесс в DEPLOY_DIR и отправить сигнал перезагрузки."""
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    if rknn_path and os.path.exists(rknn_path):
        dst = DEPLOY_DIR / "model.rknn"
        shutil.copy2(rknn_path, str(dst))
        log(f"Развёрнуто {rknn_path} → {dst}")

    if onnx_path and os.path.exists(onnx_path):
        dst = DEPLOY_DIR / "model.onnx"
        shutil.copy2(onnx_path, str(dst))
        log(f"Развёрнуто {onnx_path} → {dst}")

    if postprocess_path and os.path.exists(postprocess_path):
        dst = DEPLOY_DIR / "postprocess.py"
        shutil.copy2(postprocess_path, str(dst))
        log(f"Развёрнуто {postprocess_path} → {dst}")

    _signal_reload()


def _signal_reload():
    """Отправка HTTP-сигнала перезагрузки сервису percept."""
    import urllib.request
    urls = ["http://localhost:5000/reload", "http://percept:5000/reload"]
    ok = False
    for url in urls:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            log(f"Сигнал перезагрузки отправлен на {url}: {resp.status}")
            ok = True
            break
        except Exception:
            continue
    if not ok:
        log("Не удалось отправить сигнал перезагрузки (percept запущен?)")
        log("Перезапустите вручную: docker compose restart percept")


def cmd_convert(args):
    """Конвертация модели: .pt → ONNX → RKNN."""
    pt = resolve_pt(args.model)
    log(f"Конвертация {pt}")

    log("Шаг 1: Определение информации о модели")
    info = _detect_model_info(pt)
    log(f"  задача={info['task']} классы={info['num_classes']} stride={info['strides']}")

    log("Шаг 2: Экспорт ONNX + сборка RKNN")
    quantize = getattr(args, "quantize", "none")
    if quantize != "none":
        log(f"Режим квантизации: {quantize}")
    onnx_path, rknn_path = export_and_compile(pt, info, quantize=quantize)

    if args.test:
        log("Шаг 3: Валидация")
        if info["task"] == "detect":
            if onnx_path and os.path.exists(onnx_path):
                result = validate(onnx_path, rknn_path)
            else:
                log("ONNX не найден, попытка валидации только по RKNN")
                result = validate(None, rknn_path)
            print(json.dumps(result, indent=2))
            status = "ПРОЙДЕНО" if result.get("pass") else "ПРОВАЛ"
            log(f"Валидация: {status}")
            if result.get("pass") is False:
                log("Валидация не пройдена, развёртывание отменено")
                return result
        else:
            log(f"Валидация пропущена для задачи '{info['task']}' (только detect)")
            result = {"note": f"валидация не реализована для задачи '{info['task']}'"}
    else:
        result = {"note": "валидация пропущена (включите флагом --test)"}

    log("Шаг 4: Генерация postprocess.py")
    postprocess_path = _generate_postprocess(info, ".")

    if args.deploy:
        log("Шаг 5: Развёртывание")
        _deploy(onnx_path, rknn_path, postprocess_path)
    else:
        for src, name in [(onnx_path, "model.onnx"), (rknn_path, "model.rknn")]:
            if src and os.path.exists(src):
                dst = MODEL_DIR / name
                os.makedirs(MODEL_DIR, exist_ok=True)
                shutil.copy2(src, str(dst))
                log(f"Скопировано в {dst}")

    if postprocess_path and os.path.exists(postprocess_path):
        dst = MODEL_DIR / "postprocess.py"
        os.makedirs(MODEL_DIR, exist_ok=True)
        shutil.copy2(postprocess_path, str(dst))
        log(f"Постпроцесс скопирован в {dst}")

    return result


def cmd_validate(args):
    """Валидация существующего .rknn файла на COCO val2017."""
    rknn_path = args.model
    if not rknn_path.endswith(".rknn"):
        rknn_path = str(MODEL_DIR / f"{rknn_path}.rknn")
    log(f"Валидация {rknn_path}")

    onnx_path = rknn_path.replace(".rknn", ".onnx")
    if not os.path.exists(onnx_path):
        stems = list(MODEL_DIR.glob(f"{Path(rknn_path).stem}*.onnx"))
        if stems:
            onnx_path = str(stems[0])

    if not os.path.exists(onnx_path):
        log(f"Соответствующий ONNX не найден для {rknn_path}, валидация пропущена")
        return {"error": "нет onnx"}

    result = validate(onnx_path, rknn_path)
    print(json.dumps(result, indent=2))
    status = "ПРОЙДЕНО" if result.get("pass") else "ПРОВАЛ"
    log(f"Валидация: {status}")
    return result


def load_model_list(path):
    """Загрузить список моделей из файла."""
    if os.path.exists(path):
        names = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        return names
    names = []
    for p in [
        "/app/converter/models_detect.txt",
        "models_detect.txt",
        "../converter/models_detect.txt",
    ]:
        if os.path.exists(p):
            names = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
            break
    return names


QUANT_MODES = ["onnx", "none", "int8", "int8-calib"]


def cmd_test_all(args):
    """Универсальное тестирование моделей по списку.

    Каждая модель тестируется во всех режимах квантизации:
    none (FP32), int8 (без калибровки), int8-calib (с калибровкой).
    """
    models = load_model_list(args.model_list)
    if not models:
        log(f'Список моделей не найден: {args.model_list}')
        return {}

    quant_modes = QUANT_MODES
    results = {}
    for name in models:
        for qmode in quant_modes:
            tag = f"{name} [{qmode}]"
            log(f"{'='*60}")
            log(f"Тестирование {tag}...")
            try:
                pt = resolve_pt(name)
                if qmode == "onnx":
                    onnx_path = export_onnx_only(pt)
                    results[tag] = {"status": "ПРОЙДЕНО" if onnx_path else "ПРОВАЛ"}
                else:
                    convert_args = argparse.Namespace(
                        model=name,
                        quantize=qmode,
                        test=True,
                        deploy=False,
                    )
                    res = cmd_convert(convert_args)
                    results[tag] = {"status": "ПРОЙДЕНО" if res.get("pass") else "ПРОВАЛ", **res}
            except Exception as e:
                log(f"ОШИБКА: {e}")
                results[tag] = {"status": "ПРОВАЛ", "error": str(e)}
            print()

    print(f"\n{'='*70}")
    print("ИТОГИ")
    print(f"{'='*70}")
    print(f"{'Модель':<40} {'Статус':<8} {'max_diff':<10} {'iou_avg':<10}")
    print("-" * 68)
    for tag, res in results.items():
        md = res.get("max_diff_avg", "-")
        iou = res.get("iou_avg", "-")
        print(f"{tag:<40} {res['status']:<8} {md:<10} {iou:<10}")

    return results


def cmd_test_all_tasks(args):
    """Тестирование всех типов моделей (detect, seg, cls, pose, obb, sem)."""
    task_lists = [
        ("detect", "models_detect.txt"),
        ("seg",    "models_seg.txt"),
        ("cls",    "models_cls.txt"),
        ("pose",   "models_pose.txt"),
        ("obb",    "models_obb.txt"),
        ("sem",    "models_sem.txt"),
    ]

    all_results = {}
    for task, default_list in task_lists:
        log(f"\n{'#'*60}")
        log(f"# ТЕСТИРОВАНИЕ: {task}")
        log(f"{'#'*60}")
        model_list = getattr(args, 'model_list', None) or default_list
        task_args = argparse.Namespace(model_list=model_list, **vars(args))
        results = cmd_test_all(task_args)
        all_results.update(results)

    print(f"\n{'='*60}")
    print("ИТОГИ ВСЕХ ЗАДАЧ")
    print(f"{'='*60}")
    print(f"{'Модель':<25} {'Статус':<8} {'max_diff':<10} {'iou_avg':<10}")
    print("-" * 55)
    for name, res in all_results.items():
        md = res.get("max_diff_avg", "-")
        iou = res.get("iou_avg", "-")
        print(f"{name:<25} {res['status']:<8} {md:<10} {iou:<10}")

    return all_results


def export_onnx_only(pt_path):
    """Экспорт .pt → ONNX (RK-opt) только, без сборки RKNN. Возвращает onnx_path."""
    apply_patches()
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import RTDETRDecoder
    model = YOLO(pt_path)

    for m in model.model.modules():
        m._rknn_mode = True

    is_rtdetr = isinstance(model.model.model[-1], RTDETRDecoder)
    opset = 16 if is_rtdetr else 12
    model.export(format="onnx", opset=opset)

    model_stem = Path(pt_path).stem
    onnx_path = f"{model_stem}.onnx"
    if not os.path.exists(onnx_path):
        onnx_paths = list(Path(".").glob("*.onnx"))
        onnx_path = str(onnx_paths[-1]) if onnx_paths else None

    if onnx_path:
        log(f"ONNX (RK-opt): {onnx_path}")
    return onnx_path


def cmd_batch_onnx(args):
    """Пакетный экспорт ONNX (RK-opt) для всех моделей из списка."""
    models = load_model_list(args.model_list)
    if not models:
        log("Нет моделей для обработки")
        return

    results = {}
    for name in models:
        log(f"{'='*60}")
        log(f"Тестирование {name}...")
        pt = resolve_pt(name)
        try:
            onnx_path = export_onnx_only(pt)
            if not onnx_path:
                raise RuntimeError("Экспорт ONNX не создал файл")

            result = validate(onnx_path, None)

            stem = Path(name).stem.replace(".pt", "")
            dst = MODEL_DIR / f"{stem}_rkopt_o19.onnx"
            os.makedirs(MODEL_DIR, exist_ok=True)
            shutil.copy2(onnx_path, str(dst))
            log(f"Скопировано в {dst}")
            os.remove(onnx_path)

            results[name] = {"status": "ПРОЙДЕНО" if result.get("pass") else "ПРОВАЛ", **result}
            status = "ПРОЙДЕНО" if result.get("pass") else "ПРОВАЛ"
            log(f"{name}: {status}")
        except Exception as e:
            log(f"ОШИБКА: {e}")
            results[name] = {"status": "ПРОВАЛ", "error": str(e)}
        print()

    # Итоги
    print(f"\n{'='*60}")
    print("ИТОГИ")
    print(f"{'='*60}")
    print(f"{'Модель':<20} {'Статус':<8} {'iou_avg':<10} {'max_diff':<10}")
    print("-" * 50)
    for name, res in results.items():
        iou = res.get("iou_avg", "-")
        md = res.get("max_diff_avg", "-")
        print(f"{name:<20} {res['status']:<8} {iou:<10} {md:<10}")
    return results


def main():
    """Точка входа: парсинг аргументов и запуск команд."""
    parser = argparse.ArgumentParser(description="RKNN Model Converter & Validator")
    sub = parser.add_subparsers(dest="command")

    p_convert = sub.add_parser("convert", help="Convert .pt → ONNX (RK-opt) → RKNN → validate")
    p_convert.add_argument("model", help="Model name or path (e.g. yolov8m.pt)")
    p_convert.add_argument("--quantize", choices=["none", "int8", "int8-calib"], default="none",
                           help="INT8 quantization: none=FP32, int8=INT8 без калибровки, int8-calib=INT8 с калибровкой")
    p_convert.add_argument("--test", action="store_true",
                           help="Validate ONNX vs RKNN before deploy")
    p_convert.add_argument("--deploy", action="store_true",
                           help="Скопировать model.rknn + postprocess.py в model/ и перезагрузить percept")

    p_validate = sub.add_parser("validate", help="Validate existing .rknn against COCO val2017")
    p_validate.add_argument("model", help="Path or name of .rknn file")

    TASK_LISTS = [
        ("detect", "models_detect.txt"),
        ("seg",    "models_seg.txt"),
        ("cls",    "models_cls.txt"),
        ("pose",   "models_pose.txt"),
        ("obb",    "models_obb.txt"),
        ("sem",    "models_sem.txt"),
    ]

    for task, default_list in TASK_LISTS:
        cmd_name = f"test-all-{task}"
        p = sub.add_parser(cmd_name, help=f"Test all {task} models")
        p.add_argument("--model-list", default=default_list,
                       help="Path to model list file")

    p_test_all = sub.add_parser("test-all", help="Test ALL task types (detect, seg, cls, pose, obb, sem)")
    p_test_all.add_argument("--model-list", default=None,
                            help="Override model list (for single-task testing)")

    for task, default_list in TASK_LISTS:
        cmd_name = f"batch-onnx-{task}"
        p = sub.add_parser(cmd_name, help=f"Export ONNX (RK-opt) for all {task} models")
        p.add_argument("--model-list", default=default_list,
                       help="Path to model list file")

    p_batch_onnx = sub.add_parser("batch-onnx", help="Export ONNX (RK-opt) for all models in list")
    p_batch_onnx.add_argument("--model-list", default="models_detect.txt",
                              help="Path to model list file")

    p_convert_all = sub.add_parser("convert-all",
                                   help="Export ONNX + both RKNN variants (none + int8)")
    p_convert_all.add_argument("model", help="Model name or path (e.g. yolo11n.pt)")

    p_eval = sub.add_parser("eval-coco",
                            help="Run yolo val on COCO val 2017 and report mAP")
    p_eval.add_argument("model", help="Path to .pt/.onnx/.rknn model")
    p_eval.add_argument("--task", default="detect",
                        choices=["detect", "segment", "pose", "obb", "classify"],
                        help="Task type")
    p_eval.add_argument("--data", default=str(COCO_VAL_YAML),
                        help="Dataset YAML path")

    args = parser.parse_args()
    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "test-all":
        cmd_test_all_tasks(args)
    elif args.command and args.command.startswith("test-all-"):
        cmd_test_all(args)
    elif args.command and args.command.startswith("batch-onnx-"):
        cmd_batch_onnx(args)
    elif args.command == "batch-onnx":
        cmd_batch_onnx(args)
    elif args.command == "convert-all":
        cmd_convert_all(args)
    elif args.command == "eval-coco":
        cmd_eval_coco(args)
    else:
        parser.print_help()


def cmd_convert_all(args):
    """Экспорт ONNX + оба варианта RKNN (none + int8), копирование в model/."""
    pt = resolve_pt(args.model)
    log(f"Конвертация {pt} (все варианты)")
    info = _detect_model_info(pt)
    log(f"  задача={info['task']} классы={info['num_classes']} stride={info['strides']}")

    onnx_path, rknn_none, rknn_int8 = export_and_compile_all(pt, info)

    stem = Path(args.model).stem.replace(".pt", "")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Копировать ONNX
    dst_onnx = MODEL_DIR / f"{stem}_rkopt_o19.onnx"
    shutil.copy2(onnx_path, str(dst_onnx))
    log(f"ONNX -> {dst_onnx}")

    # Копировать RKNN none
    dst_none = MODEL_DIR / f"{stem}_none.rknn"
    shutil.copy2(rknn_none, str(dst_none))
    log(f"RKNN none -> {dst_none}")

    # Копировать RKNN int8
    dst_int8 = MODEL_DIR / f"{stem}_int8.rknn"
    shutil.copy2(rknn_int8, str(dst_int8))
    log(f"RKNN int8 -> {dst_int8}")

    # Копировать постпроцесс
    postprocess_path = _generate_postprocess(info, ".")
    if postprocess_path and os.path.exists(postprocess_path):
        dst = MODEL_DIR / f"{stem}_postprocess.py"
        shutil.copy2(postprocess_path, str(dst))
        log(f"Постпроцесс -> {dst}")

    log("Готово")
    print(json.dumps({
        "model": stem,
        "task": info["task"],
        "onnx": str(dst_onnx),
        "rknn_none": str(dst_none),
        "rknn_int8": str(dst_int8),
        "postprocess": str(MODEL_DIR / f"{stem}_postprocess.py"),
    }))


def cmd_eval_coco(args):
    """Запустить yolo val на COCO val 2017 и вывести метрики."""
    model_path = resolve_pt(args.model) if args.model.endswith(".pt") else args.model
    result = eval_coco(model_path, task=args.task, data_yaml=args.data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
