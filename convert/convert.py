#!/usr/bin/env python
"""RKNN Model Converter with caching.

Unified command: convert MODEL --fp16|--int8|--int8-calib|--onnx

Cache structure: cache/{model}/{model}.pt|onnx|_fp16.rknn|_int8.rknn|_int8-calib.rknn
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# convert.py использует `from patches import rknn_patch` (lazy, в apply_patches).
# Добавляем свою директорию в sys.path, чтобы модуль импортировался и как
# convert.convert (из benchmark/run_3way_eval.py), и как скрипт.
_CONVERT_DIR = Path(__file__).resolve().parent
if str(_CONVERT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONVERT_DIR))

# Подготовка данных: setup_datasets из корня проекта гарантирует, что
# мини-датасеты ultralytics (coco128 и т.п.) качаются в datasets/, а не в кэш.
_ROOT_DIR = _CONVERT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
from setup_datasets import ensure_mini_dataset, ensure_ultralytics_dir  # noqa: E402

ensure_ultralytics_dir()

CACHE_DIR = Path("/app/cache")
MODEL_DIR = Path("/app/convert/model")
WEIGHTS_DIR = Path("/app/convert/weights")

TEMPLATES_DIR = Path(__file__).parent / "templates"

COCO_VAL_DIR = Path("/app/datasets/coco_val2017")
COCO_VAL_IMG_DIR = COCO_VAL_DIR / "images" / "val2017"
COCO_VAL_YAML = COCO_VAL_DIR / "coco_val.yaml"
DOTA_VAL_DIR = Path("/app/datasets/dotav1")
DOTA_VAL_IMG_DIR = DOTA_VAL_DIR / "images" / "val"
CITYSCAPES_VAL_DIR = Path("/app/datasets/cityscapes")
CITYSCAPES_VAL_IMG_DIR = CITYSCAPES_VAL_DIR / "images" / "val"
IMAGENET_VAL_DIR = Path("/app/datasets/imagenet")
IMAGENET_VAL_IMG_DIR = IMAGENET_VAL_DIR / "val"

ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
DOWNLOAD_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Quantize modes
# ---------------------------------------------------------------------------

QUANT_FLAGS = ("fp16", "int8", "int8-calib", "onnx")


def _rknn_suffix(quantize: str) -> str:
    """Суффикс файла RKNN для типа квантизации."""
    return f"_{quantize}.rknn"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[CONVERT] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Ultralytics patches
# ---------------------------------------------------------------------------

def apply_patches():
    from patches import rknn_patch
    rknn_patch.apply()
    log("Патчи RKNN применены")


# ---------------------------------------------------------------------------
# Pretrained model detection
# ---------------------------------------------------------------------------

def _load_ultralytics_models() -> set[str]:
    """Загрузить множество имён pretrained моделей ultralytics."""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
        return {Path(n).stem for n in GITHUB_ASSETS_NAMES}
    except Exception:
        pass
    # Fallback: загрузить из файлов списков моделей
    names = set()
    for p in Path(__file__).parent.glob("models_*.txt"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    return names


ULTRALYTICS_MODELS = _load_ultralytics_models()


def _is_pretrained(name: str) -> bool:
    """Проверить, является ли имя pretrained моделью ultralytics."""
    return name in ULTRALYTICS_MODELS


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def _hash_file(path: Path, chunk_size: int = 8192) -> str:
    """SHA256 хеш файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

def _cache_model_dir(model_name: str) -> Path:
    """Путь к директории кеша модели."""
    return CACHE_DIR / model_name


def _target_file(model_name: str, quantize: str) -> Path:
    """Путь к целевому файлу в кеше."""
    d = _cache_model_dir(model_name)
    if quantize == "onnx":
        return d / f"{model_name}.onnx"
    return d / f"{model_name}{_rknn_suffix(quantize)}"


def _pt_file(model_name: str) -> Path:
    """Путь к .pt файлу в кеше."""
    return _cache_model_dir(model_name) / f"{model_name}.pt"


def _onnx_file(model_name: str) -> Path:
    """Путь к .onnx файлу в кеше."""
    return _cache_model_dir(model_name) / f"{model_name}.onnx"


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

def _check_collision(model_name: str, model_input: Path | str) -> None:
    """Проверить коллизии имён для кастомных файлов.

    Raises RuntimeError при коллизии.
    """
    if _is_pretrained(model_name):
        raise RuntimeError(
            f"Имя кастомной модели '{model_name}' совпадает с pretrained моделью Ultralytics. "
            f"Переименуйте файл перед конвертацией."
        )

    cache_dir = _cache_model_dir(model_name)
    if not cache_dir.exists():
        return

    # Проверяем бинарное совпадение с существующим кешем
    model_input = Path(model_input)
    if model_input.exists():
        existing_pt = cache_dir / f"{model_name}.pt"
        existing_onnx = cache_dir / f"{model_name}.onnx"
        existing_files = list(cache_dir.glob(f"{model_name}*.rknn")) + \
                         ([existing_onnx] if existing_onnx.exists() else []) + \
                         ([existing_pt] if existing_pt.exists() else [])

        if existing_files:
            # Сравниваем с первым найденным файлом
            ref = existing_files[0]
            if model_input.suffix == ref.suffix:
                if _hash_file(model_input) != _hash_file(ref):
                    raise RuntimeError(
                        f"Кеш содержит другую модель с именем '{model_name}' в {cache_dir}. "
                        f"Удалите {cache_dir} или переименуйте файл."
                    )
                # Файлы совпадают — ок
            else:
                raise RuntimeError(
                    f"Кеш содержит файлы модели '{model_name}' с другим расширением. "
                    f"Удалите {cache_dir} или переименуйте файл."
                )


# ---------------------------------------------------------------------------
# Cache chain logic
# ---------------------------------------------------------------------------

# Цепочка: .pt → .onnx → _fp16.rknn / _int8.rknn / _int8-calib.rknn
CHAIN = ["pt", "onnx", "fp16", "int8", "int8-calib"]


def _chain_index(quantize: str) -> int:
    """Индекс целевого формата в цепочке."""
    if quantize == "onnx":
        return 1
    return QUANT_FLAGS.index(quantize) + 2  # fp16=2, int8=3, int8-calib=4


def _find_nearest(model_name: str, quantize: str) -> tuple[str, Path] | None:
    """Найти ближайший промежуточный файл в кеше.

    Возвращает (формат, путь) или None.
    """
    cache_dir = _cache_model_dir(model_name)
    if not cache_dir.exists():
        return None

    target_idx = _chain_index(quantize)

    # Идём от конца к началу, ищем первый существующий файл
    for i in range(target_idx - 1, -1, -1):
        fmt = CHAIN[i]
        if fmt == "pt":
            p = _pt_file(model_name)
        elif fmt == "onnx":
            p = _onnx_file(model_name)
        else:
            p = _target_file(model_name, fmt)
        if p.exists():
            return (fmt, p)
    return None


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

def _dataset_name(task: str) -> str:
    return {
        "detect": "COCO val2017",
        "segment": "COCO val2017",
        "classify": "ImageNet val",
        "pose": "COCO val2017",
        "obb": "DOTAv1",
        "semantic": "Cityscapes",
    }.get(task, "COCO val2017")


def _ensure_coco_labels(task: str) -> None:
    """Сгенерировать yolo-labels (detect/pose/seg) для эталонного .val().

    Нужны только для ad-hoc .val() ultralytics (eval/калибровка читают JSON напрямую).
    Генерация идемпотентна — при готовых labels мгновенный skip.
    """
    if task not in ("detect", "segment", "pose"):
        return
    try:
        from setup_coco_labels import ensure_coco_labels
        ensure_coco_labels(task)
    except Exception as e:
        log(f"Предупреждение: не удалось сгенерировать labels для {task}: {e}")


def _ensure_dataset(task: str = "detect") -> Path:
    """Проверить наличие нужного датасета для калибровки.

    Если датасета нет — скачивает через setup_datasets.py.
    Возвращает корень датасета (path в data YAML).
    """
    import subprocess as _sp

    SCRIPT = Path(__file__).parent / "setup_datasets.py"
    DATA_DIR = "/app/datasets"

    flag_map = {
        "obb": "--dota",
        "semantic": "--cityscapes",
        "classify": "--imagenet",
    }
    flag = flag_map.get(task, "--coco")

    # Где лежат изображения (для проверки наличия)
    img_dir_map = {
        "obb": DOTA_VAL_IMG_DIR,
        "semantic": CITYSCAPES_VAL_IMG_DIR,
        "classify": IMAGENET_VAL_IMG_DIR,
    }
    img_dir = img_dir_map.get(task, COCO_VAL_IMG_DIR)

    # Корень датасета (path в YAML)
    # COCO/DOTA/Cityscapes: img_dir = .../images/val* → root = parent.parent
    # ImageNet: img_dir = .../val → root = parent
    root_dir_map = {
        "obb": DOTA_VAL_DIR,
        "semantic": CITYSCAPES_VAL_DIR,
        "classify": IMAGENET_VAL_DIR,
    }
    root_dir = root_dir_map.get(task, COCO_VAL_DIR)

    if img_dir.exists():
        n = sum(1 for _ in img_dir.iterdir()) if img_dir.is_dir() else 0
        if n > 0:
            log(f"Датасет для {task} уже есть: {n} файлов в {img_dir}")
            _ensure_coco_labels(task)
            return root_dir

    log(f"Датасет для {task} не найден, скачиваю...")
    try:
        _sp.run(
            [sys.executable, str(SCRIPT), "--data-dir", DATA_DIR, flag],
            check=True, timeout=7200,
        )
    except _sp.CalledProcessError as e:
        raise RuntimeError(
            f"Не удалось скачать датасет для {task}: "
            f"python3 {SCRIPT} --data-dir {DATA_DIR} {flag}\n"
            f"Код возврата: {e.returncode}"
        ) from e

    # Повторная проверка после скачивания
    if img_dir.exists() and any(img_dir.iterdir()):
        _ensure_coco_labels(task)
        return root_dir

    raise RuntimeError(
        f"Датасет для {task} не найден после скачивания в {img_dir}. "
        f"Проверьте: python3 {SCRIPT} --data-dir {DATA_DIR} {flag}"
    )


# ---------------------------------------------------------------------------
# Calibration YAML generation
# ---------------------------------------------------------------------------

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


def _make_calib_yaml(task: str = "detect") -> str:
    data_dir = _ensure_dataset(task)
    if task == "obb":
        return _make_dota_calib_yaml(data_dir)
    elif task == "semantic":
        return _make_cityscapes_calib_yaml(data_dir)
    elif task == "classify":
        return _make_imagenet_calib_yaml(data_dir)
    else:
        return _make_coco_calib_yaml(data_dir, task)


def _make_coco_calib_yaml(data_dir: Path, task: str = "detect") -> str:
    n = len(list(COCO_VAL_IMG_DIR.glob("*.jpg")))
    log(f"Калибровочный набор: COCO val2017 — {n} изображений")

    lines = [
        f"path: {data_dir}",
        "train: images/val2017",
        "val: images/val2017",
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
    n = len(list(DOTA_VAL_IMG_DIR.glob("*.png")))
    log(f"Калибровочный набор: DOTAv1 val — {n} изображений")

    lines = [
        f"path: {data_dir}",
        "train: images/val",
        "val: images/val",
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
        "  13: helicopter",
        "  14: container-crane",
    ]

    calib_yaml = Path("/tmp/calib.yaml")
    calib_yaml.write_text("\n".join(lines) + "\n")
    log(f"Калибровочный YAML: {calib_yaml} (задача=obb)")
    return str(calib_yaml)


def _make_cityscapes_calib_yaml(data_dir: Path) -> str:
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
        "train: images/val",
        "val: images/val",
        f"nc: {len(CITYSCAPES_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(CITYSCAPES_CLASSES):
        lines.append(f"  {i}: {name}")

    calib_yaml = Path("/tmp/calib.yaml")
    calib_yaml.write_text("\n".join(lines) + "\n")
    log(f"Калибровочный YAML: {calib_yaml} (задача=semantic)")
    return str(calib_yaml)


def _make_imagenet_calib_yaml(data_dir: Path) -> str:
    val_dir = data_dir / "val"
    n_classes = len([d for d in val_dir.iterdir() if d.is_dir()]) if val_dir.exists() else 0
    n_images = sum(1 for _ in val_dir.rglob("*.JPEG")) if val_dir.exists() else 0
    log(f"Калибровочный набор: ImageNet val — {n_images} изображений, {n_classes} классов")

    lines = [
        f"path: {data_dir}",
        "train: val",
        "val: val",
        f"nc: {n_classes}",
        "names:",
    ]

    calib_yaml = Path("/tmp/calib.yaml")
    calib_yaml.write_text("\n".join(lines) + "\n")
    log(f"Калибровочный YAML: {calib_yaml} (задача=classify)")
    return str(calib_yaml)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(url, dst):
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


# ---------------------------------------------------------------------------
# PT resolution (with cache)
# ---------------------------------------------------------------------------

def resolve_pt(name: str) -> str:
    """Найти или скачать .pt файл модели.

    Для pretrained моделей кеширует в cache/{name}/{name}.pt.
    """
    # Если это путь к файлу — вернуть как есть
    p = Path(name)
    if p.exists():
        return str(p)

    # Если name без расширения — добавить .pt
    if not name.endswith(".pt"):
        name = f"{name}.pt"

    stem = Path(name).stem

    # Проверить кеш
    cached = _pt_file(stem)
    if cached.exists():
        log(f"[PT] из кеша: {cached}")
        return str(cached)

    # Проверить WEIGHTS_DIR
    p = WEIGHTS_DIR / name
    if p.exists():
        # Скопировать в кеш
        cache_dir = _cache_model_dir(stem)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / name
        shutil.copy2(str(p), str(dst))
        log(f"[PT] скопирован из weights: {dst}")
        return str(dst)

    # Скачать pretrained
    if _is_pretrained(stem):
        cache_dir = _cache_model_dir(stem)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / name
        url = f"{ASSETS_URL}/{name}"
        log(f"[PT] скачивание {stem} из {url}")
        if _download(url, str(dst)):
            return str(dst)

    # Fallback: попробовать скачать в текущую директорию
    dst = Path(name)
    if not dst.exists():
        url = f"{ASSETS_URL}/{name}"
        log(f"[PT] скачивание {name} из {url}")
        if _download(url, str(dst)):
            return str(dst)

    return str(dst)


# ---------------------------------------------------------------------------
# Model info detection
# ---------------------------------------------------------------------------

def _detect_model_info(pt_path):
    """Загрузить модель YOLO и извлечь параметры."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import RTDETRDecoder
    model = YOLO(pt_path)

    task = getattr(model, "task", "detect")
    head = model.model.model[-1]
    is_rtdetr = isinstance(head, RTDETRDecoder)

    if is_rtdetr:
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
        "head_type": type(head).__name__,
    }


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_onnx_only(pt_path):
    """Экспорт .pt → ONNX (RK-opt). Возвращает onnx_path."""
    apply_patches()
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import RTDETRDecoder
    model = YOLO(pt_path)

    for m in model.model.modules():
        m._rknn_mode = True
        # yolo26 end2end: fuse() иначе удаляет cv2/cv3 (one2many) и оставляет one2one —
        # RK-opt получил бы другие головы, чем std. Экспортируем всегда с one2many.
        if hasattr(m, "end2end"):
            m.end2end = False
    if hasattr(model, "end2end"):
        model.end2end = False

    is_rtdetr = isinstance(model.model.model[-1], RTDETRDecoder)
    opset = 16 if is_rtdetr else 12
    pt_dir = str(Path(pt_path).parent)
    model.export(format="onnx", opset=opset, project=pt_dir, name=".")

    model_stem = Path(pt_path).stem
    onnx_path = str(Path(pt_dir) / f"{model_stem}.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = None

    if onnx_path:
        log(f"ONNX (RK-opt): {onnx_path}")
    return onnx_path


def _calib_img_dir(data_root: Path) -> Path:
    """Найти каталог с изображениями внутри датасета для калибровки RKNN.

    Обходит рекурсивно (coco128: images/train2017, coco8-pose: images/train,
    imagenet: вложенные каталоги классов). Возвращает каталог первого .jpg.
    """
    root = Path(data_root)
    for jpg in root.rglob("*.jpg"):
        return jpg.parent
    for jpg in root.rglob("*.jpeg"):
        return jpg.parent
    for jpg in root.rglob("*.png"):
        return jpg.parent
    for jpg in root.rglob("*.bmp"):
        return jpg.parent
    raise RuntimeError(f"Нет изображений для калибровки в {root}")


def _build_rknn(onnx_path, quantize, calib_dir=None):
    """Собрать RKNN из ONNX файла.

    Сборка во временный каталог (не рядом с onnx!): когда onnx уже лежит в кеше,
    запись рядом дала бы путь, совпадающий с целевым rknn-файлом, и последующий
    copy2(src, dst) падал бы 'same file'. Плюс при падении сборки в кеше не
    остаётся обрубленного rknn, который потом приняли бы за валидный.
    """
    from rknn.api import RKNN

    onnx_abs = Path(onnx_path).resolve()
    stem = onnx_abs.stem
    suffix = "_int8.rknn" if quantize else "_fp16.rknn"
    tmp_dir = Path(tempfile.mkdtemp(prefix="rknn_build_"))
    rknn_path = str(tmp_dir / f"{stem}{suffix}")

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
                exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
                imgs = []
                for ext in exts:
                    imgs.extend(Path(calib_dir).glob(ext))
                for img_path in sorted(imgs)[:512]:
                    f.write(str(img_path) + "\n")
            ret = rknn.build(do_quantization=True, dataset=calib_txt)
        else:
            log("Сборка RKNN (int8, без калибровки)")
            ret = rknn.build(do_quantization=True)
    else:
        log("Сборка RKNN (fp16)")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError(f"rknn.build не удался: {ret}")

    rknn.export_rknn(rknn_path)
    rknn.release()
    log(f"RKNN ({'int8' if quantize else 'fp16'}): {rknn_path}")
    return rknn_path


# ---------------------------------------------------------------------------
# Dataset.txt logging
# ---------------------------------------------------------------------------

def _log_dataset_txt():
    dataset_candidates = [
        Path("/tmp/dataset.txt"),
        Path("dataset.txt"),
        Path("calib_dataset.txt"),
    ]
    dataset_candidates.extend(Path(".").glob("*dataset*.txt"))
    dataset_candidates.extend(Path("/tmp").glob("*dataset*.txt"))

    found = False
    for ds_path in dataset_candidates:
        if ds_path.exists() and ds_path.stat().st_size > 0:
            try:
                lines = ds_path.read_text().strip().splitlines()
                total = len(lines)
                log(f"=== DATASET.TXT ({ds_path}): {total} изображений ===")
                show_lines = lines[:10]
                if total > 15:
                    show_lines.append(f"  ... ({total - 15} изображений пропущено) ...")
                    show_lines.extend(lines[-5:])
                elif total > 10:
                    show_lines.extend(lines[10:])
                for line in show_lines:
                    log(f"  {line}")
                log("=== КОНЕЦ DATASET.TXT ===")
                found = True
                break
            except Exception as e:
                log(f"Ошибка чтения {ds_path}: {e}")

    if not found:
        log("ВНИМАНИЕ: dataset.txt не найден!")


# ---------------------------------------------------------------------------
# Postprocess generation
# ---------------------------------------------------------------------------

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
        log(f"Шаблон '{template_name}' не найден, используется detect")
        template_path = TEMPLATES_DIR / "detect_postprocess.py"

    template = template_path.read_text()

    class_names = [info["names"].get(i, f"cls_{i}") for i in range(info["num_classes"])]

    is_obb = task == "obb"
    is_pose = task == "pose"
    is_seg = task == "segment"
    is_cls = task == "classify"
    is_nodfl = info.get("reg_max", 16) <= 1
    # OBB26 (yolo26) отдаёт angle в виде RAW (без sigmoid), OBB (yolo11) — через
    # (sigmoid(x)-0.25)*pi. Это важно только для RK-opt per-stride ветки.
    is_obb26 = info.get("head_type") == "OBB26"
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
        "CONF_THRESH": "0.25",
        "IOU_THRESH": "0.7",
        "NO_DFL": "True" if is_nodfl else "False",
        "ANGLE_SIGMOID": "False" if is_obb26 else "True",
    }

    for key, val in substitutions.items():
        template = template.replace("{{" + key + "}}", val)

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / "postprocess.py"
    out_path.write_text(template)
    log(f"Сгенерирован {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Deploy to model/
# ---------------------------------------------------------------------------

def _deploy_to_model(model_name: str, quantize: str, info: dict) -> None:
    """Скопировать файлы из кеша в model/ и сгенерировать postprocess.py."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Копировать модель
    if quantize == "onnx":
        src = _onnx_file(model_name)
        dst = MODEL_DIR / "model.onnx"
    else:
        src = _target_file(model_name, quantize)
        dst = MODEL_DIR / "model.rknn"

    if src.exists():
        shutil.copy2(str(src), str(dst))
        log(f"Развёрнуто {src} → {dst}")

    # Скопировать .onnx (для ONNX CPU mode)
    if quantize == "onnx":
        onnx_src = _onnx_file(model_name)
        if onnx_src.exists():
            shutil.copy2(str(onnx_src), str(MODEL_DIR / "model.onnx"))

    # Генерация postprocess.py
    if info:
        postprocess_path = _generate_postprocess(info, ".")
        if postprocess_path and os.path.exists(postprocess_path):
            dst_pp = MODEL_DIR / "postprocess.py"
            shutil.copy2(postprocess_path, str(dst_pp))
            log(f"Постпроцесс → {dst_pp}")

    # Записать model_name.txt и quantize.txt
    (MODEL_DIR / "model_name.txt").write_text(model_name)
    (MODEL_DIR / "quantize.txt").write_text(quantize)
    log(f"model_name={model_name}, quantize={quantize}")

    _signal_reload()


def _signal_reload():
    import urllib.request
    urls = ["http://localhost:5000/reload", "http://percept:5000/reload"]
    for url in urls:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            log(f"Сигнал перезагрузки: {url}: {resp.status}")
            return
        except Exception:
            continue
    log("Не удалось отправить сигнал перезагрузки (percept запущен?)")


# ---------------------------------------------------------------------------
# Shared utilities for benchmark validation
# ---------------------------------------------------------------------------

def preprocess(frame, input_size=640):
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_left = round(pad_w / 2 - 0.1)
    pad_top = round(pad_h / 2 - 0.1)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
    return tensor, scale, (pad_w, pad_h)


def run_onnx(sess, tensor):
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: tensor / 255.0})


def run_rknn(rknn, tensor):
    nhwc = tensor.transpose(0, 2, 3, 1)
    return rknn.inference(inputs=[nhwc])


# ---------------------------------------------------------------------------
# List cache
# ---------------------------------------------------------------------------

def cmd_list_cache():
    """Показать содержимое кеша."""
    if not CACHE_DIR.exists():
        log("Кеш пуст")
        return

    models = sorted(d.name for d in CACHE_DIR.iterdir() if d.is_dir())
    if not models:
        log("Кеш пуст")
        return

    print(f"{'Модель':<25} {'Файлы'}")
    print("-" * 70)
    for model in models:
        d = CACHE_DIR / model
        files = sorted(f.name for f in d.iterdir() if f.is_file())
        sizes = []
        for f in d.iterdir():
            if f.is_file():
                sz = f.stat().st_size
                if sz > 1024 * 1024:
                    sizes.append(f"{sz / 1024 / 1024:.1f}MB")
                else:
                    sizes.append(f"{sz / 1024:.0f}KB")
        pairs = [f"{n} ({s})" for n, s in zip(files, sizes)]
        print(f"{model:<25} {', '.join(pairs)}")

    total_size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
    print(f"\nВсего: {len(models)} моделей, {total_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# Main convert function
# ---------------------------------------------------------------------------

def run_convert(model_input: str, quantize: str, info: dict | None = None) -> Path:
    """Единая точка входа: конвертация модели с кешированием.

    Принимает имя модели (pretrained) или путь к файлу (.pt/.onnx/.rknn).
    Возвращает путь к целевому файлу в кеше.
    """
    model_input_path = Path(model_input)

    # Определяем имя модели
    if model_input_path.is_absolute() or model_input_path.suffix in (".pt", ".onnx", ".rknn"):
        # Кастомный файл
        model_name = model_input_path.stem
        if model_input_path.suffix == ".rknn":
            model_name = model_input_path.stem  # "yolo26n-seg_int8" → need to strip suffix
            # Попробовать найти имя без суффикса квантизации
            for q in QUANT_FLAGS:
                suffix = _rknn_suffix(q)
                if model_name.endswith(suffix):
                    model_name = model_name[:-len(suffix)]
                    break
    else:
        # Имя модели (pretrained)
        model_name = model_input_path.stem
        if not model_name:
            model_name = str(model_input_path)

    log(f"Модель: {model_name}, квантизация: {quantize}")

    # Создать директорию кеша
    cache_dir = _cache_model_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Проверка коллизий для кастомных файлов
    if model_input_path.exists():
        _check_collision(model_name, model_input_path)

    # Проверка целевого файла в кеше
    target = _target_file(model_name, quantize)
    if target.exists():
        log(f"Из кеша: {target}")
        return target

    # ONNX-only режим
    if quantize == "onnx":
        # Ищем .pt в кеше или скачиваем
        pt = _pt_file(model_name)
        if not pt.exists():
            resolved = Path(resolve_pt(model_name))
            if resolved.exists():
                if resolved != _pt_file(model_name):
                    shutil.copy2(str(resolved), str(_pt_file(model_name)))
                pt = _pt_file(model_name)
            else:
                pt = resolved

        if not pt.exists():
            raise RuntimeError(f".pt файл не найден для {model_name}")

        log("Экспорт ONNX (RK-opt)")
        onnx_path = export_onnx_only(str(pt))
        if onnx_path:
            if Path(onnx_path) != target:
                shutil.copy2(onnx_path, str(target))
            log(f"ONNX сохранён в кеш: {target}")
        return target

    # RKNN режимы: fp16, int8, int8-calib
    # Ищем ближайшее промежуточное звено
    nearest = _find_nearest(model_name, quantize)
    if nearest:
        fmt, src = nearest
        log(f"Промежуточное звено: {fmt} ({src})")

        if fmt == "pt":
            # Есть .pt → нужен .onnx
            log("Экспорт ONNX из .pt")
            onnx_path = export_onnx_only(str(src))
            if onnx_path:
                onnx_dst = _onnx_file(model_name)
                if Path(onnx_path) != onnx_dst:
                    shutil.copy2(onnx_path, str(onnx_dst))
                log(f"ONNX → кеш: {onnx_dst}")
            else:
                raise RuntimeError("Экспорт ONNX не удался")
            src = onnx_dst

        if fmt in ("pt", "onnx"):
            # Есть .onnx → собираем RKNN
            onnx_src = _onnx_file(model_name) if fmt == "pt" else src
            info_local = info or _detect_model_info(str(_pt_file(model_name)) if _pt_file(model_name).exists() else str(onnx_src))

            if quantize == "int8-calib":
                task = info_local.get("task", "detect")
                calib_dir = _calib_img_dir(_ensure_dataset(task))
                log(f"Сборка RKNN int8-calib: задача={task}, датасет={calib_dir}")
                rknn_path = _build_rknn(str(onnx_src), quantize=True, calib_dir=calib_dir)
                shutil.copy2(rknn_path, str(target))
            elif quantize == "int8":
                task = info_local.get("task", "detect")
                calib_dir = _calib_img_dir(ensure_mini_dataset(task))
                log(f"Сборка RKNN int8 (без калибровки): задача={task}, мини-датасет={calib_dir}")
                rknn_path = _build_rknn(str(onnx_src), quantize=True, calib_dir=calib_dir)
                shutil.copy2(rknn_path, str(target))
            else:
                # fp16
                rknn_path = _build_rknn(str(onnx_src), quantize=False)
                shutil.copy2(rknn_path, str(target))

            log(f"RKNN → кеш: {target}")
            return target

        if fmt.startswith("int8") or fmt == "fp16":
            # Есть другой RKNN → конвертируем через ONNX
            # Нужен .onnx
            onnx_src = _onnx_file(model_name)
            if not onnx_src.exists():
                raise RuntimeError(f"ONNX не найден для {model_name}, необходим для конвертации из {fmt}")

            info_local = info or {}
            if quantize == "int8-calib":
                task = info_local.get("task", "detect")
                calib_dir = _calib_img_dir(_ensure_dataset(task))
                log(f"Сборка RKNN int8-calib из ONNX: датасет={calib_dir}")
                rknn_path = _build_rknn(str(onnx_src), quantize=True, calib_dir=calib_dir)
                shutil.copy2(rknn_path, str(target))
            elif quantize == "int8":
                task = info_local.get("task", "detect")
                calib_dir = _calib_img_dir(ensure_mini_dataset(task))
                log(f"Сборка RKNN int8 из ONNX: мини-датасет={calib_dir}")
                rknn_path = _build_rknn(str(onnx_src), quantize=True, calib_dir=calib_dir)
                shutil.copy2(rknn_path, str(target))
            else:
                rknn_path = _build_rknn(str(onnx_src), quantize=False)
                shutil.copy2(rknn_path, str(target))

            log(f"RKNN → кеш: {target}")
            return target

    # Полная конвертация: ничего в кеше
    log("Полная конвертация")

    # Получаем .pt
    pt = _pt_file(model_name)
    if not pt.exists():
        # Попробовать скачать/скопировать
        resolved = Path(resolve_pt(model_name))
        if resolved.exists():
            if resolved != pt:
                shutil.copy2(str(resolved), str(pt))
        else:
            raise RuntimeError(f".pt файл не найден для {model_name}")

    # Определяем info
    if info is None:
        info = _detect_model_info(str(pt))
    log(f"  задача={info['task']} классы={info['num_classes']} stride={info['strides']}")

    # Экспорт ONNX
    log("Экспорт ONNX")
    onnx_path = export_onnx_only(str(pt))
    if onnx_path:
        onnx_dst = _onnx_file(model_name)
        if Path(onnx_path) != onnx_dst:
            shutil.copy2(onnx_path, str(onnx_dst))
        log(f"ONNX → кеш: {onnx_dst}")
    else:
        raise RuntimeError("Экспорт ONNX не удался")

    # Сборка RKNN
    if quantize == "int8-calib":
        task = info.get("task", "detect")
        calib_dir = _calib_img_dir(_ensure_dataset(task))
        log(f"Сборка RKNN int8-calib: задача={task}, датасет={calib_dir}")
        rknn_path = _build_rknn(str(onnx_dst), quantize=True, calib_dir=calib_dir)
        shutil.copy2(rknn_path, str(target))
    elif quantize == "int8":
        task = info.get("task", "detect")
        calib_dir = _calib_img_dir(ensure_mini_dataset(task))
        log(f"Сборка RKNN int8: задача={task}, мини-датасет={calib_dir}")
        rknn_path = _build_rknn(str(onnx_dst), quantize=True, calib_dir=calib_dir)
        shutil.copy2(rknn_path, str(target))
    else:
        rknn_path = _build_rknn(str(onnx_dst), quantize=False)
        shutil.copy2(rknn_path, str(target))

    log(f"RKNN → кеш: {target}")
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RKNN Model Converter with caching",
        usage="convert MODEL [--fp16|--int8|--int8-calib|--onnx] | --list-cache",
    )
    parser.add_argument("model", nargs="?", help="Имя модели или путь к файлу")
    parser.add_argument("--fp16", action="store_const", dest="quantize", const="fp16",
                        help="RKNN FP16 (по умолчанию)")
    parser.add_argument("--int8", action="store_const", dest="quantize", const="int8",
                        help="RKNN INT8 без калибровки")
    parser.add_argument("--int8-calib", action="store_const", dest="quantize", const="int8-calib",
                        help="RKNN INT8 с калибровкой")
    parser.add_argument("--onnx", action="store_const", dest="quantize", const="onnx",
                        help="Только ONNX (CPU)")
    parser.add_argument("-l", "--list-cache", action="store_true",
                        help="Показать кеш моделей")

    args = parser.parse_args()

    if args.list_cache:
        cmd_list_cache()
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    quantize = args.quantize or "fp16"

    try:
        target = run_convert(args.model, quantize)
        pt_path = _pt_file(args.model)
        info = _detect_model_info(str(pt_path)) if pt_path.exists() else {}
        _deploy_to_model(args.model, quantize, info)
        print(json.dumps({
            "status": "OK",
            "model": args.model,
            "quantize": quantize,
            "target": str(target),
        }))
    except Exception as e:
        print(json.dumps({
            "status": "FAIL",
            "model": args.model,
            "quantize": quantize,
            "error": str(e),
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
