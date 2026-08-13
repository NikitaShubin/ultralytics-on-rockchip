#!/usr/bin/env python3
"""Загрузка и подготовка датасетов для калибровки и валидации."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / 'datasets'

DOTAV1_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip'
CITYSCAPES_HF_REPO = 'Antreas/Cityscapes'
IMAGENET_VAL_URL = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
IMAGENET_VALPREP_URL = 'https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh'

# COCO: официальный источник images.cocodataset.org может быть недоступен
# (DPI/гео-блокировка), поэтому указаны фолбэк-зеркала на HuggingFace.
COCO_ANN_ZIP_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
# Зеркало аннотаций: по одному JSON на файл (instances + person_keypoints).
COCO_ANN_MIRROR_JSONS = (
    ('https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json',
     'instances_val2017.json'),
    ('https://huggingface.co/datasets/merve/coco/resolve/main/annotations/person_keypoints_val2017.json',
     'person_keypoints_val2017.json'),
)
# Изображения val2017: официальный zip или зеркало (kaggle-дамп 5000 jpg).
COCO_IMAGES_URLS = (
    'http://images.cocodataset.org/zips/val2017.zip',
    'https://huggingface.co/datasets/baohao/coco2017/resolve/main/coco_val2017.zip',
)
# У зеркала baohao/coco2017 файлы лежат во вложенных папках kaggle/input/...
COCO_IMAGES_STRIP_PREFIXES = ('kaggle/input/coco-2017-dataset/coco2017/',)

# Встроенные мини-датасеты ultralytics, используемые для INT8-конвертации
# (TASK2CALIBRATIONDATA) и smoke-проверок. Ключи — задачи проекта.
MINI_DATASETS = {
    'detect': 'coco128.yaml',
    'seg': 'coco128-seg.yaml',
    'pose': 'coco8-pose.yaml',
    'obb': 'dota128.yaml',
    'cls': 'imagenet100',
    'sem': 'cityscapes8.yaml',
}
TASK_ALIASES = {'segment': 'seg', 'classify': 'cls', 'semantic': 'sem'}

# Полные датасеты задач (для eval / int8-calib)
FULL_DATASETS = {
    'detect': 'coco_val2017',
    'seg': 'coco_val2017',
    'pose': 'coco_val2017',
    'obb': 'dotav1',
    'cls': 'imagenet',
    'sem': 'cityscapes',
}


def ensure_dir(path: Path) -> None:
    """Создать директорию и проверить права на запись."""
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        print(f'ОШИБКА: Нет прав на запись в {path}', file=sys.stderr)
        print(f'  Возможно, директория создана от root (Docker mount).', file=sys.stderr)
        print(f'  Исправление: sudo chown -R $(id -u):$(id -g) {path}', file=sys.stderr)
        sys.exit(1)


def download_file(urls, dest: Path) -> None:
    """Загрузка файла: сначала в .tmp, потом rename (защита от битых загрузок).

    Принимает один URL или список URL (фолбэк-источники — перебираются по порядку,
    пока один не скачается полностью). Поддерживает докачку (-C -): если .tmp
    существует и больше нуля, загрузка продолжится с места обрыва.
    """
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        raise ValueError('Нет источников для скачивания')

    if dest.exists() and dest.stat().st_size > 0:
        print(f'Файл уже существует: {dest}')
        return

    tmp = dest.with_suffix('.tmp')
    last_err = None
    for idx, url in enumerate(urls):
        # Докачка возможна только если .tmp остался от ТОГО ЖЕ источника (первая попытка);
        # при смене URL другой файл докачивать нельзя.
        resume = idx == 0 and tmp.exists() and tmp.stat().st_size > 1_000_000
        if not resume:
            tmp.unlink(missing_ok=True)
        if resume:
            print(f'Докачка {tmp} ({tmp.stat().st_size // (1024*1024)} MB)...')
        print(f'Загрузка {url}...')
        try:
            subprocess.run(
                ['curl', '-fSL', '--progress-bar', '-L', '--retry', '2',
                 '--retry-delay', '5', '--connect-timeout', '20',
                 '--speed-limit', '1000', '--speed-time', '15',
                 '-C', '-', '-o', str(tmp), url],
                check=True,
            )
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f'Загрузка вернула пустой файл: {url}')
            tmp.rename(dest)
            print(f'Загружен: {dest} ({dest.stat().st_size // (1024*1024)} MB)')
            return
        except (subprocess.CalledProcessError, RuntimeError) as e:
            last_err = e
            print(f'    Не удалось ({e}). Пробую следующий источник...', file=sys.stderr)
            tmp.unlink(missing_ok=True)

    raise RuntimeError(
        f'Все источники недоступны: {urls}\nПоследняя ошибка: {last_err}'
    )


def _extract_zip(zip_path: Path, dest: Path, strip_prefixes=()) -> None:
    """Распаковать zip-архив в dest, опционально убирая префикс путей.

    strip_prefixes — кортеж префиксов; для каждой записи первый совпавший префикс
    вырезается из пути (нужно для зеркал, где архив лежит во вложенных папках,
    напр. kaggle/input/.../).
    """
    validate_zip(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for entry in z.namelist():
            name = entry
            for pref in strip_prefixes:
                if entry.startswith(pref):
                    name = entry[len(pref):]
                    break
            if not name or name.startswith('__MACOSX/') or name.startswith('.DS_Store'):
                continue
            target = dest / name
            if entry.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as src, open(target, 'wb') as dst:
                    dst.write(src.read())


def validate_zip(path: Path) -> None:
    """Проверить что файл — валидный zip-архив."""
    try:
        with zipfile.ZipFile(path, 'r') as z:
            z.testzip()
    except zipfile.BadZipFile:
        content = path.read_text(errors='replace')[:200]
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f'{path.name} — битый архив. Содержимое файла:\n{content}\n'
            f'Удалён. Попробуйте снова.'
        )


def validate_tar(path: Path) -> None:
    """Проверить что файл — валидный tar-архив (первые 3 блока)."""
    import tarfile
    try:
        with tarfile.open(path, 'r:*') as tar:
            # Читаем первые 3 записи для быстрой проверки
            for i, _ in enumerate(tar):
                if i >= 2:
                    break
    except (tarfile.TarError, EOFError, OSError) as e:
        size_mb = path.stat().st_size // (1024 * 1024)
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f'{path.name} ({size_mb} MB) — битый tar-архив: {e}\n'
            f'Удалён. Попробуйте снова.'
        )


def _ensure_coco_annotations(coco_dir: Path, ann_dir: Path) -> None:
    """Аннотации COCO: официальный zip или фолбэк-зеркало на HF (по одному JSON).

    Возможные файлы (detect/seg/obb используют instances, pose — person_keypoints).
    """
    ann_zip = DATASETS_DIR / 'annotations_trainval2017.zip'
    try:
        download_file(COCO_ANN_ZIP_URL, ann_zip)
    except RuntimeError as e:
        print(f'Официальный источник недоступен ({e}).', file=sys.stderr)
        print('Качаю аннотации с зеркала HuggingFace (merve/coco)...')
        ann_dir.mkdir(parents=True, exist_ok=True)
        for url, name in COCO_ANN_MIRROR_JSONS:
            dest = ann_dir / name
            if not dest.exists() or dest.stat().st_size == 0:
                download_file(url, dest)
        return

    print('Распаковка аннотаций...')
    _extract_zip(ann_zip, coco_dir)
    ann_zip.unlink(missing_ok=True)


def _ensure_coco_images(coco_dir: Path, images_dir: Path) -> None:
    """Изображения COCO val2017: официальный zip или зеркало HF (kaggle-дамп)."""
    img_zip = DATASETS_DIR / 'val2017.zip'
    download_file(COCO_IMAGES_URLS, img_zip)
    print('Распаковка изображений (это может занять время)...')
    _extract_zip(img_zip, coco_dir / 'images', COCO_IMAGES_STRIP_PREFIXES)
    img_zip.unlink(missing_ok=True)


def setup_coco_val2017() -> None:
    """Загрузка COCO val2017 (5000 изображений + аннотации)."""
    coco_dir = DATASETS_DIR / 'coco_val2017'
    ensure_dir(coco_dir)

    images_dir = coco_dir / 'images' / 'val2017'
    ann_dir = coco_dir / 'annotations'

    # Проверка: всё уже на месте
    if (images_dir.exists() and any(images_dir.glob('*.jpg'))
            and ann_dir.exists() and any(ann_dir.glob('*.json'))):
        # Убираем оставшиеся архивы (распаковано, архив не нужен)
        for leftover in ['annotations_trainval2017.zip', 'val2017.zip']:
            (DATASETS_DIR / leftover).unlink(missing_ok=True)
        print('COCO val2017 уже загружена')
        return

    # Аннотации
    if not (ann_dir.exists() and any(ann_dir.glob('*.json'))):
        _ensure_coco_annotations(coco_dir, ann_dir)

    # Изображения
    if not (images_dir.exists() and any(images_dir.glob('*.jpg'))):
        _ensure_coco_images(coco_dir, images_dir)

    print(f'COCO val2017 готова: {coco_dir}')


def setup_dota_v1() -> None:
    """Загрузка DOTAv1 val split (458 изображений + YOLO OBB аннотации).

    Скачивает полный DOTAv1.zip (~1.9 ГБ), извлекает только val split,
    удаляет zip-архив.
    """
    dota_dir = DATASETS_DIR / 'dotav1'
    ensure_dir(dota_dir)
    images_val = dota_dir / 'images' / 'val'
    labels_val = dota_dir / 'labels' / 'val'

    # Проверка: всё уже на месте
    if (images_val.exists()
            and (any(images_val.glob('*.png')) or any(images_val.glob('*.jpg')))
            and labels_val.exists() and any(labels_val.glob('*.txt'))):
        n_img = len(list(images_val.iterdir()))
        n_lbl = len(list(labels_val.glob('*.txt')))
        # Убираем оставшийся архив (распакован — не нужен)
        (DATASETS_DIR / 'DOTAv1.zip').unlink(missing_ok=True)
        print(f'DOTAv1 val уже загружена: {n_img} изображений, {n_lbl} аннотаций')
        return

    print('Загрузка DOTAv1 (~1.9 ГБ)...')
    zip_path = DATASETS_DIR / 'DOTAv1.zip'
    if not zip_path.exists():
        download_file(DOTAV1_URL, zip_path)

    print('Распаковка val split из DOTAv1.zip...')
    validate_zip(zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Список файлов val split (strip top-level dir)
        val_files = [f for f in z.namelist() if '/val/' in f]
        print(f'  Найдено {len(val_files)} файлов в val split')

        for f in val_files:
            # Strip top-level directory (DOTAv1/ или dota/)
            parts = f.split('/')
            if len(parts) > 1 and '.' not in parts[0]:
                stripped = '/'.join(parts[1:])
            else:
                stripped = f
            if not stripped:
                continue
            target = dota_dir / stripped
            if f.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(f) as src, open(target, 'wb') as dst:
                    dst.write(src.read())

    # Удаляем zip
    zip_path.unlink(missing_ok=True)

    # Удаляем train split и прочие файлы если попали
    for sub in ['train', 'test']:
        for parent in [dota_dir / 'images', dota_dir / 'labels']:
            d = parent / sub
            if d.exists():
                import shutil
                shutil.rmtree(d)

    # Удаляем лишние файлы из корня (LICENSE, README, и т.д.)
    for f in dota_dir.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)

    n_img = len(list(images_val.iterdir())) if images_val.exists() else 0
    n_lbl = len(list(labels_val.glob('*.txt'))) if labels_val.exists() else 0
    print(f'DOTAv1 val готова: {n_img} изображений, {n_lbl} аннотаций')
    print(f'  Директория: {dota_dir}')


def setup_cityscapes_val() -> None:
    """Загрузка Cityscapes val split (500 изображений + маски).

    Источник: Antreas/Cityscapes на HuggingFace (parquet val split, 3 шарда,
    изображения в нормальном PNG). Имена — официальные (из path маски,
    `<city>_<seq>_<frame>`), совпадают со стемами масок.
    """
    cs_dir = DATASETS_DIR / 'cityscapes'
    ensure_dir(cs_dir)
    images_val = cs_dir / 'images' / 'val'
    masks_val = cs_dir / 'masks' / 'val'

    # Проверка: всё уже на месте
    if (images_val.exists()
            and len(list(images_val.glob('*.png'))) >= 450
            and masks_val.exists()
            and len(list(masks_val.glob('*.png'))) >= 450):
        n_img = len(list(images_val.glob('*.png')))
        n_mask = len(list(masks_val.glob('*.png')))
        print(f'Cityscapes val уже загружена: {n_img} изображений, {n_mask} масок')
        return

    # Список parquet-файлов val split Antreas/Cityscapes (3 шарда)
    parquet_urls = [
        'https://huggingface.co/datasets/Antreas/Cityscapes/resolve/main/data/val-00000-of-00003.parquet',
        'https://huggingface.co/datasets/Antreas/Cityscapes/resolve/main/data/val-00001-of-00003.parquet',
        'https://huggingface.co/datasets/Antreas/Cityscapes/resolve/main/data/val-00002-of-00003.parquet',
    ]

    images_val.mkdir(parents=True, exist_ok=True)
    masks_val.mkdir(parents=True, exist_ok=True)

    # Проверяем, есть ли pyarrow (нужен для чтения parquet)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pyarrow
        except ImportError:
            print('Установка pyarrow...')
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--user', 'pyarrow', '-q',
            ])
            import pyarrow.parquet as pq

    from io import BytesIO
    from PIL import Image

    idx = 0
    for url in parquet_urls:
        fname = url.split('/')[-1]
        cache_path = cs_dir / f'val_{fname}'

        if not cache_path.exists():
            print(f'Загрузка {fname}...')
            subprocess.run(
                ['curl', '-fSL', '-o', str(cache_path), url],
                check=True,
            )

        print(f'Извлечение из {fname}...')
        table = pq.read_table(str(cache_path))

        img_col = table.column('image').combine_chunks()
        seg_col = table.column('semantic_segmentation').combine_chunks()

        for row_idx in range(len(table)):
            # Официальное имя из path маски: <city>_<seq>_<frame>_gtFine_labelIds.png
            seg_path = seg_col.field('path')[row_idx].as_py()
            if not seg_path or not seg_path.endswith('_gtFine_labelIds.png'):
                raise ValueError(f'Неожиданный path маски: {seg_path}')
            stem = Path(seg_path).name.replace('_gtFine_labelIds.png', '')

            image_path = images_val / f'{stem}.png'
            mask_path = masks_val / f'{stem}.png'

            if not image_path.exists():
                img_bytes = img_col.field('bytes')[row_idx].as_py()
                Image.open(BytesIO(img_bytes)).save(image_path)
            if not mask_path.exists():
                mask_bytes = seg_col.field('bytes')[row_idx].as_py()
                Image.open(BytesIO(mask_bytes)).save(mask_path)

            idx += 1

            if idx % 50 == 0:
                print(f'  Сохранено {idx}/500...')

        # Освобождаем память
        del table

    # Удаляем parquet-кэш
    for f in cs_dir.glob('val_*.parquet'):
        f.unlink(missing_ok=True)

    n_img = len(list(images_val.glob('*.png')))
    n_mask = len(list(masks_val.glob('*.png')))
    print(f'Cityscapes val готова: {n_img} изображений, {n_mask} масок')
    print(f'  Директория: {cs_dir}')


def normalize_image_extensions(root: Path) -> int:
    """Привести расширения изображений к нижнему регистру (.JPEG → .jpg).

    rknn-toolkit2 поддерживает только .jpg .jpeg .png .bmp (нижний регистр).
    Возвращает количество переименованных файлов.
    """
    renamed = 0
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        ext = p.suffix
        if ext and ext != ext.lower():
            new_path = p.with_suffix(ext.lower())
            # Если файл с таким именем уже есть — удаляем текущий (дубликат)
            if new_path.exists() and new_path != p:
                p.unlink()
            else:
                p.rename(new_path)
            renamed += 1
    return renamed


def ensure_ultralytics_dir() -> Path:
    """Гарантировать, что мини-датасеты ultralytics качаются в datasets/ проекта.

    Задаёт YOLO_DATASETS_DIR (и дублирует в settings ultralytics), чтобы
    встроенные мини-датасеты (coco128, coco8-pose, dota128, imagenet100, ...)
    попадали в папку проекта, а не в пользовательский кэш. Вызывать ДО
    первого использования ultralytics (импорт/check_det_dataset).
    """
    ensure_dir(DATASETS_DIR)
    os.environ.setdefault('YOLO_DATASETS_DIR', str(DATASETS_DIR))
    try:
        from ultralytics.utils import SETTINGS
        if SETTINGS.get('datasets_dir') != str(DATASETS_DIR):
            SETTINGS.update({'datasets_dir': str(DATASETS_DIR)})
    except Exception:
        pass
    return DATASETS_DIR


def _normalize_mini_extensions(data_dir: Path) -> None:
    """Привести расширения мини-датасета к нижнему регистру после скачивания.

    rknn-toolkit2 поддерживает только .jpg .jpeg .png .bmp (нижний регистр),
    а ultralytics скачивает imagenet100 с .JPEG файлами.
    """
    if not data_dir.exists():
        return
    n = normalize_image_extensions(data_dir)
    if n:
        print(f'  Нормализовано {n} расширений в {data_dir}')


def ensure_mini_dataset(task: str) -> Path:
    """Обеспечить встроенный мини-датасет ultralytics для задачи (в datasets/).

    Скачивает ТОЛЬКО то, что нужно текущей задаче (coco128 / coco128-seg /
    coco8-pose / dota128 / imagenet100 / cityscapes8). Используется для
    INT8-конвертации без калибровки и smoke-проверок. Идемпотентно.
    """
    ensure_ultralytics_dir()
    task = TASK_ALIASES.get(task, task)
    key = MINI_DATASETS.get(task)
    if key is None:
        raise ValueError(f'Неизвестная задача для мини-датасета: {task}')

    if task == 'cls':
        from ultralytics.data.utils import check_cls_dataset
        check_cls_dataset(key)
        data_dir = DATASETS_DIR / 'imagenet100'
        _normalize_mini_extensions(data_dir)
        return data_dir

    from ultralytics.data.utils import check_det_dataset
    info = check_det_dataset(key)
    data_dir = Path(info.get('path', str(DATASETS_DIR / Path(key).stem)))
    _normalize_mini_extensions(data_dir)
    return data_dir


def setup_imagenet_val() -> None:
    """Загрузка ImageNet val split (50000 изображений, 1000 классов)."""
    imagenet_dir = DATASETS_DIR / 'imagenet'
    val_dir = imagenet_dir / 'val'
    ensure_dir(imagenet_dir)

    # Всё уже на месте
    if val_dir.exists():
        n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
        n_images = sum(1 for _ in val_dir.rglob('*.JPEG'))
        if n_classes >= 999 and n_images >= 49000:
            print(f'ImageNet val уже загружена: {n_images} изображений, {n_classes} классов')
            normalize_image_extensions(val_dir)
            # Убираем оставшийся tar-архив (распакован — не нужен)
            (DATASETS_DIR / 'ILSVRC2012_img_val.tar').unlink(missing_ok=True)
            return

    # Если val_dir есть но неполный — удаляем и начинаем заново
    if val_dir.exists():
        import shutil
        n = sum(1 for _ in val_dir.rglob('*'))
        print(f'Неполная распаковка ({n} файлов), удаление...')
        shutil.rmtree(val_dir)

    # Удаляем flat JPEG если есть (остаток неполной распаковки в корне)
    for f in imagenet_dir.glob('ILSVRC*.JPEG'):
        f.unlink()

    # Скачивание tar (curl -C - докачивает с места прерывания)
    tar_path = DATASETS_DIR / 'ILSVRC2012_img_val.tar'
    if not tar_path.exists() or tar_path.stat().st_size < 5_000_000_000:
        print('Загрузка ImageNet val (~6.3 GB)...')
        download_file(IMAGENET_VAL_URL, tar_path)

    # Распаковка
    val_dir.mkdir(parents=True, exist_ok=True)
    print('Распаковка ImageNet val...')
    import tarfile
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(val_dir)
    print('Распаковка завершена')

    # Архив успешно распакован — удаляем, чтобы не занимать место (~6.3 GB)
    tar_path.unlink(missing_ok=True)

    # Организация по классам
    valprep_sh = imagenet_dir / 'valprep.sh'
    if not valprep_sh.exists():
        print('Скачивание valprep.sh...')
        subprocess.run(
            ['curl', '-fSL', '-o', str(valprep_sh), IMAGENET_VALPREP_URL],
            check=True,
        )
    flat_in_val = list(val_dir.glob('ILSVRC*.JPEG'))
    if flat_in_val:
        print('Организация по поддиректориям классов...')
        subprocess.run(['bash', str(valprep_sh)], cwd=str(val_dir), check=True)

    # Нормализуем расширения (.JPEG → .jpg)
    renamed = normalize_image_extensions(val_dir)
    if renamed:
        print(f'  Нормализовано {renamed} расширений (.JPEG → .jpg)')

    n_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
    n_images = sum(1 for _ in val_dir.rglob('*.jpg'))
    print(f'ImageNet val готова: {n_images} изображений, {n_classes} классов')
    print(f'  Директория: {imagenet_dir}')


def _setup_full_dataset(task: str) -> None:
    """Загрузить полный датасет задачи (для eval / int8-calib)."""
    task = TASK_ALIASES.get(task, task)
    if task in ('detect', 'seg', 'pose'):
        setup_coco_val2017()
    elif task == 'obb':
        setup_dota_v1()
    elif task == 'sem':
        setup_cityscapes_val()
    elif task == 'cls':
        setup_imagenet_val()
    else:
        raise ValueError(f'Неизвестная задача: {task}')


def ensure_full_dataset(task: str) -> Path:
    """Программный доступ к полному датасету задачи (качает, если отсутствует).

    Используется eval/3way для автоскачивания полного датасета с аннотациями
    (GT для метрик). Возвращает путь к датасету.
    """
    task = TASK_ALIASES.get(task, task)
    name = FULL_DATASETS.get(task)
    if name is None:
        raise ValueError(f'Неизвестная задача: {task}')
    data_dir = DATASETS_DIR / name
    _setup_full_dataset(task)
    return data_dir


def main() -> None:
    global DATASETS_DIR

    parser = argparse.ArgumentParser(
        description='Загрузка датасетов для калибровки и валидации',
    )
    parser.add_argument('task', nargs='?', default=None,
                        help='Задача: detect|segment|pose|obb|classify|semantic')
    parser.add_argument('--mini', action='store_true',
                        help='Скачать встроенный мини-датасет задачи (для INT8/smoke)')
    parser.add_argument('--full', action='store_true',
                        help='Скачать полный датасет задачи (для eval/int8-calib)')
    parser.add_argument('--coco', action='store_true', help='Загрузить COCO val2017')
    parser.add_argument('--dota', action='store_true', help='Загрузить DOTAv1 val')
    parser.add_argument('--cityscapes', action='store_true', help='Загрузить Cityscapes val')
    parser.add_argument('--imagenet', action='store_true', help='Загрузить ImageNet val (6.3 GB, 1000 классов)')
    parser.add_argument('--all', action='store_true', help='Загрузить все датасеты')
    parser.add_argument('--data-dir', default=None,
                        help='Куда скачивать датасеты (по умолчанию: datasets/ рядом со скриптом)')
    args = parser.parse_args()

    if args.data_dir:
        DATASETS_DIR = Path(args.data_dir)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_ultralytics_dir()

    print('=== Подготовка датасетов ===')
    print()

    if args.task:
        # Режим по задаче: качать только нужное (--mini / --full / оба)
        if args.mini or not (args.full or args.coco or args.dota or args.cityscapes or args.imagenet):
            print(f'[mini] {args.task}: {MINI_DATASETS.get(TASK_ALIASES.get(args.task, args.task))}')
            ensure_mini_dataset(args.task)
            print()
        if args.full:
            print(f'[full] {args.task}: {FULL_DATASETS.get(TASK_ALIASES.get(args.task, args.task))}')
            _setup_full_dataset(args.task)
            print()
    else:
        # Legacy-режим
        if not (args.coco or args.dota or args.cityscapes or args.imagenet or args.all):
            args.all = True

        if args.all or args.coco:
            setup_coco_val2017()
            print()

        if args.all or args.dota:
            setup_dota_v1()
            print()

        if args.all or args.cityscapes:
            setup_cityscapes_val()
            print()

        if args.all or args.imagenet:
            setup_imagenet_val()
            print()

    print('=== Все датасеты готовы ===')
    print(f'Директория: {DATASETS_DIR}')


if __name__ == '__main__':
    main()
