#!/usr/bin/env python3
"""Загрузка и подготовка датасетов для калибровки и валидации."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / 'datasets'


def download_file(url: str, dest: Path) -> None:
    """Загрузка файла с прогрессом."""
    print(f'Загрузка {url}...')
    subprocess.run(
        ['wget', '-q', '--show-progress', '-O', str(dest), url],
        check=True,
    )


def setup_coco_val2017() -> None:
    """Загрузка COCO val2017 (5000 изображений + аннотации)."""
    coco_dir = DATASETS_DIR / 'coco_val2017'
    coco_dir.mkdir(parents=True, exist_ok=True)

    images_dir = coco_dir / 'images' / 'val2017'
    ann_dir = coco_dir / 'annotations'

    # Проверка: всё уже на месте
    if (images_dir.exists() and any(images_dir.glob('*.jpg'))
            and ann_dir.exists() and any(ann_dir.glob('*.json'))):
        print('COCO val2017 уже загружена')
        return

    # Аннотации
    if not (ann_dir.exists() and any(ann_dir.glob('*.json'))):
        ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        ann_zip = DATASETS_DIR / 'annotations_trainval2017.zip'
        if not ann_zip.exists():
            download_file(ann_url, ann_zip)

        print('Распаковка аннотаций...')
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(coco_dir)

        ann_zip.unlink(missing_ok=True)

    # Изображения
    if not (images_dir.exists() and any(images_dir.glob('*.jpg'))):
        img_url = 'http://images.cocodataset.org/zips/val2017.zip'
        img_zip = DATASETS_DIR / 'val2017.zip'
        if not img_zip.exists():
            download_file(img_url, img_zip)

        print('Распаковка изображений (это может занять время)...')
        with zipfile.ZipFile(img_zip, 'r') as z:
            z.extractall(coco_dir / 'images')

        img_zip.unlink(missing_ok=True)

    print(f'COCO val2017 готова: {coco_dir}')


def main() -> None:
    """Главная функция: загрузка всех необходимых датасетов."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    print('=== Подготовка датасетов ===')
    print()

    setup_coco_val2017()

    print()
    print('=== Все датасеты готовы ===')
    print(f'Директория: {DATASETS_DIR}')


if __name__ == '__main__':
    main()
