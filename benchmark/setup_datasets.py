#!/usr/bin/env python3
"""Download benchmark test datasets into benchmark/data/.

Runs inside the converter container (has urllib + ultralytics).
Usage: python3 setup_datasets.py <data_dir>
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    'dota8': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip',
    'cityscapes8': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/cityscapes8.zip',
}


def download(data_dir: Path, name: str, url: str) -> None:
    target = data_dir / name
    if target.exists():
        print(f'[SKIP] {name} already exists at {target}')
        return

    zip_path = data_dir / f'{name}.zip'
    print(f'[DOWNLOAD] {name} from {url}')
    urllib.request.urlretrieve(url, zip_path)

    print(f'[EXTRACT] {name} → {target}')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    zip_path.unlink()
    print(f'[DONE] {name}')


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/app/benchmark/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DATASETS.items():
        download(data_dir, name, url)

    print(f'\nAll datasets ready in {data_dir}')
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            images = list(d.rglob('*.*'))
            n = sum(1 for f in images if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
            print(f'  {d.name}: {n} images')


if __name__ == '__main__':
    main()
