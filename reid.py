"""On-demand скачивание ReID-энкодера (Фаза 2) — как в ultralytics.

В репозитории НЕТ весов: при первом включении TRACKER_REID код сам скачивает
ONNX osnet в смонтированный `model/reid/` (rw-маунт в compose). Повторные
запуски используют кеш. Зеркало переопределяется env `TRACKER_REID_URL`.

Модель: osnet_x0_25_msmt17 (~0.25M параметров, ONNX 0.9 МБ) — вход
[N,3,256,128], выход (N,512) эмбеддингов; конвертируется в RKNN (NPU).

Запуск вручную: python3 reid.py  (просто докачивает и печатает путь).
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

REID_DIR = Path(os.environ.get('REID_DIR', '/app/model/reid'))

# Проверенные источники в порядке приоритета (proxy-окружение ниже).
_SOURCES: dict[str, list[str]] = {
    'osnet_x0_25_msmt17': [
        'https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/'
        'osnet_x0_25_msmt17.onnx',
    ],
}
# Гост-размер не guardar; считается валидным файлом всё > 0.5 МБ.
_MIN_WEIGHT_BYTES = 500_000


def ensure_reid_weights(model: str | None = None, timeout: int | None = None
                        ) -> Path:
    """Скачать веса `model` (osnet-семейство) в REID_DIR, вернуть путь.

    Поведение как в ultralytics: файл уже есть → сразу путь; отсутствует →
    докачка из списка URL (env `TRACKER_REID_URL` имеет высщий приоритет).
    Гарантирует атомарность: пишем в .part → rename (обрыв не оставляет
    обрубка, который потом принимается за рабочий файл).
    """
    model = model or os.environ.get('TRACKER_REID_MODEL', 'osnet_x0_25_msmt17')
    REID_DIR.mkdir(parents=True, exist_ok=True)
    dest = REID_DIR / f'{model}.onnx'
    if dest.is_file() and dest.stat().st_size > _MIN_WEIGHT_BYTES:
        log.info('ReID: веса уже на месте (%s)', dest)
        return dest

    timeout = timeout if timeout is not None else int(
        os.environ.get('REID_DOWNLOAD_TIMEOUT', '600'))
    urls = [os.environ.get('TRACKER_REID_URL', '')] + _SOURCES.get(model, [])
    tmp = dest.with_suffix(dest.suffix + '.part')
    for url in filter(None, urls):
        try:
            log.info('ReID: скачиваю %s → %s', model, tmp)
            # HF: resolve → редирект на CDN; urllib обрабатывает сам.
            req = urllib.request.Request(
                url, headers={'User-Agent': 'percept/1.0 (reid downloader)'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(tmp, 'wb') as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            size = tmp.stat().st_size
            if size <= _MIN_WEIGHT_BYTES:
                raise RuntimeError(f'слишком мал файл ({size} Б) — мусор/HTML')
            tmp.replace(dest)
            log.info('ReID: скачан %s (%s Б) → %s', model, size, dest)
            return dest
        except Exception as exc:
            log.warning('ReID: источник %s не удался: %s', url, exc)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f'не удалось скачать веса ReID {model}; задайте TRACKER_REID_URL '
        'или выключите TRACKER_REID')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(ensure_reid_weights())
