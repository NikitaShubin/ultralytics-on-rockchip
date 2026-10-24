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

import cv2
import numpy as np

log = logging.getLogger(__name__)

REID_DIR = Path(os.environ.get('REID_DIR', '/app/model/reid'))

# ImageNet-нормализация osnet (torchreid build_transform).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

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


class ReidEngine:
    """Эмбеддинг-энкодер ReID: osnet_x0_25 (RKNN/NPU либо ONNX/CPU).

    `embed(frame, boxes)` — кропы (xyxy в координатах кадра) →
    (K, 512) L2-нормированные эмбеддинги. Фиксированный батч 16: если боксов
    больше — паддинг нулями и слайс; меньше — паддинг и слайс.

    RKNN-модель (/app/model/reid/osnet_x0_25_msmt17.rknn) создаётся:
      docker compose -f convert/docker-compose.yml run --rm \
          --entrypoint python3 convert /app/convert/reid_rknn.py
    Normalization (mean/std ImageNet) запечены в RKNN; ONNX-путь делает их сам.

    Цены (замеры на RK3588): batch16 = ~67–100 мс (int8/fp16 NPU) или ~76 мс
    (CPU) — поэтому ReID по умолчанию выключен и включается env TRACKER_REID.
    """

    BATCH = 16

    def __init__(self, backend: str = 'rknn', weights_dir: str | None = None) -> None:
        backend = 'onnx' if backend == 'auto' or backend not in ('rknn', 'onnx') else backend
        self.backend = backend
        base = Path(weights_dir) if weights_dir else REID_DIR
        self._rknn_mod = None
        self._sess = None
        try:
            if backend == 'rknn':
                from rknnlite.api import RKNNLite as _mod
                self._rknn_mod = _mod
                rknn_path = base / 'osnet_x0_25_msmt17.rknn'
                if not rknn_path.is_file():
                    raise FileNotFoundError(rknn_path)
                self._rk = _mod(verbose=False)
                self._rk.load_rknn(str(rknn_path))
                if self._rk.init_runtime():
                    raise RuntimeError('rknnlite init_runtime не прошёл')
                log.info('ReID: RKNN NPU (osnet_x0_25)')
            else:
                import onnxruntime as ort
                onnx_path = base / 'osnet_x0_25_msmt17.onnx'
                if not onnx_path.is_file():
                    onnx_path = Path(str(ensure_reid_weights()))
                self._sess = ort.InferenceSession(
                    str(onnx_path), providers=['CPUExecutionProvider'])
                log.info('ReID: ONNX CPU (osnet_x0_25)')
        except Exception as exc:
            log.error('ReID: движок не поднялся (%s backend): %s', backend, exc)
            raise

    def embed(self, frame: np.ndarray, boxes: list | np.ndarray
              ) -> tuple[list[int], np.ndarray]:
        """Эмбеддинги кропов. Возвращает (индексы_боксов, feats (K,512))."""
        k = min(len(boxes), self.BATCH)
        if k == 0 or frame is None:
            return [], np.zeros((0, 512), np.float32)
        crops = np.zeros((self.BATCH, 3, 256, 128), np.uint8)
        for i in range(k):
            x1, y1, x2, y2 = (float(v) for v in boxes[i][:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = cv2.cvtColor(frame[y1:y2 + 1, x1:x2 + 1], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
            crops[i] = crop.transpose(2, 0, 1)
        feats = self._infer(crops)[:k]
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        return list(range(k)), feats / np.maximum(norm, 1e-9)

    def _infer(self, crops: np.ndarray) -> np.ndarray:
        if self._rknn_mod is not None:
            out = self._rk.inference(inputs=[crops])[0]
        else:
            x = crops.astype(np.float32) / 255.0
            x = (x - _MEAN[None, :, None, None]) / _STD[None, :, None, None]
            out = self._sess.run(None, {self._sess.get_inputs()[0].name: x})[0]
        return np.asarray(out, dtype=np.float32)

