"""Конверсия ReID-энкодера (osnet) ONNX → RKNN для NPU Orange Pi 5 Pro.

Запуск в convert-контейнере (rknn-toolkit2 + numpy + cv2; model/reid — часть
маунта ../model):
  docker compose -f convert/docker-compose.yml run --rm \
      --entrypoint python3 convert /app/convert/reid_rknn.py

Env:
  REID_QUANT=int8|fp16  (int8 по умолчанию)
  REID_CALIB_SRC=/app/video/people.mp4  — видео для РЕАЛЬНОЙ калибровки
      (рандомные кропы); если видео нет — синтетические патчи (плохо всюду:
      шумовая калибровка int8 коллапсирует выход).

Нормализация ImageNet ([0.485,0.456,0.406]/[0.229,0.224,0.225]) запекается в
RKNN (mean/std в config) — рантайм подаёт сырые RGB-пиксели [0,255].
Выход модели: (16,512) сырые эмбеддинги; норму L2 делает рантайм.
"""

from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np
from rknn.api import RKNN

SRC = os.environ.get('REID_SRC', '/app/convert/model/reid/osnet_x0_25_msmt17.onnx')
DST = os.environ.get('REID_DST', '/app/convert/model/reid/osnet_x0_25_msmt17.rknn')
CALIB_DIR = '/reid_calib'
W, H, BATCH = 128, 256, 16
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _crops_from_video(path: str, n: int = 24) -> list[np.ndarray]:
    """Случайные кропы людей-подобных боксов из видео (BATCH×3×256×128 uint8)."""
    rng = random.Random(2025)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise RuntimeError(f'нет кадров в {path}')
    batches = []
    for _ in range(n):
        idx = rng.randrange(total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        # Случш. боксы людей-подобных пропорций (h:w = 2:1), чтобы попасть
        # на реальные объекты/фоны — статистика активаций близка к прод.
        arrs = []
        for _ in range(BATCH):
            bh = rng.randint(h // 3, int(h * 0.9))
            bw = max(24, int(bh / 2))
            x0 = rng.randint(0, max(1, w - bw))
            y0 = rng.randint(0, max(1, h - bh))
            crop = frame[y0:y0 + bh, x0:x0 + bw]
            crop = cv2.cvtColor(cv2.resize(crop, (W, H),
                                           interpolation=cv2.INTER_LINEAR),
                                cv2.COLOR_BGR2RGB)
            arrs.append(crop.transpose(2, 0, 1))
        arr = np.stack(arrs).astype(np.uint8)  # (16,3,256,128)
        batches.append(arr)
    cap.release()
    return batches


def _synthetic(n: int = 24) -> list[np.ndarray]:
    rng = np.random.default_rng(12345)
    return [rng.integers(0, 256, (BATCH, 3, H, W), dtype=np.uint8)
            for _ in range(n)]


def _write_calib(batches: list[np.ndarray]) -> str:
    os.makedirs(CALIB_DIR, exist_ok=True)
    lines = []
    for i, arr in enumerate(batches):
        path = f'{CALIB_DIR}/c{i:03d}.npy'
        np.save(path, arr)
        lines.append(path)
    lst = f'{CALIB_DIR}/calib.txt'
    with open(lst, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return lst


def main() -> None:
    quant = os.environ.get('REID_QUANT', 'int8').lower()
    calib_src = os.environ.get('REID_CALIB_SRC', '/app/video/people.mp4')
    rk = RKNN(verbose=False)
    cfg = dict(
        target_platform='rk3588',
        mean_values=[[0.485, 0.456, 0.406]],
        std_values=[[0.229, 0.224, 0.225]],
    )
    if quant == 'int8':
        cfg.update(quantized_dtype='w8a8', quantized_algorithm='normal',
                   quantized_method='channel')
    print('-- config', quant)
    ret = rk.config(**cfg)
    if ret:
        sys.exit(f'config fail {ret}')
    print('-- load_onnx', SRC)
    ret = rk.load_onnx(model=SRC)
    if ret:
        sys.exit(f'load_onnx fail {ret}')
    if quant == 'int8':
        # РЕАЛЬНАЯ калибровка из видео — иначе int8-выход коллапсирует.
        try:
            batches = _crops_from_video(calib_src)
            print(f'-- калибровка из видео: {len(batches)} батчей')
        except Exception as exc:
            print(f'-- калибровка из видео недоступна ({exc}), синтетика')
            batches = _synthetic()
        dataset = _write_calib(batches)
        print('-- build (int8)')
        ret = rk.build(do_quantization=True, dataset=dataset)
    else:
        print('-- build (fp16)')
        ret = rk.build(do_quantization=False)
    if ret:
        sys.exit(f'build fail {ret}')
    print('-- export', DST)
    ret = rk.export_rknn(DST)
    if ret:
        sys.exit(f'export fail {ret}')
    print('OK:', DST)


if __name__ == '__main__':
    main()
