#!/usr/bin/env python3
"""Эксперимент: батч-инференс на NPU (RK3588).

Собирает std ONNX yolo26n с batch=8 → RKNN fp16 static [8,3,640,640],
замеряет: infer батча-8 одним вызовом vs 8 вызовов batch-1 (существующий кеш).
Рабочие файлы не трогает — всё во временной папке /tmp/bs8_test.

Запуск в convert-контейнере (rknnlite + реальный NPU):
  docker compose -f convert/docker-compose.yml run --rm --entrypoint python \
    convert /app/benchmark/prof_batch_infer.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NAME = 'yolo26n'
BATCH = 8
IMGSZ = 640
PT = ROOT / 'cache' / NAME / f'{NAME}.pt'
BS1_RKNN = ROOT / 'cache' / NAME / f'{NAME}_fp16.rknn'

TMP = Path(tempfile.mkdtemp(prefix='bs8_test_'))
ONNX_BS8 = TMP / f'{NAME}_bs{BATCH}.onnx'
RKNN_BS8 = TMP / f'{NAME}_bs{BATCH}.rknn'


def export_onnx_bs8():
    print(f'[1] Экспорт std ONNX batch={BATCH}...', flush=True)
    from ultralytics import YOLO
    model = YOLO(str(PT), task='detect')
    model.end2end = False
    for m in model.model.modules():
        if hasattr(m, 'end2end'):
            m.end2end = False
    out = model.export(format='onnx', opset=12, batch=BATCH)
    src = Path(out)
    import shutil
    shutil.copy2(str(src), str(ONNX_BS8))
    print(f'    ONNX: {ONNX_BS8} ({ONNX_BS8.stat().st_size/1e6:.1f} MB)', flush=True)


def build_rknn():
    print(f'[2] Сборка RKNN fp16 static [{BATCH},3,{IMGSZ},{IMGSZ}]...', flush=True)
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    ret = rknn.load_onnx(model=str(ONNX_BS8))
    if ret != 0:
        raise RuntimeError(f'load_onnx: {ret}')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f'build: {ret}')
    rknn.export_rknn(str(RKNN_BS8))
    rknn.release()
    print(f'    RKNN: {RKNN_BS8}', flush=True)


def bench_infer(rknn, tensor, iters=30):
    for _ in range(3):
        rknn.inference(inputs=[tensor])
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        rknn.inference(inputs=[tensor])
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    med = ts[len(ts) // 2]
    return med, min(ts)


def main():
    export_onnx_bs8()
    build_rknn()

    from rknnlite.api import RKNNLite

    print('[3] Замер на реальном NPU...', flush=True)

    # bs1: существующий кеш yolo26n_fp16.rknn
    lite1 = RKNNLite()
    lite1.load_rknn(str(BS1_RKNN))
    lite1.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    t1 = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32) * 255.0
    med1, _ = bench_infer(lite1, t1)
    print(f'    bs1 ×8 (последоват.): {med1*8:.1f} ms на 8 кадров ({med1:.1f} ms/кадр)', flush=True)

    # bs8: экспериментальная модель
    lite8 = RKNNLite()
    lite8.load_rknn(str(RKNN_BS8))
    lite8.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    t8 = np.random.rand(BATCH, 3, IMGSZ, IMGSZ).astype(np.float32) * 255.0
    med8, _ = bench_infer(lite8, t8)
    print(f'    bs8 ×1 (батч):       {med8:.1f} ms на 8 кадров ({med8/8:.1f} ms/кадр)', flush=True)

    print(f'\n    УСКОРЕНИЕ батча: {med1*8/med8:.2f}x (bs1×8 = {med1*8:.0f}ms vs bs8 = {med8:.0f}ms)')
    print(f'    ТМП: {TMP} (удалить вручную при желании)')

    lite1.release()
    lite8.release()


if __name__ == '__main__':
    main()
