#!/usr/bin/env python3
"""Бенчмарк FPS для одной модели с использованием production Pipeline.

Запускается внутри контейнера percept.
Использование: python3 fps_bench.py <model_name> <task> --temp-dir <dir> --data-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/app')
from pipeline import Pipeline  # noqa: E402
from config import Backend  # noqa: E402

NPU_LOAD_PATH = '/sys/kernel/debug/rknpu/load'


def _sample_npu_load(samples: list[float], stop_event: threading.Event) -> None:
    """Сэмплирование загрузки NPU из sysfs."""
    pattern = re.compile(r'Core\d+:\s*(\d+)%')
    while not stop_event.is_set():
        try:
            with open(NPU_LOAD_PATH) as f:
                content = f.read()
            matches = pattern.findall(content)
            if matches:
                avg_load = sum(int(m) for m in matches) / len(matches)
                samples.append(avg_load)
        except (FileNotFoundError, PermissionError, OSError):
            pass
        stop_event.wait(0.2)


def load_test_images(data_dir: str, limit: int = 100) -> list[np.ndarray]:
    """Загрузить тестовые изображения из директории."""
    data_dir = Path(data_dir)
    images: list[np.ndarray] = []
    for f in sorted(data_dir.rglob('*.*')):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            if any(p.name in ('labels', 'label') for p in f.parents):
                continue
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)
                if len(images) >= limit:
                    break
    return images


def benchmark_backend(
    model_name: str,
    temp_dir: str,
    images: list[np.ndarray],
    backend: Backend,
    warmup_sec: int = 10,
    measure_sec: int = 20,
    collect_npu_load: bool = False,
) -> dict:
    """Бенчмарк FPS на заданном бэкенде (CPU или NPU)."""
    pipe = Pipeline(model_dir=temp_dir, model_name=model_name, backend=backend)

    n_images = len(images)
    idx = 0

    def next_image():
        nonlocal idx
        img = images[idx % n_images]
        idx += 1
        return img

    t_start = time.monotonic()
    n_warmup = 0
    while time.monotonic() - t_start < warmup_sec:
        img = next_image()
        pipe.detect(img)
        n_warmup += 1

    npu_load_samples: list[float] = []
    stop_event = threading.Event()
    sampler_thread: threading.Thread | None = None
    if collect_npu_load:
        sampler_thread = threading.Thread(
            target=_sample_npu_load, args=(npu_load_samples, stop_event), daemon=True
        )
        sampler_thread.start()

    t_start = time.monotonic()
    timestamps = []
    while time.monotonic() - t_start < measure_sec:
        img = next_image()
        pipe.detect(img)
        timestamps.append(time.monotonic())

    if sampler_thread:
        stop_event.set()
        sampler_thread.join(timeout=2)

    elapsed = timestamps[-1] - t_start if timestamps else 0
    n_measure = len(timestamps)
    fps = n_measure / elapsed if elapsed > 0 else 0

    fps_std = 0.0
    if len(timestamps) >= 2:
        intervals = np.diff(timestamps)
        fps_values = 1.0 / intervals
        fps_std = float(np.std(fps_values, ddof=1))

    if hasattr(pipe, '_backend') and pipe._backend.rknn:
        pipe._backend.rknn.release()

    npu_load_avg = round(float(np.mean(npu_load_samples)), 1) if npu_load_samples else None

    return {
        'backend': backend.value,
        'fps': round(fps, 2),
        'fps_std': round(fps_std, 2),
        'n_frames': n_measure,
        'n_warmup': n_warmup,
        'elapsed_s': round(elapsed, 2),
        'warmup_s': warmup_sec,
        'input_size': pipe.input_size,
        'npu_load_avg': npu_load_avg,
    }


def _load_input_size(model_name: str, temp_dir: str) -> int:
    """Загрузить input_size из файла постпроцесса."""
    import importlib.util
    post_path = Path(temp_dir) / f'{model_name}_postprocess.py'
    if post_path.exists():
        spec = importlib.util.spec_from_file_location('_pp_validation', str(post_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, 'INPUT_SIZE', 640)
    return 640


def run_validation(
    model_name: str,
    temp_dir: str,
    images: list[np.ndarray],
) -> dict:
    """Валидация ONNX vs RKNN на NPU hardware."""
    onnx_path = Path(temp_dir) / f'{model_name}.onnx'
    rknn_path = Path(temp_dir) / f'{model_name}.rknn'

    if not onnx_path.exists() or not rknn_path.exists():
        return {'error': 'onnx или rknn не найдены', 'max_diff_avg': -1, 'max_diff_max': -1, 'n_images': 0}

    input_size = _load_input_size(model_name, temp_dir)

    import onnxruntime as rt
    from rknnlite.api import RKNNLite

    sess = rt.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        return {'error': f'ошибка загрузки RKNN {ret}', 'max_diff_avg': -1, 'max_diff_max': -1, 'n_images': 0}
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        rknn.release()
        return {'error': f'ошибка инициализации RKNN {ret}', 'max_diff_avg': -1, 'max_diff_max': -1, 'n_images': 0}

    max_diffs = []
    for img in images[:16]:
        h, w = img.shape[:2]
        scale = min(input_size / w, input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w = input_size - new_w
        pad_h = input_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)

        onnx_out = sess.run(None, {sess.get_inputs()[0].name: tensor / 255.0})
        rknn_out = rknn.inference(inputs=[tensor.transpose(0, 2, 3, 1)])

        if not isinstance(onnx_out, (list, tuple)):
            onnx_out = [onnx_out]

        max_diff = 0.0
        for o, r in zip(onnx_out, rknn_out):
            diff = np.abs(o.astype(np.float32) - r.astype(np.float32)).max()
            max_diff = max(max_diff, float(diff))
        max_diffs.append(max_diff)

    rknn.release()

    return {
        'max_diff_avg': float(np.mean(max_diffs)) if max_diffs else -1,
        'max_diff_max': float(np.max(max_diffs)) if max_diffs else -1,
        'n_images': len(max_diffs),
        'validation_mode': 'hardware',
    }


def main() -> None:
    """Основная функция бенчмарка."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Model name (e.g. yolo11n)')
    parser.add_argument('task', help='Task type (detect, segment, pose, obb, classify, semantic)')
    parser.add_argument('--temp-dir', default='/app/benchmark/temp_models', help='Model files directory')
    parser.add_argument('--data-dir', default='/app/coco_val2017/images/val2017', help='Test images directory')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup seconds per backend')
    parser.add_argument('--measure', type=int, default=20, help='Measure seconds per backend')
    args = parser.parse_args()

    images = load_test_images(args.data_dir)
    if not images:
        sys.stderr.write(f'[ОШИБКА] нет тестовых изображений в {args.data_dir}\n')
        sys.exit(1)

    sys.stderr.write(f'[ИНФО] загружено {len(images)} тестовых изображений из {args.data_dir}\n')
    sys.stderr.write(f'[ИНФО] бенчмарк: {args.model_name}, задача={args.task}, '
                     f'прогрев={args.warmup}с, измерение={args.measure}с\n')

    results = {
        'model': args.model_name,
        'task': args.task,
        'cpu': None,
        'npu': None,
    }

    for backend in [Backend.CPU, Backend.NPU]:
        try:
            sys.stderr.write(f'[ИНФО] бенчмарк {backend.value}...\n')
            bench = benchmark_backend(
                args.model_name,
                args.temp_dir,
                images,
                backend,
                warmup_sec=args.warmup,
                measure_sec=args.measure,
                collect_npu_load=(backend == Backend.NPU),
            )
            sys.stderr.write(f'[ИНФО] {backend.value} FPS={bench["fps"]} '
                             f'(n={bench["n_frames"]}, {bench["elapsed_s"]}с)\n')
            results[backend.value] = bench
        except Exception as e:
            import traceback
            sys.stderr.write(f'[ОШИБКА] {backend.value} не удался: {e}\n')
            sys.stderr.write(traceback.format_exc() + '\n')
            results[backend.value] = {'backend': backend.value, 'error': str(e), 'fps': -1}

    validation = run_validation(args.model_name, args.temp_dir, images)
    sys.stderr.write(f'[ВАЛИДАЦИЯ] max_diff_avg={validation.get("max_diff_avg", -1):.4f} '
                     f'max={validation.get("max_diff_max", -1):.4f}\n')
    results['validation'] = validation

    print(json.dumps(results))


if __name__ == '__main__':
    main()
