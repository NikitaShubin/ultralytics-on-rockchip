#!/usr/bin/env python3
"""Бенчмарк FPS для одной модели с использованием production Pipeline.

Запускается внутри контейнера percept.
Использование: python3 fps_bench.py <model_name> <task> --temp-dir <dir> --data-dir <dir> [--quantize none|int8|both]

Для каждой модели тестирует:
  ONNX на CPU     — FPS (без CPU load, всегда ~100%)
  RKNN FP16 на NPU — FPS + NPU load + CPU load
  RKNN INT8 на NPU — FPS + NPU load + CPU load
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
    """Сэмплирование загрузки NPU из sysfs (per-core)."""
    pattern = re.compile(r'Core(\d+):\s*(\d+)%')
    while not stop_event.is_set():
        try:
            with open(NPU_LOAD_PATH) as f:
                content = f.read()
            matches = pattern.findall(content)
            if matches:
                avg_load = sum(int(m) for _, m in matches) / len(matches)
                samples.append(avg_load)
        except (FileNotFoundError, PermissionError, OSError):
            pass
        stop_event.wait(0.2)


def _sample_npu_load_per_core(samples_per_core: dict[int, list[float]], stop_event: threading.Event) -> None:
    """Сэмплирование загрузки NPU per-core из sysfs."""
    pattern = re.compile(r'Core(\d+):\s*(\d+)%')
    while not stop_event.is_set():
        try:
            with open(NPU_LOAD_PATH) as f:
                content = f.read()
            matches = pattern.findall(content)
            for core_id, load in matches:
                cid = int(core_id)
                if cid not in samples_per_core:
                    samples_per_core[cid] = []
                samples_per_core[cid].append(int(load))
        except (FileNotFoundError, PermissionError, OSError):
            pass
        stop_event.wait(0.2)


def _read_cpu_stat_per_core() -> dict[int, tuple[int, int]]:
    """Прочитать время CPU per-core из /proc/stat. Возвращает {core: (idle, total)}."""
    stats = {}
    try:
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu') and not line.startswith('cpu '):
                    parts = line.split()
                    core_id = int(parts[0][3:])
                    values = [int(x) for x in parts[1:9]]
                    idle = values[3] + values[4]
                    total = sum(values)
                    stats[core_id] = (idle, total)
    except (OSError, ValueError):
        pass
    return stats


def _sample_cpu_load_per_core(samples_per_core: dict[int, list[float]], stop_event: threading.Event) -> None:
    """Сэмплирование загрузки CPU per-core из /proc/stat."""
    prev_stats = _read_cpu_stat_per_core()
    while not stop_event.is_set():
        stop_event.wait(0.5)
        cur_stats = _read_cpu_stat_per_core()
        for core_id, (cur_idle, cur_total) in cur_stats.items():
            if core_id in prev_stats:
                prev_idle, prev_total = prev_stats[core_id]
                d_idle = cur_idle - prev_idle
                d_total = cur_total - prev_total
                if d_total > 0:
                    usage = (1.0 - d_idle / d_total) * 100.0
                    if core_id not in samples_per_core:
                        samples_per_core[core_id] = []
                    samples_per_core[core_id].append(round(usage, 1))
        prev_stats = cur_stats


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


def _load_input_size(model_name: str, temp_dir: str) -> int:
    """Загрузить input_size из файла постпроцесса."""
    import importlib.util
    post_path = Path(temp_dir) / f'{model_name}_postprocess.py'
    if not post_path.exists():
        post_path = Path(temp_dir) / 'postprocess.py'
    if post_path.exists():
        spec = importlib.util.spec_from_file_location('_pp', str(post_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, 'INPUT_SIZE', 640)
    return 640


def benchmark_backend(
    model_name: str,
    temp_dir: str,
    images: list[np.ndarray],
    backend: Backend,
    rknn_suffix: str = '',
    warmup_sec: int = 10,
    measure_sec: int = 20,
    collect_npu_load: bool = False,
    collect_cpu_load: bool = False,
) -> dict:
    """Бенчмарк FPS на заданном бэкенде (CPU или NPU)."""
    pipe = Pipeline(model_dir=temp_dir, model_name=model_name, backend=backend,
                    rknn_suffix=rknn_suffix)

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
    npu_load_per_core: dict[int, list[float]] = {}
    cpu_load_per_core: dict[int, list[float]] = {}
    stop_event = threading.Event()
    sampler_threads: list[threading.Thread] = []

    if collect_npu_load:
        t = threading.Thread(target=_sample_npu_load, args=(npu_load_samples, stop_event), daemon=True)
        t.start()
        sampler_threads.append(t)
        t2 = threading.Thread(target=_sample_npu_load_per_core, args=(npu_load_per_core, stop_event), daemon=True)
        t2.start()
        sampler_threads.append(t2)

    if collect_cpu_load:
        t = threading.Thread(target=_sample_cpu_load_per_core, args=(cpu_load_per_core, stop_event), daemon=True)
        t.start()
        sampler_threads.append(t)

    t_start = time.monotonic()
    timestamps = []
    while time.monotonic() - t_start < measure_sec:
        img = next_image()
        pipe.detect(img)
        timestamps.append(time.monotonic())

    stop_event.set()
    for t in sampler_threads:
        t.join(timeout=2)

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
    cpu_load_avg = round(float(np.mean([v for cores in cpu_load_per_core.values() for v in cores])), 1) if cpu_load_per_core else None

    npu_per_core = {}
    for cid in sorted(npu_load_per_core.keys()):
        vals = npu_load_per_core[cid]
        npu_per_core[str(cid)] = round(float(np.mean(vals)), 1) if vals else 0

    cpu_per_core = {}
    for cid in sorted(cpu_load_per_core.keys()):
        vals = cpu_load_per_core[cid]
        cpu_per_core[str(cid)] = round(float(np.mean(vals)), 1) if vals else 0

    return {
        'backend': backend.value,
        'rknn_suffix': rknn_suffix,
        'fps': round(fps, 2),
        'fps_std': round(fps_std, 2),
        'n_frames': n_measure,
        'n_warmup': n_warmup,
        'elapsed_s': round(elapsed, 2),
        'warmup_s': warmup_sec,
        'input_size': pipe.input_size,
        'npu_load_avg': npu_load_avg,
        'npu_load_per_core': npu_per_core,
        'cpu_load_avg': cpu_load_avg,
        'cpu_load_per_core': cpu_per_core,
    }


def main() -> None:
    """Основная функция бенчмарка."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Имя модели (напр. yolo11n)')
    parser.add_argument('task', help='Тип задачи (detect, segment, pose, obb, classify, semantic)')
    parser.add_argument('--temp-dir', default='/app/benchmark/temp_models', help='Директория с моделями')
    parser.add_argument('--data-dir', default='/app/datasets/coco_val2017/images/val2017', help='Директория тестовых изображений')
    parser.add_argument('--warmup', type=int, default=10, help='Секунды прогрева')
    parser.add_argument('--measure', type=int, default=20, help='Секунды измерения')
    parser.add_argument('--quantize', choices=['onnx', 'none', 'int8', 'int8-calib', 'all'], default='all',
                        help='Какие варианты тестировать: onnx=только CPU, none=FP16, int8=INT8, int8-calib=INT8 с калибровкой, all=все')
    args = parser.parse_args()

    images = load_test_images(args.data_dir)
    if not images:
        sys.stderr.write(f'[ОШИБКА] нет тестовых изображений в {args.data_dir}\n')
        sys.exit(1)

    sys.stderr.write(f'[ИНФО] загружено {len(images)} тестовых изображений из {args.data_dir}\n')
    sys.stderr.write(f'[ИНФО] бенчмарк: {args.model_name}, задача={args.task}, '
                     f'прогрев={args.warmup}с, измерение={args.measure}с, '
                     f'квантизация={args.quantize}\n')

    temp_dir = Path(args.temp_dir)
    results = {
        'model': args.model_name,
        'task': args.task,
    }

    def _has_rknn(suffix: str) -> bool:
        """Проверить наличие RKNN модели: именованный файл или model.rknn."""
        if suffix:
            named = temp_dir / f'{args.model_name}{suffix}.rknn'
            if named.exists():
                return True
        return (temp_dir / 'model.rknn').exists()

    # Определяем какие RKNN варианты тестировать
    rknn_variants = []
    if args.quantize in ('none', 'all'):
        if _has_rknn('_fp16'):
            rknn_variants.append(('none', 'NPU FP16'))
        else:
            sys.stderr.write(f'[ПРОПУСК] {_has_rknn.__name__}: fp16.rknn не найден\n')
    if args.quantize in ('int8', 'all'):
        if _has_rknn('_int8'):
            rknn_variants.append(('int8', 'NPU INT8'))
        else:
            sys.stderr.write(f'[ПРОПУСК] int8.rknn не найден\n')
    if args.quantize in ('int8-calib', 'all'):
        if _has_rknn('_int8-calib'):
            rknn_variants.append(('int8-calib', 'NPU INT8-calib'))
        else:
            sys.stderr.write(f'[ПРОПУСК] int8-calib.rknn не найден\n')

    # ONNX на CPU (если запрошен)
    has_onnx = (temp_dir / f'{args.model_name}.onnx').exists() or (temp_dir / 'model.onnx').exists()
    if has_onnx and args.quantize in ('onnx', 'all'):
        try:
            sys.stderr.write(f'[ИНФО] бенчмарк ONNX CPU...\n')
            bench = benchmark_backend(
                args.model_name, args.temp_dir, images, Backend.CPU,
                rknn_suffix='',
                warmup_sec=args.warmup, measure_sec=args.measure,
                collect_npu_load=False, collect_cpu_load=False,
            )
            sys.stderr.write(f'[ИНФО] ONNX CPU FPS={bench["fps"]} '
                             f'(n={bench["n_frames"]}, {bench["elapsed_s"]}с)\n')
            results['cpu'] = bench
        except Exception as e:
            import traceback
            sys.stderr.write(f'[ОШИБКА] ONNX CPU: {e}\n')
            sys.stderr.write(traceback.format_exc() + '\n')
            results['cpu'] = {'backend': 'cpu', 'error': str(e), 'fps': -1}
    else:
        sys.stderr.write(f'[ПРОПУСК] {args.model_name}.onnx не найден\n')
        results['cpu'] = {'backend': 'cpu', 'error': 'onnx не найден', 'fps': -1}

    # RKNN варианты на NPU
    for suffix, label in rknn_variants:
        try:
            sys.stderr.write(f'[ИНФО] бенчмарк {label}...\n')
            bench = benchmark_backend(
                args.model_name, args.temp_dir, images, Backend.NPU,
                rknn_suffix={'none': '_fp16', 'int8': '_int8', 'int8-calib': '_int8-calib'}.get(suffix, ''),
                warmup_sec=args.warmup, measure_sec=args.measure,
                collect_npu_load=True, collect_cpu_load=True,
            )
            sys.stderr.write(f'[ИНФО] {label} FPS={bench["fps"]} '
                             f'(n={bench["n_frames"]}, {bench["elapsed_s"]}с, '
                             f'NPU={bench["npu_load_avg"]}%, CPU={bench["cpu_load_avg"]}%)\n')
            results[f'npu_{suffix}'] = bench
        except Exception as e:
            import traceback
            sys.stderr.write(f'[ОШИБКА] {label}: {e}\n')
            sys.stderr.write(traceback.format_exc() + '\n')
            results[f'npu_{suffix}'] = {'backend': 'npu', 'rknn_suffix': suffix, 'error': str(e), 'fps': -1}

    # Если all — также запускаем onnx-only, none, int8, int8-calib по отдельности
    if args.quantize == 'all':
        # Уже запущены все варианты выше
        pass

    print(json.dumps(results))


if __name__ == '__main__':
    main()
