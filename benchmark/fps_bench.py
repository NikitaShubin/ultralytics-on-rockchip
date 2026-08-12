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

from telemetry import (  # noqa: E402
    NPU_LOAD_PATH,
    read_npu_temp, read_npu_freq, read_cpu_freqs, read_cpu_stat_per_core,
    cpu_core_max_freq, read_thermal_max, throttle_reasons, wait_no_throttle,
)


def _sample_telemetry(npu_temp_samples: list[float], npu_freq_samples: list[int],
                      cpu_freq_per_core: dict[int, list[int]],
                      thermal_samples: list[float], stop_event: threading.Event) -> None:
    """Сэмплирование температуры NPU, частот NPU/CPU и температуры всех зон."""
    while not stop_event.is_set():
        t = read_npu_temp()
        if t is not None:
            npu_temp_samples.append(t)
        f = read_npu_freq()
        if f is not None:
            npu_freq_samples.append(f)
        for cid, f in read_cpu_freqs().items():
            cpu_freq_per_core.setdefault(cid, []).append(f)
        t = read_thermal_max()
        if t is not None:
            thermal_samples.append(t)
        stop_event.wait(0.2)


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


def _sample_cpu_load_per_core(samples_per_core: dict[int, list[float]], stop_event: threading.Event) -> None:
    """Сэмплирование загрузки CPU per-core из /proc/stat."""
    prev_stats = read_cpu_stat_per_core()
    while not stop_event.is_set():
        stop_event.wait(0.5)
        cur_stats = read_cpu_stat_per_core()
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
    temp_samples: list[float] = []
    npu_freq_samples: list[int] = []
    cpu_freq_per_core: dict[int, list[int]] = {}
    thermal_samples: list[float] = []
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

    t = threading.Thread(target=_sample_telemetry,
                         args=(temp_samples, npu_freq_samples, cpu_freq_per_core, thermal_samples, stop_event),
                         daemon=True)
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

    npu_temp_avg = round(float(np.mean(temp_samples)), 1) if temp_samples else None
    npu_temp_min = round(float(min(temp_samples)), 1) if temp_samples else None
    npu_temp_max = round(float(max(temp_samples)), 1) if temp_samples else None
    npu_freq_avg = round(float(np.mean(npu_freq_samples)) / 1e6, 0) if npu_freq_samples else None
    npu_freq_min = round(float(min(npu_freq_samples)) / 1e6, 0) if npu_freq_samples else None
    thermal_max = round(float(max(thermal_samples)), 1) if thermal_samples else None

    cpu_freq_ratio = None
    if cpu_freq_per_core:
        ratios = []
        for cid, vals in cpu_freq_per_core.items():
            mx = cpu_core_max_freq(cid)
            if not mx:
                continue
            loads = cpu_load_per_core.get(cid)
            loaded = bool(loads) and float(np.mean(loads)) > 50.0
            if not loaded:
                continue
            ratios.append(float(np.mean(vals)) / mx)
        cpu_freq_ratio = round(min(ratios), 2) if ratios else None

    reasons = throttle_reasons(npu_freq_avg, npu_load_avg, cpu_freq_ratio, thermal_max)

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
        'npu_temp_avg': npu_temp_avg,
        'npu_temp_min': npu_temp_min,
        'npu_temp_max': npu_temp_max,
        'npu_freq_avg': npu_freq_avg,
        'npu_freq_min': npu_freq_min,
        'thermal_max_c': thermal_max,
        'cpu_freq_ratio': cpu_freq_ratio,
        'throttled': bool(reasons),
        'throttle_reason': '; '.join(reasons) or None,
    }


def _bench_median(repeats: int, bench_fn, throttle_timeout: int = 600) -> dict:
    """Запустить benchmark_backend `repeats` раз, вернуть повтор с медианным fps.

    Перед каждым повтором ждёт выхода из троттлинга (температура любой
    thermal-зоны). Если за throttle_timeout секунд не дождались — возвращает
    результат с ошибкой 'throttle_timeout' (fps=-1, throttled=True), и повтор
    НЕ запускается.
    """
    runs = []
    for _ in range(max(1, repeats)):
        if not wait_no_throttle(throttle_timeout):
            return {'fps': -1, 'fps_std': None, 'n_frames': 0, 'n_warmup': 0,
                    'elapsed_s': None, 'input_size': None,
                    'npu_load_avg': None, 'npu_load_per_core': {},
                    'cpu_load_avg': None, 'cpu_load_per_core': {},
                    'npu_temp_avg': None, 'npu_temp_min': None, 'npu_temp_max': None,
                    'npu_freq_avg': None, 'npu_freq_min': None, 'thermal_max_c': None,
                    'cpu_freq_ratio': None,
                    'throttled': True, 'throttle_reason': 'throttle_timeout',
                    'error': 'throttle_timeout'}
        runs.append(bench_fn())
    fps_list = [r['fps'] for r in runs]
    fps_list_sorted = sorted(fps_list)
    med = fps_list_sorted[len(fps_list_sorted) // 2]
    chosen = next(r for r in runs if r['fps'] == med)
    chosen['fps_min'] = min(fps_list)
    chosen['fps_max'] = max(fps_list)
    chosen['n_repeats'] = len(fps_list)
    return chosen


def main() -> None:
    """Основная функция бенчмарка."""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Имя модели (напр. yolo11n)')
    parser.add_argument('task', help='Тип задачи (detect, segment, pose, obb, classify, semantic)')
    parser.add_argument('--temp-dir', default='/app/benchmark/temp_models', help='Директория с моделями')
    parser.add_argument('--data-dir', default='/app/datasets/coco_val2017/images/val2017', help='Директория тестовых изображений')
    parser.add_argument('--warmup', type=int, default=10, help='Секунды прогрева')
    parser.add_argument('--measure', type=int, default=20, help='Секунды измерения на повтор')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Сколько раз повторить каждый вариант (возвращается медиана fps)')
    parser.add_argument('--quantize', choices=['onnx', 'none', 'int8', 'int8-calib', 'all'], default='all',
                        help='Какие варианты тестировать: onnx=только CPU, none=FP16, int8=INT8, int8-calib=INT8 с калибровкой, all=все')
    parser.add_argument('--throttle-timeout', type=int, default=600,
                        help='Секунд ждать выхода из троттлинга перед повтором; при таймауте вариант пропускается')
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
            sys.stderr.write(f'[ИНФО] бенчмарк ONNX CPU (repeats={args.repeats})...\n')
            bench = _bench_median(args.repeats, lambda: benchmark_backend(
                args.model_name, args.temp_dir, images, Backend.CPU,
                rknn_suffix='',
                warmup_sec=args.warmup, measure_sec=args.measure,
                collect_npu_load=False, collect_cpu_load=True,
            ), throttle_timeout=args.throttle_timeout)
            sys.stderr.write(f'[ИНФО] ONNX CPU FPS={bench["fps"]} '
                             f'(n={bench["n_frames"]}, {bench["elapsed_s"]}с, '
                             f'min={bench.get("fps_min")}, max={bench.get("fps_max")})\n')
            if bench.get('throttled'):
                sys.stderr.write(f'[ПРЕДУПРЕЖДЕНИЕ] ONNX CPU: троттлинг ({bench.get("throttle_reason")})\n')
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
            sys.stderr.write(f'[ИНФО] бенчмарк {label} (repeats={args.repeats})...\n')
            bench = _bench_median(args.repeats, lambda s=suffix, l=label: benchmark_backend(
                args.model_name, args.temp_dir, images, Backend.NPU,
                rknn_suffix={'none': '_fp16', 'int8': '_int8', 'int8-calib': '_int8-calib'}.get(s, ''),
                warmup_sec=args.warmup, measure_sec=args.measure,
                collect_npu_load=True, collect_cpu_load=True,
            ), throttle_timeout=args.throttle_timeout)
            sys.stderr.write(f'[ИНФО] {label} FPS={bench["fps"]} '
                             f'(n={bench["n_frames"]}, {bench["elapsed_s"]}с, '
                             f'NPU={bench["npu_load_avg"]}%, CPU={bench["cpu_load_avg"]}%, '
                             f'temp={bench.get("npu_temp_avg")}°C[{bench.get("npu_temp_min")}..{bench.get("npu_temp_max")}], '
                             f'min={bench.get("fps_min")}, max={bench.get("fps_max")})\n')
            if bench.get('throttled'):
                sys.stderr.write(f'[ПРЕДУПРЕЖДЕНИЕ] {label}: троттлинг ({bench.get("throttle_reason")})\n')
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

    sys.stdout.flush()
    sys.stdout.write('\n' + json.dumps(results) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
