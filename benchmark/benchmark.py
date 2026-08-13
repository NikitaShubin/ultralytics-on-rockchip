#!/usr/bin/env python3
"""Оркестратор бенчмарка — запускается на HOST (Orange Pi 5 Pro).

Четыре фазы:
  convert  — конвертация .pt → .onnx + _fp16.rknn + _int8.rknn + _int8-calib.rknn
  validate — валидация ONNX vs RKNN (реальный NPU, fallback — симулятор)
  bench    — FPS + нагрузка на CPU/NPU (ARM железо)
  eval     — точность (eval.py, все 4 quantize-варианта)

Использование:
    python3 benchmark.py --phase convert                    # конвертация
    python3 benchmark.py --phase validate                   # валидация
    python3 benchmark.py --phase bench                      # FPS
    python3 benchmark.py --phase eval                       # точность
    python3 benchmark.py --phase all                        # всё (convert → validate → bench → eval)
    python3 benchmark.py --phase bench --quantize int8      # только INT8 FPS
    python3 benchmark.py --phase all --model yolo11n
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from models import FULL as MODELS, PILOT as PILOT_MODELS

BENCHMARK_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BENCHMARK_DIR.parent
CACHE_DIR = PROJECT_DIR / 'cache'
DATA_DIR = BENCHMARK_DIR / 'data'
RESULTS_DIR = BENCHMARK_DIR / 'results'
CONVERT_COMPOSE = PROJECT_DIR / 'convert' / 'docker-compose.yml'
WEB_COMPOSE = PROJECT_DIR / 'docker-compose.yml'
RESULTS_FILE = RESULTS_DIR / 'results.json'
SUMMARY_TSV = RESULTS_DIR / 'summary.tsv'
LOG_FILE = RESULTS_DIR / 'benchmark.log'

COCO_VAL_DIR = PROJECT_DIR / 'datasets' / 'coco_val2017'
COCO_VAL_CONTAINER = '/app/datasets/coco_val2017'

DOTA_VAL_DIR = PROJECT_DIR / 'datasets' / 'dotav1'
DOTA_VAL_CONTAINER = '/app/datasets/dotav1'

CITYSCAPES_VAL_DIR = PROJECT_DIR / 'datasets' / 'cityscapes'
CITYSCAPES_VAL_CONTAINER = '/app/datasets/cityscapes'

IMAGENET_VAL_DIR = PROJECT_DIR / 'datasets' / 'imagenet'
IMAGENET_VAL_CONTAINER = '/app/datasets/imagenet'

WARMUP_S = int(os.environ.get('BENCH_WARMUP', '10'))
MEASURE_S = int(os.environ.get('BENCH_MEASURE', '30'))
BENCH_REPEATS = int(os.environ.get('BENCH_REPEATS', '3'))
CONVERT_TIMEOUT = int(os.environ.get('BENCH_CONVERT_TIMEOUT', '7200'))
EVAL_TIMEOUT = int(os.environ.get('BENCH_EVAL_TIMEOUT', '14400'))
QUANTIZE = os.environ.get('BENCH_QUANTIZE', 'all')
PERCEPT_CONTAINER = 'percept'

# quantize-варианты eval (eval.py --backend/--rknn-suffix и файл в cache/<model>/)
EVAL_VARIANTS = {
    'onnx': {'backend': 'cpu', 'suffix': '', 'file': lambda n: f'{n}.onnx'},
    'fp16': {'backend': 'npu', 'suffix': '_fp16', 'file': lambda n: f'{n}_fp16.rknn'},
    'int8': {'backend': 'npu', 'suffix': '_int8', 'file': lambda n: f'{n}_int8.rknn'},
    'int8-calib': {'backend': 'npu', 'suffix': '_int8-calib', 'file': lambda n: f'{n}_int8-calib.rknn'},
}

QUANTIZE_TO_EVAL = {
    'onnx': ['onnx'],
    'none': ['fp16'],
    'int8': ['int8'],
    'int8-calib': ['int8-calib'],
    'all': ['onnx', 'fp16', 'int8', 'int8-calib'],
}

# Имена задач: models.py → eval.py
TASK_TO_EVAL = {
    'detect': 'detect',
    'seg': 'segment',
    'cls': 'classify',
    'pose': 'pose',
    'obb': 'obb',
    'sem': 'semantic',
}

SETUP_SCRIPT = PROJECT_DIR / 'setup_datasets.py'
DATASET_FLAG = {
    'coco_val2017': '--coco',
    'dotav1': '--dota',
    'cityscapes': '--cityscapes',
    'imagenet': '--imagenet',
}


def ensure_dataset(model) -> None:
    """Убедиться что датасет для модели доступен (скачать если нужно)."""
    flag = DATASET_FLAG.get(model.dataset)
    if not flag:
        return
    log(f'[DATASET] Проверяю {model.dataset}...')
    subprocess.run(
        [sys.executable, str(SETUP_SCRIPT), flag],
        cwd=str(PROJECT_DIR),
        timeout=7200,
        check=True,
    )


def log(msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------------------
# Управление контейнером percept (освобождение NPU для bench)
# ---------------------------------------------------------------------------

def _container_running(name: str) -> bool:
    r = subprocess.run(
        ['docker', 'inspect', '-f', '{{.State.Running}}', name],
        capture_output=True, text=True, timeout=10,
    )
    return r.returncode == 0 and r.stdout.strip() == 'true'


def _container_has_restart_policy(name: str) -> bool:
    r = subprocess.run(
        ['docker', 'inspect', '-f', '{{.HostConfig.RestartPolicy.Name}}', name],
        capture_output=True, text=True, timeout=10,
    )
    return r.returncode == 0 and r.stdout.strip() != 'no'


def stop_percept() -> bool:
    """Остановить persistent контейнер percept, если запущен. Возвращает True если был запущен."""
    if not _container_running(PERCEPT_CONTAINER):
        return False
    log(f'[PERCEPT] Останавливаю контейнер {PERCEPT_CONTAINER} для освобождения NPU...')
    subprocess.run(
        ['docker', 'stop', PERCEPT_CONTAINER],
        capture_output=True, text=True, timeout=30,
    )
    time.sleep(2)
    if _container_running(PERCEPT_CONTAINER):
        log(f'[PERCEPT] Контейнер не остановился, принудительно kill...')
        subprocess.run(
            ['docker', 'kill', PERCEPT_CONTAINER],
            capture_output=True, text=True, timeout=10,
        )
        time.sleep(1)
    log(f'[PERCEPT] Контейнер остановлен')
    return True


def restart_percept(was_running: bool) -> None:
    """Перезапустить контейнер percept, если он был запущен до bench."""
    if not was_running:
        return
    if _container_running(PERCEPT_CONTAINER):
        return
    log(f'[PERCEPT] Перезапускаю контейнер {PERCEPT_CONTAINER}...')
    subprocess.run(
        ['docker', 'start', PERCEPT_CONTAINER],
        capture_output=True, text=True, timeout=30,
    )
    time.sleep(2)
    if _container_running(PERCEPT_CONTAINER):
        log(f'[PERCEPT] Контейнер запущен')
    else:
        log(f'[PERCEPT] Не удалось перезапустить контейнер')


def _container_data_path(data_dir: Path) -> str:
    """Сопоставление host data_dir с путём контейнера."""
    if data_dir.name == 'coco_val2017' or data_dir == COCO_VAL_DIR:
        return f'{COCO_VAL_CONTAINER}/images/val2017'
    if data_dir.name == 'dotav1' or data_dir == DOTA_VAL_DIR:
        return f'{DOTA_VAL_CONTAINER}/images/val'
    if data_dir.name == 'cityscapes' or data_dir == CITYSCAPES_VAL_DIR:
        return f'{CITYSCAPES_VAL_CONTAINER}/images/val'
    if data_dir.name == 'imagenet' or data_dir == IMAGENET_VAL_DIR:
        return f'{IMAGENET_VAL_CONTAINER}/val'
    return f'/app/benchmark/data/{data_dir.name}'


def _container_dataset_root(data_dir: Path) -> str:
    """Корень датасета в контейнере (для eval.py, который сам ищет аннотации)."""
    if data_dir.name == 'coco_val2017' or data_dir == COCO_VAL_DIR:
        return COCO_VAL_CONTAINER
    if data_dir.name == 'dotav1' or data_dir == DOTA_VAL_DIR:
        return DOTA_VAL_CONTAINER
    if data_dir.name == 'cityscapes' or data_dir == CITYSCAPES_VAL_DIR:
        return CITYSCAPES_VAL_CONTAINER
    if data_dir.name == 'imagenet' or data_dir == IMAGENET_VAL_DIR:
        return IMAGENET_VAL_CONTAINER
    return f'/app/benchmark/data/{data_dir.name}'


# ---------------------------------------------------------------------------
# Фаза: convert
# ---------------------------------------------------------------------------

def run_converter(model_name: str, task: str, output_dir: Path, data_dir: Path) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(CONVERT_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{PROJECT_DIR / "datasets"}:/app/datasets',
        '-e', f'CONVERT_QUANTIZE={QUANTIZE}',
        '-e', 'KEEP_MODELS=1',
        '--entrypoint', 'python',
        'convert',
        '/app/benchmark/convert_bench.py',
        model_name, task, f'/app/cache/{model_name}',
        data_path,
    ]
    log(f'ЗАПУСК: {" ".join(cmd[:-6])} ...')
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
        elapsed = time.time() - t0
        log(f'[ТАЙМИНГ] convert завершён за {elapsed:.0f}с')
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] конвертер превысил таймаут {CONVERT_TIMEOUT}с')
        return None

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            log(f'[convert stdout] {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            log(f'[convert stderr] {line}')

    if result.returncode != 0:
        log(f'[ОШИБКА] конвертер завершился с кодом {result.returncode}')
        return None

    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Фаза: validate
# ---------------------------------------------------------------------------

def run_validate(model_name: str, task: str, temp_dir: Path, data_dir: Path, input_size: int = 640) -> dict | None:
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(CONVERT_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{PROJECT_DIR / "datasets"}:/app/datasets',
        '--entrypoint', 'python',
        'convert',
        '/app/benchmark/validate_bench.py',
        model_name, task, f'/app/cache/{model_name}',
        data_path, str(input_size),
    ]
    log(f'ЗАПУСК: {" ".join(cmd[:-6])} ...')
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0
        log(f'[ТАЙМИНГ] validate завершён за {elapsed:.0f}с')
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] валидация превысила таймаут 1800с')
        return None

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            log(f'[validate stdout] {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            log(f'[validate stderr] {line}')

    if result.returncode != 0:
        log(f'[ОШИБКА] валидация завершилась с кодом {result.returncode}')
        return None

    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Фаза: bench
# ---------------------------------------------------------------------------

def run_fps(model_name: str, task: str, cache_dir: Path, data_dir: Path,
            quantize: str = 'both') -> dict | None:
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(WEB_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{CACHE_DIR}:/app/cache',
        '-v', f'{PROJECT_DIR / "datasets"}:/app/datasets',
        '-e', 'TEST_MODE=1',
        '-e', 'WEB_MODE=0',
        '-e', 'CAM_MODE=file',
        '-e', 'PYTHONPATH=/app',
        '-e', f'BENCH_WARMUP={WARMUP_S}',
        '-e', f'BENCH_MEASURE={MEASURE_S}',
        'percept', 'python3', '/app/benchmark/fps_bench.py',
        model_name, task,
        '--temp-dir', f'/app/cache/{model_name}',
        '--data-dir', data_path,
        '--warmup', str(WARMUP_S),
        '--measure', str(MEASURE_S),
        '--repeats', str(BENCH_REPEATS),
        '--quantize', quantize,
    ]
    log(f'ЗАПУСК: {" ".join(cmd[:-9])} ...')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] FPS-тест превысил таймаут 1200с')
        return None

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            log(f'[percept stdout] {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            log(f'[percept stderr] {line}')

    if result.returncode != 0:
        log(f'[ОШИБКА] percept завершился с кодом {result.returncode}')
        return None

    marker = '{"model"'
    idx = result.stdout.rfind(marker)
    if idx < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(result.stdout[idx:])[0]
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Фаза: eval (accuracy)
# ---------------------------------------------------------------------------

def run_eval(model_name: str, task: str, model_dir: Path, data_dir: Path,
             variant: str) -> dict | None:
    """Запустить eval.py (convert-контейнер) для одного quantize-варианта.

    onnx → backend cpu; rknn (fp16/int8/int8-calib) → backend npu
    (реальный NPU через rknnlite в convert-контейнере), суффикс через --rknn-suffix.
    """
    v = EVAL_VARIANTS[variant]
    model_file = model_dir / v['file'](model_name)
    if not model_file.exists():
        log(f'[ПРОПУСК] eval {variant}: нет модели {model_file}')
        return None
    data_path = _container_dataset_root(data_dir)
    task_arg = TASK_TO_EVAL.get(task, task)
    cmd = [
        'docker', 'compose', '-f', str(CONVERT_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{PROJECT_DIR / "datasets"}:/app/datasets',
        '--entrypoint', 'python',
        'convert',
        '/app/benchmark/eval.py',
        task_arg, data_path,
        '--model-dir', f'/app/cache/{model_name}',
        '--model-name', model_name,
        '--backend', v['backend'],
    ]
    if v['suffix']:
        cmd += ['--rknn-suffix', v['suffix']]
    log(f'ЗАПУСК eval {variant}: {" ".join(cmd[-10:])} ...')
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=EVAL_TIMEOUT)
        elapsed = time.time() - t0
        log(f'[ТАЙМИНГ] eval {variant} завершён за {elapsed:.0f}с')
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] eval {variant} превысил таймаут {EVAL_TIMEOUT}с')
        return None

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            log(f'[eval stdout] {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            log(f'[eval stderr] {line}')

    if result.returncode != 0:
        log(f'[ОШИБКА] eval {variant} завершился с кодом {result.returncode}')
        return None

    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def model_is_complete(result: dict) -> bool:
    """Проверить, завершены ли все фазы для модели (convert + validate + bench + eval)."""
    if result.get('conversion_status') != 'OK':
        return False
    v = result.get('validation', {})
    has_val = (v.get('fp16', {}).get('max_diff_avg', -1) >= 0
               or v.get('int8', {}).get('max_diff_avg', -1) >= 0)
    if not has_val:
        return False
    fps = result.get('fps', {})
    has_fps = any(
        fps.get(k, {}).get('fps', -1) > 0
        for k in ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib']
    )
    if not has_fps:
        return False
    eval_result = result.get('eval', {}) or {}
    done_keys = {k for k, val in eval_result.items()
                 if isinstance(val, dict) and val.get('elapsed_s', 0) > 0}
    need_keys = QUANTIZE_TO_EVAL.get(QUANTIZE, ['onnx', 'fp16', 'int8', 'int8-calib'])
    if not all(vr in done_keys for vr in need_keys):
        return False
    return True


def load_results() -> list[dict]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def upsert_result(results: list[dict], entry: dict) -> None:
    for j, r in enumerate(results):
        if r['model'] == entry['model']:
            results[j] = entry
            return
    results.append(entry)


def _per_core_cols(fps_data: dict) -> list:
    """Извлечь per-core колонки: NPU + CPU для одного варианта."""
    npu_pc = fps_data.get('npu_load_per_core', {})
    cpu_pc = fps_data.get('cpu_load_per_core', {})
    cols = []
    for i in sorted(int(k) for k in npu_pc.keys()):
        cols.append(npu_pc.get(str(i), ''))
    for i in sorted(int(k) for k in cpu_pc.keys()):
        cols.append(cpu_pc.get(str(i), ''))
    return cols


def write_tsv(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TSV, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        # Определяем количество ядер из первого замера
        n_npu_cores = 0
        n_cpu_cores = 0
        for r in results:
            for key in ('npu_none', 'npu_int8', 'npu_int8-calib'):
                d = (r.get('fps', {}) or {}).get(key, {}) or {}
                n_npu_cores = max(n_npu_cores, len(d.get('npu_load_per_core', {})))
                n_cpu_cores = max(n_cpu_cores, len(d.get('cpu_load_per_core', {})))
            if n_npu_cores and n_cpu_cores:
                break
        if not n_npu_cores:
            n_npu_cores = 3
        if not n_cpu_cores:
            n_cpu_cores = 8

        # Заголовок
        header = [
            'Модель', 'Задача', 'Разрешение',
            'CPU_fps', 'CPU_fps_std',
        ]
        for variant in ('FP16', 'INT8', 'INT8cal'):
            header += [
                f'NPU_{variant}_fps', f'NPU_{variant}_fps_std',
                f'NPU_{variant}_load', f'CPU_load_{variant.lower()}',
            ]
            header += [f'NPU_{variant}_core{i}' for i in range(n_npu_cores)]
            header += [f'CPU_{variant}_core{i}' for i in range(n_cpu_cores)]
        header += [
            'top1', 'top5', 'map50', 'map50_95',
            'max_diff_avg', 'max_diff_max',
            'Троттлинг', 'Статус',
        ]
        writer.writerow(header)

        # Строки
        for r in results:
            model = r.get('model', '?')
            task = r.get('task', '?')
            isz = r.get('input_size', 640)
            input_size = f'{isz}x{isz}' if isinstance(isz, int) else str(isz)

            cpu = (r.get('fps', {}) or {}).get('cpu', {}) or {}
            npu_fp16 = (r.get('fps', {}) or {}).get('npu_none', {}) or {}
            npu_int8 = (r.get('fps', {}) or {}).get('npu_int8', {}) or {}
            npu_int8cal = (r.get('fps', {}) or {}).get('npu_int8-calib', {}) or {}

            v = r.get('validation', {})
            v_fp16 = v.get('fp16', {}) or {}
            v_int8 = v.get('int8', {}) or {}
            max_diff = v_fp16.get('max_diff_avg', v_int8.get('max_diff_avg', -1))
            max_diff_max = v_fp16.get('max_diff_max', v_int8.get('max_diff_max', -1))

            ev = r.get('eval', {}) or {}
            # Метрики берём из onnx-варианта (эталон); если нет — из первого доступного
            ref = ev.get('onnx') or next((v for v in ev.values() if isinstance(v, dict) and v), {}) or {}
            top1 = ref.get('top1', '')
            top5 = ref.get('top5', '')
            map50 = ref.get('map50', ref.get('inmem_map50', ''))
            map50_95 = ref.get('map50_95', ref.get('inmem_map50_95', ''))

            row = [
                model, task, input_size,
                cpu.get('fps', -1), cpu.get('fps_std', 0),
            ]
            for variant_data in (npu_fp16, npu_int8, npu_int8cal):
                row += [
                    variant_data.get('fps', -1), variant_data.get('fps_std', 0),
                    variant_data.get('npu_load_avg', ''), variant_data.get('cpu_load_avg', ''),
                ]
                row += _per_core_cols(variant_data)
            row += [
                top1, top5, map50, map50_95,
                max_diff, max_diff_max,
            ]
            throttle_parts = []
            for vd in (cpu, npu_fp16, npu_int8, npu_int8cal):
                if vd.get('throttled'):
                    tag = 'cpu' if vd.get('backend') == 'cpu' else (vd.get('rknn_suffix') or '?')
                    throttle_parts.append(f'{tag}:{vd.get("throttle_reason")}')
            row.append('; '.join(throttle_parts) if throttle_parts else '')
            row.append(r.get('status', '?'))
            writer.writerow(row)
    log(f'[ГОТОВО] TSV-сводка записана в {SUMMARY_TSV}')


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def phase_is(phases: list[str], name: str) -> bool:
    """Проверить, входит ли фаза в список запрошенных."""
    return 'all' in phases or name in phases


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Оркестратор YOLO бенчмарка")
    parser.add_argument("--phase", nargs='+', default=['convert', 'validate'],
                        choices=['convert', 'validate', 'bench', 'eval', 'all'],
                        help="Фазы бенчмарка (можно перечислить несколько)")
    parser.add_argument("--quantize", choices=["onnx", "none", "int8", "int8-calib", "all"], default="all",
                        help="Какие варианты RKNN тестировать: onnx=CPU only, none=FP16, int8=INT8, int8-calib=INT8 с калибровкой, all=все")
    parser.add_argument("--model", default=None,
                        help="Запуск для одной модели (напр. yolo26x-seg)")
    parser.add_argument("--pilot", action="store_true",
                        help="Только PILOT-модели (минимальные размеры)")
    cli_args, _ = parser.parse_known_args()

    global QUANTIZE
    QUANTIZE = cli_args.quantize

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    phases = cli_args.phase
    log(f'Фазы: {phases}, квантизация: {QUANTIZE}')

    # Фильтрация моделей
    models = MODELS
    if cli_args.pilot:
        models = PILOT_MODELS
    if cli_args.model:
        models = [m for m in MODELS if m.name == cli_args.model]
        if not models:
            log(f'[ОШИБКА] модель {cli_args.model!r} не найдена в models.py')
            sys.exit(1)
    log(f'Бенчмарк начат: {datetime.now().isoformat()}')
    log(f'Mоделей={len(models)}, Warmup={WARMUP_S}с, Measure={MEASURE_S}с')

    existing = load_results()

    # Остановить percept перед bench-фазой (освободить NPU)
    percept_was_running = False
    if phase_is(phases, 'bench'):
        percept_was_running = stop_percept()

    try:
        for i, model in enumerate(models):
            log(f'\n{"="*60}')
            log(f'[{i+1}/{len(models)}] {model.name} ({model.task})')

            if model.dataset == 'coco_val2017':
                model_data_dir = COCO_VAL_DIR
            elif model.dataset == 'dotav1':
                model_data_dir = DOTA_VAL_DIR
            elif model.dataset == 'cityscapes':
                model_data_dir = CITYSCAPES_VAL_DIR
            else:
                model_data_dir = DATA_DIR / model.dataset

            model_cache_dir = CACHE_DIR / model.name

            # Убедиться что датасет готов (скачивание если нужно)
            ensure_dataset(model)

            # Полный skip если все фазы уже завершены
            existing_entry = next((r for r in existing if r['model'] == model.name), None)
            if existing_entry and model_is_complete(existing_entry):
                log(f'[ПРОПУСК] все фазы завершены')
                continue

            # --- Фаза: convert ---
            if phase_is(phases, 'convert'):
                # Resume: пропускаем convert только если ВСЕ нужные файлы на месте.
                # (ранее проверялся только _fp16.rknn — это была дыра: при наличии fp16
                # отсутствующие int8/int8-calib тихо пропускались и не дособирались)
                need_convert = QUANTIZE_TO_EVAL.get(QUANTIZE, ['onnx', 'fp16', 'int8', 'int8-calib'])
                missing = [
                    v for v in need_convert
                    if not (model_cache_dir / EVAL_VARIANTS[v]['file'](model.name)).exists()
                ]
                if not missing:
                    log(f'[ПРОПУСК] convert: все файлы ({", ".join(need_convert)}) уже на месте')
                    if model.name not in {r['model'] for r in existing}:
                        existing.append({
                            'model': model.name,
                            'task': model.task,
                            'conversion_status': 'OK',
                            'input_size': model.input_size,
                            'validation': {},
                            'fps': {},
                            'status': 'ОЖИДАНИЕ',
                        })
                        save_results(existing)
                else:
                    conv_result = run_converter(model.name, model.task, model_cache_dir, model_data_dir)
                    if conv_result is None or conv_result.get('status') != 'OK':
                        log(f'[ПРОВАЛ] конвертация не удалась для {model.name}')
                        result_entry = {
                            'model': model.name,
                            'task': model.task,
                            'conversion_status': 'ПРОВАЛ',
                            'conversion_error': conv_result.get('error', '') if conv_result else '',
                            'validation': {},
                            'fps': {},
                            'input_size': model.input_size,
                            'status': 'ПРОВАЛ',
                        }
                        upsert_result(existing, result_entry)
                        save_results(existing)
                        continue
                    # Обновляем conversion_status
                    for j, r in enumerate(existing):
                        if r['model'] == model.name:
                            existing[j]['conversion_status'] = 'OK'
                            existing[j]['input_size'] = conv_result.get('input_size', model.input_size)
                            break
                    else:
                        result_entry = {
                            'model': model.name,
                            'task': model.task,
                            'conversion_status': 'OK',
                            'input_size': conv_result.get('input_size', model.input_size),
                            'validation': {},
                            'fps': {},
                            'status': 'ОЖИДАНИЕ',
                        }
                        existing.append(result_entry)
                    save_results(existing)

            # --- Фаза: validate ---
            if phase_is(phases, 'validate'):
                # Resume: пропускаем если уже валидировано
                if model.name in {r['model'] for r in existing
                                  if r.get('validation', {}).get('fp16', {}).get('max_diff_avg', -1) >= 0
                                  or r.get('validation', {}).get('int8', {}).get('max_diff_avg', -1) >= 0}:
                    log(f'[ПРОПУСК] validate: уже валидировано')
                elif not (model_cache_dir / f'{model.name}_fp16.rknn').exists():
                    log(f'[ПРОПУСК] validate: нет сконвертированных моделей')
                else:
                    val_result = run_validate(model.name, model.task, model_cache_dir, model_data_dir,
                                             input_size=model.input_size)
                    if val_result is None:
                        log(f'[ПРОВАЛ] валидация не удалась для {model.name}')
                        if model.name not in {r['model'] for r in existing}:
                            existing.append({
                                'model': model.name,
                                'task': model.task,
                                'conversion_status': 'OK',
                                'input_size': model.input_size,
                                'validation': {},
                                'fps': {},
                                'status': 'ПРОВАЛ',
                            })
                    else:
                        validations = val_result.get('validations', {})
                        found = False
                        for j, r in enumerate(existing):
                            if r['model'] == model.name:
                                existing[j]['validation'] = validations
                                found = True
                                break
                        if not found:
                            existing.append({
                                'model': model.name,
                                'task': model.task,
                                'conversion_status': 'OK',
                                'input_size': model.input_size,
                                'validation': validations,
                                'fps': {},
                                'status': 'ОЖИДАНИЕ',
                            })
                        # Определяем статус по max_diff
                        for j, r in enumerate(existing):
                            if r['model'] == model.name:
                                max_diffs = []
                                for vk, vv in validations.items():
                                    md = vv.get('max_diff_avg', -1)
                                    if md >= 0:
                                        max_diffs.append(md)
                                if max_diffs:
                                    worst = max(max_diffs)
                                    existing[j]['status'] = 'ПРОЙДЕНО' if worst < 0.5 else 'ПРОВАЛ'
                                break
                    save_results(existing)

            # --- Фаза: bench ---
            if phase_is(phases, 'bench'):
                # Создать запись если модели нет в results.json
                if model.name not in {r['model'] for r in existing}:
                    existing.append({
                        'model': model.name,
                        'task': model.task,
                        'input_size': model.input_size,
                        'conversion_status': 'N/A',
                        'validation': {},
                        'fps': {},
                        'eval': {},
                        'status': 'ОЖИДАНИЕ',
                    })
                    save_results(existing)

                bench_done = False
                for r in existing:
                    if r['model'] == model.name:
                        fps_existing = r.get('fps', {})
                        if any(fps_existing.get(k, {}).get('fps', -1) > 0
                               for k in ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib']):
                            log(f'[ПРОПУСК] bench: уже замерено (results.json)')
                            bench_done = True
                        break

                if not bench_done:
                    if not (model_cache_dir / f'{model.name}.onnx').exists() and \
                       not (model_cache_dir / f'{model.name}_fp16.rknn').exists():
                        log(f'[ПРОВАЛ] bench: нет моделей в cache/ и нет результатов')
                        for j, r in enumerate(existing):
                            if r['model'] == model.name:
                                existing[j]['fps'] = {}
                                existing[j]['status'] = 'ПРОВАЛ'
                                break
                        save_results(existing)
                    else:
                        fps_result = run_fps(model.name, model.task, model_cache_dir, model_data_dir,
                                             quantize=QUANTIZE)
                        if fps_result is None:
                            log(f'[ПРОВАЛ] FPS-тест не удалась для {model.name}')
                            fps = {}
                        else:
                            fps = {}
                            for key in ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib']:
                                b = fps_result.get(key, {}) or {}
                                fps[key] = {
                                    'fps': b.get('fps', -1),
                                    'fps_std': b.get('fps_std', 0),
                                    'n_frames': b.get('n_frames', 0),
                                    'n_repeats': b.get('n_repeats'),
                                    'fps_min': b.get('fps_min'),
                                    'fps_max': b.get('fps_max'),
                                    'npu_load_avg': b.get('npu_load_avg'),
                                    'npu_load_per_core': b.get('npu_load_per_core', {}),
                                    'cpu_load_avg': b.get('cpu_load_avg'),
                                    'cpu_load_per_core': b.get('cpu_load_per_core', {}),
                                    'npu_temp_avg': b.get('npu_temp_avg'),
                                    'npu_temp_min': b.get('npu_temp_min'),
                                    'npu_temp_max': b.get('npu_temp_max'),
                                    'npu_freq_avg': b.get('npu_freq_avg'),
                                    'thermal_max_c': b.get('thermal_max_c'),
                                    'cpu_freq_ratio': b.get('cpu_freq_ratio'),
                                    'throttled': b.get('throttled'),
                                    'throttle_reason': b.get('throttle_reason'),
                                    'error': b.get('error'),
                                }

                        for j, r in enumerate(existing):
                            if r['model'] == model.name:
                                existing[j]['fps'] = fps
                                has_data = any(
                                    fps.get(k, {}).get('fps', -1) > 0
                                    for k in ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib']
                                )
                                if has_data:
                                    existing[j]['status'] = 'ПРОЙДЕНО'
                                elif fps:
                                    existing[j]['status'] = 'ПРОВАЛ'
                                break
                        save_results(existing)

                        cpu_fps = fps.get('cpu', {}).get('fps', -1)
                        npu_fp16_fps = fps.get('npu_none', {}).get('fps', -1)
                        npu_int8_fps = fps.get('npu_int8', {}).get('fps', -1)
                        npu_int8c_fps = fps.get('npu_int8-calib', {}).get('fps', -1)
                        log(f'[ГОТОВО] {model.name}: CPU={cpu_fps} FP16={npu_fp16_fps} INT8={npu_int8_fps} INT8-calib={npu_int8c_fps}')

            # --- Фаза: eval (accuracy) ---
            if phase_is(phases, 'eval'):
                # Создать запись если модели нет в results.json
                if model.name not in {r['model'] for r in existing}:
                    existing.append({
                        'model': model.name,
                        'task': model.task,
                        'input_size': model.input_size,
                        'conversion_status': 'N/A',
                        'validation': {},
                        'fps': {},
                        'eval': {},
                        'status': 'ОЖИДАНИЕ',
                    })
                    save_results(existing)

                eval_variants = QUANTIZE_TO_EVAL.get(QUANTIZE,
                                                     ['onnx', 'fp16', 'int8', 'int8-calib'])
                if model.name not in {r['model'] for r in existing if r.get('eval', {})}:
                    eval_results = {}
                else:
                    eval_results = next(r for r in existing if r['model'] == model.name)['eval'] or {}
                done_keys = {k for k, val in eval_results.items()
                             if isinstance(val, dict) and val.get('elapsed_s', 0) > 0}
                if all(vr in done_keys for vr in eval_variants):
                    log(f'[ПРОПУСК] eval: все варианты уже выполнены (results.json)')
                else:
                    for variant in eval_variants:
                        if variant in done_keys:
                            log(f'[ПРОПУСК] eval {variant}: уже выполнено (results.json)')
                            continue
                        eval_result = run_eval(model.name, model.task, model_cache_dir,
                                               model_data_dir, variant)
                        if eval_result is None:
                            log(f'[ОШИБКА] eval {variant} не удался для {model.name}')
                            eval_result = {}
                        eval_results[variant] = eval_result
                    for j, r in enumerate(existing):
                        if r['model'] == model.name:
                            existing[j]['eval'] = eval_results
                            break
                    save_results(existing)
                    log(f'[ГОТОВО] eval {model.name}: {json.dumps(eval_results, ensure_ascii=False)}')

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log(f'[ПРЕРВАНО] Ctrl+C')
        else:
            log(f'[ОШИБКА] {e}')
    finally:
        restart_percept(percept_was_running)

    write_tsv(existing)

    log(f'\n{"="*60}')
    log(f'БЕНЧМАРК ЗАВЕРШЁН: {len(existing)} моделей')
    log(f'Результаты: {RESULTS_FILE}')
    log(f'Сводка: {SUMMARY_TSV}')

    print(f'\nСводка ({SUMMARY_TSV}):')
    print(SUMMARY_TSV.read_text() if SUMMARY_TSV.exists() else '(пусто)')


if __name__ == '__main__':
    main()
