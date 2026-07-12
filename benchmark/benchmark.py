#!/usr/bin/env python3
"""Оркестратор бенчмарка — запускается на HOST.

Итерирует модели из models.py, вызывает конвертер для конвертации+валидации,
затем percept для FPS-теста, собирает результаты, генерирует TSV-отчёт.

Использование:
    python3 benchmark.py                          # полный бенчмарк
    python3 benchmark.py --model yolo26x-seg      # одна модель
    python3 benchmark.py --model yolo26x-seg --quantize int8
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from models import FULL as MODELS

BENCHMARK_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = BENCHMARK_DIR.parent
TEMP_DIR = BENCHMARK_DIR / 'temp_models'
DATA_DIR = BENCHMARK_DIR / 'data'
RESULTS_DIR = BENCHMARK_DIR / 'results'
CONVERTER_COMPOSE = PROJECT_DIR / 'converter' / 'docker-compose.yml'
WEB_COMPOSE = PROJECT_DIR / 'docker-compose.yml'
RESULTS_FILE = RESULTS_DIR / 'results.json'
SUMMARY_TSV = RESULTS_DIR / 'summary.tsv'
LOG_FILE = RESULTS_DIR / 'benchmark.log'

COCO_VAL_DIR = PROJECT_DIR / 'datasets' / 'coco_val2017'
COCO_VAL_CONTAINER = '/app/coco_val2017'

WARMUP_S = int(os.environ.get('BENCH_WARMUP', '10'))
MEASURE_S = int(os.environ.get('BENCH_MEASURE', '20'))
CONVERT_TIMEOUT = int(os.environ.get('BENCH_CONVERT_TIMEOUT', '7200'))
PARALLEL = os.environ.get('BENCH_PARALLEL', '0') == '1'
QUANTIZE = os.environ.get('BENCH_QUANTIZE', 'none')


def log(msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def _container_data_path(data_dir: Path) -> str:
    """Сопоставление host data_dir с путём контейнера."""
    if data_dir.name == 'coco_val2017' or data_dir == COCO_VAL_DIR:
        return f'{COCO_VAL_CONTAINER}/images/val2017'
    return f'/app/benchmark/data/{data_dir.name}'


def run_converter(model_name: str, task: str, output_dir: Path, data_dir: Path) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = _container_data_path(data_dir)
    keep = os.environ.get('KEEP_MODELS', '0')
    cmd = [
        'docker', 'compose', '-f', str(CONVERTER_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{COCO_VAL_DIR}:{COCO_VAL_CONTAINER}:ro',
        '-e', f'CONVERT_QUANTIZE={QUANTIZE}',
        '-e', f'KEEP_MODELS={keep}',
        '--entrypoint', 'python',
        'converter',
        '/app/benchmark/convert_bench.py',
        model_name, task, f'/app/benchmark/temp_models/{model_name}',
        data_path,
    ]
    log(f'ЗАПУСК: {" ".join(cmd[:-6])} ...')
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
        elapsed = time.time() - t0
        log(f'[ТАЙМИНГ] run_converter завершён за {elapsed:.0f}с')
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] конвертер превысил таймаут {CONVERT_TIMEOUT}с')
        return None

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            log(f'[converter stdout] {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            log(f'[converter stderr] {line}')

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


def run_fps(model_name: str, task: str, temp_dir: Path, data_dir: Path) -> dict | None:
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(WEB_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{COCO_VAL_DIR}:{COCO_VAL_CONTAINER}:ro',
        '-e', 'TEST_MODE=1',
        '-e', 'WEB_MODE=0',
        '-e', 'CAM_MODE=file',
        '-e', 'PYTHONPATH=/app',
        '-e', f'BENCH_WARMUP={WARMUP_S}',
        '-e', f'BENCH_MEASURE={MEASURE_S}',
        'percept', 'python3', '/app/benchmark/fps_bench.py',
        model_name, task,
        '--temp-dir', f'/app/benchmark/temp_models/{model_name}',
        '--data-dir', data_path,
        '--warmup', str(WARMUP_S),
        '--measure', str(MEASURE_S),
    ]
    log(f'ЗАПУСК: {" ".join(cmd[:-8])} ...')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] FPS-тест превысил таймаут 600с')
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

    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def cleanup_temp(model_name: str) -> None:
    model_dir = f'/app/benchmark/temp_models/{model_name}'
    result = subprocess.run(
        ['docker', 'run', '--rm', '-v', f'{BENCHMARK_DIR}:/app/benchmark', 'alpine',
         'rm', '-rf', model_dir],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        log(f'[ОЧИСТКА] удалено temp_models/{model_name}')
    else:
        log(f'[ОЧИСТКА] ошибка: {result.stderr.strip()}')


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


def write_tsv(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TSV, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'Модель', 'Задача', 'Разрешение',
            'CPU_fps', 'CPU_fps_std', 'NPU_fps', 'NPU_fps_std',
            'max_diff_avg', 'max_diff_max',
            'NPU_load_avg',
            'Статус',
        ])
        for r in results:
            model = r.get('model', '?')
            task = r.get('task', '?')
            isz = r.get('input_size', 640)
            input_size = f'{isz}x{isz}' if isinstance(isz, int) else str(isz)
            v = r.get('validation', {})
            cpu = (r.get('fps', {}) or {}).get('cpu', {}) or {}
            npu = (r.get('fps', {}) or {}).get('npu', {}) or {}
            cpu_fps = cpu.get('fps', -1)
            cpu_fps_std = cpu.get('fps_std', 0)
            npu_fps = npu.get('fps', -1)
            npu_fps_std = npu.get('fps_std', 0)
            md_avg = v.get('max_diff_avg', -1)
            md_max = v.get('max_diff_max', -1)
            npu_load = (r.get('fps', {}) or {}).get('npu', {}) or {}
            npu_load_val = npu_load.get('npu_load_avg', '')
            status = r.get('status', '?')
            if cpu_fps == -1 and npu_fps == -1 and 'error' in r:
                status = 'ПРОВАЛ:' + r['error'][:60]
            writer.writerow([model, task, input_size, cpu_fps, cpu_fps_std, npu_fps, npu_fps_std, md_avg, md_max, npu_load_val, status])
    log(f'[ГОТОВО] TSV-сводка записана в {SUMMARY_TSV}')


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Оркестратор YOLO бенчмарка")
    parser.add_argument("--quantize", choices=["none", "int8"], default=None,
                        help="Режим INT8 квантизации для конвертера")
    parser.add_argument("--model", default=None,
                        help="Запуск бенчмарка для одной модели (напр. yolo26x-seg)")
    cli_args, _ = parser.parse_known_args()
    if cli_args.quantize is not None:
        global QUANTIZE
        QUANTIZE = cli_args.quantize

    # Фильтрация моделей
    models = MODELS
    if cli_args.model:
        models = [m for m in MODELS if m.name == cli_args.model]
        if not models:
            log(f'[ОШИБКА] модель {cli_args.model!r} не найдена в models.py')
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log(f'Бенчмарк начат: {datetime.now().isoformat()}')
    log(f'Warmup={WARMUP_S}с, Measure={MEASURE_S}с, Моделей={len(models)}')

    existing = load_results()
    passed = {r['model'] for r in existing if r.get('status', '').startswith('ПРОЙДЕНО')}

    for i, model in enumerate(models):
        log(f'\n{"="*60}')
        log(f'[{i+1}/{len(models)}] {model.name} ({model.task})')

        if model.name in passed:
            log(f'[ПРОПУСК] уже пройдена, удалите {RESULTS_FILE} для перезапуска')
            continue

        if model.name in {r['model'] for r in existing}:
            log(f'[ПОВТОР] предыдущий результат — ПРОВАЛ/прерывание')

        model_temp_dir = TEMP_DIR / model.name
        if model.dataset == 'coco_val2017':
            model_data_dir = COCO_VAL_DIR
        else:
            model_data_dir = DATA_DIR / model.dataset

        conv_result = run_converter(model.name, model.task, model_temp_dir, model_data_dir)
        if conv_result is None:
            log(f'[ПРОВАЛ] конвертация не удалась для {model.name}')
            result_entry = {
                'model': model.name,
                'task': model.task,
                'status': 'ПРОВАЛ',
                'error': 'конвертация не удалась',
                'validation': {},
                'fps': {},
                'input_size': model.input_size,
            }
            upsert_result(existing, result_entry)
            save_results(existing)
            cleanup_temp(model.name)
            continue

        conv_ok = conv_result.get('status') == 'OK'
        if not conv_ok:
            log(f'[ПРОВАЛ] конвертация не удалась для {model.name}')
            result_entry = {
                'model': model.name,
                'task': model.task,
                'status': 'ПРОВАЛ',
                'error': 'конвертация не удалась',
                'validation': {},
                'fps': {},
                'input_size': model.input_size,
            }
            upsert_result(existing, result_entry)
            save_results(existing)
            cleanup_temp(model.name)
            continue

        fps_result = run_fps(model.name, model.task, model_temp_dir, model_data_dir)
        if fps_result is None:
            log(f'[ПРОВАЛ] FPS-тест не удался для {model.name}')
            fps = {'cpu': {'fps': -1, 'error': 'FPS-тест не удался'}, 'npu': {'fps': -1, 'error': 'FPS-тест не удался'}}
            validation = {}
        else:
            fps = {}
            for backend in ['cpu', 'npu']:
                b = fps_result.get(backend, {}) or {}
                fps[backend] = {
                    'fps': b.get('fps', -1),
                    'fps_std': b.get('fps_std', 0),
                    'n_frames': b.get('n_frames', 0),
                    'npu_load_avg': b.get('npu_load_avg'),
                    'error': b.get('error'),
                }
            validation = fps_result.get('validation', {})

        v = validation or {}
        max_diff = v.get('max_diff_avg', -1)
        threshold = 0.5
        if max_diff >= 0:
            status = 'ПРОЙДЕНО' if max_diff < threshold else 'ПРОВАЛ'
        else:
            status = 'ПРОЙДЕНО' if conv_ok else 'ПРОВАЛ'
        if fps.get('cpu', {}).get('fps', -1) == -1 and fps.get('npu', {}).get('fps', -1) == -1:
            if fps.get('cpu', {}).get('error') or fps.get('npu', {}).get('error'):
                status = 'ПРОВАЛ'

        result_entry = {
            'model': model.name,
            'task': model.task,
            'status': status,
            'validation': validation,
            'fps': fps,
            'input_size': conv_result.get('input_size', model.input_size),
        }
        upsert_result(existing, result_entry)
        save_results(existing)
        cleanup_temp(model.name)

        cpu_fps = fps.get('cpu', {}).get('fps', -1)
        npu_fps = fps.get('npu', {}).get('fps', -1)
        md_avg = validation.get('max_diff_avg', -1)
        log(f'[ГОТОВО] {model.name}: статус={status} '
            f'CPU={cpu_fps} NPU={npu_fps} max_diff={md_avg}')

    write_tsv(existing)

    log(f'\n{"="*60}')
    log(f'БЕНЧМАРК ЗАВЕРШЁН: {len(existing)} моделей')
    log(f'Результаты: {RESULTS_FILE}')
    log(f'Сводка: {SUMMARY_TSV}')

    print(f'\nСводка ({SUMMARY_TSV}):')
    print(SUMMARY_TSV.read_text() if SUMMARY_TSV.exists() else '(пусто)')


if __name__ == '__main__':
    main()
