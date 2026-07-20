#!/usr/bin/env python3
"""Оркестратор бенчмарка — запускается на HOST.

Три фазы:
  convert  — конвертация .pt → .onnx + _none.rknn + _int8.rknn (x86 ПК)
  validate — валидация ONNX vs RKNN (x86 ПК)
  bench    — FPS + нагрузка на CPU/NPU (ARM железо)

Использование:
    python3 benchmark.py --phase convert                    # конвертация (ПК)
    python3 benchmark.py --phase validate                   # валидация (ПК)
    python3 benchmark.py --phase bench                      # FPS (ARM)
    python3 benchmark.py --phase convert validate           # конвертация + валидация (ПК)
    python3 benchmark.py --phase all                        # всё (convert → validate → bench)
    python3 benchmark.py --phase all --clean                # всё с авто-очисткой temp_models
    python3 benchmark.py --phase bench --quantize int8      # только INT8 FPS
    python3 benchmark.py --phase bench --clean --model yolo11n
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

DOTA_VAL_DIR = PROJECT_DIR / 'datasets' / 'dotav1'
DOTA_VAL_CONTAINER = '/app/dotav1'

CITYSCAPES_VAL_DIR = PROJECT_DIR / 'datasets' / 'cityscapes'
CITYSCAPES_VAL_CONTAINER = '/app/cityscapes'

WARMUP_S = int(os.environ.get('BENCH_WARMUP', '10'))
MEASURE_S = int(os.environ.get('BENCH_MEASURE', '20'))
CONVERT_TIMEOUT = int(os.environ.get('BENCH_CONVERT_TIMEOUT', '7200'))
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
    if data_dir.name == 'dotav1' or data_dir == DOTA_VAL_DIR:
        return f'{DOTA_VAL_CONTAINER}/images/val'
    if data_dir.name == 'cityscapes' or data_dir == CITYSCAPES_VAL_DIR:
        return f'{CITYSCAPES_VAL_CONTAINER}/images/val'
    return f'/app/benchmark/data/{data_dir.name}'


# ---------------------------------------------------------------------------
# Фаза: convert
# ---------------------------------------------------------------------------

def run_converter(model_name: str, task: str, output_dir: Path, data_dir: Path) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(CONVERTER_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{COCO_VAL_DIR}:{COCO_VAL_CONTAINER}:ro',
        '-v', f'{DOTA_VAL_DIR}:{DOTA_VAL_CONTAINER}:ro',
        '-v', f'{CITYSCAPES_VAL_DIR}:{CITYSCAPES_VAL_CONTAINER}:ro',
        '-e', f'CONVERT_QUANTIZE={QUANTIZE}',
        '-e', 'KEEP_MODELS=1',
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
        log(f'[ТАЙМИНГ] convert завершён за {elapsed:.0f}с')
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


# ---------------------------------------------------------------------------
# Фаза: validate
# ---------------------------------------------------------------------------

def run_validate(model_name: str, task: str, temp_dir: Path, data_dir: Path) -> dict | None:
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(CONVERTER_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{COCO_VAL_DIR}:{COCO_VAL_CONTAINER}:ro',
        '-v', f'{DOTA_VAL_DIR}:{DOTA_VAL_CONTAINER}:ro',
        '-v', f'{CITYSCAPES_VAL_DIR}:{CITYSCAPES_VAL_CONTAINER}:ro',
        '--entrypoint', 'python',
        'converter',
        '/app/benchmark/validate_bench.py',
        model_name, task, f'/app/benchmark/temp_models/{model_name}',
        data_path,
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

def run_fps(model_name: str, task: str, temp_dir: Path, data_dir: Path,
            quantize: str = 'both') -> dict | None:
    data_path = _container_data_path(data_dir)
    cmd = [
        'docker', 'compose', '-f', str(WEB_COMPOSE), 'run', '--rm',
        '-v', f'{BENCHMARK_DIR}:/app/benchmark',
        '-v', f'{COCO_VAL_DIR}:{COCO_VAL_CONTAINER}:ro',
        '-v', f'{DOTA_VAL_DIR}:{DOTA_VAL_CONTAINER}:ro',
        '-v', f'{CITYSCAPES_VAL_DIR}:{CITYSCAPES_VAL_CONTAINER}:ro',
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
        '--quantize', quantize,
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


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

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
            'CPU_fps', 'CPU_fps_std',
            'NPU_FP32_fps', 'NPU_FP32_fps_std', 'NPU_FP32_load', 'CPU_load_fp32',
            'NPU_INT8_fps', 'NPU_INT8_fps_std', 'NPU_INT8_load', 'CPU_load_int8',
            'max_diff_avg', 'max_diff_max',
            'Статус',
        ])
        for r in results:
            model = r.get('model', '?')
            task = r.get('task', '?')
            isz = r.get('input_size', 640)
            input_size = f'{isz}x{isz}' if isinstance(isz, int) else str(isz)

            cpu = (r.get('fps', {}) or {}).get('cpu', {}) or {}
            npu_fp32 = (r.get('fps', {}) or {}).get('npu_none', {}) or {}
            npu_int8 = (r.get('fps', {}) or {}).get('npu_int8', {}) or {}

            v = r.get('validation', {})
            v_fp32 = v.get('fp32', {}) or {}
            v_int8 = v.get('int8', {}) or {}
            max_diff = v_fp32.get('max_diff_avg', v_int8.get('max_diff_avg', -1))
            max_diff_max = v_fp32.get('max_diff_max', v_int8.get('max_diff_max', -1))

            writer.writerow([
                model, task, input_size,
                cpu.get('fps', -1), cpu.get('fps_std', 0),
                npu_fp32.get('fps', -1), npu_fp32.get('fps_std', 0),
                npu_fp32.get('npu_load_avg', ''), npu_fp32.get('cpu_load_avg', ''),
                npu_int8.get('fps', -1), npu_int8.get('fps_std', 0),
                npu_int8.get('npu_load_avg', ''), npu_int8.get('cpu_load_avg', ''),
                max_diff, max_diff_max,
                r.get('status', '?'),
            ])
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
                        choices=['convert', 'validate', 'bench', 'all'],
                        help="Фазы бенчмарка (можно перечислить несколько)")
    parser.add_argument("--quantize", choices=["none", "int8", "both"], default="both",
                        help="Какие варианты RKNN тестировать (по умолчанию: оба)")
    parser.add_argument("--model", default=None,
                        help="Запуск для одной модели (напр. yolo26x-seg)")
    parser.add_argument("--clean", action="store_true",
                        help="Удалить temp_models после завершения")
    cli_args, _ = parser.parse_known_args()

    global QUANTIZE
    QUANTIZE = cli_args.quantize

    phases = cli_args.phase
    log(f'Фазы: {phases}, квантизация: {QUANTIZE}, clean: {cli_args.clean}')

    # Фильтрация моделей
    models = MODELS
    if cli_args.model:
        models = [m for m in MODELS if m.name == cli_args.model]
        if not models:
            log(f'[ОШИБКА] модель {cli_args.model!r} не найдена в models.py')
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log(f'Бенчмарк начат: {datetime.now().isoformat()}')
    log(f'Mоделей={len(models)}, Warmup={WARMUP_S}с, Measure={MEASURE_S}с')

    existing = load_results()

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

        model_temp_dir = TEMP_DIR / model.name

        # --- Фаза: convert ---
        if phase_is(phases, 'convert'):
            # Resume: пропускаем если уже сконвертировано (есть _none.rknn)
            if (model_temp_dir / f'{model.name}_none.rknn').exists():
                log(f'[ПРОПУСК] convert: уже сконвертировано')
            elif model.name in {r['model'] for r in existing if r.get('conversion_status') == 'OK'}:
                log(f'[ПРОПУСК] convert: уже сконвертировано (results.json)')
            else:
                conv_result = run_converter(model.name, model.task, model_temp_dir, model_data_dir)
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
                              if r.get('validation', {}).get('fp32', {}).get('max_diff_avg', -1) >= 0
                              or r.get('validation', {}).get('int8', {}).get('max_diff_avg', -1) >= 0}:
                log(f'[ПРОПУСК] validate: уже валидировано')
            elif not (model_temp_dir / f'{model.name}_none.rknn').exists():
                log(f'[ПРОПУСК] validate: нет сконвертированных моделей')
            else:
                val_result = run_validate(model.name, model.task, model_temp_dir, model_data_dir)
                if val_result is None:
                    log(f'[ПРОВАЛ] валидация не удалась для {model.name}')
                else:
                    validations = val_result.get('validations', {})
                    # Обновляем results.json
                    for j, r in enumerate(existing):
                        if r['model'] == model.name:
                            existing[j]['validation'] = validations
                            # Определяем статус по max_diff
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
            if not (model_temp_dir / f'{model.name}.onnx').exists() and \
               not (model_temp_dir / f'{model.name}_none.rknn').exists():
                log(f'[ПРОПУСК] bench: нет моделей в temp_models/')
                continue

            fps_result = run_fps(model.name, model.task, model_temp_dir, model_data_dir,
                                 quantize=QUANTIZE)
            if fps_result is None:
                log(f'[ПРОВАЛ] FPS-тест не удалась для {model.name}')
                fps = {}
            else:
                fps = {}
                for key in ['cpu', 'npu_none', 'npu_int8']:
                    b = fps_result.get(key, {}) or {}
                    fps[key] = {
                        'fps': b.get('fps', -1),
                        'fps_std': b.get('fps_std', 0),
                        'n_frames': b.get('n_frames', 0),
                        'npu_load_avg': b.get('npu_load_avg'),
                        'cpu_load_avg': b.get('cpu_load_avg'),
                        'error': b.get('error'),
                    }

            # Обновляем results.json
            for j, r in enumerate(existing):
                if r['model'] == model.name:
                    existing[j]['fps'] = fps
                    # Пересчитываем статус
                    has_data = any(
                        fps.get(k, {}).get('fps', -1) > 0
                        for k in ['cpu', 'npu_none', 'npu_int8']
                    )
                    if has_data:
                        existing[j]['status'] = 'ПРОЙДЕНО'
                    elif fps:
                        existing[j]['status'] = 'ПРОВАЛ'
                    break
            save_results(existing)

            cpu_fps = fps.get('cpu', {}).get('fps', -1)
            npu_fp32_fps = fps.get('npu_none', {}).get('fps', -1)
            npu_int8_fps = fps.get('npu_int8', {}).get('fps', -1)
            log(f'[ГОТОВО] {model.name}: CPU={cpu_fps} FP32={npu_fp32_fps} INT8={npu_int8_fps}')

        # --- Очистка ---
        if cli_args.clean and (phase_is(phases, 'bench') or phase_is(phases, 'all')):
            cleanup_temp(model.name)

    write_tsv(existing)

    log(f'\n{"="*60}')
    log(f'БЕНЧМАРК ЗАВЕРШЁН: {len(existing)} моделей')
    log(f'Результаты: {RESULTS_FILE}')
    log(f'Сводка: {SUMMARY_TSV}')

    print(f'\nСводка ({SUMMARY_TSV}):')
    print(SUMMARY_TSV.read_text() if SUMMARY_TSV.exists() else '(пусто)')


if __name__ == '__main__':
    main()
