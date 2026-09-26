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
def _read_timeout(name: str) -> int | None:
    """Таймаут из env; None/отсутствие → ждать сколько нужно (без потолка)."""
    raw = os.environ.get(name)
    return int(raw) if raw else None


CONVERT_TIMEOUT = _read_timeout('BENCH_CONVERT_TIMEOUT')
EVAL_TIMEOUT = _read_timeout('BENCH_EVAL_TIMEOUT')
VALIDATE_TIMEOUT = _read_timeout('BENCH_VALIDATE_TIMEOUT')
BENCH_TIMEOUT = _read_timeout('BENCH_TIMEOUT')
REPORT_TIMEOUT = _read_timeout('BENCH_REPORT_TIMEOUT')
QUANTIZE = os.environ.get('BENCH_QUANTIZE', 'all')
CLEANUP_AFTER = os.environ.get('BENCH_CLEANUP', '1') == '1'
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

# Ожидаемые ключи FPS по сценарию квантизации (fps_bench.py → results.json)
QUANTIZE_TO_FPS = {
    'onnx': ['cpu'],
    'none': ['npu_none'],
    'int8': ['npu_int8'],
    'int8-calib': ['npu_int8-calib'],
    'all': ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib'],
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


# Сбои в текущем прогоне: («модель [фаза]», причина). Непустой список → exit != 0.
FAILURES: list[str] = []


def _fail(where: str, detail: str) -> None:
    """Зафиксировать сбой этапа в текущем прогоне (влияет на exit code)."""
    msg = f'{where}: {detail}'
    FAILURES.append(msg)
    log(f'[ПРОВАЛ] {msg}')


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

def _run_container(compose_file: Path, service: str, args: list[str],
                   timeout: int | None, tag: str,
                   env: list[str] | None = None,
                   volumes: list[str] | None = None,
                   entrypoint: str = 'python') -> subprocess.CompletedProcess | None:
    """Общий запуск скрипта в docker-compose контейнере.

    Возвращает CompletedProcess при успешном выходе (rc==0), None при
    таймауте или ненулевом коде. stdout/stderr логируются построчно.
    При таймауте (timeout задан и истёк) процесс убивается, а накопленный
    частичный stdout сохраняется в возвращаемом объекте и логируется.
    """
    cmd = ['docker', 'compose', '-f', str(compose_file), 'run', '--rm']
    for v in (volumes or []):
        cmd += ['-v', v]
    for e in (env or []):
        cmd += ['-e', e]
    cmd += ['--entrypoint', entrypoint, service] + args
    log(f'ЗАПУСК: {" ".join(cmd[:-6])} ...')
    t0 = time.time()

    def _run() -> subprocess.CompletedProcess | None:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            log(f'[ОШИБКА] {tag} не удалось запустить: {e}')
            return None
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            log(f'[ОШИБКА] {tag} превысил таймаут {timeout}с, принудительно останавливаю...')
            proc.kill()
            out, err = proc.communicate()
            _dump_partial(out, err, tag)
            return subprocess.CompletedProcess(
                proc.args, -9, stdout=out, stderr=err)
        return subprocess.CompletedProcess(
            proc.args, proc.returncode, stdout=out, stderr=err)

    result = _run()
    log(f'[ТАЙМИНГ] {tag} завершён за {time.time() - t0:.0f}с')

    if result is None:
        return None
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            log(f'[{tag} stdout] {line}')
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            log(f'[{tag} stderr] {line}')

    if result.returncode != 0:
        log(f'[ОШИБКА] {tag} завершился с кодом {result.returncode}')
        return None
    return result


def _dump_partial(out: str, err: str, tag: str) -> None:
    """Логировать фрагмент накопленного stdout/stderr убитого процесса."""
    for stream_name, data in (('stdout', out), ('stderr', err)):
        if not data:
            continue
        tail = data.strip().split('\n')[-50:]
        log(f'[{tag} {stream_name} до таймаута]')
        for line in tail:
            log(f'[{tag} {stream_name}] {line}')


def _last_json_line(stdout: str) -> dict | None:
    """Последняя JSON-строка в stdout (результат запущенного скрипта)."""
    if not stdout:
        return None
    for line in reversed(stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def run_converter(model_name: str, task: str, output_dir: Path, data_dir: Path) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = _container_data_path(data_dir)
    result = _run_container(
        CONVERT_COMPOSE, 'convert',
        ['/app/benchmark/convert_bench.py', model_name, task, f'/app/cache/{model_name}', data_path],
        CONVERT_TIMEOUT, 'convert',
        env=[f'CONVERT_QUANTIZE={QUANTIZE}', 'KEEP_MODELS=1'],
        volumes=[f'{BENCHMARK_DIR}:/app/benchmark', f'{PROJECT_DIR / "datasets"}:/app/datasets'],
    )
    return _last_json_line(result.stdout) if result else None


# ---------------------------------------------------------------------------
# Фаза: validate
# ---------------------------------------------------------------------------

def run_validate(model_name: str, task: str, temp_dir: Path, data_dir: Path, input_size: int = 640) -> dict | None:
    data_path = _container_data_path(data_dir)
    result = _run_container(
        CONVERT_COMPOSE, 'convert',
        ['/app/benchmark/validate_bench.py', model_name, task, f'/app/cache/{model_name}',
         data_path, str(input_size)],
        VALIDATE_TIMEOUT, 'validate',
        volumes=[f'{BENCHMARK_DIR}:/app/benchmark', f'{PROJECT_DIR / "datasets"}:/app/datasets'],
    )
    return _last_json_line(result.stdout) if result else None


# ---------------------------------------------------------------------------
# Фаза: bench
# ---------------------------------------------------------------------------

def run_fps(model_name: str, task: str, cache_dir: Path, data_dir: Path,
            quantize: str = 'both') -> dict | None:
    """FPS-бенчмарк в convert-контейнере (НЕ percept!): там есть PIL/torch/
    torchvision/ultralytics (их требуют cls/obb постпроцессы) + rknn-библиотеки
    и /dev/dri для реального NPU."""
    data_path = _container_data_path(data_dir)
    env = ['TEST_MODE=1', 'WEB_MODE=0', 'CAM_MODE=file', 'PYTHONPATH=/app',
           f'BENCH_WARMUP={WARMUP_S}', f'BENCH_MEASURE={MEASURE_S}']
    result = _run_container(
        CONVERT_COMPOSE, 'convert',
        ['/app/benchmark/fps_bench.py', model_name, task,
         '--temp-dir', f'/app/cache/{model_name}',
         '--data-dir', data_path,
         '--warmup', str(WARMUP_S),
         '--measure', str(MEASURE_S),
         '--repeats', str(BENCH_REPEATS),
         '--quantize', quantize],
        BENCH_TIMEOUT, 'bench',
        env=env,
        volumes=[f'{BENCHMARK_DIR}:/app/benchmark',
                 f'{CACHE_DIR}:/app/cache',
                 f'{PROJECT_DIR / "datasets"}:/app/datasets'],
    )
    if result is None:
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
    Бросает FileNotFoundError, если файла модели нет (ожидаемый вариант не
    тестируется) — вызывающий код отличает это от сбоя процесса eval.
    """
    v = EVAL_VARIANTS[variant]
    model_file = model_dir / v['file'](model_name)
    if not model_file.exists():
        raise FileNotFoundError(f'нет модели {model_file} (конвертация не выполнена)')
    data_path = _container_dataset_root(data_dir)
    task_arg = TASK_TO_EVAL.get(task, task)
    args = ['/app/benchmark/eval.py', task_arg, data_path,
            '--model-dir', f'/app/cache/{model_name}',
            '--model-name', model_name,
            '--backend', v['backend']]
    if v['suffix']:
        args += ['--rknn-suffix', v['suffix']]
    result = _run_container(
        CONVERT_COMPOSE, 'convert', args, EVAL_TIMEOUT, f'eval {variant}',
        env=['MALLOC_ARENA_MAX=2'],
        volumes=[f'{BENCHMARK_DIR}:/app/benchmark', f'{PROJECT_DIR / "datasets"}:/app/datasets'],
    )
    return _last_json_line(result.stdout) if result else None


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _eval_done(entry: dict, quantize: str) -> bool:
    """Все ожидаемые (по сценарию квантизации) eval-варианты выполнены."""
    eval_result = entry.get('eval', {}) or {}
    done_keys = {k for k, val in eval_result.items()
                 if isinstance(val, dict) and val.get('elapsed_s', 0) > 0}
    need_keys = QUANTIZE_TO_EVAL.get(quantize, ['onnx', 'fp16', 'int8', 'int8-calib'])
    return all(vr in done_keys for vr in need_keys)


def _bench_done(entry: dict, quantize: str) -> bool:
    """Все ожидаемые (по сценарию квантизации) FPS-замеры получены."""
    fps = entry.get('fps', {})
    need = QUANTIZE_TO_FPS.get(quantize, ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib'])
    return all(fps.get(k, {}).get('fps', -1) > 0 for k in need)


def _validate_done(entry: dict) -> bool:
    v = entry.get('validation', {})
    return (v.get('fp16', {}).get('max_diff_avg', -1) >= 0
            or v.get('int8', {}).get('max_diff_avg', -1) >= 0)


def _convert_files_missing(model_cache_dir: Path, model_name: str, quantize: str) -> list[str]:
    """Недостающие файлы конвертации по сценарию квантизации."""
    need_convert = QUANTIZE_TO_EVAL.get(quantize, ['onnx', 'fp16', 'int8', 'int8-calib'])
    return [
        v for v in need_convert
        if not (model_cache_dir / EVAL_VARIANTS[v]['file'](model_name)).exists()
    ]


def _cleanup_model_cache(model_cache_dir: Path, model_name: str) -> None:
    """Удалить .pt веса после успешного завершения всех фаз.

    Файлы в cache/ root-owned (созданы в контейнере) — удаление
    через docker compose run --rm (где есть root-доступ).
    """
    if not CLEANUP_AFTER:
        return
    pt = model_cache_dir / f'{model_name}.pt'
    if not pt.exists():
        return
    sz = pt.stat().st_size
    _run_container(
        CONVERT_COMPOSE, 'convert',
        ['-c', f'rm -f /app/cache/{model_name}/{model_name}.pt'],
        timeout=30, tag=f'cleanup {model_name}',
        entrypoint='bash',
    )
    if not pt.exists():
        log(f'[ОЧИСТКА] {model_name}: удалён {model_name}.pt ({sz / 1e6:.1f} МБ)')
    else:
        log(f'[ОЧИСТКА] {model_name}: не удалось удалить {model_name}.pt')


def model_is_complete(result: dict, phases: list[str], quantize: str) -> bool:
    """Завершены ли запрошенные фазы для модели (сценарно-зависимо).

    Проверяются только те фазы, которые входят в `phases`; ожидаемые eval/fps
    варианты определяются `quantize`. Для частичных сценариев (например
    `--phase bench --quantize int8`) не требует невыполненных фаз.
    """
    if phase_is(phases, 'convert') and result.get('conversion_status') != 'OK':
        return False
    if phase_is(phases, 'validate') and not _validate_done(result):
        return False
    if phase_is(phases, 'bench') and not _bench_done(result, quantize):
        return False
    if phase_is(phases, 'eval') and not _eval_done(result, quantize):
        return False
    return True


def missing_parts(entry: dict, phases: list[str], quantize: str) -> list[str]:
    """Список незавершённых фаз/вариантов (только из запрошенного сценария)."""
    parts = []
    if phase_is(phases, 'convert') and entry.get('conversion_status') != 'OK':
        parts.append('convert')
    if phase_is(phases, 'validate') and not _validate_done(entry):
        parts.append('validate')
    if phase_is(phases, 'bench'):
        need = QUANTIZE_TO_FPS.get(quantize, ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib'])
        fps = entry.get('fps', {})
        parts += [f'bench:{k}' for k in need if fps.get(k, {}).get('fps', -1) <= 0]
    if phase_is(phases, 'eval'):
        eval_result = entry.get('eval', {}) or {}
        done_keys = {k for k, val in eval_result.items()
                     if isinstance(val, dict) and val.get('elapsed_s', 0) > 0}
        need_keys = QUANTIZE_TO_EVAL.get(quantize, ['onnx', 'fp16', 'int8', 'int8-calib'])
        parts += [f'eval:{k}' for k in need_keys if k not in done_keys]
    return parts


def recompute_status(results: list[dict], quantize: str) -> None:
    """Итоговый статус по фактическим данным (полный сценарий, честно)."""
    full = ['convert', 'validate', 'bench', 'eval']
    for r in results:
        r['status'] = 'ПРОЙДЕНО' if model_is_complete(r, full, quantize) else 'ПРОВАЛ'


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


def run_report(report_dir: Path | None = None) -> None:
    """Собрать отчёт с графиками (make_report.py в convert-контейнере).

    report_dir — каталог для отчёта (`--report-dir`); по умолчанию
    benchmark/results/report.
    """
    report_dir = (report_dir or (RESULTS_DIR / 'report')).resolve()
    log(f'[ОТЧЁТ] Генерация отчёта с графиками (make_report.py) в {report_dir}...')
    # Хостовый путь -> путь в контейнере (results монтируется в /app/benchmark/results)
    try:
        container_out = Path('/app/benchmark/results') / report_dir.relative_to(RESULTS_DIR.resolve())
    except ValueError:
        log(f'[ОТЧЁТ] ОШИБКА: --report-dir вне {RESULTS_DIR} не поддерживается (volumes внутри results)')
        _fail('report', f'--report-dir вне {RESULTS_DIR}')
        return
    result = _run_container(
        CONVERT_COMPOSE, 'convert',
        ['/app/benchmark/make_report.py', '--out', str(container_out)],
        REPORT_TIMEOUT, 'report',
        volumes=[f'{RESULTS_DIR}:/app/benchmark/results'],
    )
    if result is None:
        _fail('report', 'генерация отчёта не удалась')
        return
    summary = _last_json_line(result.stdout)
    if not summary:
        log('[ОТЧЁТ] не удалось распарсить JSON-ответ make_report.py')
        return
    log(f'[ОТЧЁТ] графиков построено: {summary.get("graphs")}')

    report_md = report_dir / 'report.md'
    if report_md.exists():
        log(f'[ГОТОВО] отчёт: {report_md}')
    else:
        log(f'[ОТЧЁТ] отчёт не найден: {report_md}')


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Оркестратор YOLO бенчмарка")
    parser.add_argument("--phase", nargs='+', default=['convert', 'validate', 'bench', 'eval'],
                        choices=['convert', 'validate', 'bench', 'eval', 'all'],
                        help="Фазы бенчмарка (по умолчанию — все: convert validate bench eval)")
    parser.add_argument("--quantize", choices=["onnx", "none", "int8", "int8-calib", "all"], default="all",
                        help="Какие варианты RKNN тестировать: onnx=CPU only, none=FP16, int8=INT8, int8-calib=INT8 с калибровкой, all=все")
    parser.add_argument("--model", default=None,
                        help="Запуск для одной модели (напр. yolo26x-seg)")
    parser.add_argument("--pilot", action="store_true",
                        help="Только PILOT-модели (минимальные размеры)")
    parser.add_argument("--force", action="store_true",
                        help="Перезапускать eval/bench даже если результат уже есть в results.json")
    parser.add_argument("--skip-report", action="store_true",
                        help="Не генерировать отчёт с графиками в конце прогона")
    parser.add_argument("--report-dir", default=None,
                        help="Каталог для отчёта (по умолчанию benchmark/results/report)")
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
    processed: list[str] = []  # модели, обработанные в этом прогоне (для итога)

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
            try:
                ensure_dataset(model)
            except Exception as e:
                log(f'[ОШИБКА] {model.name} [dataset]: {e}')
                _fail(f'{model.name} [dataset]', str(e))
                save_results(existing)
                continue

            # Полный skip если все фазы уже завершены (кроме --force).
            # Для convert фазы дополнительно проверяем наличие файлов: conversion_status
            # мог быть выставлен 'OK' раньше, когда дособрались не все варианты
            # (например sem: onnx+fp16 готовы, а int8/int8-calib нет) — тогда не скипаем.
            existing_entry = next((r for r in existing if r['model'] == model.name), None)
            if existing_entry and not cli_args.force and model_is_complete(existing_entry, phases, QUANTIZE):
                if phase_is(phases, 'convert') and _convert_files_missing(model_cache_dir, model.name, QUANTIZE):
                    log(f'[ИНФО] convert неполон: {", ".join(_convert_files_missing(model_cache_dir, model.name, QUANTIZE))} — дособираю')
                else:
                    log(f'[ПРОПУСК] все фазы завершены')
                    continue
            processed.append(model.name)

            # --- Фаза: convert ---
            if phase_is(phases, 'convert'):
                # Resume: пропускаем convert только если ВСЕ нужные файлы на месте.
                # (ранее проверялся только _fp16.rknn — это была дыра: при наличии fp16
                # отсутствующие int8/int8-calib тихо пропускались и не дособирались)
                missing = _convert_files_missing(model_cache_dir, model.name, QUANTIZE)
                if not missing:
                    all_convert = QUANTIZE_TO_EVAL.get(QUANTIZE, ['onnx', 'fp16', 'int8', 'int8-calib'])
                    log(f'[ПРОПУСК] convert: все файлы ({", ".join(all_convert)}) уже на месте')
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
                        for j, r in enumerate(existing):
                            if r['model'] == model.name and r.get('conversion_status') != 'OK':
                                existing[j]['conversion_status'] = 'OK'
                                existing[j].pop('conversion_error', None)
                                log(f'[ИНФО] {model.name}: файлы convert на месте, исправлен conversion_status')
                                save_results(existing)
                                break
                else:
                    conv_result = run_converter(model.name, model.task, model_cache_dir, model_data_dir)
                    if conv_result is None or conv_result.get('status') != 'OK':
                        _fail(f'{model.name} [convert]',
                              conv_result.get('error', 'конвертация не удалась') if conv_result
                              else 'конвертация вернула None/таймаут')
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
                        _fail(f'{model.name} [validate]', 'валидация не удалась')
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
                need_fps = QUANTIZE_TO_FPS.get(QUANTIZE, ['cpu', 'npu_none', 'npu_int8', 'npu_int8-calib'])
                for r in existing:
                    if r['model'] == model.name:
                        fps_existing = r.get('fps', {})
                        if not cli_args.force and all(fps_existing.get(k, {}).get('fps', -1) > 0 for k in need_fps):
                            log(f'[ПРОПУСК] bench: уже замерено (results.json)')
                            bench_done = True
                        elif cli_args.force:
                            log('[--force] bench: перезамер всех вариантов')
                        break

                if not bench_done:
                    if not (model_cache_dir / f'{model.name}.onnx').exists() and \
                       not (model_cache_dir / f'{model.name}_fp16.rknn').exists():
                        _fail(f'{model.name} [bench]', 'нет моделей в cache/ и нет результатов')
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
                            _fail(f'{model.name} [bench]', 'FPS-тест не удался')
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
                                missing_fps = [k for k in need_fps
                                               if fps.get(k, {}).get('fps', -1) <= 0]
                                if missing_fps:
                                    _fail(f'{model.name} [bench]',
                                          f'не получены FPS: {", ".join(missing_fps)}')
                                    existing[j]['status'] = 'ПРОВАЛ'
                                else:
                                    existing[j]['status'] = 'ПРОЙДЕНО'
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
                if not cli_args.force and all(vr in done_keys for vr in eval_variants):
                    log(f'[ПРОПУСК] eval: все варианты уже выполнены (results.json)')
                else:
                    if cli_args.force:
                        log('[--force] eval: пересчёт всех вариантов')
                    for variant in eval_variants:
                        if not cli_args.force and variant in done_keys:
                            log(f'[ПРОПУСК] eval {variant}: уже выполнено (results.json)')
                            continue
                        try:
                            eval_result = run_eval(model.name, model.task, model_cache_dir,
                                                   model_data_dir, variant)
                        except FileNotFoundError as e:
                            _fail(f'{model.name} [eval]', f'{variant}: {e}')
                            eval_result = {'error': str(e)}
                        if eval_result is None:
                            _fail(f'{model.name} [eval]', f'{variant}: процесс упал/таймаут')
                            eval_result = {}
                        eval_results[variant] = eval_result
                    for j, r in enumerate(existing):
                        if r['model'] == model.name:
                            existing[j]['eval'] = eval_results
                            break
                    save_results(existing)
                    log(f'[ГОТОВО] eval {model.name}: {json.dumps(eval_results, ensure_ascii=False)}')

                # Очистка тяжёлых файлов (.pt) после всех фаз
                try:
                    entry = next((r for r in existing if r['model'] == model.name), None)
                    if entry and model_is_complete(entry, phases, QUANTIZE):
                        _cleanup_model_cache(model_cache_dir, model.name)
                except Exception as e:
                    log(f'[ОЧИСТКА] {model.name}: ошибка очистки (не критично): {e}')

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log(f'[ПРЕРВАНО] Ctrl+C')
            _fail('main', 'Ctrl+C')
        else:
            log(f'[ОШИБКА] {e}')
            _fail('main', f'необработанное исключение: {e}')
    finally:
        restart_percept(percept_was_running)

    recompute_status(existing, QUANTIZE)
    write_tsv(existing)
    save_results(existing)

    log(f'\n{"="*60}')
    log(f'БЕНЧМАРК ЗАВЕРШЁН: {len(existing)} моделей')

    incomplete: list[tuple[str, list[str]]] = []
    for name in processed:
        entry = next((r for r in existing if r['model'] == name), None)
        if entry is None:
            continue
        missing = missing_parts(entry, phases, QUANTIZE)
        if missing:
            incomplete.append((name, missing))
            log(f'[ИТОГ] {name}: НЕПОЛНО ({", ".join(missing)})')
        else:
            log(f'[ИТОГ] {name}: OK')

    if not FAILURES and not incomplete and not cli_args.skip_report:
        run_report(Path(cli_args.report_dir) if cli_args.report_dir else None)
    elif not cli_args.skip_report:
        log('[ОТЧЁТ] пропущен: есть сбои или неполные модели')

    if FAILURES:
        log(f'\nСбои в прогоне ({len(FAILURES)}):')
        for failure in FAILURES:
            log(f'  - {failure}')
    for name, missing in incomplete:
        log(f'  [ИТОГ] {name}: не хватает {", ".join(missing)}')

    log(f'Результаты: {RESULTS_FILE}')
    log(f'Сводка: {SUMMARY_TSV}')

    print(f'\nСводка ({SUMMARY_TSV}):')
    print(SUMMARY_TSV.read_text() if SUMMARY_TSV.exists() else '(пусто)')

    if FAILURES or incomplete:
        log(f'БЕНЧМАРК ЗАВЕРШЁН С ОШИБКАМИ (exit=1)')
        sys.exit(1)
    log(f'БЕНЧМАРК ЗАВЕРШЁН УСПЕШНО (exit=0)')


if __name__ == '__main__':
    main()
