#!/usr/bin/env python3
"""Исследование эффективности параллельного постпроцесса eval (NMS/маски).

Запускает существующий eval.py через воркер-подпроцесс, подменяя число воркеров
путла (monkeypatch `_big_core_count`), и параллельно сэмплирует загрузку CPU
per-core, частоты и температуры. Production-код eval.py НЕ изменяется.

Запуск (в convert-контейнере, где есть torch/rknnlite и NPU):
  docker compose -f convert/docker-compose.yml run --rm --entrypoint python \
    convert /app/benchmark/probe_parallel.py [--limit 300] [--workers 1,4,6,8]

По умолчанию авто-детектит модели в cache/ (как в /app/cache), задачи по суффиксу
имени и датасеты по задаче. Каждый конфиг можно перезапускать — уже выполненные
пропускаются (resume по results/parallel_probe.json).

Примеры:
  # быстрая проверка работы скрипта (пойдёт в отдельный файл, не перезапишет результаты)
  python probe_parallel.py --smoke
  # полное исследование всех доступных моделей: fp16+onnx, 1/4/6/8 воркеров, 300 кадров
  python probe_parallel.py --limit 300 --workers 1,4,6,8 --variants fp16,onnx
  # одна модель, все 4 варианта, 8 воркеров
  python probe_parallel.py --models yolo26n-seg --variants all --workers 1,8 --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    'onnx': {'backend': 'cpu', 'suffix': '', 'file': lambda n: f'{n}.onnx'},
    'fp16': {'backend': 'npu', 'suffix': '_fp16', 'file': lambda n: f'{n}_fp16.rknn'},
    'int8': {'backend': 'npu', 'suffix': '_int8', 'file': lambda n: f'{n}_int8.rknn'},
    'int8-calib': {'backend': 'npu', 'suffix': '_int8-calib', 'file': lambda n: f'{n}_int8-calib.rknn'},
}

MODEL_TASK_SUFFIXES = [('-seg', 'segment'), ('-pose', 'pose'), ('-obb', 'obb'),
                       ('-cls', 'classify'), ('-sem', 'semantic')]

TASK_DATA = {
    'detect': 'coco_val2017',
    'segment': 'coco_val2017',
    'pose': 'coco_val2017',
    'obb': 'dotav1',
    'classify': 'imagenet',
    'semantic': 'cityscapes',
}


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def model_task(name: str) -> str:
    for suf, task in MODEL_TASK_SUFFIXES:
        if name.endswith(suf):
            return task
    return 'detect'


def resolve_model_dir(name: str) -> str:
    c = Path('/app/cache', name)
    if c.exists():
        return str(c)
    h = ROOT / 'cache' / name
    if h.exists():
        return str(h)
    return str(c)


def resolve_data_dir(task: str) -> str:
    d = TASK_DATA.get(task)
    if not d:
        return ''
    c = Path('/app/datasets', d)
    if c.exists():
        return str(c)
    h = ROOT / 'datasets' / d
    if h.exists():
        return str(h)
    return str(c)


def _is_model_dir(p: Path) -> bool:
    if p.name.startswith('.'):
        return False
    try:
        for f in p.iterdir():
            if f.suffix.lower() in ('.pt', '.onnx', '.rknn'):
                return True
    except OSError:
        return False
    return False


def discover_models() -> list[tuple[str, str]]:
    cache_dirs = [Path('/app/cache') if Path('/app/cache').exists() else ROOT / 'cache']
    out = []
    for cd in cache_dirs:
        if not cd.exists():
            continue
        for p in sorted(cd.iterdir()):
            if p.is_dir() and _is_model_dir(p):
                out.append((p.name, model_task(p.name)))
    seen = set()
    uniq = []
    for name, task in out:
        if name not in seen:
            seen.add(name)
            uniq.append((name, task))
    return uniq


# ---------------------------------------------------------------------------
# Sampler: CPU per-core load / частоты / температуры / NPU
# ---------------------------------------------------------------------------

def _read_stat_per_core() -> dict[int, tuple[int, int]]:
    out = {}
    try:
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu') and not line.startswith('cpu '):
                    parts = line.split()
                    v = [int(x) for x in parts[1:9]]
                    out[int(parts[0][3:])] = (sum(v), v[3] + v[4])
    except (OSError, ValueError):
        pass
    return out


def _read_cpu_freqs() -> dict[int, int]:
    out = {}
    try:
        base = Path('/sys/devices/system/cpu')
        for d in sorted(base.glob('cpu[0-9]*')):
            try:
                out[int(d.name[3:])] = int((d / 'cpufreq' / 'scaling_cur_freq').read_text().strip())
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return out


def _read_max_freqs() -> dict[int, int]:
    out = {}
    try:
        base = Path('/sys/devices/system/cpu')
        for d in sorted(base.glob('cpu[0-9]*')):
            try:
                out[int(d.name[3:])] = int((d / 'cpufreq' / 'cpuinfo_max_freq').read_text().strip())
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return out


def _read_simple_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_thermal_max() -> float | None:
    temps = []
    try:
        for z in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
            try:
                temps.append(float((z / 'temp').read_text().strip()) / 1000.0)
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return max(temps) if temps else None


def _read_npu_load() -> float | None:
    import re
    try:
        with open('/sys/kernel/debug/rknpu/load') as f:
            m = re.findall(r'Core\d+:\s*(\d+)%', f.read())
        if m:
            return sum(int(x) for x in m) / len(m)
    except (OSError, ValueError):
        pass
    return None


def _read_npu_temp() -> float | None:
    v = _read_simple_int('/sys/class/thermal/thermal_zone6/temp')
    return v / 1000.0 if v is not None else None


def _read_npu_freq() -> int | None:
    return _read_simple_int('/sys/kernel/debug/rknpu/freq')


class Sampler:
    def __init__(self, interval: float = 1.0):
        self.interval = max(0.2, interval)
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None
        self.cpu_load: dict[int, list[float]] = {}
        self.cpu_freq: dict[int, list[int]] = {}
        self.npu_load: list[float] = []
        self.npu_temp: list[float] = []
        self.npu_freq: list[int] = []
        self.thermal: list[float] = []
        self._prev: dict[int, tuple[int, int]] | None = None

    def start(self) -> None:
        self._prev = _read_stat_per_core()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=max(5, self.interval * 2 + 1))

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.interval)
            cur = _read_stat_per_core()
            if cur and self._prev:
                for cid, (ct, cidle) in cur.items():
                    prev = self._prev.get(cid)
                    if prev:
                        dt = ct - prev[0]
                        di = cidle - prev[1]
                        if dt > 0:
                            self.cpu_load.setdefault(cid, []).append(
                                100.0 * (1.0 - di / dt))
            self._prev = cur
            for cid, f in _read_cpu_freqs().items():
                self.cpu_freq.setdefault(cid, []).append(f)
            for val, store in ((_read_npu_load(), self.npu_load),
                               (_read_npu_temp(), self.npu_temp),
                               (_read_npu_freq(), self.npu_freq),
                               (_read_thermal_max(), self.thermal)):
                if val is not None:
                    store.append(val)

    def summary(self, max_freqs: dict[int, int]) -> dict:
        def avg(vals):
            return round(sum(vals) / len(vals), 1) if vals else None

        big = {cid for cid, mx in max_freqs.items() if mx == (max(max_freqs.values()) if max_freqs else 0)}
        little = set(max_freqs) - big
        per_core_load = {cid: avg(v) for cid, v in sorted(self.cpu_load.items())}

        def cluster_avg(cores):
            vals = [per_core_load[c] for c in sorted(cores) if per_core_load.get(c) is not None]
            return avg(vals)

        ratios = []
        for cid, freqs in self.cpu_freq.items():
            mx = max_freqs.get(cid)
            loads = self.cpu_load.get(cid, [])
            loaded = bool(loads) and (sum(loads) / len(loads)) > 50.0
            if mx and loaded and freqs:
                ratios.append((sum(freqs) / len(freqs)) / mx)
        return {
            'n_samples': len(self.thermal) or len(next(iter(self.cpu_load.values()), [])),
            'cpu_load_per_core': per_core_load,
            'big_load_avg': cluster_avg(big),
            'little_load_avg': cluster_avg(little),
            'freq_ratio_min': round(min(ratios), 3) if ratios else None,
            'npu_load_avg': avg(self.npu_load),
            'npu_temp_min': round(min(self.npu_temp), 1) if self.npu_temp else None,
            'npu_temp_max': round(max(self.npu_temp), 1) if self.npu_temp else None,
            'npu_freq_min_mhz': round(min(self.npu_freq) / 1e6) if self.npu_freq else None,
            'thermal_max_c': round(max(self.thermal), 1) if self.thermal else None,
        }


# ---------------------------------------------------------------------------
# Воркер: eval.py с подменённым числом воркеров
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> int:
    n_workers = int(os.environ.get('PROBE_WORKERS', '4'))
    sys.path.insert(0, str(ROOT))
    import benchmark.eval as ev
    ev._big_core_count = lambda: n_workers
    sys.stderr.write(f'[PROBE] воркер с N={n_workers}\n')
    sys.stderr.flush()
    res = ev.evaluate(args.task, args.data_dir, model_dir=args.model_dir,
                      backend=args.backend, model_name=args.model_name,
                      limit=args.limit, rknn_suffix=args.rknn_suffix)
    print('EVAL_RESULT ' + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Родитель
# ---------------------------------------------------------------------------

def wait_cool(max_temp: float, timeout: float, label: str) -> float:
    t0 = time.monotonic()
    while True:
        t = _read_thermal_max()
        if t is not None and t <= max_temp:
            return t
        if time.monotonic() - t0 > timeout:
            log(f'[ОЖИДАНИЕ] {label}: не остыло до {max_temp}°C за {timeout:.0f}с '
                f'(сейчас {t if t is not None else "?"}°C) — продолжаю')
            return t
        time.sleep(5)


def metrics_equal(eval_res: dict) -> bool:
    return bool(eval_res.get('error') is None and 'metrics_equal' in eval_res)


def run_one(cfg: dict, args: argparse.Namespace, results: dict) -> dict:
    key = f"{cfg['model']}|{cfg['variant']}|{cfg['workers']}"
    if not args.force and key in results:
        log(f'[SKIP] {key} — уже выполнен (--force для повтора)')
        return results[key]

    model, variant, workers = cfg['model'], cfg['variant'], cfg['workers']
    v = VARIANTS[variant]
    model_dir = cfg['model_dir']
    model_file = Path(model_dir) / v['file'](model)
    if not model_file.exists():
        log(f'[ПРОПУСК] {key}: нет файла {model_file}')
        results[key] = {'key': key, 'model': model, 'task': cfg['task'],
                        'variant': variant, 'workers': workers,
                        'status': 'skip', 'error': f'нет модели {model_file.name}'}
        return results[key]
    data_dir = cfg['data_dir']
    if not data_dir:
        log(f'[ПРОПУСК] {key}: нет датасета для задачи {cfg["task"]}')
        results[key] = {'key': key, 'model': model, 'task': cfg['task'],
                        'variant': variant, 'workers': workers,
                        'status': 'skip', 'error': 'нет датасета'}
        return results[key]

    log(f'[ПРОГОН] {key}  task={cfg["task"]} limit={args.limit} '
        f'backend={v["backend"]} suffix={v["suffix"]!r}')

    if args.cool:
        t = wait_cool(args.max_cool_temp, args.cool_timeout, f'{key}')
        log(f'[ТЕМП] {key}: старт при {t if t is not None else "?"}°C (порог {args.max_cool_temp}°C)')
        time.sleep(2)

    sampler = Sampler(interval=args.probe_interval)
    sampler.start()

    log_dir = ROOT / 'cache' / 'probe_parallel_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'{model}_{variant}_w{workers}.log'
    env = dict(os.environ)
    env['PROBE_WORKERS'] = str(workers)
    cmd = [sys.executable, str(Path(__file__).resolve()),
           '--worker', cfg['task'], data_dir,
           '--model-dir', model_dir, '--model-name', model,
           '--backend', v['backend'], '--limit', str(args.limit)]
    if v['suffix']:
        cmd += ['--rknn-suffix', v['suffix']]

    t0 = time.monotonic()
    start_temp = _read_thermal_max()
    ret = None
    try:
        ret = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=args.worker_timeout)
    except subprocess.TimeoutExpired:
        log(f'[ОШИБКА] {key}: таймаут {args.worker_timeout}с')
        ret = None
    wall_s = time.monotonic() - t0
    end_temp = _read_thermal_max()
    sampler.stop()

    with open(log_file, 'w') as f:
        f.write('=== STDOUT ===\n')
        f.write(ret.stdout if ret and ret.stdout else '')
        f.write('\n=== STDERR ===\n')
        f.write(ret.stderr if ret and ret.stderr else '')

    eval_res = None
    status = 'ok'
    error = None
    if ret is None:
        status, error = 'error', 'таймаут воркера'
    elif ret.returncode != 0:
        status, error = 'error', f'код возврата {ret.returncode}'
    else:
        idx = ret.stdout.rfind('EVAL_RESULT ')
        if idx < 0:
            status, error = 'error', 'нет EVAL_RESULT в stdout'
        else:
            try:
                payload = ret.stdout[idx + len('EVAL_RESULT '):].strip()
                if payload:
                    payload = payload.splitlines()[0]
                eval_res = json.loads(payload)
            except json.JSONDecodeError as e:
                status, error = 'error', f'битый JSON: {e}'

    entry = {
        'key': key, 'model': model, 'task': cfg['task'],
        'variant': variant, 'workers': workers,
        'status': status, 'error': error,
        'wall_s': round(wall_s, 1),
        'start_temp_c': round(start_temp, 1) if start_temp is not None else None,
        'end_temp_c': round(end_temp, 1) if end_temp is not None else None,
        'cpu': sampler.summary(_read_max_freqs()),
        'log_file': str(log_file),
    }
    if eval_res is not None:
        entry['eval'] = eval_res
    results[key] = entry
    save_results(results, args.output)
    log(f'[ГОТОВО] {key}: wall={wall_s:.1f}с status={status} '
        f'{("error=" + error) if error else ""}')
    return entry


def save_results(results: dict, output) -> None:
    output = Path(output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        tmp.replace(output)
    except OSError as e:
        log(f'[ПРЕДУПРЕЖДЕНИЕ] не удалось сохранить результаты: {e}')


_TELEMETRY_PREFIXES = ('npu_', 'cpu_', 'thermal_', 'throttled')


def _metric_keys(ev_res: dict) -> dict:
    return {k: v for k, v in ev_res.items()
            if isinstance(v, (int, float)) and not k.startswith(_TELEMETRY_PREFIXES)
            and k not in ('elapsed_s', 'total', 'errors', 'predictions')}


def build_summary(results: dict) -> list[dict]:
    by_cfg: dict[tuple, list[dict]] = {}
    for entry in results.values():
        if entry.get('status') != 'ok':
            continue
        by_cfg.setdefault((entry['model'], entry['task'], entry['variant']), []).append(entry)

    rows = []
    for (model, task, variant), entries in sorted(by_cfg.items()):
        entries.sort(key=lambda e: e['workers'])
        base = None
        eq = True
        for e in entries:
            keys = _metric_keys(e.get('eval') or {})
            if base is None:
                base = keys
            elif base != keys:
                eq = False
        rows.append({
            'model': model, 'task': task, 'variant': variant,
            'metrics_equal': eq,
            'runs': [
                {
                    'workers': e['workers'], 'wall_s': e['wall_s'],
                    'big_load_avg': (e.get('cpu') or {}).get('big_load_avg'),
                    'little_load_avg': (e.get('cpu') or {}).get('little_load_avg'),
                    'freq_ratio_min': (e.get('cpu') or {}).get('freq_ratio_min'),
                    'thermal_max_c': (e.get('cpu') or {}).get('thermal_max_c'),
                    'npu_load_avg': (e.get('cpu') or {}).get('npu_load_avg'),
                    'map50': (e.get('eval') or {}).get('map50'),
                } for e in entries
            ],
        })
    return rows


def _fmt(v, fmt: str, empty: str = '     -') -> str:
    return empty if v is None else f'{v:{fmt}}'


def print_table(rows: list[dict]) -> None:
    if not rows:
        log('Нет завершённых прогонов.')
        return
    hdr = (f'{"model":<14}{"task":<9}{"variant":<11}{"N":>3}{"wall_s":>9}'
           f'{"A76%":>7}{"A55%":>7}{"freq":>6}{"tmax":>6}{"npu%":>6}')
    print()
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        for run in r['runs']:
            print(f'{r["model"]:<14}{r["task"]:<9}{r["variant"]:<11}{run["workers"]:>3}'
                  f'{run["wall_s"]:>9.1f}'
                  f'{_fmt(run.get("big_load_avg"), ">6.0f", "      -")}'
                  f'{_fmt(run.get("little_load_avg"), ">6.0f", "      -")}'
                  f'{_fmt(run.get("freq_ratio_min"), ">5.2f", "     -")}'
                  f'{_fmt(run.get("thermal_max_c"), ">5.0f", "     -")}'
                  f'{_fmt(run.get("npu_load_avg"), ">5.1f", "     -")}')
        eq = 'ДА' if r['metrics_equal'] else 'НЕТ!'
        print(f'  метрики эквивалентны по N: {eq}')
    print()


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('task', nargs='?', default='')
    p.add_argument('data_dir', nargs='?', default='')
    p.add_argument('--model-dir', default='')
    p.add_argument('--model-name', default='')
    p.add_argument('--backend', default='')
    p.add_argument('--rknn-suffix', default='')

    p.add_argument('--limit', type=int, default=300)
    p.add_argument('--workers', default='1,4,6,8')
    p.add_argument('--variants', default='fp16,onnx')
    p.add_argument('--models', default='')
    p.add_argument('--tasks', default='')
    p.add_argument('--output', default=str(ROOT / 'cache' / 'probe_parallel.json'))
    p.add_argument('--force', action='store_true')
    p.add_argument('--cool', action='store_true', default=True)
    p.add_argument('--no-cool', dest='cool', action='store_false')
    p.add_argument('--max-cool-temp', type=float, default=60.0)
    p.add_argument('--cool-timeout', type=float, default=900.0)
    p.add_argument('--probe-interval', type=float, default=1.0)
    p.add_argument('--worker-timeout', type=float, default=3600.0)
    p.add_argument('--smoke', action='store_true', help='быстрая проверка: limit 50, 1/8, fp16')
    return p.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    if args.worker:
        return run_worker(args)

    if args.smoke:
        args.limit = 50
        args.workers = '1,8'
        args.variants = 'fp16'
        args.output = str(ROOT / 'cache' / 'probe_parallel_smoke.json')

    workers_list = [int(w) for w in args.workers.split(',') if w.strip().isdigit()]
    variants_list = [v for v in args.variants.split(',') if v.strip()]
    if not workers_list:
        log('[ОШИБКА] пустой --workers')
        return 1
    if not variants_list:
        log('[ОШИБКА] пустой --variants')
        return 1
    for v in variants_list:
        if v not in VARIANTS:
            log(f'[ОШИБКА] неизвестный вариант {v!r} (есть: {", ".join(VARIANTS)})')
            return 1

    output = Path(args.output)
    results = {}
    if output.exists():
        try:
            results = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            log(f'[ПРЕДУПРЕЖДЕНИЕ] битый {output} — начинаю заново')

    selected = args.models.split(',') if args.models else None
    selected_tasks = set(args.tasks.split(',')) if args.tasks else None

    log(f'Исследование: limit={args.limit} workers={workers_list} '
        f'variants={variants_list} охлаждение={args.max_cool_temp}°C')
    if selected:
        models = [(m.strip(), model_task(m.strip())) for m in selected if m.strip()]
        models = [m for m in models if m[0]]
    else:
        models = discover_models()
    if not models:
        log('[ПРЕДУПРЕЖДЕНИЕ] нет моделей в cache/ — укажите --models')
        return 1
    if selected_tasks:
        models = [m for m in models if m[1] in selected_tasks]
    if not models:
        log('[ПРЕДУПРЕЖДЕНИЕ] после фильтрации по задачам моделей не осталось')
        return 1
    log(f'Модели: {", ".join(f"{m}({t})" for m, t in models)}')

    try:
        for model, task in models:
            model_dir = resolve_model_dir(model)
            data_dir = resolve_data_dir(task)
            for variant in variants_list:
                for workers in workers_list:
                    run_one({'model': model, 'task': task, 'variant': variant,
                             'workers': workers, 'model_dir': model_dir,
                             'data_dir': data_dir}, args, results)
    except KeyboardInterrupt:
        log('[ПРЕРВАНО] сохраняю частичные результаты')
        save_results(results, output)

    save_results(results, output)
    rows = build_summary(results)
    print_table(rows)
    log(f'Результаты: {output}')
    log(f'Полные логи прогонов: {ROOT / "cache" / "probe_parallel_logs"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
