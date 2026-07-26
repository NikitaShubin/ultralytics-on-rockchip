#!/usr/bin/env python3
"""Стресс-тест Orange Pi 3B (RK3588S): CPU + NPU + RAM + температура.

Использование:
    python3 stress_test.py                     # все компоненты, 5 минут
    python3 stress_test.py --duration 600      # 10 минут
    python3 stress_test.py --cpu-only           # только CPU
    python3 stress_test.py --npu-cores 0,1,2 --npu-streams 3
    python3 stress_test.py --log results.json   # записать в JSON

CPU/RAM запускаются с nice=19 (низкий приоритет). NPU — нормальный приоритет.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── Детекция rknnlite ──────────────────────────────────────────────────────

try:
    from rknnlite.api import RKNNLite as _RKNNLite
    HAS_RKNNLITE = True
except ImportError:
    HAS_RKNNLITE = False

PROJECT_DIR = Path(__file__).parent.resolve()

# ─── Температура ───────────────────────────────────────────────────────────

THERMAL_ZONES: dict[str, str] = {}


def init_thermal() -> dict[str, str]:
    """Читает thermal_zone* → name → path."""
    zones = {}
    base = Path('/sys/class/thermal')
    for zone_dir in sorted(base.glob('thermal_zone*')):
        type_file = zone_dir / 'type'
        temp_file = zone_dir / 'temp'
        if type_file.exists() and temp_file.exists():
            name = type_file.read_text().strip()
            zones[name] = str(temp_file)
    return zones


def read_temps(zones: dict[str, str]) -> dict[str, float]:
    """Читает текущие температуры (°C)."""
    temps = {}
    for name, path in zones.items():
        try:
            raw = int(Path(path).read_text().strip())
            temps[name] = raw / 1000.0
        except (ValueError, OSError):
            pass
    return temps


# ─── NPU стресс-тест ──────────────────────────────────────────────────────

NPU_CORES_MAP = {
    0: 'NPU_CORE_0',
    1: 'NPU_CORE_1',
    2: 'NPU_CORE_2',
    3: 'NPU_CORE_0_1',
    4: 'NPU_CORE_0_1_2',
}


class NPUStress:
    """Бесконечный RKNN инференс на выбранных ядрах.

    На каждое ядро запускается `streams` потоков, каждый со своим RKNNRuntime,
    чтобы реально загрузить NPU (один поток = ~30-50% одного ядра).
    Автоматически выбирает: прямой rknnlite (если доступен) или Docker.
    """

    def __init__(self, model_path: str, core_ids: list[int],
                 fps_counter: list, streams_per_core: int = 1):
        self.model_path = model_path
        self.core_ids = core_ids
        self.fps_counter = fps_counter  # shared [total_frames, stop_flag]
        self.streams_per_core = streams_per_core
        self._threads: list[threading.Thread] = []
        self._docker_proc: subprocess.Popen | None = None

    def start(self):
        if HAS_RKNNLITE:
            self._start_local()
        else:
            self._start_docker()

    def stop(self):
        self.fps_counter[1] = 1  # stop flag
        if self._docker_proc:
            self._docker_proc.terminate()
            try:
                self._docker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._docker_proc.kill()
            self._docker_proc = None
        _stop_npu_stress_in_container()
        for t in self._threads:
            t.join(timeout=5)

    def _start_local(self):
        for core_id in self.core_ids:
            for stream_idx in range(self.streams_per_core):
                t = threading.Thread(
                    target=self._run,
                    args=(core_id, stream_idx),
                    daemon=True,
                )
                self._threads.append(t)
                t.start()

    def _run(self, core_id: int, stream_idx: int):
        from rknnlite.api import RKNNLite

        core_name = NPU_CORES_MAP.get(core_id, f'NPU_CORE_{core_id}')
        core_mask = getattr(RKNNLite, core_name, None)
        if core_mask is None:
            return

        # каждый RKNN runtime — отдельный экземпляр
        rknn = RKNNLite()
        ret = rknn.load_rknn(self.model_path)
        if ret != 0:
            print(f'[NPU] core{core_id} stream{stream_idx}: load_rknn failed ({ret})')
            return

        ret = rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            print(f'[NPU] core{core_id} stream{stream_idx}: init_runtime failed ({ret})')
            return

        dummy = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
        label = f'core{core_id}/s{stream_idx}'
        print(f'[NPU] {label}: инференс запущен')

        while not self.fps_counter[1]:
            try:
                rknn.inference(inputs=[dummy])
                self.fps_counter[0] += 1
            except Exception as e:
                print(f'[NPU] {label}: ошибка — {e}')
                time.sleep(1)

        rknn.release()
        print(f'[NPU] {label}: остановлен')

    def _start_docker(self):
        """Запуск NPU стресса через Docker (rknnlite недоступен на хосте)."""
        ensure_percept_container()

        script_dir = Path(__file__).parent.resolve()
        container_script = '/tmp/npu_stress_oc.py'

        # Создаём скрипт инференса
        import tempfile
        host_script = tempfile.mktemp(suffix='.py')
        cores_arg = ','.join(str(c) for c in self.core_ids)
        with open(host_script, 'w') as f:
            f.write(f'''\
import sys, signal, threading, time, numpy as np
CORES_MAP = {{
    0: 'NPU_CORE_0', 1: 'NPU_CORE_1', 2: 'NPU_CORE_2',
    3: 'NPU_CORE_0_1', 4: 'NPU_CORE_0_1_2',
}}
model_path = sys.argv[1]
core_ids = [int(x) for x in sys.argv[2].split(',')]
streams = int(sys.argv[3])
stop = threading.Event()
def run(core_id, idx):
    from rknnlite.api import RKNNLite
    core_name = CORES_MAP.get(core_id, f'NPU_CORE_{{core_id}}')
    mask = getattr(RKNNLite, core_name, None)
    if mask is None: return
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        print(f'[NPU] core{{core_id}}/s{{idx}}: load failed', file=sys.stderr); return
    if rknn.init_runtime(core_mask=mask) != 0:
        print(f'[NPU] core{{core_id}}/s{{idx}}: init failed', file=sys.stderr); return
    dummy = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
    print(f'[NPU] core{{core_id}}/s{{idx}}: started', flush=True)
    while not stop.is_set():
        try:
            rknn.inference(inputs=[dummy])
        except Exception as e:
            print(f'[NPU] core{{core_id}}/s{{idx}}: {{e}}', file=sys.stderr)
            time.sleep(1)
    rknn.release()
    print(f'[NPU] core{{core_id}}/s{{idx}}: stopped', flush=True)
for cid in core_ids:
    for s in range(streams):
        t = threading.Thread(target=run, args=(cid, s), daemon=True)
        t.start()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
while True: time.sleep(60)
''')

        # Копируем скрипт и модель в контейнер
        subprocess.run(['docker', 'cp', host_script, f'percept:{container_script}'],
                       capture_output=True)
        os.unlink(host_script)

        stress_basename = Path(self.model_path).name
        container_model = f'/app/model/{stress_basename}'
        subprocess.run(['docker', 'cp', self.model_path, f'percept:{container_model}'],
                       capture_output=True)

        _stop_npu_stress_in_container()
        time.sleep(1)

        cores_arg = ','.join(str(c) for c in self.core_ids)
        self._docker_proc = subprocess.Popen(
            ['docker', 'exec', 'percept', 'python3', container_script,
             container_model, cores_arg, str(self.streams_per_core)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        total = len(self.core_ids) * self.streams_per_core
        print(f'[NPU] Docker: {len(self.core_ids)} ядра × {self.streams_per_core} потоков = {total} (PID {self._docker_proc.pid})')


# ─── Docker container management ─────────────────────────────────────────────

_percept_container_started = False


def ensure_percept_container():
    """Проверяет/запускает контейнер percept для NPU инференса."""
    global _percept_container_started
    result = subprocess.run(
        ['docker', 'inspect', '-f', '{{.State.Running}}', 'percept'],
        capture_output=True, text=True,
    )
    if result.stdout.strip() == 'true':
        return True
    print('[NPU] Контейнер percept не запущен, запускаю...')
    subprocess.run(
        ['docker', 'compose', '-f', str(PROJECT_DIR / 'docker-compose.yml'),
         'up', '-d', 'percept'],
        capture_output=True, text=True,
    )
    for _ in range(30):
        time.sleep(1)
        result = subprocess.run(
            ['docker', 'inspect', '-f', '{{.State.Running}}', 'percept'],
            capture_output=True, text=True,
        )
        if result.stdout.strip() == 'true':
            time.sleep(2)
            print('[OK] Контейнер percept запущен')
            _percept_container_started = True
            return True
    print('[ОШИБКА] Не удалось запустить контейнер percept')
    return False


def stop_percept_if_started():
    """Останавливает контейнер percept, если был запущен автоматически."""
    global _percept_container_started
    if _percept_container_started:
        print('[NPU] Останавливаю контейнер percept...')
        subprocess.run(
            ['docker', 'compose', '-f', str(PROJECT_DIR / 'docker-compose.yml'),
             'stop', 'percept'],
            capture_output=True,
        )
        _percept_container_started = False


def _stop_npu_stress_in_container():
    """Останавливает NPU stress процессы внутри контейнера."""
    subprocess.run(
        ['docker', 'exec', 'percept', 'python3', '-c',
         'import os,signal,glob\n'
         'for f in glob.glob("/proc/[0-9]*/cmdline"):\n'
         ' try:\n'
         '  pid=int(f.split("/")[2])\n'
         '  if pid<=1: continue\n'
         '  cmd=open(f,"rb").read().decode("utf-8",errors="replace")\n'
         '  if "npu_stress" in cmd: os.kill(pid,signal.SIGKILL)\n'
         ' except: pass'],
        capture_output=True,
    )


# ─── CPU стресс-тест ──────────────────────────────────────────────────────

class CPUStress:
    """Запускает `stress` на всех ядрах."""

    def __init__(self, n_workers: int | None = None, nice: int = 19):
        self.n_workers = n_workers or os.cpu_count()
        self.nice = nice
        self._proc: subprocess.Popen | None = None

    def start(self):
        cmd = [
            'nice', '-n', str(self.nice),
            'stress',
            '--cpu', str(self.n_workers),
            '--timeout', '999d',
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f'[CPU] stress запущен: {self.n_workers} воркеров, nice={self.nice} (PID {self._proc.pid})')

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print(f'[CPU] stress остановлен')


# ─── RAM стресс-тест ──────────────────────────────────────────────────────

class RAMStress:
    """Запускает `stress` с нагрузкой на память."""

    def __init__(self, mb_per_worker: int = 512, n_workers: int = 2, nice: int = 19):
        self.mb = mb_per_worker
        self.n_workers = n_workers
        self.nice = nice
        self._proc: subprocess.Popen | None = None

    def start(self):
        cmd = [
            'nice', '-n', str(self.nice),
            'stress',
            '--vm', str(self.n_workers),
            '--vm-bytes', f'{self.mb}M',
            '--vm-keep',
            '--timeout', '999d',
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f'[RAM] stress запущен: {self.n_workers} × {self.mb} МБ, nice={self.nice} (PID {self._proc.pid})')

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            print('[RAM] stress остановлен')


# ─── Мониторинг ───────────────────────────────────────────────────────────

class Monitor:
    """Периодически печатает температуры, FPS, CPU load."""

    def __init__(
        self,
        thermal_zones: dict[str, str],
        npu_fps_counter: list,
        interval: float = 2.0,
        log_file: str | None = None,
    ):
        self.zones = thermal_zones
        self.npu_fps = npu_fps_counter
        self.interval = interval
        self.log_file = log_file
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: list[dict] = []
        self._prev_npu_frames = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.log_file:
            Path(self.log_file).write_text(json.dumps(self._history, indent=2))

    def _run(self):
        header_printed = False
        while not self._stop.is_set():
            temps = read_temps(self.zones)
            npu_total = self.npu_fps[0]
            npu_fps = (npu_total - self._prev_npu_frames) / self.interval
            self._prev_npu_frames = npu_total

            # CPU load из /proc/stat
            cpu_load = self._read_cpu_load()

            # NPU load
            npu_load = self._read_npu_load()

            # RAM
            ram = self._read_ram()

            entry = {
                'time': datetime.now().isoformat(),
                'temps': temps,
                'npu_fps': round(npu_fps, 1),
                'npu_total_frames': npu_total,
                'cpu_load': round(cpu_load, 1),
                'npu_load': npu_load,
                'ram_used_mb': ram['used_mb'],
                'ram_total_mb': ram['total_mb'],
            }
            self._history.append(entry)

            if not header_printed:
                self._print_header(temps)
                header_printed = True

            self._print_row(entry)
            self._stop.wait(self.interval)

    def _print_header(self, temps: dict):
        cols = ['ВРЕМЯ', 'CPU%']
        for name in sorted(temps.keys()):
            cols.append(f'T:{name}')
        cols += ['NPU%', 'NPUFPS', 'NPUФР', 'RAM_МБ']
        print('  '.join(f'{c:>10}' for c in cols))
        print('-' * (12 * len(cols)))

    def _print_row(self, entry: dict):
        ts = entry['time'][11:19]
        cols = [ts, f"{entry['cpu_load']:.0f}%"]
        for name in sorted(entry['temps'].keys()):
            cols.append(f"{entry['temps'][name]:.1f}°")
        # Средний NPU load по всем ядрам
        npu_load = entry.get('npu_load', {})
        if npu_load:
            avg = sum(npu_load.values()) / len(npu_load)
            cols.append(f"{avg:.0f}%")
        else:
            cols.append('-')
        cols.append(f"{entry['npu_fps']:.1f}")
        cols.append(f"{entry['npu_total_frames']}")
        cols.append(f"{entry['ram_used_mb']}")
        print('  '.join(f'{c:>10}' for c in cols))

    def _read_cpu_load(self) -> float:
        try:
            with open('/proc/stat') as f:
                line = f.readline()
            parts = line.split()
            # user nice system idle iowait irq softirq steal
            vals = [int(x) for x in parts[1:9]]
            idle = vals[3] + vals[4]
            total = sum(vals)
            # дельта между вызовами
            if not hasattr(self, '_prev_cpu'):
                self._prev_cpu = (idle, total)
                return 0.0
            prev_idle, prev_total = self._prev_cpu
            self._prev_cpu = (idle, total)
            d_idle = idle - prev_idle
            d_total = total - prev_total
            if d_total == 0:
                return 0.0
            return (1.0 - d_idle / d_total) * 100.0
        except Exception:
            return 0.0

    def _read_npu_load(self) -> dict[str, int]:
        """Читает NPU load в процентах по ядрам."""
        try:
            raw = Path('/sys/kernel/debug/rknpu/load').read_text().strip()
            result = {}
            for part in raw.split(','):
                part = part.strip()
                if 'Core' in part and ':' in part:
                    # "Core0: 32%" или "NPU load:  Core0: 32%"
                    colon_parts = part.rsplit(':', 1)
                    if len(colon_parts) == 2:
                        name = colon_parts[0].strip().split()[-1]  # "Core0"
                        val = colon_parts[1].strip().rstrip('%')
                        result[name] = int(val)
            return result
        except Exception:
            return {}

    def _read_ram(self) -> dict:
        try:
            with open('/proc/meminfo') as f:
                info = {}
                for line in f:
                    parts = line.split()
                    info[parts[0].rstrip(':')] = int(parts[1])
            total = info['MemTotal'] // 1024
            avail = info['MemAvailable'] // 1024
            return {'total_mb': total, 'used_mb': total - avail}
        except Exception:
            return {'total_mb': 0, 'used_mb': 0}


# ─── NPU load (rknpu/debug) ──────────────────────────────────────────────

def read_npu_load() -> str | None:
    try:
        return Path('/sys/kernel/debug/rknpu/load').read_text().strip()
    except OSError:
        return None


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Стресс-тест Orange Pi 3B (RK3588S)')
    parser.add_argument('--duration', type=int, default=300,
                        help='Длительность в секундах (по умолчанию: 300)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Только CPU + RAM (без NPU)')
    parser.add_argument('--npu-only', action='store_true',
                        help='Только NPU')
    parser.add_argument('--no-ram', action='store_true',
                        help='Без нагрузки на RAM')
    parser.add_argument('--npu-cores', default='0,1,2',
                        help='NPU ядра через запятую (по умолчанию: 0,1,2 — все)')
    parser.add_argument('--npu-streams', type=int, default=8,
                        help='Потоков инференса на ядро NPU (по умолчанию: 8)')
    parser.add_argument('--model', default=None,
                        help='RKNN модель (по умолчанию: yolo26x-seg.rknn)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Интервал мониторинга в секундах')
    parser.add_argument('--log', default=None,
                        help='Путь для JSON-лога результатов')
    parser.add_argument('--ram-mb', type=int, default=1024,
                        help='МБ на RAM воркер (по умолчанию: 1024)')
    parser.add_argument('--nice', type=int, default=19,
                        help='Приоритет CPU nice (по умолчанию: 19 = мин)')
    args = parser.parse_args()

    # --- Низкий приоритет ---
    # НЕ ставим nice на главный процесс — RKNN зависает при низком приоритете.
    # nice применяется только к подпроцессам stress (CPU/RAM) через 'nice -n 19'.
    # Для NPU потоков nice не нужен — RKNN должен работать на нормальном приоритете.

    # Модель
    if args.model:
        model_path = args.model
    else:
        # Ищем стресс-модель: stress_model.rknn → model.rknn → генерация
        candidates = [
            PROJECT_DIR / 'model' / 'stress_model.rknn',
            PROJECT_DIR / 'model' / 'model.rknn',
            PROJECT_DIR / 'model' / 'yolo26x-seg.rknn',
        ]
        model_path = None
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break
        if model_path is None:
            # Автогенерация стресс-модели
            print('[NPU] RKNN модель не найдена, генерирую synth_stress...')
            onnx_path = PROJECT_DIR / 'model' / 'synth_stress.onnx'
            rknn_path = PROJECT_DIR / 'model' / 'stress_model.rknn'
            if not onnx_path.exists():
                subprocess.run(
                    ['python3', str(PROJECT_DIR / 'convert' / 'synth_stress.py'),
                     '-o', str(onnx_path)],
                    check=True,
                )
            if not rknn_path.exists():
                subprocess.run(
                    ['docker', 'compose', '-f',
                     str(PROJECT_DIR / 'convert' / 'docker-compose.yml'),
                     'run', '--rm', '--entrypoint', 'python', 'convert', '-c',
                     f'from rknn.api import RKNN; rknn=RKNN(); '
                     f'rknn.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform="rk3588"); '
                     f'rknn.load_onnx(model="{onnx_path}"); '
                     f'rknn.build(do_quantization=False); '
                     f'rknn.export_rknn("{rknn_path}"); rknn.release(); print("OK")'],
                    check=True,
                )
            model_path = str(rknn_path)
            print(f'[OK] Стресс-модель: {model_path}')
    if not Path(model_path).exists():
        print(f'[ОШИБКА] модель не найдена: {model_path}')
        sys.exit(1)

    # NPU ядра
    core_ids = [int(x) for x in args.npu_cores.split(',')]

    # thermal
    thermal_zones = init_thermal()
    print(f'Термозоны: {list(thermal_zones.keys())}')

    # Начальные температуры
    temps_start = read_temps(thermal_zones)
    print(f'Температуры ДО: {", ".join(f"{k}={v:.1f}°C" for k, v in sorted(temps_start.items()))}')

    # NPU FPS счётчик [total_frames, stop_flag]
    npu_fps_counter = [0, 0]

    # Запуск компонентов
    cpu_stress = None
    ram_stress = None
    npu_stress = None
    monitor = None

    try:
        # CPU
        if not args.npu_only:
            cpu_stress = CPUStress(nice=args.nice)
            cpu_stress.start()

        # RAM
        if not args.no_ram and not args.npu_only:
            n_ram_workers = max(1, os.cpu_count() // 2)
            ram_stress = RAMStress(mb_per_worker=args.ram_mb, n_workers=n_ram_workers, nice=args.nice)
            ram_stress.start()

        # NPU
        if not args.cpu_only:
            npu_stress = NPUStress(
                model_path, core_ids, npu_fps_counter,
                streams_per_core=args.npu_streams,
            )
            npu_stress.start()
            time.sleep(2)

        # Мониторинг
        monitor = Monitor(
            thermal_zones, npu_fps_counter,
            interval=args.interval,
            log_file=args.log,
        )
        monitor.start()

        total_npu = len(core_ids) * args.npu_streams
        print(f'\n{"="*60}')
        print(f'СТРЕСС-ТЕСТ: {args.duration}с (nice={args.nice})')
        if not args.npu_only:
            print(f'  CPU: {os.cpu_count()} ядер (stress)')
        if not args.no_ram and not args.npu_only:
            print(f'  RAM: {n_ram_workers} × {args.ram_mb} МБ')
        if not args.cpu_only:
            print(f'  NPU: {len(core_ids)} ядра × {args.npu_streams} потоков = {total_npu} RKNN рантаймов')
            print(f'       модель: {Path(model_path).name}')
        print(f'{"="*60}\n')

        time.sleep(args.duration)

    except KeyboardInterrupt:
        print('\n\nПрервано пользователем')
    finally:
        if monitor:
            monitor.stop()
        if npu_stress:
            npu_stress.stop()
        stop_percept_if_started()
        if cpu_stress:
            cpu_stress.stop()
        if ram_stress:
            ram_stress.stop()

        temps_end = read_temps(thermal_zones)
        npu_load = read_npu_load()

        print(f'\n{"="*60}')
        print('РЕЗУЛЬТАТЫ')
        print(f'{"="*60}')
        print(f'\nТемпературы ДО:')
        for k, v in sorted(temps_start.items()):
            print(f'  {k}: {v:.1f}°C')
        print(f'\nТемпературы ПОСЛЕ:')
        for k, v in sorted(temps_end.items()):
            delta = v - temps_start.get(k, v)
            print(f'  {k}: {v:.1f}°C  (Δ {delta:+.1f}°C)')
        if npu_load:
            print(f'\nNPU load: {npu_load}')
        print(f'\nВсего NPU кадров: {npu_fps_counter[0]}')
        if monitor and monitor._history:
            fps_values = [e['npu_fps'] for e in monitor._history if e['npu_fps'] > 0]
            if fps_values:
                print(f'NPU FPS: avg={sum(fps_values)/len(fps_values):.1f} '
                      f'min={min(fps_values):.1f} max={max(fps_values):.1f}')
        if args.log:
            print(f'\nЛог: {args.log}')


if __name__ == '__main__':
    main()
