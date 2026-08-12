"""Общая телеметрия: частоты NPU/CPU, температуры, детекция троттлинга.

Единый источник функций и критериев троттлинга для fps_bench.py и eval.py.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

NPU_LOAD_PATH = '/sys/kernel/debug/rknpu/load'
NPU_TEMP_PATH = '/sys/class/thermal/thermal_zone6/temp'
NPU_FREQ_PATH = '/sys/kernel/debug/rknpu/freq'
NPU_MAX_FREQ_HZ = 1_000_000_000      # RK3588: макс. частота NPU
THROTTLE_TEMP_C = 80.0               # порог температуры (любая зона) → троттлинг
THROTTLE_FREQ_RATIO = 0.9            # avg-частота < 90% от max → троттлинг
THROTTLE_LOAD_PCT = 50.0             # загрузка, с которой частота значима


def read_npu_temp() -> float | None:
    """Температура NPU из thermal-зоны (в °C) или None."""
    try:
        with open(NPU_TEMP_PATH) as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def read_npu_freq() -> int | None:
    """Текущая частота NPU в Hz из sysfs (или None)."""
    try:
        with open(NPU_FREQ_PATH) as f:
            return int(f.read().strip())
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def read_npu_load() -> float | None:
    """Текущая загрузка NPU в % (среднее по ядрам) или None."""
    import re
    try:
        with open(NPU_LOAD_PATH) as f:
            content = f.read()
        matches = re.findall(r'Core(\d+):\s*(\d+)%', content)
        if matches:
            return sum(int(m) for _, m in matches) / len(matches)
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return None


def cpu_core_max_freq(core_id: int) -> int | None:
    """Максимальная частота CPU-ядра в Hz (из cpuinfo_max_freq)."""
    try:
        p = Path(f'/sys/devices/system/cpu/cpu{core_id}/cpufreq/cpuinfo_max_freq')
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def read_cpu_freqs() -> dict[int, int]:
    """Текущая частота каждого CPU-ядра в Hz."""
    freqs = {}
    for d in sorted(Path('/sys/devices/system/cpu').glob('cpu[0-9]*')):
        try:
            cid = int(d.name[3:])
            cur = (d / 'cpufreq' / 'scaling_cur_freq').read_text().strip()
            freqs[cid] = int(cur)
        except (OSError, ValueError):
            pass
    return freqs


def read_cpu_stat_per_core() -> dict[int, tuple[int, int]]:
    """Время CPU per-core из /proc/stat: {core: (idle, total)}."""
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


def read_cpu_loads(interval: float = 0.1) -> dict[int, float]:
    """Загрузка CPU per-core в % (одна пара отсчётов через interval)."""
    prev = read_cpu_stat_per_core()
    time.sleep(interval)
    cur = read_cpu_stat_per_core()
    loads = {}
    for core_id, (cur_idle, cur_total) in cur.items():
        if core_id in prev:
            p_idle, p_total = prev[core_id]
            d_idle = cur_idle - p_idle
            d_total = cur_total - p_total
            if d_total > 0:
                loads[core_id] = (1.0 - d_idle / d_total) * 100.0
    return loads


def read_thermal_max() -> float | None:
    """Максимальная температура по всем thermal-зонам в °C (или None)."""
    temps = []
    for z in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        try:
            temps.append(float((z / 'temp').read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def throttle_reasons(npu_freq_avg: float | None, npu_load_avg: float | None,
                     cpu_freq_ratio: float | None, thermal_max: float | None) -> list[str]:
    """Причины троттлинга по агрегатам за окно. Пустой список = троттлинга нет.

    - NPU: частота <90% max ТОЛЬКО при загрузке NPU >50% (недогрузка — не троттлинг);
    - CPU: avg-частота <90% max по НАГРУЖЕННЫМ ядрам (load>50%) — в idle ядра легитимно снижают частоту;
    - температура любой зоны >=80°C — безусловный признак.
    """
    reasons = []
    if (npu_freq_avg is not None and npu_load_avg is not None and npu_load_avg > THROTTLE_LOAD_PCT
            and npu_freq_avg < NPU_MAX_FREQ_HZ / 1e6 * THROTTLE_FREQ_RATIO):
        reasons.append(f'npu_freq={npu_freq_avg:.0f}MHz@load={npu_load_avg:.0f}%')
    if cpu_freq_ratio is not None and cpu_freq_ratio < THROTTLE_FREQ_RATIO:
        reasons.append(f'cpu_freq={cpu_freq_ratio:.2f}')
    if thermal_max is not None and thermal_max >= THROTTLE_TEMP_C:
        reasons.append(f'temp={thermal_max:.0f}C')
    return reasons


def wait_no_throttle(timeout_sec: int, poll: float = 5.0) -> bool:
    """Ждать выхода из троттлинга. True — готово к замеру, False — таймаут.

    Критерий ожидания — только температура: в idle частоты легитимно падают.
    """
    deadline = time.monotonic() + max(0, timeout_sec)
    while True:
        t = read_thermal_max()
        if t is None or t < THROTTLE_TEMP_C:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


class TelemetrySampler:
    """Фоновый сэмплер частот/температур на время долгого процесса (eval).

    Собирает частоты NPU/CPU, загрузку NPU/CPU per-core и температуры.
    Не нагружает систему: интервал по умолчанию 2с.
    """

    def __init__(self, collect_cpu: bool = True, interval: float = 2.0):
        self._stop = threading.Event()
        self._interval = interval
        self._collect_cpu = collect_cpu
        self._thread: threading.Thread | None = None
        self.npu_freq: list[int] = []
        self.npu_load: list[float] = []
        self.cpu_freq: dict[int, list[int]] = {}
        self.cpu_load: dict[int, list[float]] = {}
        self.npu_temp: list[float] = []
        self.thermal: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            f = read_npu_freq()
            if f is not None:
                self.npu_freq.append(f)
            l = read_npu_load()
            if l is not None:
                self.npu_load.append(l)
            if self._collect_cpu:
                for cid, f in read_cpu_freqs().items():
                    self.cpu_freq.setdefault(cid, []).append(f)
                for cid, l in read_cpu_loads(interval=0.1).items():
                    self.cpu_load.setdefault(cid, []).append(l)
            t = read_npu_temp()
            if t is not None:
                self.npu_temp.append(t)
            t = read_thermal_max()
            if t is not None:
                self.thermal.append(t)
            self._stop.wait(self._interval)

    def summarize(self) -> dict:
        """Агрегаты за окно: температуры, частоты, ratio, throttled-флаги."""
        npu_temp_avg = round(float(np.mean(self.npu_temp)), 1) if self.npu_temp else None
        npu_temp_min = round(float(min(self.npu_temp)), 1) if self.npu_temp else None
        npu_temp_max = round(float(max(self.npu_temp)), 1) if self.npu_temp else None
        npu_freq_avg = round(float(np.mean(self.npu_freq)) / 1e6, 0) if self.npu_freq else None
        npu_freq_min = round(float(min(self.npu_freq)) / 1e6, 0) if self.npu_freq else None
        npu_load_avg = round(float(np.mean(self.npu_load)), 1) if self.npu_load else None
        thermal_max = round(float(max(self.thermal)), 1) if self.thermal else None

        cpu_freq_ratio = None
        if self.cpu_freq and self.cpu_load:
            ratios = []
            for cid, vals in self.cpu_freq.items():
                mx = cpu_core_max_freq(cid)
                if not mx:
                    continue
                loads = self.cpu_load.get(cid)
                loaded = bool(loads) and float(np.mean(loads)) > THROTTLE_LOAD_PCT
                if not loaded:
                    continue
                ratios.append(float(np.mean(vals)) / mx)
            cpu_freq_ratio = round(min(ratios), 2) if ratios else None

        reasons = throttle_reasons(npu_freq_avg, npu_load_avg, cpu_freq_ratio, thermal_max)
        return {
            'npu_temp_avg': npu_temp_avg,
            'npu_temp_min': npu_temp_min,
            'npu_temp_max': npu_temp_max,
            'npu_freq_avg': npu_freq_avg,
            'npu_freq_min': npu_freq_min,
            'npu_load_avg': npu_load_avg,
            'thermal_max_c': thermal_max,
            'cpu_freq_ratio': cpu_freq_ratio,
            'throttled': bool(reasons),
            'throttle_reason': '; '.join(reasons) or None,
        }
