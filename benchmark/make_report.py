#!/usr/bin/env python3
"""Генерация отчёта с графиками производительности (запускается В convert-контейнере).

Читает benchmark/results/results.json (монтируется writable), пишет в
benchmark/results/report/:
  - 11 PNG:
      * detect/seg/pose/obb/cls: <task>_onnx_fp16.png  (onnx — сплошная, fp16 — пунктирная)
                                 <task>_int8.png       (int8 — сплошная, int8-calib — пунктирная)
      * sem_all.png — все 4 варианта на одном изображении
  - report.md — подробный отчёт на русском со встроенными PNG и таблицами.

Ось X — время обработки одного кадра (1000/fps), ось Y — основная метрика:
  detect: map50; seg: inmem_seg_map50; pose: inmem_pose_map50;
  obb: map50; cls: top1; sem: inmem_miou.

Цвет линий фиксирован ПО ПОКОЛЕНИЮ на всех графиках (глобальная палитра),
модели одного семейства соединены линией; подписи — размеры (n/s/m/l/x и спец.).
В легенде сначала идут образцы стилей линий (сплошная/пунктирная), затем семьи.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path('/app/benchmark')
RESULTS_FILE = ROOT / 'results' / 'results.json'
REPORT_DIR = ROOT / 'results' / 'report'


def _parse_args() -> tuple[Path, Path]:
    """--out — куда писать отчёт; --results — откуда читать журнал результатов."""
    import argparse
    parser = argparse.ArgumentParser(description='Генерация отчёта производительности YOLO на Orange Pi 5 Pro')
    parser.add_argument('--out', type=Path, default=REPORT_DIR,
                        help='каталог для отчёта (report.md + PNG), по умолчанию benchmark/results/report')
    parser.add_argument('--results', type=Path, default=RESULTS_FILE,
                        help='results.json (журнал), по умолчанию benchmark/results/results.json')
    args, _ = parser.parse_known_args()
    return args.out, args.results

sys.path.insert(0, str(ROOT))
from models import FULL as MODELS  # noqa: E402

TASKS = ['detect', 'seg', 'pose', 'obb', 'cls', 'sem']

# (ключ, метка, fps-источник)
VARIANTS = [
    ('onnx', 'onnx (CPU)', 'cpu'),
    ('fp16', 'fp16 (NPU)', 'npu_none'),
    ('int8', 'int8 (NPU)', 'npu_int8'),
    ('int8-calib', 'int8-calib (NPU)', 'npu_int8-calib'),
]
VARIANT_LABEL = {k: lbl for k, lbl, _f in VARIANTS}

MPL_DPI = 200

# Основная метрика по задаче (решение пользователя 2026-09-20)
METRIC = {
    'detect': 'map50',
    'seg': 'inmem_seg_map50',
    'pose': 'inmem_pose_map50',
    'obb': 'map50',
    'cls': 'top1',
    'sem': 'inmem_miou',
}

TASK_LABEL = {
    'detect': 'Детекция',
    'seg': 'Сегментация',
    'pose': 'Ключевые точки',
    'obb': 'Повёрнутые боксы',
    'cls': 'Классификация',
    'sem': 'Семантическая сегментация',
}

# Глобальная палитра поколений: ОДИН цвет для каждого семейства на ВСЕХ графиках.
# Цвета подобраны контрастными (хорошо различимы), совпадают на всех графиках.
FAMILY_ORDER = ['yolov3', 'yolov5', 'yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolo12', 'yolo26', 'прочее']
FAMILY_COLORS = {
    'yolov3': '#e6194b',
    'yolov5': '#f58231',
    'yolov8': '#4363d8',
    'yolov9': '#3cb44b',
    'yolov10': '#911eb4',
    'yolo11': '#42d4f4',
    'yolo12': '#f032e6',
    'yolo26': '#9a6324',
    'прочее': '#808080',
}

# стиль линии/маркер по варианту (для объединённых графиков и sem_all).
# Пунктирные узоры заданы МЕЛКИМ ШАГОМ ((off, on) в pt), чтобы образец линии
# в легенде (короткий отрезок) показывал несколько штрихов и был различим.
VARIANT_LS = {
    'onnx': '-',
    'fp16': (0, (2.8, 1.6)),
    'int8': '-',
    'int8-calib': (0, (2.1, 1.6)),
}
VARIANT_MK = {'onnx': 'o', 'fp16': 's', 'int8': 'o', 'int8-calib': 'D'}

PAIRS = [
    ('onnx_fp16', ('onnx', 'fp16'), 'onnx (CPU) — сплошная, fp16 (NPU) — пунктирная'),
    ('int8', ('int8', 'int8-calib'), 'int8 — сплошная, int8-calib — пунктирная'),
]


def family_of(name: str) -> str:
    for fam in ('yolov10', 'yolov9', 'yolov8', 'yolov5', 'yolov3', 'yolo11', 'yolo12', 'yolo26'):
        if name.startswith(fam):
            return fam
    return 'прочее'


def size_of(name: str, model_map: dict[str, tuple[str, str]]) -> str:
    if name in model_map:
        return model_map[name][1].lstrip('-')
    return name


def collect(models: list[dict], variant: str, fps_key: str, task: str,
            model_map: dict[str, tuple[str, str]]) -> dict[str, list[tuple[str, str, float, float]]]:
    """Собрать точки (name, size, time_ms, metric) по семействам для одного варианта."""
    fam_points: dict[str, list[tuple[str, str, float, float]]] = {}
    metric_key = METRIC[task]
    for r in sorted(models, key=lambda x: x.get('model', '')):
        name = r.get('model')
        fps_data = (r.get('fps') or {}).get(fps_key) or {}
        fps = fps_data.get('fps')
        eval_data = (r.get('eval') or {}).get(variant) or {}
        metric = eval_data.get(metric_key)
        if fps is None or fps <= 0 or not isinstance(metric, (int, float)):
            print(f'[ПРОПУСК] {task}/{variant}: {name}')
            continue
        fam = family_of(name)
        size = size_of(name, model_map)
        fam_points.setdefault(fam, []).append((name, size, 1000.0 / fps, float(metric)))
    return fam_points


def plot_family_lines(ax, fam_points: dict, variant: str, variant_label: str | None = None,
                      annotate: bool = True, markersize: int = 5,
                      fontsize: int = 8, alpha: float = 0.9) -> list[Line2D]:
    """Нанести точки/линии семейств (цвет — по поколению, стиль/маркер — по варианту).

    variant_label: если задан, каждая линия помечается как '<fam> (<variant_label>)'
    (нужно для полной легенды объединённых графиков), иначе просто '<fam>'.
    """
    handles = []
    for fam in FAMILY_ORDER:
        if fam not in fam_points:
            continue
        color = FAMILY_COLORS[fam]
        spts = sorted(fam_points[fam], key=lambda p: p[2])
        xs = [p[2] for p in spts]
        ys = [p[3] for p in spts]
        label = f'{fam} ({variant_label})' if variant_label else fam
        (ln,) = ax.plot(xs, ys, marker=VARIANT_MK[variant], linestyle=VARIANT_LS[variant],
                        linewidth=1.5, markersize=markersize, color=color, label=label)
        handles.append(ln)
        if annotate:
            for px, py, (_n, size, _t, _m) in zip(xs, ys, spts):
                ax.annotate(size, (px, py), textcoords='offset points',
                            xytext=(4, 3), fontsize=fontsize, color=color, alpha=alpha)
    return handles


def setup_axes(ax, title: str, metric_key: str, all_times: list[float]) -> None:
    # Лог-шкала при разбросе >= 10x (порог выбран по int8-int8-calib:
    # высокие/низкие времена в паре сильно различаются, линейная шкала сплющивает детали).
    if min(all_times) > 0 and max(all_times) / min(all_times) >= 10:
        ax.set_xscale('log')
        xlabel = 'Время обработки кадра, мс (лог. шкала)'
    else:
        xlabel = 'Время обработки кадра, мс'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)


def save_fig(fig, fname: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORT_DIR / fname, dpi=MPL_DPI)
    plt.close(fig)


def md_pair_table(md: list[str], pts_a: dict, pts_b: dict,
                  variant_a: str, variant_b: str, metric_key: str) -> None:
    """Сводная таблица двух вариантов: модель, время/fps/метрика для каждого."""
    all_names = sorted({p[0] for fp in pts_a.values() for p in fp}
                       | {p[0] for fp in pts_b.values() for p in fp})
    md.append('| Модель | Семейство | Размер '
              f'| {variant_a}: мс | {variant_a}: fps | {variant_a}: метрика '
              f'| {variant_b}: мс | {variant_b}: fps | {variant_b}: метрика |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for name in all_names:
        pa = next((p for fp in pts_a.values() for p in fp if p[0] == name), None)
        pb = next((p for fp in pts_b.values() for p in fp if p[0] == name), None)
        size = (pa or pb)[1]
        fam = family_of(name)
        ca = f'{pa[2]:.1f} | {1000.0 / pa[2]:.1f} | {pa[3]:.4f}' if pa else '— | — | —'
        cb = f'{pb[2]:.1f} | {1000.0 / pb[2]:.1f} | {pb[3]:.4f}' if pb else '— | — | —'
        md.append(f'| {name} | {fam} | {size} | {ca} | {cb} |')
    md.append('')


def main() -> None:
    global REPORT_DIR, RESULTS_FILE
    REPORT_DIR, RESULTS_FILE = _parse_args()

    if not RESULTS_FILE.exists():
        print(json.dumps({'error': f'нет {RESULTS_FILE}', 'graphs': 0}))
        sys.exit(1)

    results = json.loads(RESULTS_FILE.read_text())
    model_map = {m.name: (m.family, m.size) for m in MODELS}
    by_task: dict[str, list[dict]] = {t: [] for t in TASKS}
    for r in results:
        task = r.get('task')
        if task in by_task:
            by_task[task].append(r)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n_graphs = 0
    md = []

    md.append('# Отчёт о производительности моделей YOLO на Orange Pi 5 Pro')
    md.append('')
    md.append(f'- Дата генерации: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    md.append(f'- Моделей в results.json: {len(results)}')
    md.append('- Железо: Orange Pi 5 Pro, RK3588 (8 ядер, NPU)')
    md.append('- Ось X — время обработки одного кадра (мс = 1000/fps), ось Y — основная метрика')
    md.append('- Линиями соединены модели одного семейства; цвет поколения одинаков на ВСЕХ графиках')
    md.append('- В легенде каждая линия — отдельная запись вида «семейство (вариант)»')
    md.append('- цвет поколения одинаков на всех графиках.')
    md.append('')

    for task in TASKS:
        models = by_task[task]
        if not models:
            continue
        md.append(f'## {TASK_LABEL[task]} ({task})')
        md.append('')
        metric_key = METRIC[task]

        if task == 'sem':
            # ---- sem: все 4 варианта на одном изображении (одно поколение YOLO26) ----
            fig, ax = plt.subplots(figsize=(10, 6.5))
            all_times: list[float] = []
            handles = []
            for variant, label, fps_key in VARIANTS:
                pts = collect(models, variant, fps_key, task, model_map)
                color = FAMILY_COLORS['yolo26']
                for fam, rows in pts.items():
                    spts = sorted(rows, key=lambda p: p[2])
                    xs = [p[2] for p in spts]
                    ys = [p[3] for p in spts]
                    all_times.extend(xs)
                    (h,) = ax.plot(xs, ys, marker=VARIANT_MK[variant], linestyle=VARIANT_LS[variant],
                                   linewidth=1.5, markersize=5, color=color, label=label)
                    handles.append(h)
                    for px, py, (_n, size, _t, _m) in zip(xs, ys, spts):
                        ax.annotate(size, (px, py), textcoords='offset points',
                                    xytext=(4, 3), fontsize=8, color=color, alpha=0.9)
            ax.legend(handles=handles, fontsize=9, handlelength=3.2, handleheight=1.5)
            setup_axes(ax, f'{TASK_LABEL[task]} — все варианты', metric_key, all_times)
            save_fig(fig, 'sem_all.png')
            n_graphs += 1

            md.append('![sem все варианты](sem_all.png)')
            md.append('')
            for variant, label, fps_key in VARIANTS:
                pts = collect(models, variant, fps_key, task, model_map)
                md.append(f'**{label}:**')
                md.append('')
                md.append('| Модель | Размер | Время кадра, мс | fps | метрика |')
                md.append('|---|---|---|---|---|')
                for fam in FAMILY_ORDER:
                    if fam not in pts:
                        continue
                    for name, size, t, m in sorted(pts[fam], key=lambda p: p[2]):
                        md.append(f'| {name} | {size} | {t:.1f} | {1000.0 / t:.1f} | {m:.4f} |')
                md.append('')
            continue

        # ---- обычные задачи: onnx+fp16 и int8+int8-calib ----
        for suffix, variants, caption in PAIRS:
            v_a, v_b = variants
            src_a = next(f for k, _l, f in VARIANTS if k == v_a)
            src_b = next(f for k, _l, f in VARIANTS if k == v_b)
            pts_a = collect(models, v_a, src_a, task, model_map)
            pts_b = collect(models, v_b, src_b, task, model_map)
            if not pts_a and not pts_b:
                continue
            fig, ax = plt.subplots(figsize=(10, 6.5))
            all_times = [p[2] for fp in pts_a.values() for p in fp] \
                        + [p[2] for fp in pts_b.values() for p in fp]
            handles = plot_family_lines(ax, pts_a, v_a, variant_label=v_a,
                                        markersize=5, fontsize=8, alpha=0.9)
            handles += plot_family_lines(ax, pts_b, v_b, variant_label=v_b,
                                         markersize=4, fontsize=7, alpha=0.7)
            ax.legend(handles=handles, fontsize=8, loc='best', handlelength=3.2, handleheight=1.5)
            setup_axes(ax, f'{TASK_LABEL[task]} — {caption}', metric_key, all_times)
            save_fig(fig, f'{task}_{suffix}.png')
            n_graphs += 1

            md.append(f'![{task} {suffix}]({task}_{suffix}.png)')
            md.append('')
            md.append(f'### {caption}')
            md.append('')
            md_pair_table(md, pts_a, pts_b, VARIANT_LABEL[v_a], VARIANT_LABEL[v_b], metric_key)

    (REPORT_DIR / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    # DPI-метаданные PNG (LibreOffice Writer считает размер из pixels/dpi):
    # 2000 px при dpi=300 -> ~16.9 см, в ширину полей A4.
    from PIL import Image
    for png in REPORT_DIR.glob('*.png'):
        with Image.open(png) as im:
            im.convert('RGB').save(png, dpi=(300, 300))

    print(json.dumps({
        'report': str(REPORT_DIR / 'report.md'),
        'graphs': n_graphs,
        'models': len(results),
    }))


if __name__ == '__main__':
    main()