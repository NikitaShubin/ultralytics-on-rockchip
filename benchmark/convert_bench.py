#!/usr/bin/env python3
"""Конвертация одной модели для бенчмарка.

Запускается внутри контейнера convert.
Использование: python3 convert_bench.py <model_name> <task> <output_dir> [test_data_dir]

Конвертирует через кеширующий convert.run_convert() (не export_and_compile_all):
каждый quantize-вариант (onnx/fp16/int8/int8-calib) проверяет target в кеше
и дособирает только недостающее из ближайшего звена (_find_nearest).

Создаёт в output_dir:
  {model_name}.onnx           — ONNX (для CPU FPS)
  {model_name}_fp16.rknn       — FP16 RKNN (для NPU FPS)
  {model_name}_int8.rknn      — INT8 RKNN (для NPU FPS)
  {model_name}_int8-calib.rknn — INT8 RKNN с калибровкой
  {model_name}_postprocess.py — Постпроцесс
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, '/app/convert')
import convert as converter  # noqa: E402

ULTRALYTICS_CACHE = Path.home() / '.cache' / 'ultralytics' / 'models'


def cleanup_model(model_name: str) -> None:
    """Очистить промежуточные файлы модели."""
    for base in [Path('.'), converter.WEIGHTS_DIR, converter.MODEL_DIR, Path('/app/convert')]:
        if base.exists():
            for f in base.glob(f'{model_name}*'):
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink(missing_ok=True)
            for d in base.glob(f'{model_name}_rknn_model'):
                shutil.rmtree(d, ignore_errors=True)
            for f in base.glob(f'{model_name}_rkopt_o19*'):
                f.unlink(missing_ok=True)

    if ULTRALYTICS_CACHE.exists():
        for f in ULTRALYTICS_CACHE.glob(f'{model_name}*'):
            f.unlink(missing_ok=True)

    for tmp_dir in [Path('/tmp'), Path.home() / '.cache']:
        for f in tmp_dir.rglob(f'{model_name}*'):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> dict:
    """Основная функция: конвертация модели."""
    model_name = sys.argv[1]
    task = sys.argv[2]
    output_dir = Path(sys.argv[3])

    sys.stderr.write(json.dumps({'phase': 'start', 'model': model_name, 'task': task}) + '\n')

    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = converter.resolve_pt(model_name)
    sys.stderr.write(json.dumps({'phase': 'resolve_pt', 'pt_path': str(pt_path)}) + '\n')

    info = converter._detect_model_info(pt_path)

    t_start = time.time()

    def phase(name, **extra):
        extra['phase'] = name
        extra['elapsed'] = round(time.time() - t_start, 1)
        sys.stderr.write(json.dumps(extra) + '\n')

    phase('model_info', task=info.get('task', task), classes=info['num_classes'], strides=info['strides'])

    # Конвертация через кеширующий run_convert (а не export_and_compile_all):
    # каждый вариант проверяет target в кеше и дособирает только недостающее
    # из ближайшего звена (_find_nearest). Порядок важен: onnx → fp16 → int8 → int8-calib,
    # чтобы кеш накапливался (fp16/int8 собираются из уже готового .onnx).
    variants = ['onnx', 'fp16', 'int8', 'int8-calib']
    targets = {}
    for q in variants:
        phase(f'convert_{q}')
        try:
            targets[q] = converter.run_convert(model_name, q, info)
            sys.stderr.write(json.dumps({'phase': f'convert_{q}_done', 'path': str(targets[q])}) + '\n')
        except Exception as e:
            sys.stderr.write(json.dumps({'phase': f'convert_{q}_fail', 'error': str(e)}) + '\n')
    phase('convert_all_done',
          onnx=str(targets.get('onnx', '') or ''),
          rknn_none=str(targets.get('fp16', '') or ''),
          rknn_int8=str(targets.get('int8', '') or ''),
          rknn_int8cal=str(targets.get('int8-calib', '') or ''))

    postprocess_path = converter._generate_postprocess(info, str(output_dir))
    sys.stderr.write(json.dumps({'phase': 'postprocess', 'path': postprocess_path or ''}) + '\n')

    # run_convert сам кладёт файлы в кеш под правильными именами
    # (cache/<model>/<model>.onnx, _fp16.rknn, _int8.rknn, _int8-calib.rknn),
    # поэтому копирование не требуется — только проверка наличия.
    copied = []
    for q, fname in [('onnx', f'{model_name}.onnx'),
                     ('fp16', f'{model_name}_fp16.rknn'),
                     ('int8', f'{model_name}_int8.rknn'),
                     ('int8-calib', f'{model_name}_int8-calib.rknn')]:
        if (output_dir / fname).exists():
            copied.append(q)

    if postprocess_path and os.path.exists(postprocess_path):
        dst = output_dir / f'{model_name}_postprocess.py'
        if Path(postprocess_path).resolve() != dst.resolve():
            shutil.copy2(postprocess_path, dst)
        copied.append('postprocess')

    status = 'OK' if len(copied) >= 3 else 'FAIL'

    if os.environ.get('KEEP_MODELS', '1') == '0':
        cleanup_model(model_name)
        sys.stderr.write(f'[ОЧИСТКА] промежуточные файлы {model_name} удалены\n')
    else:
        sys.stderr.write(f'[СОХРАНЕНИЕ] модели сохранены (KEEP_MODELS=1)\n')

    result = {
        'model': model_name,
        'task': task,
        'input_size': info.get('input_size', 640),
        'status': status,
        'copied': copied,
    }
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        result = {
            'model': sys.argv[1] if len(sys.argv) > 1 else '?',
            'task': sys.argv[2] if len(sys.argv) > 2 else '?',
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc(),
        }
        model_name = result['model']
        if model_name != '?':
            cleanup_model(model_name)
        print(json.dumps(result))
        sys.exit(1)
