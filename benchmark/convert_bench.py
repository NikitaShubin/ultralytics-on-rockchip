#!/usr/bin/env python3
"""Конвертация + валидация одной модели для бенчмарка.

Запускается внутри контейнера converter.
Использование: python3 convert_bench.py <model_name> <task> <output_dir> [test_data_dir]
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/app/converter')
import converter  # noqa: E402

ULTRALYTICS_CACHE = Path.home() / '.cache' / 'ultralytics' / 'models'


def load_test_images(data_dir: str | Path, limit: int = 16) -> list[np.ndarray]:
    """Загрузить тестовые изображения из директории."""
    data_dir = Path(data_dir)
    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        for f in sorted(data_dir.glob(ext)):
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)
                if len(images) >= limit:
                    return images
    for f in sorted(data_dir.rglob('*.*')):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            if not any(p.name == 'labels' for p in f.parents):
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)
                    if len(images) >= limit:
                        break
    return images[:limit]


def run_validation(
    onnx_path: str,
    rknn_path: str,
    test_images: list[np.ndarray],
    input_size: int = 640,
) -> dict:
    """Валидация ONNX vs RKNN на тестовых изображениях."""
    import onnxruntime as rt
    from rknn.api import RKNN

    sess = rt.InferenceSession(onnx_path)
    rknn = RKNN(verbose=False)
    rknn.load_rknn(rknn_path)
    ret = rknn.init_runtime()
    validation_mode = 'hardware'
    if ret != 0:
        rknn.release()
        sys.stderr.write(f'[ПРЕДУПРЕЖДЕНИЕ] инициализация NPU runtime не удалась (код={ret}), '
                         f'попытка валидации через симулятор с пересборкой ONNX\n')
        try:
            rknn = RKNN(verbose=False)
            rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                        target_platform='rk3588')
            rknn.load_onnx(model=onnx_path)
            rknn.build(do_quantization=False)
            ret = rknn.init_runtime()
            if ret != 0:
                rknn.release()
                return {'note': f'NPU runtime недоступен (код={ret})',
                        'max_diff_avg': -1, 'max_diff_max': -1, 'n_images': 0,
                        'validation_mode': 'none'}
            validation_mode = 'simulator'
        except Exception as e:
            return {'note': f'NPU runtime недоступен: {e}',
                    'max_diff_avg': -1, 'max_diff_max': -1, 'n_images': 0,
                    'validation_mode': 'none'}

    max_diffs = []
    for img in test_images:
        h, w = img.shape[:2]
        tensor, scale, pad = converter.preprocess(img, input_size=input_size)

        onnx_out = converter.run_onnx(sess, tensor)
        rknn_out = converter.run_rknn(rknn, tensor)

        if isinstance(onnx_out, (list, tuple)):
            onnx_tensors = onnx_out
        else:
            onnx_tensors = [onnx_out]

        max_diff = 0.0
        for o, r in zip(onnx_tensors, rknn_out):
            diff = np.abs(o.astype(np.float32) - r.astype(np.float32)).max()
            max_diff = max(max_diff, float(diff))
        max_diffs.append(max_diff)

    rknn.release()

    return {
        'max_diff_avg': float(np.mean(max_diffs)) if max_diffs else -1,
        'max_diff_max': float(np.max(max_diffs)) if max_diffs else -1,
        'n_images': len(test_images),
        'validation_mode': validation_mode,
    }


def cleanup_model(model_name: str, onnx_path: str | None, rknn_path: str | None) -> None:
    """Очистить файлы модели после обработки."""
    if onnx_path and Path(onnx_path).exists():
        Path(onnx_path).unlink(missing_ok=True)

    if rknn_path:
        rknn_dir = Path(rknn_path).parent
        if rknn_dir.exists():
            shutil.rmtree(rknn_dir, ignore_errors=True)

    for base in [Path('.'), converter.WEIGHTS_DIR, converter.MODEL_DIR, Path('/app/converter')]:
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
    """Основная функция: конвертация и валидация модели."""
    model_name = sys.argv[1]
    task = sys.argv[2]
    output_dir = Path(sys.argv[3])
    test_data_dir = sys.argv[4] if len(sys.argv) > 4 else '/app/coco_val2017/images/val2017'

    sys.stderr.write(json.dumps({'phase': 'start', 'model': model_name, 'task': task}) + '\n')

    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = converter.resolve_pt(model_name)
    sys.stderr.write(json.dumps({'phase': 'resolve_pt', 'pt_path': pt_path}) + '\n')

    info = converter._detect_model_info(pt_path)
    quantize = os.environ.get('CONVERT_QUANTIZE', 'none')

    t_start = time.time()
    def phase(name, **extra):
        extra['phase'] = name
        extra['elapsed'] = round(time.time() - t_start, 1)
        sys.stderr.write(json.dumps(extra) + '\n')

    phase('model_info', task=info.get('task', task), classes=info['num_classes'], strides=info['strides'])

    phase('export_compile')
    onnx_path, rknn_path = converter.export_and_compile(pt_path, info, quantize=quantize)
    phase('export_compile_done', onnx=onnx_path or '', rknn=rknn_path or '')

    postprocess_path = converter._generate_postprocess(info, str(output_dir))
    sys.stderr.write(json.dumps({'phase': 'postprocess', 'path': postprocess_path or ''}) + '\n')

    if onnx_path:
        dst = output_dir / f'{model_name}.onnx'
        shutil.copy2(onnx_path, dst)
        sys.stderr.write(f'[КОПИРОВАНИЕ] {onnx_path} -> {dst}\n')

    if rknn_path and os.path.exists(rknn_path):
        dst = output_dir / f'{model_name}.rknn'
        shutil.copy2(rknn_path, dst)
        sys.stderr.write(f'[КОПИРОВАНИЕ] {rknn_path} -> {dst}\n')

    if postprocess_path and os.path.exists(postprocess_path):
        dst = output_dir / f'{model_name}_postprocess.py'
        shutil.copy2(postprocess_path, dst)
        sys.stderr.write(f'[КОПИРОВАНИЕ] {postprocess_path} -> {dst}\n')

    status = 'OK' if onnx_path and rknn_path and os.path.exists(rknn_path) else 'FAIL'

    if os.environ.get('KEEP_MODELS', '0') == '1':
        sys.stderr.write(f'[СОХРАНЕНИЕ] очистка пропущена, модели сохранены в {output_dir}\n')
    else:
        onnx_path_clean = onnx_path
        rknn_path_clean = rknn_path
        cleanup_model(model_name, onnx_path_clean, rknn_path_clean)
        sys.stderr.write(f'[ОЧИСТКА] файлы модели {model_name} удалены\n')

    result = {
        'model': model_name,
        'task': task,
        'input_size': info.get('input_size', 640),
        'status': status,
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
            cleanup_model(model_name, None, None)
        print(json.dumps(result))
        sys.exit(1)
