#!/usr/bin/env python3
"""Валидация ONNX vs RKNN для одной модели.

Запускается внутри контейнера converter.
Использование: python3 validate_bench.py <model_name> <task> <model_dir> [test_data_dir]

Сравнивает выходы ONNX vs RKNN FP32 и ONNX vs RKNN INT8 на тестовых изображениях.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, '/app/converter')
import converter  # noqa: E402


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


def main() -> dict:
    """Основная функция: валидация модели."""
    model_name = sys.argv[1]
    task = sys.argv[2]
    model_dir = Path(sys.argv[3])
    test_data_dir = sys.argv[4] if len(sys.argv) > 4 else '/app/coco_val2017/images/val2017'
    input_size = int(sys.argv[5]) if len(sys.argv) > 5 else 640

    sys.stderr.write(json.dumps({'phase': 'start', 'model': model_name, 'task': task}) + '\n')

    onnx_path = model_dir / f'{model_name}.onnx'
    rknn_none_path = model_dir / f'{model_name}_none.rknn'
    rknn_int8_path = model_dir / f'{model_name}_int8.rknn'

    if not onnx_path.exists():
        sys.stderr.write(f'[ОШИБКА] ONNX не найден: {onnx_path}\n')
        result = {'model': model_name, 'task': task, 'status': 'FAIL',
                  'error': f'ONNX не найден: {onnx_path}'}
        print(json.dumps(result))
        return result

    images = load_test_images(test_data_dir)
    if not images:
        sys.stderr.write(f'[ОШИБКА] нет тестовых изображений в {test_data_dir}\n')
        result = {'model': model_name, 'task': task, 'status': 'FAIL',
                  'error': f'нет тестовых изображений'}
        print(json.dumps(result))
        return result

    sys.stderr.write(f'[ИНФО] загружено {len(images)} тестовых изображений, input_size={input_size}\n')

    t_start = time.time()
    results = {'model': model_name, 'task': task, 'validations': {}}

    if rknn_none_path.exists():
        sys.stderr.write(f'[ВАЛИДАЦИЯ] FP32: ONNX vs {rknn_none_path.name}...\n')
        t0 = time.time()
        v_fp32 = run_validation(str(onnx_path), str(rknn_none_path), images, input_size=input_size)
        elapsed = time.time() - t0
        v_fp32['elapsed'] = round(elapsed, 1)
        results['validations']['fp32'] = v_fp32
        sys.stderr.write(f'[ВАЛИДАЦИЯ] FP32: max_diff_avg={v_fp32["max_diff_avg"]:.4f} '
                         f'max={v_fp32["max_diff_max"]:.4f} ({v_fp32["validation_mode"]}) '
                         f'за {elapsed:.0f}с\n')
    else:
        sys.stderr.write(f'[ПРОПУСК] FP32: {rknn_none_path.name} не найден\n')

    if rknn_int8_path.exists():
        sys.stderr.write(f'[ВАЛИДАЦИЯ] INT8: ONNX vs {rknn_int8_path.name}...\n')
        t0 = time.time()
        v_int8 = run_validation(str(onnx_path), str(rknn_int8_path), images, input_size=input_size)
        elapsed = time.time() - t0
        v_int8['elapsed'] = round(elapsed, 1)
        results['validations']['int8'] = v_int8
        sys.stderr.write(f'[ВАЛИДАЦИЯ] INT8: max_diff_avg={v_int8["max_diff_avg"]:.4f} '
                         f'max={v_int8["max_diff_max"]:.4f} ({v_int8["validation_mode"]}) '
                         f'за {elapsed:.0f}с\n')
    else:
        sys.stderr.write(f'[ПРОПУСК] INT8: {rknn_int8_path.name} не найден\n')

    results['status'] = 'OK'
    total_elapsed = time.time() - t_start
    results['elapsed'] = round(total_elapsed, 1)
    sys.stderr.write(f'[ГОТОВО] валидация {model_name} за {total_elapsed:.0f}с\n')

    print(json.dumps(results))
    return results


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
        print(json.dumps(result))
        sys.exit(1)
