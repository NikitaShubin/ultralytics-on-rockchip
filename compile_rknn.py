#!/usr/bin/env python3
"""Compile the already-exported ONNX to RKNN with different normalization settings."""

import os, sys
sys.path = [p for p in sys.path if 'ultralytics' not in p]
sys.path.insert(0, '/workspace/ultralytics_yolov8')

OUTDIR = '/workspace/webcam/model'
ONNX_PATH = os.path.join(OUTDIR, 'yolov8m-fixed.onnx')

from rknn.api import RKNN

for norm_name, mean_vals, std_vals, out_name in [
    ('CORRECT (0/255)',  [0,0,0],   [255,255,255], 'yolov8m-norm0.rknn'),
    ('OLD (114/255)',    [114,114,114], [255,255,255], 'yolov8m-norm114.rknn'),
]:
    print(f"\n=== Compiling: {norm_name} ===", flush=True)
    rknn = RKNN(verbose=False)

    ret = rknn.config(
        target_platform='rk3588',
        mean_values=[mean_vals],
        std_values=[std_vals],
        quantized_dtype='asymmetric_quantized-16',
        quantized_algorithm='normal',
        output_optimize=True,
    )
    assert ret == 0, f'config: {ret}'

    ret = rknn.load_onnx(ONNX_PATH)
    assert ret == 0, f'load_onnx: {ret}'

    ret = rknn.build(do_quantization=False, dataset='')
    assert ret == 0, f'build: {ret}'

    rknn_path = os.path.join(OUTDIR, out_name)
    ret = rknn.export_rknn(rknn_path)
    assert ret == 0, f'export: {ret}'
    rknn.release()
    size = os.path.getsize(rknn_path) / 1024 / 1024
    print(f'Saved: {rknn_path} ({size:.1f} MB)', flush=True)

print('\nDone!', flush=True)
print(f'Models in {OUTDIR}:', flush=True)
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith('.rknn'):
        print(f'  {f}', flush=True)
