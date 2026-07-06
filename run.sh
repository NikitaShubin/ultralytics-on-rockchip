#!/bin/bash
set -e
cd /workspace

echo "=== Step 1/3: Export ONNX (no DFL) ==="
python3 -m pip install -e /workspace/ultralytics_yolov8 --no-deps -q
python3 -c "
from ultralytics import YOLO
model = YOLO('/workspace/yolov8m.pt')
model.export(format='rknn', imgsz=640, opset=12)
import os, shutil
src = '/workspace/yolov8m.onnx'
dst = '/workspace/webcam/model/yolov8m-nodfl.onnx'
shutil.move(src, dst)
print(f'ONNX: {dst} ({os.path.getsize(dst)/1024/1024:.1f} MB)')
"

echo ""
echo "=== Step 2/3: Convert ONNX -> RKNN ==="
python3 -c "
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588', mean_values=[[114,114,114]], std_values=[[255,255,255]],
            quantized_dtype='asymmetric_quantized-16', quantized_algorithm='normal', output_optimize=1)
rknn.load_onnx('/workspace/webcam/model/yolov8m-nodfl.onnx')
rknn.build(do_quantization=False, dataset='')
rknn.export_rknn('/workspace/webcam/model/yolov8m-nodfl.rknn')
rknn.release()
import os
print(f'RKNN: /workspace/webcam/model/yolov8m-nodfl.rknn ({os.path.getsize(\"/workspace/webcam/model/yolov8m-nodfl.rknn\")/1024/1024:.1f} MB)')
"

echo ""
echo "=== Step 3/3: Verify ==="
python3 -c "
from rknnlite.api import RKNNLite
import numpy as np
rknn = RKNNLite()
rknn.load_rknn('/workspace/webcam/model/yolov8m-nodfl.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
out = rknn.inference(inputs=[np.zeros((1,3,640,640), dtype=np.uint8)])
rknn.release()
print(f'OK — {len(out)} outputs')
for i,o in enumerate(out): print(f'  [{i}] shape={list(o.shape)} dtype={o.dtype}')
"

echo ""
echo "=== DONE ==="
