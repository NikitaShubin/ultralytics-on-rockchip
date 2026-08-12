#!/usr/bin/env python3
"""Профиль segment-ветки eval.py по фазам: infer vs pp vs coco-conv vs seg_tp.

Запуск в convert-контейнере:
  docker compose -f convert/docker-compose.yml run --rm --entrypoint python \
    convert /app/benchmark/prof_seg_pipeline.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cache' / 'yolo26n-seg'))

from pipeline import Pipeline
from percept.io import letterbox
from benchmark.eval import load_coco, detections_to_coco, _seg_tp_per_image, _seg_pre_nms, _seg_nms_finish

NAME = 'yolo26n-seg'
SUFFIX = '_fp16'
MODEL_DIR = ROOT / 'cache' / NAME
DATA = ROOT / 'datasets' / 'coco_val2017'
ANN = DATA / 'annotations' / 'instances_val2017.json'
IMGDIR = DATA / 'images' / 'val2017'

pipeline = Pipeline(
    model_dir=MODEL_DIR, model_name=NAME,
    backend='npu', rknn_suffix=SUFFIX,
)
pp = pipeline._postprocess
post_mod = sys.modules['postprocess_mod']
backend = pipeline._backend

post_mod.CONF_THRESH = 0.001
post_mod.IOU_THRESH = 0.7
post_mod.MIN_WH = 1
post_mod.MAX_DET = 300
post_mod.DEDUP_ENABLED = False
if hasattr(post_mod, 'EVAL_PRECISE'):
    post_mod.EVAL_PRECISE = True

import torch, torchvision
torch.set_num_threads(1)
_orig_nms = post_mod.nms
def _fast_nms(boxes, scores, iou_threshold):
    tb = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
    ts = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
    return torchvision.ops.nms(tb, ts, iou_threshold).numpy().astype(np.int64)
post_mod.nms = _fast_nms

images, ann_by_img, cat_ids = load_coco(str(ANN), 'segment')
imgs = list(images.items())[:30]

def seg_tp(anns, objects, sx, sy, pad_left, pad_top):
    return _seg_tp_per_image(anns, objects, sx, sy, pad_left, pad_top)

t_infer, t_pp, t_coco, t_segtp = [], [], [], []
n_diff = 0
for img_id, img_info in imgs:
    img_path = IMGDIR / img_info['file_name']
    frame = cv2.imread(str(img_path))
    h, w = frame.shape[:2]
    canvas, scale, pad = letterbox(frame, pipeline.input_size)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)

    t0 = time.perf_counter()
    outputs = backend.infer(tensor)
    t_infer.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    objects = pp(outputs, scale, pad, (h, w))
    t_pp.append((time.perf_counter() - t0) * 1000)

    pre = _seg_pre_nms(post_mod, outputs)
    if pre is None:
        conv = []
    else:
        x, proto = pre
        conv = _seg_nms_finish(post_mod, x, proto, scale, pad, (h, w), _fast_nms)
    if len(objects) != len(conv):
        n_diff += 1
    else:
        for a, b in zip(objects, conv):
            if a['class_id'] != b['class_id'] or abs(a['score'] - b['score']) > 1e-6:
                n_diff += 1
                break
            if not np.array_equal(a['masks_160'], b['masks_160']):
                n_diff += 1
                break
            if any(abs(x - y) > 1e-4 for x, y in zip(a['bbox'], b['bbox'])):
                n_diff += 1
                break
    assert conv or not objects, (img_id, len(objects), len(conv))

    pad_left = round(pad[0] / 2 - 0.1)
    pad_top = round(pad[1] / 2 - 0.1)
    t0 = time.perf_counter()
    cres = detections_to_coco(objects, img_id, cat_ids)
    t_coco.append((time.perf_counter() - t0) * 1000)

    sx = (pipeline.input_size - pad[0]) / w
    sy = (pipeline.input_size - pad[1]) / h
    anns = [a for a in ann_by_img.get(img_id, []) if not a.get('iscrowd', 0)]
    t0 = time.perf_counter()
    seg_tp(anns, objects, sx, sy, pad_left, pad_top)
    t_segtp.append((time.perf_counter() - t0) * 1000)

def med(x):
    x.sort()
    return x[len(x) // 2]

print(f'n preds/cadr:    {np.mean([len(ann_by_img.get(i,[])) for i,_ in imgs]):.0f} GT, '
      f'{np.mean([len(o) for o in []]):.0f} preds')
print(f'infer (NPU):     med={med(t_infer):6.1f} ms')
print(f'postprocess:     med={med(t_pp):6.1f} ms')
print(f'coco-conv(RLE):  med={med(t_coco):6.1f} ms')
print(f'seg_tp(metrics): med={med(t_segtp):6.1f} ms')
print(f'ЭКВИВАЛЕНТНОСТЬ: кадров с расхождением = {n_diff}/{len(imgs)}')
tot = med(t_infer) + med(t_pp) + med(t_coco) + med(t_segtp)
print(f'СУММА:           med={tot:6.1f} ms/кадр')
