#!/usr/bin/env python3
"""Сравнение скорости NMS: torchvision vs ultralytics .val() на том же кадре.

Вопрос: у ultralytics .val() при conf=0.001 тоже >1с/кадр на ARM?
"""
import time

import numpy as np
import torch
import torchvision

N = 30000
rng = np.random.default_rng(0)
cx = rng.uniform(0, 640, N).astype(np.float32)
cy = rng.uniform(0, 640, N).astype(np.float32)
w = rng.uniform(1, 64, N).astype(np.float32)
h = rng.uniform(1, 64, N).astype(np.float32)
boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
scores = rng.uniform(0, 1, N).astype(np.float32)

tb = torch.from_numpy(boxes)
ts = torch.from_numpy(scores)
for _ in range(3):
    keep = torchvision.ops.nms(tb, ts, 0.7)
t0 = time.perf_counter()
for _ in range(5):
    keep = torchvision.ops.nms(tb, ts, 0.7)
t1 = (time.perf_counter() - t0) / 5 * 1000
print(f'torchvision.ops.nms на {N}: {t1:.1f} ms, keep={len(keep)}')

# ультралайтс non_max_suppression
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

pred = torch.from_numpy(np.random.uniform(0, 1, (1, 4 + 80, 8400)).astype(np.float32))
pred[:, :4, :] = torch.tensor(rng.uniform(0, 640, (4, 8400)).astype(np.float32))
t0 = time.perf_counter()
for _ in range(3):
    out = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.7,
                              max_det=300, multi_label=True, max_time_img=1000)
t1 = (time.perf_counter() - t0) / 3 * 1000
print(f'ultralytics non_max_suppression (conf=0.001, multi_label): {t1:.1f} ms, len={len(out[0])}')
