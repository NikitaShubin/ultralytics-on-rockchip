#!/usr/bin/env python3
"""Тест параллельного NMS: последовательно vs ThreadPool vs ProcessPool.

Идея пользователя: батч кадров → NMS каждого кадра параллельно на всех ядрах CPU.
Инференс остаётся последовательным (NPU), распараллеливается только NMS.
"""
import concurrent.futures as cf
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torchvision

NCORES = min(8, os.cpu_count() or 8)
NCORES = 8  # A55 медленные — сначала половина ядер

rng = np.random.default_rng(42)
NBATCH = 8
N = 30000
torch.set_num_threads(1)

frames = []
for _ in range(NBATCH):
    cx = rng.uniform(0, 640, N).astype(np.float32)
    cy = rng.uniform(0, 640, N).astype(np.float32)
    w = rng.uniform(1, 64, N).astype(np.float32)
    h = rng.uniform(1, 64, N).astype(np.float32)
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    scores = rng.uniform(0, 1, N).astype(np.float32)
    frames.append((boxes, scores))


def nms_one(args):
    boxes, scores = args
    tb = torch.from_numpy(np.ascontiguousarray(boxes))
    ts = torch.from_numpy(np.ascontiguousarray(scores))
    return torchvision.ops.nms(tb, ts, 0.7).numpy()


def bench(tag, run):
    t0 = time.perf_counter()
    res = run()
    dt = time.perf_counter() - t0
    print(f'{tag}: {dt*1000:.0f} ms, results={len(res)}, n_keep={[len(r) for r in res]}')


bench('последовательно', lambda: [nms_one(f) for f in frames])


def run_threads():
    with cf.ThreadPoolExecutor(NCORES) as ex:
        return list(ex.map(nms_one, frames))


def run_procs():
    with cf.ProcessPoolExecutor(NCORES) as ex:
        return list(ex.map(nms_one, frames))


bench(f'ThreadPool (n={NCORES})', run_threads)
bench(f'ProcessPool (n={NCORES})', run_procs)
