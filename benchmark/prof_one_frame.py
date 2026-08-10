#!/usr/bin/env python3
"""Профилирование одного кадра: наш Pipeline (NPU) + поэтапный постпроцесс.

Показывает, где реально тратятся секунды: letterbox / infer / expand / argsort / NMS.
Запуск в convert-контейнере (rknnlite + реальный NPU):
  docker compose -f convert/docker-compose.yml run --rm --entrypoint python \
    convert /app/benchmark/prof_one_frame.py yolo26n detect /app/datasets/coco128/images/train2017
"""
import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import percept.inference as P  # noqa: E402
from pipeline import Pipeline  # noqa: E402


def t(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return (time.perf_counter() - t0) * 1000, r


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'yolo26n'
    task = sys.argv[2] if len(sys.argv) > 2 else 'detect'
    img_dir = sys.argv[3] if len(sys.argv) > 3 else '/app/datasets/coco128/images/train2017'

    imgs = sorted(Path(img_dir).glob('*.jpg'))[:3]
    if not imgs:
        print(f'НЕТ ИЗОБРАЖЕНИЙ в {img_dir}')
        return

    cache = ROOT / 'cache' / model
    pipe = Pipeline(model_dir=str(cache), model_name=model, backend='npu', rknn_suffix='_fp16')

    import sys as _s
    pp = _s.modules['postprocess_mod']
    pp.CONF_THRESH = 0.001
    pp.IOU_THRESH = 0.7
    pp.MIN_WH = 1
    pp.MAX_DET = 300
    pp.DEDUP_ENABLED = False
    pp.EVAL_PRECISE = True

    # fast NMS через torchvision (как в eval.py) или numpy-версия из шаблона
    nms_tag = 'template-numpy'
    if os.environ.get('NMS') == 'tv':
        try:
            import torch
            import torchvision
            _orig_nms = pp.nms

            def _fast_nms(boxes, scores, iou_threshold):
                tb = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))
                ts = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
                return torchvision.ops.nms(tb, ts, iou_threshold).numpy().astype(np.int64)

            pp.nms = _fast_nms
            nms_tag = 'torchvision'
        except Exception as e:
            nms_tag = f'numpy ({e})'
    print(f'NMS: {nms_tag} | eval_mode: conf={pp.CONF_THRESH} iou={pp.IOU_THRESH}')

    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # 1. letterbox + preprocess (как в ModelBackend.detect)
        dt, (canvas, scale, pad) = t(P.letterbox, frame, pipe.input_size)
        dt2, (rgb,) = t(lambda: (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32),))
        dt3, (tensor,) = t(
            lambda: (np.expand_dims(rgb.transpose(2, 0, 1), axis=0),))
        tensor = tensor / 255.0

        # 2. infer (NPU)
        dt4, outputs = t(pipe._backend.infer, tensor)
        n_out = len(outputs)
        shapes = [tuple(o.shape) for o in outputs]

        # 3. постпроцесс поэтапно
        t0 = time.perf_counter()
        if len(outputs) == 1 and outputs[0].ndim == 3 and outputs[0].shape[1] == 4 + pp.NUM_CLASSES:
            single = True
            pred = outputs[0][0].astype(np.float32)
            pred[:4] = pp.xywh2xyxy(pred[:4].T).T
            pred = pred.T
        else:
            single = False
            pred = pp._decode_per_stride(outputs)
        t_decode = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        xc = pred[:, 4:4 + pp.NUM_CLASSES].max(axis=1) > pp.CONF_THRESH
        pred = pred[xc]
        t_xc = (time.perf_counter() - t0) * 1000

        box = pred[:, :4]
        cls = pred[:, 4:4 + pp.NUM_CLASSES]
        t0 = time.perf_counter()
        i, j = np.where(cls > pp.CONF_THRESH)
        t_where = (time.perf_counter() - t0) * 1000
        n_cand = len(i)

        t0 = time.perf_counter()
        x = np.column_stack([
            box[i],
            cls[i, j, None],
            j[:, None].astype(np.float32),
        ])
        t_col = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if x.shape[0] > pp.MAX_NMS:
            filt = x[:, 4].argsort()[::-1][:pp.MAX_NMS]
            x = x[filt]
        t_sort = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        c = x[:, 5:6] * pp.MAX_WH
        boxes_nms = x[:, :4] + c
        keep = pp.nms(boxes_nms, x[:, 4], pp.IOU_THRESH)
        keep = keep[:pp.MAX_DET]
        t_nms = (time.perf_counter() - t0) * 1000

        # 5. полный postprocess (для сверки)
        t5, objects = t(pp.postprocess, outputs, scale, pad, (h, w))

        # 6. сверка: конвейерные функции eval.py == шаблонный postprocess
        import sys as _s2
        _s2.path.insert(0, str(ROOT / 'benchmark'))
        from eval import _detect_pre_nms, _detect_nms_finish
        x2 = _detect_pre_nms(pp, outputs)
        objs_pipe = _detect_nms_finish(pp, x2, scale, pad, (h, w), pp.nms) if x2 is not None else []
        same = len(objs_pipe) == len(objects)
        if same:
            for a, b in zip(objs_pipe, objects):
                if a != b:
                    same = False
                    break
        print(f'    сверка конвейер==шаблон: {"OK" if same else "РАСХОЖДЕНИЕ"} ({len(objects)} objects)')

        print(f'--- кадр {i}: {img_path.name} ({w}x{h}) — выходы {len(outputs)}: {shapes[0][1]}...')
        print(f'    letterbox={dt:.1f}ms convert={dt2+dt3:.1f}ms | '
              f'infer(NPU)={dt4:.1f}ms | decode={t_decode:.1f}ms | '
              f'xc-filter={t_xc:.1f}ms | where={t_where:.1f}ms (cand={n_cand}) | '
              f'colstack={t_col:.1f}ms | sort/top30000={t_sort:.1f}ms | NMS={t_nms:.1f}ms | '
              f'keep={len(keep)} | full_post={t5:.1f}ms | objects={len(objects)}')
        print(f'    СУММА этапов={dt+dt2+dt3+dt4+t_decode+t_xc+t_where+t_col+t_sort+t_nms:.1f}ms')


if __name__ == '__main__':
    main()
