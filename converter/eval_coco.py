#!/usr/bin/env python
"""Evaluate ONNX or RKNN model on COCO val 2017, report mAP."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_BASE = "/app/coco_val2017"
COCO_DETECT_ANN = f"{COCO_BASE}/annotations/instances_val2017.json"
COCO_POSE_ANN = f"{COCO_BASE}/annotations/person_keypoints_val2017.json"
COCO_IMG_DIR = f"{COCO_BASE}/images/val2017"

INPUT_SIZE = 640
CONF_THRESH = 0.001
IOU_THRESH = 0.7
MAX_DET = 100


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
    return tensor, scale, (pad_w, pad_h)


def dfl(position):
    BIN_COUNT = 4
    n, c, h, w = position.shape
    mc = c // BIN_COUNT
    y = position.reshape(n, BIN_COUNT, mc, h, w)
    y_exp = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
    weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * weights).sum(axis=2)


def make_anchors(grid_h, grid_w, stride):
    col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
    row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
    grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w))
    grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w))
    grid = np.concatenate([grid_x, grid_y], axis=1)
    anchor = grid + 0.5
    return anchor * stride


def postprocess_detect(outputs, scale, pad, orig_wh):
    orig_w, orig_h = orig_wh
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]
    outputs_per_head = 2
    all_boxes, all_scores, all_cls = [], [], []
    for idx in range(len(strides)):
        box_dist = outputs[idx * outputs_per_head]
        cls_scores = outputs[idx * outputs_per_head + 1]
        stride = strides[idx]
        gh = gw = grid_sizes[idx]
        col = np.arange(gw, dtype=np.float32).reshape(1, 1, 1, gw)
        row = np.arange(gh, dtype=np.float32).reshape(1, 1, gh, 1)
        grid_x = np.broadcast_to(col, (1, 1, gh, gw))
        grid_y = np.broadcast_to(row, (1, 1, gh, gw))
        grid = np.concatenate([grid_x, grid_y], axis=1)
        dbox = dfl(box_dist)
        l, t, r, b = dbox[:, 0:1], dbox[:, 1:2], dbox[:, 2:3], dbox[:, 3:4]
        x1 = (grid[:, 0:1] + 0.5 - l) * stride
        y1 = (grid[:, 1:2] + 0.5 - t) * stride
        x2 = (grid[:, 0:1] + 0.5 + r) * stride
        y2 = (grid[:, 1:2] + 0.5 + b) * stride
        boxes = np.concatenate([x1, y1, x2, y2], axis=1)
        boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
        n_cls = cls_scores.shape[1]
        scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, n_cls)
        max_cls = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        all_boxes.append(boxes)
        all_scores.append(max_cls)
        all_cls.append(cls_ids)
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)
    mask = scores > CONF_THRESH
    if not mask.any():
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if not len(boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    keep = nms_per_class(boxes, scores, cls_ids, IOU_THRESH, MAX_DET)
    return boxes[keep], scores[keep], cls_ids[keep]


def postprocess_pose(outputs, scale, pad, orig_wh):
    orig_w, orig_h = orig_wh
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]
    outputs_per_head = 3
    n_kpts = 17
    all_boxes, all_scores, all_cls, all_kpts = [], [], [], []
    for idx in range(len(strides)):
        box_dist = outputs[idx * outputs_per_head]
        cls_scores = outputs[idx * outputs_per_head + 1]
        kpt_scores = outputs[idx * outputs_per_head + 2]
        stride = strides[idx]
        gh = gw = grid_sizes[idx]
        col = np.arange(gw, dtype=np.float32).reshape(1, 1, 1, gw)
        row = np.arange(gh, dtype=np.float32).reshape(1, 1, gh, 1)
        grid_x = np.broadcast_to(col, (1, 1, gh, gw))
        grid_y = np.broadcast_to(row, (1, 1, gh, gw))
        grid = np.concatenate([grid_x, grid_y], axis=1)
        dbox = dfl(box_dist)
        l, t, r, b = dbox[:, 0:1], dbox[:, 1:2], dbox[:, 2:3], dbox[:, 3:4]
        x1 = (grid[:, 0:1] + 0.5 - l) * stride
        y1 = (grid[:, 1:2] + 0.5 - t) * stride
        x2 = (grid[:, 0:1] + 0.5 + r) * stride
        y2 = (grid[:, 1:2] + 0.5 + b) * stride
        boxes = np.concatenate([x1, y1, x2, y2], axis=1)
        boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
        n_cls = cls_scores.shape[1]
        scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, n_cls)
        max_cls = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)

        gx = np.arange(gw, dtype=np.float32).reshape(1, gw)
        gy = np.arange(gh, dtype=np.float32).reshape(gh, 1)
        grid_x_2d = np.broadcast_to(gx, (gh, gw))
        grid_y_2d = np.broadcast_to(gy, (gh, gw))

        kpts_4d = kpt_scores.reshape(n_kpts, 3, gh, gw)
        kx = (kpts_4d[:, 0, :, :] * 2 + grid_x_2d) * stride
        ky = (kpts_4d[:, 1, :, :] * 2 + grid_y_2d) * stride
        kvis = 1.0 / (1.0 + np.exp(-kpts_4d[:, 2, :, :]))
        kpts_out = np.stack([kx, ky, kvis], axis=-1).transpose(1, 2, 0, 3).reshape(-1, n_kpts * 3)

        all_boxes.append(boxes)
        all_scores.append(max_cls)
        all_cls.append(cls_ids)
        all_kpts.append(kpts_out)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)
    kpts_all = np.concatenate(all_kpts)
    mask = scores > CONF_THRESH
    if not mask.any():
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), np.empty((0, 17 * 3))
    boxes, scores, cls_ids, kpts_all = boxes[mask], scores[mask], cls_ids[mask], kpts_all[mask]
    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    x1 = np.clip((boxes[:, 0] - pad_left) / scale, 0, orig_w)
    y1 = np.clip((boxes[:, 1] - pad_top) / scale, 0, orig_h)
    x2 = np.clip((boxes[:, 2] - pad_left) / scale, 0, orig_w)
    y2 = np.clip((boxes[:, 3] - pad_top) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    for k in range(17):
        kpts_all[:, k * 3] = (kpts_all[:, k * 3] - pad_left) / scale
        kpts_all[:, k * 3 + 1] = (kpts_all[:, k * 3 + 1] - pad_top) / scale
        kpts_all[:, k * 3] = np.clip(kpts_all[:, k * 3], 0, orig_w)
        kpts_all[:, k * 3 + 1] = np.clip(kpts_all[:, k * 3 + 1], 0, orig_h)

    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    boxes, scores, cls_ids, kpts_all = boxes[valid], scores[valid], cls_ids[valid], kpts_all[valid]
    if not len(boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), np.empty((0, 17 * 3))
    keep = nms_per_class(boxes, scores, cls_ids, IOU_THRESH, MAX_DET)
    return boxes[keep], scores[keep], cls_ids[keep], kpts_all[keep]


def nms_per_class(boxes, scores, cls_ids, iou_thresh, max_det):
    unique_cls = np.unique(cls_ids)
    keep = []
    for c in unique_cls:
        mask_c = cls_ids == c
        boxes_c = boxes[mask_c]
        scores_c = scores[mask_c]
        if len(boxes_c) == 0:
            continue
        x1, y1, x2, y2 = boxes_c.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores_c.argsort()[::-1]
        keep_c = []
        while order.size > 0 and len(keep_c) < max_det:
            i = order[0]
            keep_c.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            order = order[1:][iou <= iou_thresh]
        keep.extend(np.where(mask_c)[0][keep_c])
    return np.array(keep)


def load_model(model_path):
    if model_path.endswith(".onnx"):
        import onnxruntime as rt
        sess = rt.InferenceSession(model_path)
        def infer(tensor):
            input_name = sess.get_inputs()[0].name
            return sess.run(None, {input_name: tensor / 255.0})
        return infer
    elif model_path.endswith(".rknn"):
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime(core_mask=0)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        def infer(tensor):
            nhwc = tensor.transpose(0, 2, 3, 1)
            return rknn.inference(inputs=[nhwc])
        return infer
    else:
        raise ValueError(f"Unknown model type: {model_path}")


def is_pose_model(n_outputs):
    return n_outputs == 9


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on COCO val 2017")
    parser.add_argument("model", help="Path to .onnx or .rknn model")
    parser.add_argument("--task", choices=["detect", "pose"], default=None,
                        help="Task type (auto-detected from model outputs if omitted)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of images (0=all)")
    parser.add_argument("--ann", default=None,
                        help="COCO annotation JSON path (overrides --task)")
    parser.add_argument("--img-dir", default=COCO_IMG_DIR,
                        help="COCO image directory")
    parser.add_argument("--max-det", type=int, default=100,
                        help="Max detections per image per class")
    args = parser.parse_args()

    print(f"[EVAL] Loading model: {args.model}", flush=True)
    infer_fn = load_model(args.model)

    dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    n_outputs = len(infer_fn(dummy))
    if args.task is None:
        args.task = "pose" if is_pose_model(n_outputs) else "detect"
    print(f"[EVAL] Detected {n_outputs} outputs → task={args.task}", flush=True)

    if args.ann is None:
        args.ann = COCO_POSE_ANN if args.task == "pose" else COCO_DETECT_ANN

    print(f"[EVAL] Loading annotations: {args.ann}", flush=True)
    coco = COCO(args.ann)
    img_ids = coco.getImgIds()
    if args.limit > 0:
        img_ids = img_ids[:args.limit]
    print(f"[EVAL] {len(img_ids)} images to evaluate (task={args.task})", flush=True)

    predictions = []
    t0 = time.time()
    img_dir = Path(args.img_dir)

    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        tensor, scale, pad = preprocess(frame)
        raw = infer_fn(tensor)

        if args.task == "pose":
            boxes, scores, cls_ids, kpts = postprocess_pose(raw, scale, pad, (w, h))
        else:
            boxes, scores, cls_ids = postprocess_detect(raw, scale, pad, (w, h))

        if len(scores) > args.max_det:
            topk = scores.argsort()[::-1][:args.max_det]
            boxes = boxes[topk]
            scores = scores[topk]
            cls_ids = cls_ids[topk]
            if args.task == "pose":
                kpts = kpts[topk]

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            pred = {
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(bw), float(bh)],
                "score": float(scores[j]),
            }
            if args.task == "pose":
                k = kpts[j]
                kpt_list = []
                n_visible = 0
                for kp in range(17):
                    kx, ky, kvis = k[kp * 3], k[kp * 3 + 1], k[kp * 3 + 2]
                    v = 2 if kvis > 0.5 else 1
                    kpt_list.extend([float(kx), float(ky), v])
                    if kvis > 0.5:
                        n_visible += 1
                pred["keypoints"] = kpt_list
                pred["num_keypoints"] = n_visible
            predictions.append(pred)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"[EVAL] {i+1}/{len(img_ids)} images, "
                  f"{len(predictions)} detections, "
                  f"{elapsed:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0
    print(f"[EVAL] Done: {len(img_ids)} images, "
          f"{len(predictions)} detections, "
          f"{elapsed:.0f}s total", flush=True)

    if not predictions:
        print("[EVAL] No predictions! mAP = 0", flush=True)
        return

    coco_dt = coco.loadRes(predictions)
    eval_type = "keypoints" if args.task == "pose" else "bbox"
    coco_eval = COCOeval(coco, coco_dt, eval_type)
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "model": Path(args.model).name,
        "task": args.task,
        "n_images": len(img_ids),
        "n_detections": len(predictions),
        "elapsed_s": round(elapsed, 1),
        "map50": float(coco_eval.stats[1]),
        "map50_95": float(coco_eval.stats[0]),
        "precision": float(coco_eval.stats[2]),
        "recall": float(coco_eval.stats[3]),
    }
    print(f"\n[EVAL] RESULTS: {json.dumps(results)}", flush=True)


if __name__ == "__main__":
    main()
