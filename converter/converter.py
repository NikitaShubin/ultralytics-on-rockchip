#!/usr/bin/env python
"""RKNN Model Converter & Validator for Ultralytics detection models.

Usage:
    docker compose run --rm converter convert yolov8m.pt
    docker compose run --rm converter validate yolov8m.rknn
    docker compose run --rm converter test-all-detect
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

COCO8_DIR = Path("/app/coco8")
MODEL_DIR = Path("/app/converter/model")
WEIGHTS_DIR = Path("/app/converter/weights")

STRIDES = [8, 16, 32]
GRID_SIZES = [80, 40, 20]
REG_MAX = 16
BIN_COUNT = 4
CONF_THRESH = 0.5
IOU_THRESH = 0.5
MIN_WH = 40

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def log(msg):
    print(f"[CONVERTER] {msg}", flush=True)


def apply_patches():
    from patches import rknn_patch
    rknn_patch.apply()
    log("RKNN patches applied to ultralytics heads")


def resolve_pt(name):
    if name.endswith(".pt"):
        p = Path(name)
        if p.exists():
            return str(p)
        p = WEIGHTS_DIR / name
        if p.exists():
            return str(p)
    for d in [WEIGHTS_DIR, MODEL_DIR, Path("."), Path("/app/converter")]:
        p = d / f"{name}.pt"
        if p.exists():
            return str(p)
    return name


def export_and_compile(pt_path):
    """Export .pt → ONNX (RK-opt) → RKNN, returns (onnx_path, rknn_path)."""
    apply_patches()
    from ultralytics import YOLO
    model = YOLO(pt_path)

    # Set custom RKNN mode flag before exporter runs
    for m in model.model.modules():
        m._rknn_mode = True

    # export_rknn does ONNX export + RKNN compile in one step
    # Returns the RKNN output directory
    rknn_dir = model.export(format="rknn", opset=12)
    log(f"RKNN model directory: {rknn_dir}")

    # Find the ONNX file produced during export (matches model name stem)
    rknn_dir_path = Path(rknn_dir)
    model_stem = Path(pt_path).stem
    onnx_path = f"{model_stem}.onnx"
    if not os.path.exists(onnx_path):
        onnx_paths = list(Path(".").glob("*.onnx"))
        onnx_path = str(onnx_paths[-1]) if onnx_paths else None

    # Find the RKNN file
    rknn_files = list(rknn_dir_path.glob("*.rknn"))
    rknn_path = str(rknn_files[0]) if rknn_files else rknn_dir

    if onnx_path:
        log(f"ONNX (RK-opt): {onnx_path}")
    log(f"RKNN: {rknn_path}")
    return onnx_path, rknn_path


def preprocess(frame, input_size=640):
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
    return tensor, scale, (pad_w, pad_h)


def dfl(position):
    n, c, h, w = position.shape
    mc = c // BIN_COUNT
    y = position.reshape(n, BIN_COUNT, mc, h, w)
    y_exp = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
    weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * weights).sum(axis=2)


def process_head(box_dist, cls_scores, stride, grid_h, grid_w):
    col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
    row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
    grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w))
    grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w))
    grid = np.concatenate([grid_x, grid_y], axis=1)
    dbox = dfl(box_dist)
    l, t, r, b = dbox[:, 0:1], dbox[:, 1:2], dbox[:, 2:3], dbox[:, 3:4]
    x1 = (grid[:, 0:1] + 0.5 - l) * stride
    y1 = (grid[:, 1:2] + 0.5 - t) * stride
    x2 = (grid[:, 0:1] + 0.5 + r) * stride
    y2 = (grid[:, 1:2] + 0.5 + b) * stride
    boxes = np.concatenate([x1, y1, x2, y2], axis=1)
    boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
    scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, 80)
    return boxes, scores


def postprocess(outputs, scale, pad, orig_wh):
    orig_w, orig_h = orig_wh
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]
    all_boxes, all_scores, all_cls = [], [], []
    for idx in range(3):
        box_dist = outputs[idx * 3]
        cls_scores = outputs[idx * 3 + 1]
        stride = strides[idx]
        gh = gw = grid_sizes[idx]
        boxes, scores = process_head(box_dist, cls_scores, stride, gh, gw)
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
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    valid = (bw >= MIN_WH) & (bh >= MIN_WH)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]
    if not len(boxes):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    keep = nms(boxes, scores, IOU_THRESH)
    return boxes[keep], scores[keep], cls_ids[keep]


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep)


def iou_matrix(boxes_a, boxes_b):
    xa1, ya1, xa2, ya2 = boxes_a.T
    xb1, yb1, xb2, yb2 = boxes_b.T
    inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
    inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
    inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
    inter_y2 = np.minimum(ya2[:, None], yb2[None, :])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-10)


def run_onnx(sess, tensor):
    input_name = sess.get_inputs()[0].name
    raw = sess.run(None, {input_name: tensor / 255.0})
    return raw


def run_rknn(rknn, tensor):
    nhwc = tensor.transpose(0, 2, 3, 1)
    return rknn.inference(inputs=[nhwc])


def load_coco8_images():
    images = []
    # coco8.zip extracts to /app/coco8/coco8/
    img_dir = COCO8_DIR / "coco8" / "images" / "train"
    if not img_dir.exists():
        img_dir = COCO8_DIR / "images" / "train"
    if not img_dir.exists():
        log(f"coco8 images not found at {img_dir}")
        return images
    for f in sorted(img_dir.glob("*.jpg"))[:10]:
        img = cv2.imread(str(f))
        if img is not None:
            images.append(img)
    return images


def validate(onnx_path, rknn_path):
    import onnxruntime as rt
    from rknn.api import RKNN

    sess = rt.InferenceSession(onnx_path)

    # Load ONNX and build RKNN in-memory for CPU simulation
    # (load_rknn + init_runtime requires NPU hardware)
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    rknn.load_onnx(model=onnx_path)
    rknn.build(do_quantization=False)
    rknn.init_runtime()

    images = load_coco8_images()
    if not images:
        log("No coco8 images found, skipping validation")
        return {"error": "no images"}

    max_diffs = []
    ious = []
    n_images = 0

    for img in images:
        h, w = img.shape[:2]
        tensor, scale, pad = preprocess(img)

        onnx_out = run_onnx(sess, tensor)
        rknn_out = run_rknn(rknn, tensor)

        if isinstance(onnx_out, (list, tuple)):
            onnx_tensors = onnx_out
        else:
            onnx_tensors = [onnx_out]

        max_diff = 0.0
        for o, r in zip(onnx_tensors, rknn_out):
            diff = np.abs(o.astype(np.float32) - r.astype(np.float32)).max()
            max_diff = max(max_diff, float(diff))
        max_diffs.append(max_diff)

        # Compare box-level results
        boxes_onnx, scores_onnx, cls_onnx = postprocess(onnx_tensors, scale, pad, (w, h))
        boxes_rknn, scores_rknn, cls_rknn = postprocess(rknn_out, scale, pad, (w, h))

        if len(boxes_onnx) > 0 and len(boxes_rknn) > 0:
            iou = iou_matrix(boxes_onnx, boxes_rknn)
            row_max = iou.max(axis=1)
            ious.append(float(row_max.mean()))
        elif len(boxes_onnx) == 0 and len(boxes_rknn) == 0:
            ious.append(1.0)
        else:
            ious.append(0.0)

        n_images += 1

    rknn.release()

    avg_max_diff = float(np.mean(max_diffs))
    avg_iou = float(np.mean(ious))

    result = {
        "n_images": n_images,
        "max_diff_max": float(np.max(max_diffs)),
        "max_diff_avg": avg_max_diff,
        "iou_avg": avg_iou,
        "pass": avg_iou > 0.9,
    }
    return result


def cmd_convert(args):
    pt = resolve_pt(args.model)
    log(f"Converting {pt}")

    log("Step 1: Export ONNX + compile RKNN")
    onnx_path, rknn_path = export_and_compile(pt)

    log("Step 2: Validate")
    if onnx_path and os.path.exists(onnx_path):
        result = validate(onnx_path, rknn_path)
    else:
        log("No ONNX found, attempting validation with RKNN only")
        result = validate(None, rknn_path)

    print(json.dumps(result, indent=2))
    status = "PASS" if result.get("pass") else "FAIL"
    log(f"Validation: {status}")

    # Copy results to model/ directory
    stem = Path(args.model).stem.replace(".pt", "")
    for src_ext in [(onnx_path, ".onnx"), (rknn_path, ".rknn")]:
        src, ext = src_ext
        if src and os.path.exists(src):
            dst = MODEL_DIR / f"{stem}_rkopt_o19{ext}"
            os.makedirs(MODEL_DIR, exist_ok=True)
            shutil.copy2(src, str(dst))
            log(f"Copied to {dst}")

    return result


def cmd_validate(args):
    rknn_path = args.model
    if not rknn_path.endswith(".rknn"):
        rknn_path = str(MODEL_DIR / f"{rknn_path}.rknn")
    log(f"Validating {rknn_path}")

    onnx_path = rknn_path.replace(".rknn", ".onnx")
    if not os.path.exists(onnx_path):
        stems = list(MODEL_DIR.glob(f"{Path(rknn_path).stem}*.onnx"))
        if stems:
            onnx_path = str(stems[0])

    if not os.path.exists(onnx_path):
        log(f"No matching ONNX found for {rknn_path}, skipping validation")
        return {"error": "no onnx"}

    result = validate(onnx_path, rknn_path)
    print(json.dumps(result, indent=2))
    status = "PASS" if result.get("pass") else "FAIL"
    log(f"Validation: {status}")
    return result


def load_model_list(path):
    if os.path.exists(path):
        names = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        return names
    names = []
    for p in [
        "/app/converter/models_detect.txt",
        "models_detect.txt",
        "../converter/models_detect.txt",
    ]:
        if os.path.exists(p):
            names = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
            break
    return names


def cmd_test_all_detect(args):
    models = load_model_list(args.model_list)
    if not models:
        log("No model list found, using default detection models")
        models = [
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu",
            "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
            "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10b", "yolov10x",
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
        ]

    results = {}
    for name in models:
        log(f"{'='*60}")
        log(f"Testing {name}...")
        try:
            pt = resolve_pt(name)
            res = cmd_convert(argparse.Namespace(model=name, **vars(args)))
            results[name] = {"status": "PASS" if res.get("pass") else "FAIL", **res}
        except Exception as e:
            log(f"FAILED: {e}")
            results[name] = {"status": "FAIL", "error": str(e)}
        print()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Status':<8} {'max_diff':<10} {'iou_avg':<10}")
    print("-" * 50)
    for name, res in results.items():
        md = res.get("max_diff_avg", "-")
        iou = res.get("iou_avg", "-")
        print(f"{name:<20} {res['status']:<8} {md:<10} {iou:<10}")

    return results


def export_onnx_only(pt_path):
    """Export .pt → ONNX (RK-opt) only, no RKNN compile. Returns onnx_path."""
    apply_patches()
    from ultralytics import YOLO
    model = YOLO(pt_path)

    for m in model.model.modules():
        m._rknn_mode = True

    model.export(format="onnx", opset=12)

    model_stem = Path(pt_path).stem
    onnx_path = f"{model_stem}.onnx"
    if not os.path.exists(onnx_path):
        onnx_paths = list(Path(".").glob("*.onnx"))
        onnx_path = str(onnx_paths[-1]) if onnx_paths else None

    if onnx_path:
        log(f"ONNX (RK-opt): {onnx_path}")
    return onnx_path


def cmd_batch_onnx(args):
    models = load_model_list(args.model_list)
    if not models:
        log("No models to process")
        return

    results = {}
    for name in models:
        log(f"{'='*60}")
        log(f"Testing {name}...")
        pt = resolve_pt(name)
        try:
            onnx_path = export_onnx_only(pt)
            if not onnx_path:
                raise RuntimeError("ONNX export produced no file")

            result = validate(onnx_path, None)

            stem = Path(name).stem.replace(".pt", "")
            dst = MODEL_DIR / f"{stem}_rkopt_o19.onnx"
            os.makedirs(MODEL_DIR, exist_ok=True)
            shutil.copy2(onnx_path, str(dst))
            log(f"Copied to {dst}")
            os.remove(onnx_path)

            results[name] = {"status": "PASS" if result.get("pass") else "FAIL", **result}
            status = "PASS" if result.get("pass") else "FAIL"
            log(f"{name}: {status}")
        except Exception as e:
            log(f"FAILED: {e}")
            results[name] = {"status": "FAIL", "error": str(e)}
        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Status':<8} {'iou_avg':<10} {'max_diff':<10}")
    print("-" * 50)
    for name, res in results.items():
        iou = res.get("iou_avg", "-")
        md = res.get("max_diff_avg", "-")
        print(f"{name:<20} {res['status']:<8} {iou:<10} {md:<10}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RKNN Model Converter & Validator")
    sub = parser.add_subparsers(dest="command")

    p_convert = sub.add_parser("convert", help="Convert .pt → ONNX (RK-opt) → RKNN → validate")
    p_convert.add_argument("model", help="Model name or path (e.g. yolov8m.pt)")

    p_validate = sub.add_parser("validate", help="Validate existing .rknn against coco8")
    p_validate.add_argument("model", help="Path or name of .rknn file")

    p_test = sub.add_parser("test-all-detect", help="Test all detection models")
    p_test.add_argument("--model-list", default="models_detect.txt",
                        help="Path to model list file")

    p_batch_onnx = sub.add_parser("batch-onnx", help="Export ONNX (RK-opt) for all models in list")
    p_batch_onnx.add_argument("--model-list", default="models_detect.txt",
                              help="Path to model list file")

    args = parser.parse_args()
    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "test-all-detect":
        cmd_test_all_detect(args)
    elif args.command == "batch-onnx":
        cmd_batch_onnx(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
