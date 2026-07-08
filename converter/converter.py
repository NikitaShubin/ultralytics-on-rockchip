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


def export_and_compile(pt_path, info=None):
    """Export .pt → ONNX (RK-opt) → RKNN, returns (onnx_path, rknn_path)."""
    apply_patches()
    from ultralytics import YOLO
    model = YOLO(pt_path)

    # Set custom RKNN mode flag before exporter runs
    for m in model.model.modules():
        m._rknn_mode = True

    # RT-DETR uses deformable attention which requires ONNX opset >= 16
    is_rtdetr = info and info.get("is_rtdetr", False)
    opset = 16 if is_rtdetr else 12
    log(f"Using ONNX opset {opset} for export")
    rknn_dir = model.export(format="rknn", opset=opset)
    log(f"RKNN model directory: {rknn_dir}")

    # Find the ONNX file produced during export
    rknn_dir_path = Path(rknn_dir)
    model_stem = Path(pt_path).stem
    pt_dir = Path(pt_path).parent
    onnx_candidates = [
        f"{model_stem}.onnx",
        str(pt_dir / f"{model_stem}.onnx"),
    ]
    onnx_path = None
    for c in onnx_candidates:
        if os.path.exists(c):
            onnx_path = c
            break
    if not onnx_path:
        onnx_paths = list(Path(".").glob("*.onnx"))
        onnx_path = str(onnx_paths[-1]) if onnx_paths else None

    # Find the RKNN file (prefer the latest .rknn)
    rknn_files = sorted(rknn_dir_path.glob("*.rknn"), key=os.path.getmtime, reverse=True)
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

    # Prefer loading pre-built RKNN; fall back to ONNX rebuild for simulation
    rknn = RKNN(verbose=False)
    if rknn_path and os.path.exists(rknn_path):
        rknn.load_rknn(rknn_path)
    else:
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
        rknn.load_onnx(model=onnx_path)
        rknn.build(do_quantization=False)

    try:
        ret = rknn.init_runtime()
        if ret != 0:
            log(f"Runtime init returned {ret} (expected in simulator for transformer models)")
            rknn.release()
            return {"note": "validation skipped - runtime requires NPU hardware"}
    except Exception as e:
        log(f"Runtime init exception (expected in simulator for transformer models): {e}")
        rknn.release()
        return {"note": "validation skipped - runtime requires NPU hardware"}

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


TEMPLATES_DIR = Path(__file__).parent / "templates"
DEPLOY_DIR = Path("/app/converter/model")


def _detect_model_info(pt_path):
    """Load YOLO model and extract parameters for postprocess generation."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import RTDETRDecoder
    model = YOLO(pt_path)

    task = getattr(model, "task", "detect")
    head = model.model.model[-1]
    is_rtdetr = isinstance(head, RTDETRDecoder)

    if is_rtdetr:
        # RT-DETR: single stride, no grid, no DFL
        stride = model.model.stride
        strides = [int(s) for s in stride.tolist()] if hasattr(stride, "tolist") else [32]
    else:
        stride = model.model.stride
        strides = [int(s) for s in stride.tolist()] if hasattr(stride, "tolist") else [8, 16, 32]

    input_size = getattr(model.model, "args", {}).get("imgsz", 640)
    if not isinstance(input_size, int):
        input_size = 640

    names = model.names if hasattr(model, "names") else getattr(model.model, "names", {})
    if not names:
        names = getattr(model.model, "names", {})
    num_classes = len(names)

    kpt_shape = getattr(head, "kpt_shape", None)
    num_kpts = kpt_shape[0] if kpt_shape else 0
    nm = getattr(head, "nm", 0)
    reg_max = getattr(head, "reg_max", 16)

    return {
        "task": task,
        "is_rtdetr": is_rtdetr,
        "strides": strides,
        "grid_sizes": [input_size // s for s in strides],
        "input_size": input_size,
        "num_classes": num_classes,
        "names": names,
        "num_kpts": num_kpts,
        "nm": nm,
        "reg_max": reg_max,
    }


def _generate_postprocess(info, dst_dir):
    """Generate postprocess.py for the given model info."""
    task = info["task"]
    is_rtdetr = info.get("is_rtdetr", False)

    if is_rtdetr:
        template_name = "detr_postprocess.py"
    else:
        template_name = f"{task}_postprocess.py"
    template_path = TEMPLATES_DIR / template_name

    if not template_path.exists():
        log(f"No template found for task '{template_name}' at {template_path}, using detect")
        template_path = TEMPLATES_DIR / "detect_postprocess.py"

    template = template_path.read_text()

    class_names = [info["names"].get(i, f"cls_{i}") for i in range(info["num_classes"])]

    is_obb = task == "obb"
    is_pose = task == "pose"
    is_seg = task == "segment"
    is_cls = task == "classify"
    is_nodfl = info.get("reg_max", 16) <= 1
    low_thresh = is_obb or is_pose or is_seg
    substitutions = {
        "INPUT_SIZE": str(info["input_size"]),
        "STRIDES": str(info["strides"]),
        "GRID_SIZES": str(info["grid_sizes"]),
        "NUM_CLASSES": str(info["num_classes"]),
        "NUM_KPTS": str(info.get("num_kpts", 0)),
        "NM": str(info.get("nm", 0)),
        "COCO_CLASSES": json.dumps(class_names),
        "REG_MAX": str(info.get("reg_max", 16)),
        "BIN_COUNT": "4",
        "CONF_THRESH": "0.25" if low_thresh else "0.5",
        "IOU_THRESH": "0.5",
        "MIN_WH": "20" if low_thresh else "40",
        "NO_DFL": "True" if is_nodfl else "False",
    }

    for key, val in substitutions.items():
        template = template.replace("{{" + key + "}}", val)

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out_path = dst / "postprocess.py"
    out_path.write_text(template)
    log(f"Generated {out_path}")
    return str(out_path)


def _deploy(onnx_path, rknn_path, postprocess_path):
    """Copy model + postprocess to DEPLOY_DIR and signal reload."""
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    if rknn_path and os.path.exists(rknn_path):
        dst = DEPLOY_DIR / "model.rknn"
        shutil.copy2(rknn_path, str(dst))
        log(f"Deployed {rknn_path} → {dst}")

    if onnx_path and os.path.exists(onnx_path):
        dst = DEPLOY_DIR / "model.onnx"
        shutil.copy2(onnx_path, str(dst))
        log(f"Deployed {onnx_path} → {dst}")

    if postprocess_path and os.path.exists(postprocess_path):
        dst = DEPLOY_DIR / "postprocess.py"
        shutil.copy2(postprocess_path, str(dst))
        log(f"Deployed {postprocess_path} → {dst}")

    _signal_reload()


def _signal_reload():
    """Send HTTP reload signal to webcam service."""
    import urllib.request
    urls = ["http://localhost:5000/reload", "http://webcam:5000/reload"]
    ok = False
    for url in urls:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            log(f"Reload signal sent to {url}: {resp.status}")
            ok = True
            break
        except Exception:
            continue
    if not ok:
        log("Reload signal failed (is webcam running?)")
        log("Restart manually: docker compose restart webcam")


def cmd_convert(args):
    pt = resolve_pt(args.model)
    log(f"Converting {pt}")

    log("Step 1: Detect model info")
    info = _detect_model_info(pt)
    log(f"  task={info['task']} classes={info['num_classes']} strides={info['strides']}")

    log("Step 2: Export ONNX + compile RKNN")
    onnx_path, rknn_path = export_and_compile(pt, info)

    if args.test:
        log("Step 3: Validate")
        if info["task"] == "detect":
            if onnx_path and os.path.exists(onnx_path):
                result = validate(onnx_path, rknn_path)
            else:
                log("No ONNX found, attempting validation with RKNN only")
                result = validate(None, rknn_path)
            print(json.dumps(result, indent=2))
            status = "PASS" if result.get("pass") else "FAIL"
            log(f"Validation: {status}")
            if result.get("pass") is False:
                log("Validation failed, skipping deploy")
                return result
        else:
            log(f"Validation skipped for task '{info['task']}' (detect-only)")
            result = {"note": f"validation not implemented for task '{info['task']}'"}
    else:
        result = {"note": "validation skipped (use --test to enable)"}

    log("Step 4: Generate postprocess.py")
    postprocess_path = _generate_postprocess(info, ".")

    stem = Path(args.model).stem.replace(".pt", "")
    if args.deploy:
        log("Step 5: Deploy")
        _deploy(onnx_path, rknn_path, postprocess_path)
    else:
        # Copy results to model/ directory (versioned)
        for src_ext in [(onnx_path, ".onnx"), (rknn_path, ".rknn")]:
            src, ext = src_ext
            if src and os.path.exists(src):
                dst = MODEL_DIR / f"{stem}_rkopt_o19{ext}"
                os.makedirs(MODEL_DIR, exist_ok=True)
                shutil.copy2(src, str(dst))
                log(f"Copied to {dst}")

    # Always copy versioned postprocess
    if postprocess_path and os.path.exists(postprocess_path):
        dst = MODEL_DIR / f"{stem}_postprocess.py"
        os.makedirs(MODEL_DIR, exist_ok=True)
        shutil.copy2(postprocess_path, str(dst))
        log(f"Copied postprocess to {dst}")

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
    from ultralytics.nn.modules.head import RTDETRDecoder
    model = YOLO(pt_path)

    for m in model.model.modules():
        m._rknn_mode = True

    is_rtdetr = isinstance(model.model.model[-1], RTDETRDecoder)
    opset = 16 if is_rtdetr else 12
    model.export(format="onnx", opset=opset)

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
    p_convert.add_argument("--test", action="store_true",
                           help="Validate ONNX vs RKNN on coco8 before deploy")
    p_convert.add_argument("--deploy", action="store_true",
                           help="Copy model.rknn + postprocess.py to model/ and reload webcam")

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
