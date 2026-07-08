#!/usr/bin/env python3
"""Validate ONNX (CPU) vs RKNN (NPU) on actual hardware."""

import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path("/app/model")


def load_images():
    imgs = []
    for p in MODEL_DIR.glob("*.jpg"):
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)
    return imgs


def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w, pad_h = new_size - new_w, new_size - new_h
    im = cv2.resize(img, (new_w, new_h))
    im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, scale, (pad_w, pad_h)


def preprocess(img, input_size=640):
    im, scale, pad = letterbox(img, input_size)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_tensor = im_rgb.transpose(2, 0, 1).astype(np.float32)[None]
    return im_tensor, im_rgb, scale, pad


def run_onnx(sess, tensor):
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: tensor})


def run_rknn(rknn, rgb_img, input_size=640):
    im = cv2.resize(rgb_img, (input_size, input_size))
    im = np.expand_dims(im, 0)
    return rknn.inference(inputs=[im])


def load_postprocess(pp_path):
    spec = importlib.util.spec_from_file_location("pp_mod", pp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pp_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_one(model_stem, onnx_path, rknn_path, pp_path):
    import onnxruntime as rt
    from rknnlite.api import RKNNLite

    print(f"\n{'='*60}")
    print(f"Validating: {model_stem}")
    print(f"  ONNX: {onnx_path}")
    print(f"  RKNN: {rknn_path}")
    print(f"  PP:   {pp_path}")

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        print(f"  FAIL: RKNN load error {ret}")
        return None
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print(f"  FAIL: RKNN init error {ret}")
        return None

    pp_mod = load_postprocess(pp_path)
    input_size = getattr(pp_mod, "INPUT_SIZE", 640)
    postprocess_fn = pp_mod.postprocess

    images = load_images()
    if not images:
        print("  FAIL: no test images in /app/model/")
        return None

    max_diffs = []
    n_objects_onnx = 0
    n_objects_rknn = 0
    n_matched = 0
    n_images = 0

    for img_idx, img in enumerate(images[:5]):
        h, w = img.shape[:2]
        tensor, rgb_img, scale, pad = preprocess(img, input_size)

        onnx_out = run_onnx(sess, tensor)
        rknn_out = run_rknn(rknn, rgb_img, input_size)

        max_diff = 0.0
        for o, r in zip(onnx_out, rknn_out):
            diff = np.abs(o.astype(np.float32) - r.astype(np.float32)).max()
            max_diff = max(max_diff, float(diff))
        max_diffs.append(max_diff)

        objs_onnx = postprocess_fn(onnx_out, scale, pad, (h, w))
        objs_rknn = postprocess_fn(rknn_out, scale, pad, (h, w))

        n_objects_onnx += len(objs_onnx)
        n_objects_rknn += len(objs_rknn)

        onnx_keys = {(o["class_id"], o["bbox"][0], o["bbox"][1]) for o in objs_onnx}
        rknn_keys = {(o["class_id"], o["bbox"][0], o["bbox"][1]) for o in objs_rknn}
        n_matched += len(onnx_keys & rknn_keys)
        n_images += 1

    rknn.release()

    avg_max_diff = float(np.mean(max_diffs)) if max_diffs else -1
    match_ratio = n_matched / max(n_objects_onnx, 1)

    result = {
        "model": model_stem,
        "n_images": n_images,
        "max_diff_avg": round(avg_max_diff, 4),
        "max_diff_max": round(float(np.max(max_diffs)), 4) if max_diffs else -1,
        "n_objects_onnx": n_objects_onnx,
        "n_objects_rknn": n_objects_rknn,
        "match_ratio": round(match_ratio, 4),
    }
    result["pass"] = avg_max_diff < 5.0 and match_ratio > 0.5

    status = "PASS" if result["pass"] else "FAIL"
    print(f"  max_diff_avg={avg_max_diff:.4f}  max={result['max_diff_max']:.4f}  "
          f"match={match_ratio:.4f}  onnx={n_objects_onnx}  rknn={n_objects_rknn}")
    print(f"  => {status}")
    return result


def main():
    results = {}

    models = [
        ("yolo26n-seg", "yolo26n-seg.onnx", "yolo26n-seg.rknn", "yolo26n-seg_postprocess.py"),
        ("yolo26n-pose", "yolo26n-pose.onnx", "yolo26n-pose.rknn", "yolo26n-pose_postprocess.py"),
        ("yolo26n-obb", "yolo26n-obb.onnx", "yolo26n-obb.rknn", "yolo26n-obb_postprocess.py"),
        ("yolo26n", "yolo26n_rkopt_o19.onnx", "yolo26n_rkopt_o19.rknn", "yolo26n_postprocess.py"),
        ("yolo12n", "yolo12n_rkopt_o19.onnx", "yolo12n_rkopt_o19.rknn", "yolo12n_postprocess.py"),
        ("yolo11n", "yolo11n_rkopt_o19.onnx", "yolo11n_rkopt_o19.rknn", "yolo11n_postprocess.py"),
    ]

    for stem, onnx_name, rknn_name, pp_name in models:
        onnx_path = MODEL_DIR / onnx_name
        rknn_path = MODEL_DIR / rknn_name
        pp_path = MODEL_DIR / pp_name

        if not onnx_path.exists():
            print(f"\n  SKIP {stem}: ONNX not found")
            continue
        if not rknn_path.exists():
            print(f"\n  SKIP {stem}: RKNN not found")
            continue
        if not pp_path.exists():
            print(f"\n  SKIP {stem}: PP not found")
            continue

        try:
            result = validate_one(stem, onnx_path, rknn_path, pp_path)
            if result:
                results[stem] = result
        except Exception as e:
            print(f"\n  ERROR {stem}: {e}")
            import traceback
            traceback.print_exc()
            results[stem] = {"model": stem, "error": str(e), "pass": False}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Status':<8} {'max_diff':<10} {'match':<8} {'ONNX':<6} {'RKNN':<6}")
    print("-" * 60)
    for name, res in sorted(results.items()):
        status = "PASS" if res.get("pass") else "FAIL"
        md = res.get("max_diff_avg", "-")
        match = res.get("match_ratio", "-")
        o = res.get("n_objects_onnx", "-")
        r = res.get("n_objects_rknn", "-")
        print(f"{name:<20} {status:<8} {md:<10} {match:<8} {o:<6} {r:<6}")

    all_pass = all(r.get("pass") for r in results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
