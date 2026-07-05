import random
import time
import cv2
import numpy as np
from pathlib import Path
from pipeline import COCO_CLASSES

COCO8_DIR = Path("/app/coco8")
FRAME_H, FRAME_W = 480, 640
FONT = cv2.FONT_HERSHEY_SIMPLEX


class TestObject:
    def __init__(self, pixels, class_id, class_name):
        self.pixels = pixels
        self.class_id = class_id
        self.class_name = class_name


def load_coco8_objects():
    objects = []
    for split in ["train", "val"]:
        img_dir = COCO8_DIR / "images" / split
        label_dir = COCO8_DIR / "labels" / split
        if not img_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            img_path = img_dir / (label_path.stem + ".jpg")
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, nw, nh = map(float, parts[1:])
                    x1 = int((cx - nw / 2) * w_img)
                    y1 = int((cy - nh / 2) * h_img)
                    x2 = int((cx + nw / 2) * w_img)
                    y2 = int((cy + nh / 2) * h_img)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_img, x2)
                    y2 = min(h_img, y2)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    pixels = img[y1:y2, x1:x2]
                    class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "?"
                    objects.append(TestObject(pixels, cls_id, class_name))
    return objects


def paste_object(canvas, obj, x, y, w, h):
    resized = cv2.resize(obj.pixels, (w, h), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas.shape[1], x + w)
    y2 = min(canvas.shape[0], y + h)
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return None
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    canvas[y1:y2, x1:x2] = resized[src_y1:src_y1 + ph, src_x1:src_x1 + pw]
    return (x1, y1, x2, y2)


def detect(pipeline, frame):
    h, w = frame.shape[:2]
    input_tensor, scale, pad = pipeline._preprocess(frame)
    outputs = pipeline.rknn.inference(inputs=[input_tensor])
    boxes, scores, class_ids = pipeline._postprocess(outputs, scale, pad, (w, h))
    return boxes, scores, class_ids


def generate_test_frames(pipeline, objects):
    while True:
        canvas = np.full((FRAME_H, FRAME_W, 3), 114, dtype=np.uint8)

        obj = random.choice(objects)

        obj_h, obj_w = obj.pixels.shape[:2]
        obj_aspect = obj_w / max(obj_h, 1)

        max_size = min(FRAME_W // 2, FRAME_H // 2)
        min_size = 30
        size = random.randint(min_size, max_size)

        if obj_aspect > 1.5:
            w = size
            h = int(size / obj_aspect)
        elif obj_aspect < 0.67:
            h = size
            w = int(size * obj_aspect)
        else:
            w = size
            h = size

        w = max(min_size, min(w, FRAME_W // 2))
        h = max(min_size, min(h, FRAME_H // 2))

        x = random.randint(0, FRAME_W - w)
        y = random.randint(0, FRAME_H - h)

        gt = paste_object(canvas, obj, x, y, w, h)
        if gt is None:
            continue

        gt_x1, gt_y1, gt_x2, gt_y2 = gt
        gt_w = gt_x2 - gt_x1
        gt_h = gt_y2 - gt_y1
        gt_cx = (gt_x1 + gt_x2) / 2
        gt_cy = (gt_y1 + gt_y2) / 2
        gt_aspect = gt_w / max(gt_h, 1)

        boxes, scores, class_ids = detect(pipeline, canvas)

        display = canvas.copy()

        cv2.rectangle(display, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)
        cv2.putText(display, f"GT: {obj.class_name} {gt_w}x{gt_h} ar={gt_aspect:.2f}",
                    (gt_x1, max(gt_y1 - 5, 12)), FONT, 0.4, (0, 255, 0), 1)

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            det_w = x2 - x1
            det_h = y2 - y1
            det_cx = (x1 + x2) / 2
            det_cy = (y1 + y2) / 2
            det_aspect = det_w / max(det_h, 1)

            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "?"

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(display, (int(det_cx), int(det_cy)),
                     (int(gt_cx), int(gt_cy)), (0, 255, 255), 1)

            w_ratio = det_w / max(gt_w, 1)
            h_ratio = det_h / max(gt_h, 1)
            ar_match = det_aspect / max(gt_aspect, 0.01)
            dx = det_cx - gt_cx
            dy = det_cy - gt_cy

            info = (
                f"{cls_name}: {score:.2f} "
                f"w={w_ratio:.2f}x h={h_ratio:.2f}x "
                f"ar={ar_match:.2f} "
                f"dx={dx:.0f} dy={dy:.0f}"
            )
            cv2.putText(display, info, (x1, min(y2 + 12, FRAME_H - 4)),
                        FONT, 0.4, (0, 0, 255), 1)

        info = f"obj={obj.class_name} gt={gt_w}x{gt_h} dets={len(boxes)}"
        cv2.putText(display, info, (10, FRAME_H - 10),
                    FONT, 0.5, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

        time.sleep(2)
