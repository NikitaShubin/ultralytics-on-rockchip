import numpy as np
import cv2

INPUT_SIZE = {{INPUT_SIZE}}
NUM_CLASSES = {{NUM_CLASSES}}
COCO_CLASSES = {{COCO_CLASSES}}

SEMANTIC_MODE = True

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)


def postprocess(outputs, scale, pad, orig_shape):
    orig_h, orig_w = orig_shape
    logits = outputs[0][0]
    if logits.ndim == 3:
        class_map = logits.argmax(axis=0).astype(np.int32)
    else:
        class_map = logits.argmax(axis=0).astype(np.int32)

    pad_left, pad_top = pad[0] // 2, pad[1] // 2
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    class_map = cv2.resize(
        class_map.astype(np.float32),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    class_map = class_map[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    class_map = cv2.resize(class_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return [{"class_map": class_map}]


def visualize(frame, objects):
    h, w = frame.shape[:2]
    if not objects or "class_map" not in objects[0]:
        return frame
    class_map = objects[0]["class_map"]
    if class_map.shape != (h, w):
        class_map = cv2.resize(class_map, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = (class_map >= 0) & (class_map < len(CITYSCAPES_PALETTE))
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[valid] = CITYSCAPES_PALETTE[class_map[valid].astype(int)]
    overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
    return overlay
