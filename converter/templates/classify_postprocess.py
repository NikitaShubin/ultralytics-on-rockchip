import numpy as np
import cv2

INPUT_SIZE = {{INPUT_SIZE}}
NUM_CLASSES = {{NUM_CLASSES}}
COCO_CLASSES = {{COCO_CLASSES}}


def preprocess(frame):
    """CenterCrop + resize to INPUT_SIZE. Returns (tensor, scale=1.0, pad=(0,0))."""
    h, w = frame.shape[:2]
    s = min(h, w)
    y1 = (h - s) // 2
    x1 = (w - s) // 2
    crop = frame[y1:y1 + s, x1:x1 + s]
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_LINEAR)
    tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
    return tensor, 1.0, (0, 0)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def postprocess(outputs, scale, pad, orig_shape):
    """Decode RKNN classify output into top-K predictions.

    Args:
        outputs: list with single tensor [1, NUM_CLASSES]
        scale, pad: ignored (no box decoding for classify)
        orig_shape: tuple (orig_h, orig_w)

    Returns:
        list of dicts: [{"bbox": [0,0,w,h], "score": float, "class_id": int, "class_name": str}, ...]
    """
    orig_h, orig_w = orig_shape
    logits = outputs[0][0]
    probs = softmax(logits)
    top5 = np.argsort(-probs)[:5]
    results = []
    for c in top5:
        results.append({
            "bbox": [0, 0, orig_w, orig_h],
            "score": float(probs[c]),
            "class_id": int(c),
            "class_name": COCO_CLASSES[c] if c < len(COCO_CLASSES) else str(c),
        })
    return results


def visualize(frame, objects):
    """Draw classification results on frame.

    Args:
        frame: numpy array (H, W, 3) BGR
        objects: list of dicts from postprocess()

    Returns:
        numpy array with drawn labels
    """
    h, w = frame.shape[:2]
    s = max(h, w) / 1000
    fs = max(0.8, s * 1.2)
    fthick = max(2, int(s * 2))
    for i, obj in enumerate(objects[:3]):
        label = f"{obj['class_name']}: {obj['score']:.2f}"
        y_pos = 30 + i * int(fs * 40)
        cv2.putText(frame, label, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), fthick)
    return frame
