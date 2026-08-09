import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose

INPUT_SIZE = {{INPUT_SIZE}}
NUM_CLASSES = {{NUM_CLASSES}}
COCO_CLASSES = {{COCO_CLASSES}}

# Точная копия ultralytics classify_transforms: Resize(224, bilinear) +
# CenterCrop(224) + ToTensor(/255). НЕ letterbox!
_transforms = Compose([
    Resize(INPUT_SIZE, interpolation=2),
    CenterCrop(INPUT_SIZE),
    ToTensor(),
])


def preprocess(frame):
    """Resize короткой стороны до INPUT_SIZE + CenterCrop + /255.

    Matches ultralytics classify_transforms exactly.
    Returns (tensor [0,1], scale, pad) — scale/pad dummy для cls.
    """
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = _transforms(pil).unsqueeze(0).numpy()
    return tensor, 1.0, (0, 0)


def postprocess(outputs, scale, pad, orig_shape):
    """Classify: top-1 предсказание (score = сырое значение модели).

    Args:
        outputs: list с одним тензором [1, NUM_CLASSES]
        scale, pad: игнорируются (нет бокс-декодинга для cls)
        orig_shape: tuple (orig_h, orig_w)

    Returns:
        list из одного dict: {"bbox", "score", "class_id", "class_name"}
    """
    pred = outputs[0]
    if pred.ndim == 2:
        pred = pred[0]
    elif pred.ndim == 3:
        pred = pred[0, 0]

    orig_h, orig_w = orig_shape
    cls_id = int(np.argmax(pred))
    score = float(pred[cls_id])

    return [{
        "bbox": [0, 0, orig_w, orig_h],
        "score": score,
        "class_id": cls_id,
        "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
    }]


def visualize(frame, objects):
    """Draw classification results on frame."""
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
