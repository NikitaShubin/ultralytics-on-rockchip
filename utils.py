"""Shared utilities for image preprocessing and postprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and pad image to new_size with aspect ratio preservation."""
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return canvas, scale, (pad_w, pad_h)


def scale_coords(  # noqa: PLR0913
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    pad: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale box coordinates from model space back to original image space."""
    pad_w, pad_h = pad
    x1 = (x1 - pad_w / 2) / scale
    y1 = (y1 - pad_h / 2) / scale
    x2 = (x2 - pad_w / 2) / scale
    y2 = (y2 - pad_h / 2) / scale
    return x1, y1, x2, y2


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
) -> list[int]:
    """Non-maximum suppression: return indices of kept boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def dfl(position: np.ndarray, bin_count: int = 4) -> np.ndarray:
    """Distribution Focal Loss: softmax over bins then weighted sum."""
    n, c, h, w = position.shape
    mc = c // bin_count
    y = position.reshape(n, bin_count, mc, h, w)
    y_softmax = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax /= y_softmax.sum(axis=2, keepdims=True)
    proj = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * proj).sum(axis=2)
