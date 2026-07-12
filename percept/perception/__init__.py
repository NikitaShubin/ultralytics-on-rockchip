"""Постпроцесс: DFL, NMS, decode heads."""

from __future__ import annotations

import numpy as np


def dfl(position: np.ndarray, bin_count: int = 4) -> np.ndarray:
    """Distribution Focal Loss: softmax over bins then weighted sum."""
    n, c, h, w = position.shape
    mc = c // bin_count
    y = position.reshape(n, bin_count, mc, h, w)
    y_softmax = np.exp(y - y.max(axis=2, keepdims=True))
    y_softmax /= y_softmax.sum(axis=2, keepdims=True)
    proj = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y_softmax * proj).sum(axis=2)


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
