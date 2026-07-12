"""Ввод/вывод: letterbox, scale_coords."""

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


def scale_coords(
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
