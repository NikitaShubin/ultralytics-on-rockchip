"""DetectorFrame data class — the standard container for detection results."""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class DetectorFrame:
    """Unified result of one detection cycle: image, objects, and timing."""

    frame_id: int
    image: np.ndarray | None = None
    jpeg: bytes | None = None
    objects: list[dict] = field(default_factory=list)
    capture_ts: float = 0.0
    process_start_ts: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    source: str = ''
    fps: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since processing started."""
        return (
            (time.monotonic() - self.process_start_ts) * 1000
            if self.process_start_ts
            else 0.0
        )


def objects_to_api(
    objects: list[dict],
    mask_format: str = 'contour',
) -> list[dict]:
    """Convert internal objects to JSON-safe API objects.

    Handles:
      - ndarray masks → contour / base64 (segmentation models)
      - OBB (angle) → 4-point polygon polygon
      - plain detections → pass through

    Args:
        objects: list of detection dicts as returned by postprocess
        mask_format: 'none' to strip masks, 'contour' for polygon outlines,
                     'base64' for PNG-encoded masks

    Returns:
        list of dicts safe for json.dumps
    """
    result: list[dict] = []
    for obj in objects:
        entry = dict(obj)
        mask = entry.pop('mask', None)
        if mask is not None:
            if mask_format == 'contour':
                contour = _mask_to_contour(mask, entry.get('bbox'))
                if contour:
                    entry['polygon'] = contour
            elif mask_format == 'base64':
                png = _mask_to_png_base64(mask)
                if png:
                    entry['mask_png'] = png
        if entry.get('angle') is not None and 'polygon' not in entry:
            polygon = _obb_to_polygon(entry['bbox'], entry['angle'])
            if polygon:
                entry['polygon'] = polygon
        if entry.get('angle') is not None:
            cx, cy, w, h = entry['bbox'][:4]
            entry['bbox'] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        result.append(entry)
    return result


def _mask_to_contour(
    mask: np.ndarray,
    bbox: list[int] | None = None,
) -> list[float] | None:
    """Extract the largest external contour from a binary mask as flat [x1,y1,x2,y2,...]."""
    if mask.size <= 1:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    ox, oy = (bbox[:2] if bbox else (0, 0))
    flat: list[float] = []
    for pt in c[:, 0, :]:
        flat.append(float(pt[0] + ox))
        flat.append(float(pt[1] + oy))
    return flat if flat else None


def _obb_to_polygon(
    bbox: list[int] | list[float],
    angle: float,
) -> list[float] | None:
    """Convert OBB (cx, cy, w, h, angle) to a 4-point polygon as flat [x1,y1,x2,y2,...]."""
    if len(bbox) < 4:
        return None
    cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    wx = w / 2 * cos_a
    wy = w / 2 * sin_a
    hx = -h / 2 * sin_a
    hy = h / 2 * cos_a
    return [
        cx - wx - hx, cy - wy - hy,
        cx + wx - hx, cy + wy - hy,
        cx + wx + hx, cy + wy + hy,
        cx - wx + hx, cy - wy + hy,
    ]


def _mask_to_png_base64(mask: np.ndarray) -> str | None:
    """Encode a binary mask as PNG and return a base64 data URL."""
    if mask.size <= 1:
        return None
    ok, buf = cv2.imencode('.png', mask)
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf).decode('ascii')
