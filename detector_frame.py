"""DetectorFrame — стандартный контейнер результатов детекции."""

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
    """Единый результат одного цикла детекции: изображение, объекты и тайминги."""

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
        """Миллисекунды с момента начала обработки."""
        return (
            (time.monotonic() - self.process_start_ts) * 1000
            if self.process_start_ts
            else 0.0
        )


def objects_to_api(
    objects: list[dict],
    mask_format: str = 'contour',
) -> list[dict]:
    """Преобразование внутренних объектов в JSON-безопасные API объекты.

    Обрабатывает:
      - ndarray маски → контур / base64 (модели сегментации)
      - OBB (angle) → 4-точечный полигон
      - обычные детекции → без изменений

    Аргументы:
        objects: список словарей детекций из постпроцесса
        mask_format: 'none' для удаления масок, 'contour' для полигонов,
                     'base64' для PNG-кодированных масок

    Возвращает:
        список словарей, безопасных для json.dumps
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
    """Извлечь наибольший внешний контур из бинарной маски как плоский [x1,y1,x2,y2,...]."""
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
    """Преобразовать OBB (cx, cy, w, h, angle) в 4-точечный полигон как [x1,y1,x2,y2,...]."""
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
    """Кодировать бинарную маску как PNG и вернуть base64 data URL."""
    if mask.size <= 1:
        return None
    ok, buf = cv2.imencode('.png', mask)
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf).decode('ascii')
