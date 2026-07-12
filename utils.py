"""Общие утилиты: letterbox, scale_coords, nms, dfl (обратная совместимость)."""

from __future__ import annotations

from percept.io import letterbox, scale_coords
from percept.perception import dfl, nms

__all__ = ['letterbox', 'scale_coords', 'nms', 'dfl']
