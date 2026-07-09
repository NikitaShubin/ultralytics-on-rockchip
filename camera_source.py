"""Camera source abstraction: USB, file, and auto-detection."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_NO_CAM_CACHE: dict[int, bytes] = {}


def _make_no_cam_placeholder(w: int = 640, h: int = 480) -> bytes:
    key = (w << 16) | h
    if key not in _NO_CAM_CACHE:
        img = np.full((h, w, 3), (114, 114, 114), dtype=np.uint8)
        cv2.putText(
            img,
            'Camera not connected',
            (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 50, 50),
            2,
        )
        _ret, buf = cv2.imencode('.jpg', img)
        _NO_CAM_CACHE[key] = buf.tobytes() if _ret else b''
    return _NO_CAM_CACHE[key]


class CameraSource(ABC):
    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> np.ndarray | None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class USBCamera(CameraSource):
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._device: str = ''

    def open(self) -> bool:
        for dev in sorted(Path('/dev').glob('video[0-9]*')):
            dev_str = str(dev)
            if self._try_device(dev_str):
                self._device = dev_str
                log.info('connected to %s', dev_str)
                return True
        return False

    def _try_device(self, dev: str) -> bool:
        try:
            cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                self._cap = cap
                return True
            cap.release()
        except OSError as e:
            log.debug('failed to open %s: %s', dev, e)
        return False

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        if self._device and not Path(self._device).exists():
            self.close()
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        self._device = ''

    @property
    def name(self) -> str:
        return f'usb:{self._device}' if self._device else 'usb:disconnected'


class FileSource(CameraSource):
    def __init__(self, path: str) -> None:
        self.path = path
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            return False
        log.info('opened file %s', self.path)
        return True

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    @property
    def name(self) -> str:
        return f'file:{self.path}'


class CameraManager:
    def __init__(
        self,
        cam_mode: str = 'auto',
        width: int = 640,
        height: int = 480,
        file_path: str = '',
    ) -> None:
        self.cam_mode = cam_mode
        self.width = width
        self.height = height
        self.file_path = file_path
        self._source: CameraSource | None = None
        self._fallback_count = 0
        self._max_fallback = 3

    def open(self) -> bool:
        if self.cam_mode == 'file' and self.file_path:
            self._source = FileSource(self.file_path)
            if self._source.open():
                return True
        source = USBCamera(self.width, self.height)
        if source.open():
            self._source = source
            return True
        if self.file_path:
            self._source = FileSource(self.file_path)
            return self._source.open()
        return False

    def read(self) -> np.ndarray | None:
        if self._source is None:
            return None
        frame = self._source.read()
        if frame is not None:
            self._fallback_count = 0
            return frame
        self._fallback_count += 1
        if self._fallback_count >= self._max_fallback:
            self._reconnect()
        return None

    def _reconnect(self) -> None:
        self.close()
        time.sleep(0.5)
        self.open()

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def name(self) -> str:
        return self._source.name if self._source else 'disconnected'
