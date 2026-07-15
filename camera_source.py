"""Абстракция источника изображения: USB, файл и авто-определение."""

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
    """Создать заглушку при отсутствии камеры."""
    key = (w << 16) | h
    if key not in _NO_CAM_CACHE:
        img = np.full((h, w, 3), (114, 114, 114), dtype=np.uint8)
        cv2.putText(
            img,
            'Камера не подключена',
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
    """Абстрактный базовый класс источника изображения."""

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
    """USB камера через OpenCV VideoCapture."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._device: str = ''

    def open(self) -> bool:
        """Поиск и открытие доступной USB камеры."""
        for dev in sorted(Path('/dev').glob('video[0-9]*')):
            dev_str = str(dev)
            if self._try_device(dev_str):
                self._device = dev_str
                log.info('подключено к %s', dev_str)
                return True
        return False

    def _try_device(self, dev: str) -> bool:
        """Попытка открыть устройство и считать кадр."""
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
            log.debug('не удалось открыть %s: %s', dev, e)
        return False

    def read(self) -> np.ndarray | None:
        """Считать один кадр."""
        if self._cap is None:
            return None
        if self._device and not Path(self._device).exists():
            self.close()
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        """Закрыть камеру."""
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        self._device = ''

    @property
    def name(self) -> str:
        return f'usb:{self._device}' if self._device else 'usb:disconnected'


class FileSource(CameraSource):
    """Источник изображения из видеофайла."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        """Открыть видеофайл."""
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            return False
        log.info('открыт файл %s', self.path)
        return True

    def read(self) -> np.ndarray | None:
        """Считать кадр (с зацикливанием)."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        """Закрыть файл."""
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    @property
    def name(self) -> str:
        return f'file:{self.path}'


class GigECamera(CameraSource):
    """Basler GigE Vision камера через pypylon."""

    def __init__(self, ip: str = '') -> None:
        self.ip = ip
        self._camera = None
        self._converter = None
        self._device = ''

    def _set_auto(self, feature: str, value: str) -> None:
        """Безопасная установка параметра pylon, если камера поддерживает."""
        try:
            getattr(self._camera, feature).SetValue(value)
            log.debug('gige: %s = %s', feature, value)
        except Exception:
            log.debug('gige: %s недоступен, пропущено', feature)

    def _configure_auto(self) -> None:
        """Включение авто-экспозиции, авто-усиления и авто-баланса белого."""
        if self._camera is None:
            return
        self._set_auto('ExposureAuto', 'Continuous')
        self._set_auto('GainAuto', 'Continuous')
        self._set_auto('BalanceWhiteAuto', 'Continuous')
        self._set_auto('AutoExposureTimeAbsLowerLimit', 24.0)
        self._set_auto('AutoTargetValue', 100)
        log.info('gige: авто-экспозиция/усиление/ББ включены (limit=24µs, target=100)')

    def open(self) -> bool:
        """Найти и открыть GigE камеру."""
        from pypylon import pylon
        try:
            tl = pylon.TlFactory.GetInstance()
            devices = tl.EnumerateDevices()
            if not devices:
                log.warning('GigE устройства не найдены')
                return False
            info = devices[0]
            self._device = f'{info.GetFriendlyName()} ({info.GetIpAddress()})'
            log.info('найдена GigE камера: %s', self._device)
            dev = tl.CreateDevice(info)
            self._camera = pylon.InstantCamera(dev)
            self._camera.Open()
            self._configure_auto()
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            log.info('подключено к GigE камере: %s', self._device)
            return True
        except Exception as e:
            log.error('не удалось открыть GigE камеру: %s', e)
            self.close()
            return False

    def read(self) -> np.ndarray | None:
        """Считать кадр с GigE камеры."""
        if self._camera is None or not self._camera.IsGrabbing():
            return None
        try:
            result = self._camera.RetrieveResult(3000)
            if result is None:
                return None
            if result.IsValid():
                image = self._converter.Convert(result)
                frame = image.GetArray()
                result.Release()
                return frame
            result.Release()
        except Exception as e:
            log.error('ошибка чтения GigE: %s', e)
            self.close()
        return None

    def close(self) -> None:
        """Закрыть GigE камеру."""
        if self._camera is not None:
            try:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                self._camera.Close()
            except Exception:
                pass
            self._camera = None
        self._converter = None
        self._device = ''

    @property
    def name(self) -> str:
        return f'gige:{self._device}' if self._device else 'gige:disconnected'


class CameraManager:
    """Менеджер камер: авто-определение и переподключение."""

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
        self._skip_gige = False

    def open(self) -> bool:
        """Открыть источник изображения по выбранному режиму."""
        if self.cam_mode == 'file' and self.file_path:
            self._source = FileSource(self.file_path)
            if self._source.open():
                return True

        if self.cam_mode in ('auto', 'ip') and not self._skip_gige:
            source = GigECamera()
            if source.open():
                self._source = source
                self._skip_gige = False
                return True
            if self.cam_mode == 'ip':
                return False
            self._skip_gige = True

        if self.cam_mode in ('auto', 'usb'):
            source = USBCamera(self.width, self.height)
            if source.open():
                self._source = source
                self._skip_gige = False
                return True

        if self.file_path:
            self._source = FileSource(self.file_path)
            self._skip_gige = False
            return self._source.open()

        return False

    def read(self) -> np.ndarray | None:
        """Считать кадр с повторным подключением при ошибках."""
        if self._source is None:
            self._fallback_count += 1
            if self._fallback_count >= self._max_fallback:
                self._reconnect()
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
        """Переподключение к источнику изображения."""
        self.close()
        time.sleep(0.5)
        self.open()

    def close(self) -> None:
        """Закрыть источник изображения."""
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def name(self) -> str:
        return self._source.name if self._source else 'disconnected'
