"""Абстракция источника изображения: USB, файл и авто-определение."""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_NO_CAM_CACHE: dict[int, bytes] = {}

# Типичные разрешения USB-камер (от больших к меньшим) — перебираются при
# выборе формата захвата. Реальный набор определяется возможностями камеры.
CAMERA_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (3840, 2160), (1920, 1080), (1600, 1200), (1280, 720), (1024, 768),
    (960, 720), (800, 600), (720, 480), (640, 480), (320, 240), (176, 144),
    (160, 120),
)

# Форматы пикселей в порядке проверки. MJPG — предпочтительный: меньше
# USB-трафика, работает у камер в «урезанном» режиме после сбоя USB.
_CANDIDATE_FOURCC: tuple[tuple[str, int], ...] = (
    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
)


def _device_inode(dev: str) -> int | None:
    """inode узла устройства; меняется при USB-переподключении камеры."""
    try:
        return os.stat(dev).st_ino
    except OSError:
        return None


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
    """USB камера через OpenCV VideoCapture с фоновым захватом.

    Кадры читаются в фоновом потоке. Основной поток получает последний
    готовый кадр мгновенно. USB glitch от NPU inference (DMA ~50ms)
    обрабатывается в фоне и не блокирует pipeline.

    Разрешение захвата:
      - явное (width/height) — используется как есть;
      - 'model'  — рабочее разрешение модели (target_size = input_size);
      - 'source' — максимальное разрешение, которое отдаёт камера.
    Фактический формат проверяется после открытия: камера может вернуть
    другой размер (например, при сбое USB — 176x144 вместо запрошенного).
    """

    def __init__(
        self,
        width: int = 0,
        height: int = 0,
        resolution_mode: str = 'model',
        target_size: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution_mode = resolution_mode
        self.target_size = target_size
        self._format: tuple[int, int] | None = None
        self._fourcc: str = ''
        self._desired: tuple[int, int] | None = None
        self._fmt_attempts = 0
        self._cap: cv2.VideoCapture | None = None
        self._device: str = ''
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame: np.ndarray | None = None
        self._frame_id: int = 0
        self._first_frame = threading.Event()
        self._dev_inode: int | None = None
        self._last_reopen: float = 0.0
        self._read_errors = 0
        self._watchdog: threading.Thread | None = None

    def open(self) -> bool:
        """Поиск и открытие доступной USB камеры."""
        for dev in sorted(Path('/dev').glob('video[0-9]*')):
            dev_str = str(dev)
            if self._try_device(dev_str):
                self._device = dev_str
                self._start_thread()
                log.info('подключено к %s (фоновый захват) формат=%sx%s',
                         dev_str, self._format[0] if self._format else '?',
                         self._format[1] if self._format else '?')
                return True
        return False

    def _candidates(self) -> list[tuple[int, int]]:
        """Список разрешений-кандидатов в порядке убывания приоритета."""
        if self.width > 0 and self.height > 0:
            return [(self.width, self.height)]
        if self.resolution_mode == 'source':
            return list(CAMERA_RESOLUTIONS)
        target = self.target_size or 640
        exact = [r for r in CAMERA_RESOLUTIONS if max(r) == target]
        # Если рабочего разрешения нет — берём ближайшее БОЛЬШЕЕ (кадр потом
        # приведётся к input_size моделью, качество не теряется).
        bigger = [r for r in CAMERA_RESOLUTIONS if max(r) > target]
        smaller = [r for r in CAMERA_RESOLUTIONS if max(r) < target]
        return exact + list(reversed(bigger)) + smaller

    def _desired_format(self) -> tuple[int, int] | None:
        """Желаемый формат захвата — для контроля watchdog.

        В режиме 'source' формат не контролируем (камера отдаёт свой максимум),
        поэтому None. В режиме 'model' — рабочее разрешение модели.
        """
        if self.resolution_mode == 'source':
            return None
        cands = self._candidates()
        return cands[0] if cands else None

    def _select_format(self, cap: cv2.VideoCapture) -> tuple[int, int, str] | None:
        """Подобрать поддерживаемый камерой формат (fourcc + размер).

        MJPG проверяется первым: он есть у большинства веб-камер, требует
        втрое меньше USB-трафика и работает даже когда камера не смогла
        переключиться на YUYV (после сбоя USB остаётся минимальный формат).
        """
        for name, fourcc in _CANDIDATE_FOURCC:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            for w, h in self._candidates():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if aw == w and ah == h:
                    return aw, ah, name
        return None

    def _current_format(self, cap: cv2.VideoCapture) -> tuple[int, int, str]:
        """Фактический формат камеры (используется, если ничего не применилось)."""
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        name = ''.join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip('\x00 ')
        return w, h, name

    def _try_device(self, dev: str) -> bool:
        """Попытка открыть устройство и считать кадр."""
        try:
            cap = cv2.VideoCapture(dev)
            fmt = self._select_format(cap)
            if fmt is None:
                # Ни один желаемый формат не применился (камера в урезанном
                # режиме) — принимаем фактический, но предупреждаем.
                fmt = self._current_format(cap)
                log.warning('%s: желаемый формат недоступен, берём фактический %sx%s (%s)',
                            dev, fmt[0], fmt[1], fmt[2] or '?')
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                self._cap = cap
                self._format = (fmt[0], fmt[1])
                self._fourcc = fmt[2]
                self._desired = self._desired_format()
                self._dev_inode = _device_inode(dev)
                return True
            cap.release()
        except OSError as e:
            log.debug('не удалось открыть %s: %s', dev, e)
        return False

    def _reopen_if_replugged(self) -> None:
        """Переоткрыть камеру, если она отвалилась (USB-переподключение меняет inode).

        После переподключения старый файловый дескриптор даёт ENODEV, и кадры
        застывают, пока контейнер не перезапустить.
        """
        if not self._device:
            return
        now = time.monotonic()
        if now - self._last_reopen < 2.0:
            return
        self._last_reopen = now
        inode = _device_inode(self._device)
        if inode is None or inode == self._dev_inode:
            return  # устройство не переподключалось — нечего переоткрывать
        log.warning(
            'камера %s переподключена (inode %s → %s), переоткрываем',
            self._device, self._dev_inode, inode,
        )
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            cap.release()
            return
        fmt = self._select_format(cap)
        if fmt is None:
            fmt = self._current_format(cap)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = cap
            self._format = (fmt[0], fmt[1])
            self._fourcc = fmt[2]
        self._dev_inode = inode
        self._read_errors = 0
        log.info('камера %s переоткрыта, формат %sx%s (%s)',
                 self._device, fmt[0], fmt[1], fmt[2] or '?')

    def _start_thread(self) -> None:
        """Запуск фонового потока захвата и сторожа."""
        self._running = True
        self._first_frame.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name='cam-capture')
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name='cam-watchdog')
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        """Сторож: если кадры перестали идти — переоткрыть камеру.

        После ребута или USB-переподключения V4L2 может вернуть ENODEV: буферы
        не выделяются, и read() залипает навсегда (кадр застывает, хотя камера
        давно готова). Обнаруживаем это по счётчику кадров и переоткрываем.
        """
        last_seen = -1
        stuck_since = time.monotonic()
        while self._running:
            time.sleep(1.0)
            with self._lock:
                frame_id = self._frame_id
                cap = self._cap
                fmt = self._format
            now = time.monotonic()
            if cap is None:
                # Камера была отключена: если устройство вернулось — открыть заново
                if self._device and Path(self._device).exists() and now - self._last_reopen > 5.0:
                    self._last_reopen = now
                    log.info('камера %s вернулась, открываем заново', self._device)
                    self._reopen_camera()
                continue
            if fmt is not None and now - self._last_reopen > 60.0:
                cur = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                desired = self._desired
                if (desired and cur != desired
                        and cur[0] * cur[1] < desired[0] * desired[1]):
                    self._last_reopen = now
                    self._fmt_attempts += 1
                    if self._fmt_attempts <= 5:
                        log.warning('камера отдаёт %sx%s вместо %sx%s — переоткрываем',
                                    cur[0], cur[1], desired[0], desired[1])
                        self._reopen_camera()
                        continue
                    if self._fmt_attempts == 6:
                        log.warning('формат %sx%s недоступен для камеры, оставляем как есть',
                                    cur[0], cur[1])
            if frame_id != last_seen:
                last_seen = frame_id
                stuck_since = now
                continue
            if now - stuck_since < 5.0:
                continue
            if now - self._last_reopen < 10.0:
                continue
            stuck_since = now
            self._last_reopen = now
            log.warning('захват камеры завис (кадры не обновляются 5 с) — переоткрываем %s',
                        self._device)
            self._reopen_camera()

    def _reopen_camera(self) -> None:
        """Переоткрыть устройство камеры и перезапустить поток захвата."""
        dev = self._device
        if not dev:
            return
        with self._lock:
            old = self._cap
            self._cap = None
        if old is not None:
            try:
                old.release()
            except Exception:  # noqa: BLE001 — ошибка закрытия не должна ронять поток
                pass
        try:
            cap = cv2.VideoCapture(dev)
        except Exception:  # noqa: BLE001
            return
        if not cap.isOpened():
            cap.release()
            return
        fmt = self._select_format(cap)
        if fmt is None:
            fmt = self._current_format(cap)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with self._lock:
            self._cap = cap
            self._format = (fmt[0], fmt[1])
            self._fourcc = fmt[2]
        self._dev_inode = _device_inode(dev)
        self._read_errors = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name='cam-capture')
        self._thread.start()
        log.info('камера %s переоткрыта, формат %sx%s (%s)', dev, fmt[0], fmt[1], fmt[2] or '?')

    def _capture_loop(self) -> None:
        """Фоновый поток: непрерывно читает кадры с камеры."""
        while self._running:
            cap = self._cap
            if cap is None:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if ret:
                self._read_errors = 0
                with self._lock:
                    if self._cap is not cap:
                        return  # устройство переоткрыто — этот поток больше не нужен
                    self._frame = frame
                    self._frame_id += 1
                self._first_frame.set()
                continue
            # Чтение не удалось: камера могла отвалиться (USB-переподключение)
            self._read_errors += 1
            if self._read_errors in (10, 100):
                log.warning('кадры не читаются (%s), ошибок: %d',
                            self._device, self._read_errors)
            self._reopen_if_replugged()
            if self._device and not Path(self._device).exists():
                # Устройство исчезло (USB-отвал): отдаём пустой кадр, чтобы
                # веб показал заглушку «Камера не подключена», и ждём возврата.
                with self._lock:
                    if self._cap is not None:
                        self._cap.release()
                    self._cap = None
                    self._frame = None
                log.warning('камера %s отключена, ждём её возврата', self._device)
                time.sleep(1.0)
                continue
            time.sleep(0.05)
            if self._cap is not None:
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                        self._frame_id += 1
                    self._first_frame.set()

    def read(self) -> np.ndarray | None:
        """Вернуть последний захваченный кадр (неблокирующе).

        При первом вызове блокирует до получения первого кадра.
        """
        if not self._running:
            return None
        self._first_frame.wait(timeout=2.0)
        with self._lock:
            return self._frame

    def close(self) -> None:
        """Остановить фоновый поток и закрыть камеру."""
        self._running = False
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
            self._watchdog = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        self._device = ''

    @property
    def name(self) -> str:
        if not self._device or not Path(self._device).exists():
            return 'usb:disconnected'
        return f'usb:{self._device}'


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
        width: int = 0,
        height: int = 0,
        file_path: str = '',
        resolution_mode: str = 'model',
        target_size: int = 0,
    ) -> None:
        self.cam_mode = cam_mode
        self.width = width
        self.height = height
        self.file_path = file_path
        self.resolution_mode = resolution_mode
        self.target_size = target_size
        self._source: CameraSource | None = None
        self._fallback_count = 0
        self._max_fallback = 2
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
            source = USBCamera(
                self.width,
                self.height,
                resolution_mode=self.resolution_mode,
                target_size=self.target_size,
            )
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
        time.sleep(0.2)
        self.open()

    def close(self) -> None:
        """Закрыть источник изображения."""
        if self._source is not None:
            self._source.close()
            self._source = None

    @property
    def name(self) -> str:
        return self._source.name if self._source else 'disconnected'
