"""Async цикл захвата + детекции с уведомлением потребителей."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import cv2

from camera_source import CameraManager
from detector_frame import DetectorFrame

if TYPE_CHECKING:
    from config import Config
    from pipeline import Pipeline

log = logging.getLogger(__name__)


@dataclass
class VideoLoop:
    """Async цикл: захват → детекция → уведомление всех потребителей."""

    pipeline: Pipeline
    config: Config
    consumers: list[Callable[[DetectorFrame], None]] = field(default_factory=list)
    _cam: CameraManager | None = None
    _running: bool = False
    _frame_id: int = 0
    _fps_counter: list[float] = field(default_factory=list)
    _fps: float = 0.0
    _last_err_log: float = 0.0

    def register_consumer(self, cb: Callable[[DetectorFrame], None]) -> None:
        """Регистрация callback для получения каждого обработанного кадра."""
        self.consumers.append(cb)

    async def run(self) -> None:
        """Главный цикл: открыть камеру, захватить, обработать, уведомить."""
        self._cam = CameraManager(
            cam_mode=self.config.cam_mode.value,
            width=self.config.cam_width,
            height=self.config.cam_height,
            file_path=self.config.cam_source,
            resolution_mode=self.config.cam_resolution,
            target_size=self.pipeline.input_size,
        )
        self._running = True
        while self._running:
            if not self._cam.open():
                log.warning('источник изображения недоступен, повторная попытка...')
                await asyncio.sleep(1.0)
                continue
            await self._capture_loop()
        self._cleanup()

    async def _capture_loop(self) -> None:
        """Цикл захвата и обработки кадров."""
        if self._cam is None:
            return
        while self._running:
            frame = self._cam.read()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            capture_ts = time.monotonic()
            self._frame_id += 1
            process_start = time.monotonic()
            try:
                objects, _ = self.pipeline.detect(frame)
                if self.pipeline._semantic_mode and self.pipeline._visualize:
                    frame = self.pipeline._visualize(frame, objects)
            except Exception as e:  # noqa: BLE001 — ошибка модели не должна убивать видеопоток
                if time.monotonic() - self._last_err_log > 10:
                    self._last_err_log = time.monotonic()
                    log.exception('ошибка конвейера, кадр отдаётся без объектов')
                objects = []
            total_ms = (time.monotonic() - process_start) * 1000
            self._update_fps()

            _, jpeg = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
            )
            det_frame = DetectorFrame(
                frame_id=self._frame_id,
                jpeg=jpeg.tobytes(),
                objects=objects,
                capture_ts=capture_ts,
                process_start_ts=process_start,
                total_ms=total_ms,
                source=self._cam.name,
                fps=self._fps,
            )
            for cb in self.consumers:
                cb(det_frame)
            await asyncio.sleep(0)

    def _update_fps(self) -> None:
        """Обновление скользящего FPS (окно 2 секунды)."""
        now = time.monotonic()
        self._fps_counter.append(now)
        cutoff = now - 2.0
        self._fps_counter = [t for t in self._fps_counter if t > cutoff]
        self._fps = len(self._fps_counter) / 2.0 if len(self._fps_counter) > 1 else 0.0

    def stop(self) -> None:
        """Сигнал на остановку цикла."""
        self._running = False

    def _cleanup(self) -> None:
        """Очистка ресурсов (закрытие камеры)."""
        if self._cam is not None:
            self._cam.close()
