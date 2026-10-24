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
import numpy as np

from camera_source import CameraManager
from detector_frame import DetectorFrame
from tracker import Tracker

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
    _err_ts: float = 0.0
    _err_count: int = 0
    _trk_err_ts: float = 0.0
    _trk_err_count: int = 0
    _tracker: Tracker | None = None
    _track_sig: tuple | None = None

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
        # Трекер создаётся только при TRACKER != off: дефолт-деплой не меняется.
        self._tracker = (Tracker(cmc=self.config.tracker_cmc)
                         if self.config.tracker == 'on' else None)
        self._track_sig = None
        if self._tracker is not None:
            log.info('трекер включён (CMC=%s)', self.config.tracker_cmc)
            if self.config.tracker_reid == 'on':
                try:
                    from reid import ReidEngine
                    self._tracker.embedder = ReidEngine()
                    log.info('ReID: подключён (osnet, NPU-RKNN)')
                except Exception as exc:
                    log.error('ReID не подключен (работаем без него): %s', exc)
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
            except Exception as exc:  # ошибка постпроцесса не должна убивать весь поток
                self._err_count += 1
                now = time.monotonic()
                if now - self._err_ts > 5.0:
                    log.error('ошибка обработки кадра #%d (всего за окно: %d): %s',
                              self._frame_id, self._err_count, exc, exc_info=True)
                    self._err_ts = now
                objects = []
            self._update_tracks(objects, frame)
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

    def _update_tracks(self, objects: list, frame: 'cv2.MatLike | np.ndarray') -> None:
        """Ассоциация треков с текущими объектами (если TRACKER=on).

        Сброс трекера при: смене/переоткрытии источника, зацикливании
        видеофайла (склейка = скачок сцены), смене модели до reload'а.
        Ошибка трекера изолируется тем же образом, что и постпроцесс.
        """
        if self._tracker is None:
            return
        sig = (self._cam.name if self._cam else '',
               self._cam.loops if self._cam else 0,
               self.pipeline.model_name)
        if sig != self._track_sig:
            self._tracker.reset()
            self._track_sig = sig
        try:
            self._tracker.update(objects, frame)
        except Exception as exc:
            self._trk_err_count += 1
            now = time.monotonic()
            if now - self._trk_err_ts > 5.0:
                log.error('ошибка трекера кадра #%d (всего за окно: %d): %s',
                          self._frame_id, self._trk_err_count, exc, exc_info=True)
                self._trk_err_ts = now

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
