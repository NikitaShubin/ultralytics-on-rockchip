"""Загрузка модели и инференс (RKNN/ONNX)."""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np

from percept.io import letterbox

log = logging.getLogger(__name__)

NPU_TEMP_PATH = Path('/sys/class/thermal/thermal_zone6/temp')
TEMP_WARN = 85
TEMP_INTERVAL = 30.0


class ModelBackend:
    """Абстракция бэкенда: RKNN или ONNX CPU."""

    def __init__(self, model_dir: str, model_name: str, backend: str = 'npu') -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name or 'yolo11n'
        self.backend = backend
        self.rknn: Any = None
        self._session: Any = None
        self._input_name: str = ''
        self.input_size = 640
        self._temp_warned = False
        self._last_temp_check = 0.0
        self._infer_lock = RLock()
        self._load()

    def _load(self) -> None:
        """Загрузка модели в зависимости от бэкенда."""
        if self.backend == 'cpu':
            self._load_onnx()
        else:
            self._load_rknn()

    def _load_rknn(self) -> None:
        """Загрузка RKNN модели."""
        from rknnlite.api import RKNNLite

        model_path = self.model_dir / f'{self.model_name}.rknn'
        if not model_path.exists():
            model_path = self.model_dir / 'model.rknn'
        if not model_path.exists():
            raise RuntimeError(f'Mодель RKNN не найдена: {model_path}')
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f'Не удалось загрузить модель RKNN: {ret}')
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f'Не удалось инициализировать runtime: {ret}')
        log.info('модель загружена (RKNN) модель=%s', self.model_name)

    def _load_onnx(self) -> None:
        """Загрузка ONNX модели (CPU)."""
        import onnxruntime as rt

        model_path = self.model_dir / f'{self.model_name}.onnx'
        if not model_path.exists():
            model_path = self.model_dir / 'model.onnx'
        if not model_path.exists():
            raise RuntimeError(f'Модель ONNX не найдена в {model_path}')
        self._session = rt.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider'],
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info('модель загружена (ONNX CPU) модель=%s', self.model_name)

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Запуск инференса на бэкенде."""
        if self.backend == 'cpu':
            return self._session.run(None, {self._input_name: tensor / 255.0})
        nhwc = tensor.transpose(0, 2, 3, 1)
        return self.rknn.inference(inputs=[nhwc])

    def reload(self, model_name: str = '', backend: str | None = None) -> None:
        """Перезагрузка модели."""
        if model_name:
            self.model_name = model_name
        if backend is not None:
            self.backend = backend
        log.info('перезагрузка модель=%s бэкенд=%s...', self.model_name, self.backend)
        with self._infer_lock:
            if self.rknn:
                with suppress(Exception):
                    self.rknn.release()
                self.rknn = None
            self._session = None
            self._load()
        log.info('перезагрузка завершена (модель=%s)', self.model_name)

    def check_temp(self) -> int | None:
        """Проверка температуры NPU (throttled)."""
        now = time.monotonic()
        if now - self._last_temp_check < TEMP_INTERVAL:
            return None
        self._last_temp_check = now
        try:
            raw = NPU_TEMP_PATH.read_text().strip()
            temp = int(raw) // 1000
        except OSError:
            return None
        if temp >= TEMP_WARN and not self._temp_warned:
            log.warning('температура NPU=%s°C (>=%s)', temp, TEMP_WARN)
            self._temp_warned = True
        return temp

    def detect(
        self,
        frame: np.ndarray,
        postprocess_fn,
        custom_preprocess=None,
    ) -> tuple[list[dict], dict]:
        """Запуск инференса + постпроцесс. Возвращает (объекты, метаданные)."""
        self.check_temp()
        h, w = frame.shape[:2]
        with self._infer_lock:
            if custom_preprocess:
                tensor, scale, pad = custom_preprocess(frame)
            else:
                canvas, scale, pad = letterbox(frame, self.input_size)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
                tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
            try:
                outputs = self.infer(tensor)
            except Exception:
                log.exception('ошибка инференса')
                return [], {}
        objects = postprocess_fn(outputs, scale, pad, (h, w)) if postprocess_fn else []
        metadata = {
            'orig_size': (h, w),
            'input_size': self.input_size,
        }
        return objects, metadata
