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

    def __init__(self, model_dir: str, model_name: str, backend: str = 'npu',
                 rknn_suffix: str = '') -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name or 'yolo11n'
        self.backend = backend
        self.rknn_suffix = rknn_suffix
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
        """Загрузка RKNN модели.

        RKNNLite на NPU-железе (ARM); если rknnlite недоступен (x86 ПК) —
        fallback на rknn-toolkit2 (эмулятор), чтобы eval/validate можно было
        гонять и без ARM.
        """
        model_path = self.model_dir / f'{self.model_name}{self.rknn_suffix}.rknn'
        if not model_path.exists():
            model_path = self.model_dir / f'{self.model_name}.rknn'
        if not model_path.exists():
            model_path = self.model_dir / 'model.rknn'
        if not model_path.exists():
            raise RuntimeError(f'Модель RKNN не найдена: {model_path}')

        # Импорт rknnlite/rknn-toolkit2 перезаписывает logging._nameToLevel/_levelToName
        # (уровни → односимвольные 'C','E','W','I','D'), что ломает последующий
        # импорт torch (Unknown level: 'WARNING'). Сохраняем стандартные таблицы
        # ДО импорта RKNN и восстанавливаем после всех его вызовов.
        import logging as _logging
        _name_to_level = dict(_logging._nameToLevel)
        _level_to_name = dict(_logging._levelToName)

        try:
            from rknnlite.api import RKNNLite
            self._lite = True
        except ImportError:
            from rknn.api import RKNN
            self._lite = False

        if self._lite:
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(str(model_path))
            if ret != 0:
                raise RuntimeError(f'Не удалось загрузить модель RKNN: {ret}')
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        else:
            self.rknn = RKNN(verbose=False)
            ret = self.rknn.load_rknn(str(model_path))
            if ret == 0:
                ret = self.rknn.init_runtime()
            if ret != 0:
                # rknn-toolkit2 не умеет инферить load_rknn-модель на симуляторе
                # без target → пересобираем из ONNX (как validate_bench).
                onnx_path = self.model_dir / f'{self.model_name}.onnx'
                if not onnx_path.exists():
                    onnx_path = self.model_dir / 'model.onnx'
                if not onnx_path.exists():
                    raise RuntimeError(f'Нет ONNX для пересборки симулятора: {onnx_path}')
                log.warning('init_runtime вернул %s, пересборка ONNX для симулятора', ret)
                self.rknn = RKNN(verbose=False)
                self.rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                                 target_platform='rk3588')
                self.rknn.load_onnx(model=str(onnx_path))
                self.rknn.build(do_quantization=False)
                ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f'Не удалось инициализировать runtime: {ret}')
        _logging._nameToLevel.clear()
        _logging._nameToLevel.update(_name_to_level)
        _logging._levelToName.clear()
        _logging._levelToName.update(_level_to_name)
        self._detect_input_size_from_onnx()
        log.info('модель загружена (RKNN%s) модель=%s суффикс=%s input_size=%s',
                 '' if self._lite else ' x86-эмулятор',
                 self.model_name, self.rknn_suffix, self.input_size)

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
        shape = self._session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int):
            self.input_size = shape[2]
        log.info('модель загружена (ONNX CPU) модель=%s input_size=%s',
                 self.model_name, self.input_size)

    def _detect_input_size_from_onnx(self) -> None:
        """Определить input_size из ONNX модели в model_dir (для RKNN backend)."""
        for name in (f'{self.model_name}.onnx', 'model.onnx'):
            onnx_path = self.model_dir / name
            if onnx_path.exists():
                try:
                    import onnxruntime as rt
                    sess = rt.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                    shape = sess.get_inputs()[0].shape
                    if len(shape) == 4 and isinstance(shape[2], int):
                        self.input_size = shape[2]
                        return
                except Exception:
                    pass
        pp = self.model_dir / 'postprocess.py'
        if pp.exists():
            import re
            text = pp.read_text()
            m = re.search(r'INPUT_SIZE\s*=\s*(\d+)', text)
            if m:
                self.input_size = int(m.group(1))

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Запуск инференса на бэкенде."""
        if self.backend == 'cpu':
            return self._session.run(None, {self._input_name: tensor})
        nhwc = tensor.transpose(0, 2, 3, 1)
        outputs = self.rknn.inference(inputs=[nhwc])
        if outputs is None:
            raise RuntimeError(
                f'RKNN inference вернул None (input shape={nhwc.shape}, '
                f'model={self.model_name}{self.rknn_suffix})'
            )
        return outputs

    def reload(self, model_name: str = '', backend: str | None = None,
               rknn_suffix: str | None = None) -> None:
        """Перезагрузка модели."""
        if model_name:
            self.model_name = model_name
        if backend is not None:
            self.backend = backend
        if rknn_suffix is not None:
            self.rknn_suffix = rknn_suffix
        log.info('перезагрузка модель=%s бэкенд=%s суффикс=%s...',
                 self.model_name, self.backend, self.rknn_suffix)
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
                if self.backend == 'cpu':
                    tensor = tensor / 255.0
            try:
                outputs = self.infer(tensor)
            except Exception:
                log.exception('ошибка инференса')
                return [], {}
        objects = postprocess_fn(outputs, scale, pad, (h, w)) if postprocess_fn else []
        metadata = {
            'orig_size': (h, w),
            'input_size': self.input_size,
            'scale': float(scale),
            'pad': (float(pad[0]), float(pad[1])),
        }
        return objects, metadata
