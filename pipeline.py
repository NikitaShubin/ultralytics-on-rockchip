"""Pipeline: model loading, inference, and post-processing orchestration."""

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

from config import Backend
from utils import letterbox

log = logging.getLogger(__name__)


def _ndarray_to_list(obj: Any) -> Any:
    """Recursively convert ndarray values to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(v) for v in obj]
    return obj

NPU_TEMP_PATH = Path('/sys/class/thermal/thermal_zone6/temp')
TEMP_WARN = 85
TEMP_INTERVAL = 30.0


class Pipeline:
    """Loads a model, runs inference, and returns detection objects."""

    def __init__(
        self,
        model_dir: str = '/app/model',
        model_name: str = '',
        backend: Backend = Backend.NPU,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name or 'yolo11n'
        self.backend = backend
        self.rknn: Any = None
        self._session: Any = None
        self._postprocess = None
        self._visualize = None
        self.input_size = 640
        self._temp_warned = False
        self._last_temp_check = 0.0
        self._infer_lock = RLock()
        self._load_model()
        self._load_postprocess()

    def _load_model(self) -> None:
        if self.backend == Backend.CPU:
            self._load_model_onnx()
        else:
            self._load_model_rknn()

    def _load_model_rknn(self) -> None:
        from rknnlite.api import RKNNLite  # noqa: PLC0415

        model_path = self.model_dir / f'{self.model_name}.rknn'
        if not model_path.exists():
            model_path = self.model_dir / 'model.rknn'
        if not model_path.exists():
            msg = f'RKNN model not found: {model_path}'
            raise RuntimeError(msg)
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            msg = f'Failed to load RKNN model: {ret}'
            raise RuntimeError(msg)
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            msg = f'Failed to init runtime: {ret}'
            raise RuntimeError(msg)
        log.info('pipeline started (RKNN) model=%s', self.model_name)

    def _load_model_onnx(self) -> None:
        import onnxruntime as rt  # noqa: PLC0415

        model_path = self.model_dir / f'{self.model_name}.onnx'
        if not model_path.exists():
            model_path = self.model_dir / 'model.onnx'
        if not model_path.exists():
            msg = f'ONNX model not found at {model_path}'
            raise RuntimeError(msg)
        self._session = rt.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider'],
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info('pipeline started (ONNX CPU) model=%s', self.model_name)

    def _load_postprocess(self) -> None:
        post_path = self.model_dir / f'{self.model_name}_postprocess.py'
        if not post_path.exists():
            post_path = self.model_dir / 'postprocess.py'
        if not post_path.exists():
            msg = f'postprocess.py not found at {post_path}'
            raise RuntimeError(msg)
        spec = importlib.util.spec_from_file_location('postprocess_mod', str(post_path))
        assert spec is not None  # noqa: S101
        mod = importlib.util.module_from_spec(spec)
        sys.modules['postprocess_mod'] = mod
        assert spec.loader is not None  # noqa: S101
        spec.loader.exec_module(mod)
        self._postprocess = getattr(mod, 'postprocess', None)
        self._visualize = getattr(mod, 'visualize', None)
        self._custom_preprocess = getattr(mod, 'preprocess', None)
        self._semantic_mode = getattr(mod, 'SEMANTIC_MODE', False)
        if self._postprocess is None:
            msg = 'postprocess.py must define postprocess()'
            raise RuntimeError(msg)
        self.input_size = getattr(mod, 'INPUT_SIZE', 640)
        log.info(
            'postprocess loaded from %s (input_size=%s)',
            post_path,
            self.input_size,
        )

    def reload(self, model_name: str = '', backend: Backend | None = None) -> None:
        """Reload model and postprocess (thread-safe)."""
        if model_name:
            self.model_name = model_name
        if backend is not None:
            self.backend = backend
        log.info('reloading model=%s backend=%s...', self.model_name, self.backend)
        with self._infer_lock:
            if self.rknn:
                with suppress(Exception):
                    self.rknn.release()
                self.rknn = None
            self._session = None
            self._load_model()
            if 'postprocess_mod' in sys.modules:
                del sys.modules['postprocess_mod']
            self._postprocess = None
            self._visualize = None
            self._load_postprocess()
        log.info('reload complete (model=%s)', self.model_name)

    def _infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.backend == Backend.CPU:
            if self._custom_preprocess:
                tensor = tensor.transpose(0, 3, 1, 2)
                return self._session.run(None, {self._input_name: tensor})
            return self._session.run(None, {self._input_name: tensor / 255.0})
        if self._custom_preprocess:
            return self.rknn.inference(inputs=[tensor])
        nhwc = tensor.transpose(0, 2, 3, 1)
        return self.rknn.inference(inputs=[nhwc])

    def _check_temp(self) -> int | None:
        """Check NPU temperature (throttled to once per TEMP_INTERVAL)."""
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
            log.warning('NPU temp=%s°C (>=%s)', temp, TEMP_WARN)
            self._temp_warned = True
        return temp

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[list[dict], dict]:
        """Run inference on a frame and return (objects, metadata)."""
        self._check_temp()
        h, w = frame.shape[:2]
        with self._infer_lock:
            if self._custom_preprocess:
                tensor, scale, pad = self._custom_preprocess(frame)
            else:
                canvas, scale, pad = letterbox(frame, self.input_size)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
                tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
            try:
                outputs = self._infer(tensor)
            except Exception:
                log.exception('inference failed')
                return [], {}
        objects: list[dict] = (
            self._postprocess(outputs, scale, pad, (h, w)) if self._postprocess else []
        )
        metadata: dict = {
            'orig_size': (h, w),
            'input_size': self.input_size,
        }
        return objects, metadata

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Legacy: run inference + visualize, return annotated image."""
        objects, _ = self.detect(frame)
        if self._visualize and (objects or self._semantic_mode):
            frame = self._visualize(frame, objects)
        return frame
