"""Pipeline: загрузка модели, инференс, постпроцесс."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from percept.inference import ModelBackend

log = logging.getLogger(__name__)


def _ndarray_to_list(obj: Any) -> Any:
    """Рекурсивное преобразование ndarray в list для JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(v) for v in obj]
    return obj


class Pipeline:
    """Загружает модель, запускает инференс, возвращает результаты."""

    def __init__(
        self,
        model_dir: str = '/app/model',
        model_name: str = '',
        backend: str = 'npu',
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name or 'yolo11n'
        self.backend = backend
        self._postprocess = None
        self._visualize = None
        self._custom_preprocess = None
        self._semantic_mode = False
        self.input_size = 640
        self._backend = ModelBackend(model_dir, model_name, backend)
        self._load_postprocess()

    def _load_postprocess(self) -> None:
        """Загрузка модуля постпроцесса из .py файла модели."""
        post_path = self.model_dir / f'{self.model_name}_postprocess.py'
        if not post_path.exists():
            post_path = self.model_dir / 'postprocess.py'
        if not post_path.exists():
            raise RuntimeError(f'postprocess.py не найден в {post_path}')
        spec = importlib.util.spec_from_file_location('postprocess_mod', str(post_path))
        if spec is None:
            raise RuntimeError(f'Не удалось загрузить спецификацию постпроцесса: {post_path}')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['postprocess_mod'] = mod
        if spec.loader is None:
            raise RuntimeError(f'Не удалось загрузить загрузчик постпроцесса: {post_path}')
        spec.loader.exec_module(mod)
        self._postprocess = getattr(mod, 'postprocess', None)
        self._visualize = getattr(mod, 'visualize', None)
        self._custom_preprocess = getattr(mod, 'preprocess', None)
        self._semantic_mode = getattr(mod, 'SEMANTIC_MODE', False)
        if self._postprocess is None:
            raise RuntimeError('postprocess.py должен определять функцию postprocess()')
        self.input_size = getattr(mod, 'INPUT_SIZE', 640)
        log.info(
            'постпроцесс загружен из %s (input_size=%s)',
            post_path,
            self.input_size,
        )

    def reload(self, model_name: str = '', backend: str | None = None) -> None:
        """Перезагрузка модели и постпроцесса (потокобезопасно)."""
        if model_name:
            self.model_name = model_name
        if backend is not None:
            self.backend = backend
        self._backend.reload(model_name, backend)
        if 'postprocess_mod' in sys.modules:
            del sys.modules['postprocess_mod']
        self._postprocess = None
        self._visualize = None
        self._load_postprocess()
        log.info('перезагрузка завершена (модель=%s)', self.model_name)

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[list[dict], dict]:
        """Запуск инференса на кадре. Возвращает (объекты, метаданные)."""
        return self._backend.detect(
            frame,
            self._postprocess,
            self._custom_preprocess,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Инференс + визуализация. Возвращает аннотированное изображение."""
        objects, _ = self.detect(frame)
        if self._visualize and (objects or self._semantic_mode):
            frame = self._visualize(frame, objects)
        return frame
