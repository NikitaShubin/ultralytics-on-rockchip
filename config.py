"""Конфигурация: CLI аргументы + переменные окружения."""

from __future__ import annotations

import argparse
import os
import sys
from enum import IntEnum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class Backend(StrEnum):
    NPU = 'npu'
    CPU = 'cpu'


class WebMode(IntEnum):
    OFF = 0
    WS_ONLY = 1
    WS_AND_UI = 2


class CameraMode(StrEnum):
    AUTO = 'auto'
    USB = 'usb'
    FILE = 'file'
    IP = 'ip'


class MaskFormat(StrEnum):
    NONE = 'none'
    CONTOUR = 'contour'
    BASE64 = 'base64'


class Config:
    """Все параметры конфигурации приложения."""

    def __init__(self) -> None:
        self.model: str = os.environ.get('MODEL', 'yolo11n')
        self.backend: Backend = Backend(os.environ.get('BACKEND', 'npu'))
        self.web_mode: WebMode = WebMode(int(os.environ.get('WEB_MODE', '0')))
        self.cam_mode: CameraMode = CameraMode(os.environ.get('CAM_MODE', 'auto'))
        self.test_mode: bool = os.environ.get('TEST_MODE', '0') == '1'
        self.cam_width: int = int(os.environ.get('CAM_W', '640'))
        self.cam_height: int = int(os.environ.get('CAM_H', '480'))
        self.cam_source: str = os.environ.get('CAM_SOURCE', '')
        self.cam_fps: int = int(os.environ.get('CAM_FPS', '30'))
        self.model_dir: str = os.environ.get('MODEL_DIR', '/app/model')
        self.host: str = os.environ.get('HOST', '0.0.0.0')
        self.port: int = int(os.environ.get('PORT', '5000'))
        self.jpeg_quality: int = int(os.environ.get('JPEG_QUALITY', '85'))
        self.log_level: str = os.environ.get('LOG_LEVEL', 'INFO')
        self.mask_format: MaskFormat = MaskFormat(os.environ.get('MASK_FORMAT', 'contour'))

    @classmethod
    def from_cli(cls) -> Config:
        """Создание конфигурации из CLI аргументов."""
        cfg = cls()
        parser = argparse.ArgumentParser(description='Перцепт — сервер обнаружения объектов')
        parser.add_argument('--model', default=cfg.model)
        parser.add_argument(
            '--backend',
            choices=['npu', 'cpu'],
            default=cfg.backend.value,
        )
        parser.add_argument(
            '--web-mode',
            type=int,
            choices=[0, 1, 2],
            default=cfg.web_mode.value,
        )
        parser.add_argument(
            '--cam-mode',
            choices=['auto', 'usb', 'file', 'ip'],
            default=cfg.cam_mode.value,
        )
        parser.add_argument('--cam-width', type=int, default=cfg.cam_width)
        parser.add_argument('--cam-height', type=int, default=cfg.cam_height)
        parser.add_argument('--cam-source', default=cfg.cam_source)
        parser.add_argument('--cam-fps', type=int, default=cfg.cam_fps)
        parser.add_argument('--model-dir', default=cfg.model_dir)
        parser.add_argument('--host', default=cfg.host)
        parser.add_argument('--port', type=int, default=cfg.port)
        parser.add_argument('--jpeg-quality', type=int, default=cfg.jpeg_quality)
        parser.add_argument('--log-level', default=cfg.log_level)
        parser.add_argument(
            '--mask-format',
            choices=['none', 'contour', 'base64'],
            default=cfg.mask_format.value,
        )
        parser.add_argument('--test-mode', action='store_true', default=cfg.test_mode)
        args = parser.parse_args()
        cfg.model = args.model
        cfg.backend = Backend(args.backend)
        cfg.web_mode = WebMode(args.web_mode)
        cfg.cam_mode = CameraMode(args.cam_mode)
        cfg.cam_width = args.cam_width
        cfg.cam_height = args.cam_height
        cfg.cam_source = args.cam_source
        cfg.cam_fps = args.cam_fps
        cfg.model_dir = args.model_dir
        cfg.host = args.host
        cfg.port = args.port
        cfg.jpeg_quality = args.jpeg_quality
        cfg.log_level = args.log_level
        cfg.mask_format = MaskFormat(args.mask_format)
        cfg.test_mode = args.test_mode
        return cfg
