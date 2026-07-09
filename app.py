"""Flask HTTP server for /frame, /objects, /reload, /test endpoints."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from flask import Flask, Response, request

from camera_source import _make_no_cam_placeholder
from detector_frame import objects_to_api

if TYPE_CHECKING:
    from config import WebcamConfig
    from detector_frame import DetectorFrame
    from pipeline import Pipeline
    from video_loop import VideoLoop

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
TEMPLATES = HERE / 'templates'


def _read_template(name: str) -> str:
    path = TEMPLATES / name
    return path.read_text(encoding='utf-8') if path.exists() else ''


TEST_DIR = HERE / 'test'

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _make_filename(stem: str, ext: str) -> str:
    """Generate a unique filename in TEST_DIR."""
    name = f'{stem}{ext}'
    if not (TEST_DIR / name).exists():
        return name
    for i in range(1, 100):
        name = f'{stem}_{i}{ext}'
        if not (TEST_DIR / name).exists():
            return name
    return f'{stem}_{int(time.time())}{ext}'


def _serve_image(path: Path, mimetype: str | None = None, quality: int = 90) -> Response | tuple[str, int]:
    img = cv2.imread(str(path))
    if img is None:
        return 'image not found', 404
    if mimetype is None:
        ext = path.suffix.lower()
        mimetype = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
        }.get(ext, 'image/jpeg')
    _ret, buf = cv2.imencode(
        '.jpg',
        img,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    return Response(
        buf.tobytes(),
        mimetype='image/jpeg',
        headers={'Cache-Control': 'no-cache'},
    )


def _register_test_routes(app: Flask, config: WebcamConfig, pipeline: Pipeline) -> None:
    """Register test-mode routes (/test, /test/images, /test/images/<name>, /process)."""

    @app.route('/test')
    def test_page() -> str:
        return _read_template('test.html') or 'Test page not available'

    @app.route('/test/images')
    def test_images() -> list[dict]:
        entries: list[dict] = []
        if TEST_DIR.is_dir():
            for p in sorted(TEST_DIR.iterdir()):
                if _is_image(p.name) and '_pred' not in p.name:
                    entries.append({
                        'name': p.name,
                        'url': f'/test/images/{p.name}',
                    })
        return entries

    @app.route('/test/images/<name>')
    def test_image(name: str) -> Response | tuple[str, int]:
        safe = Path(name).name
        path = TEST_DIR / safe
        if not path.is_file() or not _is_image(safe):
            return 'not found', 404
        return _serve_image(path)

    @app.route('/test/upload', methods=['POST'])
    def test_upload() -> tuple[str, int] | Response:
        file = request.files.get('image')
        if file is None:
            return 'No image provided', 400
        original_name = Path(file.filename or 'image.jpg').stem
        ext = Path(file.filename or 'image.jpg').suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            ext = '.jpg'
        safe_name = _make_filename(original_name, ext)
        dest = TEST_DIR / safe_name
        file.save(str(dest))
        log.info('saved test image: %s', safe_name)
        return Response(safe_name, mimetype='text/plain')

    @app.route('/test/images/<name>', methods=['DELETE'])
    def test_delete(name: str) -> tuple[str, int]:
        safe = Path(name).name
        path = TEST_DIR / safe
        if not path.is_file() or not _is_image(safe):
            return 'not found', 404
        path.unlink()
        log.info('deleted test image: %s', safe)
        return 'OK', 200

    @app.route('/process', methods=['POST'])
    def process_upload() -> Response | tuple[str, int]:
        file = request.files.get('image')
        if file is None:
            return 'No image provided', 400
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is None:
            return 'Invalid image', 400
        result = pipeline.process(img)
        _ret, buf = cv2.imencode(
            '.jpg',
            result,
            [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
        )
        if not _ret:
            return 'encode failed', 500
        return Response(
            buf.tobytes(),
            mimetype='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
        )


def create_app(  # noqa: C901
    config: WebcamConfig,
    pipeline: Pipeline,
    loop: VideoLoop | None = None,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    _last_objects: list[dict] = []
    _last_jpeg: bytes | None = None

    mask_format = config.mask_format.value
    if loop is not None:

        def _on_frame(frame: DetectorFrame) -> None:
            nonlocal _last_objects, _last_jpeg
            _last_objects = objects_to_api(frame.objects, mask_format)
            _last_jpeg = frame.jpeg

        loop.register_consumer(_on_frame)

    @app.route('/')
    def index() -> tuple[str, int]:
        return _read_template('index.html') or 'UI not available', 200

    @app.route('/frame')
    def frame() -> Response:
        nonlocal _last_jpeg
        if _last_jpeg is None:
            return Response(_make_no_cam_placeholder(), mimetype='image/jpeg')
        return Response(
            _last_jpeg,
            mimetype='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
        )

    @app.route('/objects')
    def objects() -> dict:
        return {'objects': _last_objects, 'count': len(_last_objects)}

    @app.route('/reload')
    def reload_pipeline() -> tuple[str, int]:
        try:
            from config import Backend
            model_name = request.args.get('model', '')
            backend_str = request.args.get('backend', '')
            backend = Backend(backend_str) if backend_str else None
            pipeline.reload(model_name=model_name, backend=backend)
        except Exception as e:
            log.exception('reload failed')
            return f'FAIL: {e}', 500
        else:
            return 'OK', 200

    if config.test_mode:
        _register_test_routes(app, config, pipeline)

    return app
