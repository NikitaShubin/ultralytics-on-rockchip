#!/usr/bin/env python3
"""Точка входа: перцепт — сервер обнаружения объектов на NPU."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from typing import TYPE_CHECKING

from waitress import serve

if TYPE_CHECKING:
    from flask import Flask

from app import create_app
from config import Config, WebMode
from pipeline import Pipeline
from video_loop import VideoLoop
from websocket_server import WebSocketServer


def setup_logging(level: str = 'INFO') -> None:
    """Настройка логирования в stdout."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
        force=True,
    )


def _run_http(app: Flask, host: str, port: int) -> None:
    """Запуск Flask/Waitress HTTP сервера (блокирующий, daemon thread)."""
    serve(app, host=host, port=port)


async def amain() -> None:
    """Async точка входа: video loop + optional WS + HTTP."""
    config = Config.from_cli()
    setup_logging(config.log_level)
    log = logging.getLogger('main')

    pipeline = Pipeline(
        model_dir=config.model_dir,
        model_name=config.model,
        backend=config.backend.value,
    )

    video_loop = VideoLoop(pipeline=pipeline, config=config)
    ws_server: WebSocketServer | None = None

    if config.web_mode >= WebMode.WS_ONLY:
        ws_port = 5001
        ws_server = WebSocketServer(
            host=config.host,
            port=ws_port,
            mask_format=config.mask_format.value,
        )
        video_loop.register_consumer(ws_server.push_frame)

    if config.web_mode >= WebMode.WS_AND_UI:
        app = create_app(config, pipeline, video_loop, ws_port=ws_port)
        http_thread = threading.Thread(
            target=_run_http,
            args=(app, config.host, config.port),
            daemon=True,
        )
        http_thread.start()
        log.info('HTTP сервер запущен на %s:%s', config.host, config.port)

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info('остановка...')
        video_loop.stop()
        stop_event.set()

    loop_task = asyncio.create_task(video_loop.run())
    ws_task: asyncio.Task | None = None
    if ws_server is not None:
        await ws_server.start()
        ws_task = asyncio.create_task(stop_event.wait())

    event_loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        event_loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    loop_task.cancel()
    if ws_task is not None:
        ws_task.cancel()
    if ws_server is not None:
        await ws_server.stop()
    log.info('остановка завершена')


def main() -> None:
    """Запуск приложения."""
    asyncio.run(amain())


if __name__ == '__main__':
    main()
