"""Async WebSocket server for streaming detections to clients."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from typing import TYPE_CHECKING

import websockets

from detector_frame import DetectorFrame, objects_to_api

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class WebSocketServer:
    """Broadcasts DetectorFrames to all connected WebSocket clients."""

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5001,
        mask_format: str = 'contour',
    ) -> None:
        self.host = host
        self.port = port
        self.mask_format = mask_format
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._last_frame: DetectorFrame | None = None
        self._server: websockets.WebSocketServer | None = None

    def push_frame(self, frame: DetectorFrame) -> None:
        """Called from video loop to broadcast a new frame to all clients."""
        self._last_frame = frame
        if not self._clients:
            return
        asyncio.get_event_loop().create_task(self._broadcast(frame))

    async def _broadcast(self, frame: DetectorFrame) -> None:
        if not self._clients:
            return
        safe_objects = objects_to_api(frame.objects, self.mask_format)
        msg = json.dumps(
            {
                'type': 'detections',
                'frame_id': frame.frame_id,
                'objects': safe_objects,
                'fps': frame.fps,
                'inference_ms': frame.inference_ms,
                'total_ms': frame.total_ms,
                'source': frame.source,
                'ts': time.time(),
            }
        )
        coros = [self._send_detections(ws, msg, frame) for ws in self._clients.copy()]
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    async def _send_detections(
        self,
        ws: websockets.WebSocketServerProtocol,
        msg: str,
        frame: DetectorFrame,
    ) -> None:
        try:
            await ws.send(msg)
            if frame.jpeg:
                header = struct.pack('!I', frame.frame_id)
                await ws.send(header + frame.jpeg)
        except websockets.ConnectionClosed:
            self._clients.discard(ws)

    async def handle(self, ws: websockets.WebSocketServerProtocol) -> None:
        """Handle a single WebSocket connection."""
        self._clients.add(ws)
        log.info('ws client connected (%d total)', len(self._clients))
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    if (
                        data.get('type') == 'frame_request'
                        and self._last_frame
                        and self._last_frame.jpeg
                    ):
                        header = struct.pack('!I', self._last_frame.frame_id)
                        await ws.send(header + self._last_frame.jpeg)
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info('ws client disconnected (%d left)', len(self._clients))

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self.handle,
            self.host,
            self.port,
        )
        log.info('ws server started on %s:%s', self.host, self.port)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            log.info('ws server stopped')
