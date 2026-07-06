import glob
import os
import select
import subprocess
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response

from pipeline import Pipeline

CAM_W, CAM_H = 640, 480
CAM_FPS = 15

app = Flask(__name__)

_NC = np.full((480, 640, 3), (114, 114, 114), dtype=np.uint8)
cv2.putText(_NC, "Camera not connected", (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
_ret, _NCJ = cv2.imencode(".jpg", _NC)
NO_CAM_JPEG = _NCJ.tobytes() if _ret else b""

pipeline = Pipeline()


class CameraManager:
    def __init__(self, width=CAM_W, height=CAM_H, fps=CAM_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = None
        self.latest_jpeg = None
        self.connected = False
        self.frame_time = 0.0
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = None
        self._proc = None

    def start(self):
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped = True
        self._kill_proc()

    def get_state(self):
        with self._lock:
            return self.latest_jpeg, self.connected, self.frame_time

    def disconnect(self):
        self._kill_proc()

    def _kill_proc(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    def _run(self):
        while not self._stopped:
            try:
                self._find_and_connect()
                if self._stopped:
                    break
                self._read_loop()
                self._on_disconnect()
            except Exception as e:
                print(f"[[CAM]] _run exception: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self._on_disconnect()
                time.sleep(1.0)

    def _on_disconnect(self):
        with self._lock:
            self.connected = False
            self.latest_jpeg = None
            self.device = None
        self._kill_proc()
        print("[[CAM]] disconnected", flush=True)

    def _find_and_connect(self):
        while not self._stopped:
            devices = sorted(glob.glob("/dev/video[0-9]*"))
            for dev in devices:
                if self._stopped:
                    return
                if self._try_camera(dev):
                    print(f"[[CAM]] connected to {dev}", flush=True)
                    return
            time.sleep(0.2)

    def _try_camera(self, dev):
        variants = [
            [
                "ffmpeg",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-analyzeduration", "0",
                "-probesize", "32",
                "-f", "v4l2",
                "-input_format", "mjpeg",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", dev,
                "-f", "image2pipe",
                "-vcodec", "copy",
                "-",
            ],
            [
                "ffmpeg",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-analyzeduration", "0",
                "-probesize", "32",
                "-f", "v4l2",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", dev,
                "-f", "image2pipe",
                "-c:v", "mjpeg",
                "-q:v", "5",
                "-",
            ],
        ]
        for idx, cmd in enumerate(variants):
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
                r, _, _ = select.select([proc.stdout], [], [], 0.5)
                if r:
                    self._proc = proc
                    self.device = dev
                    with self._lock:
                        self.connected = True
                    print(f"[[CAM]] try {dev} variant {idx} OK", flush=True)
                    return True
                proc.kill()
                print(f"[[CAM]] try {dev} variant {idx} timeout", flush=True)
            except Exception as e:
                print(f"[[CAM]] try {dev} variant {idx} error: {e}", flush=True)
        return False

    def _read_loop(self):
        buf = b""
        last_frame_time = time.monotonic()
        stat_check = time.monotonic()
        loop_start = time.monotonic()
        while not self._stopped and self._proc:
            if time.monotonic() - loop_start > 30.0:
                print("[[CAM]] break: reconnect timer", flush=True)
                self._kill_proc()
                break
            if self._proc.poll() is not None:
                print("[[CAM]] break: proc exited", flush=True)
                break
            if self.device and glob.glob(self.device) == []:
                print("[[CAM]] break: device vanished", flush=True)
                self._kill_proc()
                break
            try:
                r, _, _ = select.select([self._proc.stdout], [], [], 0.5)
            except ValueError:
                print("[[CAM]] break: select ValueError", flush=True)
                break
            if not r:
                now = time.monotonic()
                if now - stat_check > 3.0:
                    stat_check = now
                    if self.device and not os.path.exists(self.device):
                        print("[[CAM]] break: stat check failed", flush=True)
                        self._kill_proc()
                        break
                since_last = now - last_frame_time
                if since_last > 1.5:
                    print(f"[[CAM]] break: frame timeout {since_last:.1f}s", flush=True)
                    self._kill_proc()
                    break
                continue
            try:
                chunk = self._proc.stdout.read(65536)
            except Exception:
                print("[[CAM]] break: read exception", flush=True)
                break
            if not chunk:
                print("[[CAM]] break: EOF", flush=True)
                break
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                if start < 0:
                    buf = b""
                    break
                end = buf.find(b"\xff\xd9", start)
                if end < 0:
                    buf = buf[start:]
                    break
                jpeg = buf[start: end + 2]
                buf = buf[end + 2:]
                with self._lock:
                    self.latest_jpeg = jpeg
                    self.connected = True
                    self.frame_time = time.monotonic()
                last_frame_time = time.monotonic()


cam = CameraManager()
cam.start()


def generate():
    while True:
        jpeg, connected, frame_time = cam.get_state()
        stale = connected and (time.monotonic() - frame_time > 2.0)
        if not connected or jpeg is None or stale:
            if stale:
                print("[[STREAM]] stale frame >2s, disconnecting", flush=True)
                cam.disconnect()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + NO_CAM_JPEG + b"\r\n"
            time.sleep(0.2)
            continue
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = pipeline.process(frame)
        ret, out = cv2.imencode(".jpg", frame)
        if ret:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + out.tobytes() + b"\r\n"


def generate_debug():
    while True:
        jpeg, connected, frame_time = cam.get_state()
        stale = connected and (time.monotonic() - frame_time > 2.0)
        if not connected or jpeg is None or stale:
            if stale:
                print("[[STREAM]] stale frame >2s, disconnecting", flush=True)
                cam.disconnect()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + NO_CAM_JPEG + b"\r\n"
            time.sleep(0.2)
            continue
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        canvas, pad, scale = pipeline.process_canvas(frame)
        ret, out = cv2.imencode(".jpg", canvas)
        if ret:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + out.tobytes() + b"\r\n"


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
<style>
* { margin: 0; padding: 0; }
html, body { width: 100%; height: 100%; overflow: hidden; background: #000; }
img { width: 100vw; height: 100vh; object-fit: contain; }
</style>
</head>
<body>
<img src="/stream">
</body>
</html>"""


@app.route("/stream")
def stream():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/debug")
def debug():
    return Response(
        generate_debug(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/raw")
def raw_debug():
    jpeg, connected, _ = cam.get_state()
    if not connected or jpeg is None:
        return "no camera"
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    input_tensor, scale, pad = pipeline._preprocess(frame)
    outputs = pipeline._infer(input_tensor)
    from pipeline import COCO_CLASSES
    lines = [f"Frame: {w}x{h}, scale={scale:.3f}, pad={pad}"]
    lines.append(f"outputs: {len(outputs)} tensors")
    for i, o in enumerate(outputs):
        lines.append(f"  [{i}] shape={list(o.shape)} dtype={o.dtype}")
    boxes, scores, class_ids = pipeline._postprocess(outputs, scale, pad, (w, h))
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        lines.append(f"  {COCO_CLASSES[cls_id]}:{score:.2f} {x1},{y1},{x2},{y2}")
    return "<pre>" + "\n".join(lines) + "</pre>"


if __name__ == "__main__":
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    finally:
        cam.stop()
