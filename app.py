import subprocess
import threading

import cv2
import numpy as np
from flask import Flask, Response

from pipeline import Pipeline

WIDTH, HEIGHT = 640, 480
FPS = 15

app = Flask(__name__)
pipeline = Pipeline()

cmd = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-f", "v4l2",
    "-input_format", "mjpeg",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-framerate", str(FPS),
    "-i", "/dev/video0",
    "-f", "image2pipe",
    "-vcodec", "copy",
    "-",
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

latest_jpeg = None
lock = threading.Lock()


def reader_thread():
    global latest_jpeg
    buf = b""
    while True:
        chunk = proc.stdout.read(65536)
        if not chunk:
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
            jpeg = buf[start : end + 2]
            buf = buf[end + 2 :]
            with lock:
                latest_jpeg = jpeg


threading.Thread(target=reader_thread, daemon=True).start()


def generate():
    while True:
        with lock:
            jpeg = latest_jpeg
        if jpeg is None:
            continue
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = pipeline.process(frame)
        ret, out = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + out.tobytes() + b"\r\n"
        )


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


if __name__ == "__main__":
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    finally:
        proc.kill()
