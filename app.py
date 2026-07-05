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


def generate_debug():
    while True:
        with lock:
            jpeg = latest_jpeg
        if jpeg is None:
            continue
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        canvas, pad, scale = pipeline.process_canvas(frame)
        ret, out = cv2.imencode(".jpg", canvas)
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


@app.route("/debug")
def debug():
    return Response(
        generate_debug(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/raw")
def raw_debug():
    with lock:
        jpeg = latest_jpeg
    if jpeg is None:
        return "no frame"
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


_test_objects = None


def _get_test_objects():
    global _test_objects
    if _test_objects is None:
        from test_pipeline import load_coco8_objects
        _test_objects = load_coco8_objects()
    return _test_objects


@app.route("/test")
def test():
    from test_pipeline import generate_test_frames
    return Response(
        generate_test_frames(pipeline, _get_test_objects()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    finally:
        proc.kill()
