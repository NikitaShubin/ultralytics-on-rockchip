import glob
import importlib
import os
import select
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, request

from pipeline import Pipeline

CAM_W, CAM_H = 640, 480
CAM_FPS = 15

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_IMG_PATH = "/app/model/test.jpg"

app = Flask(__name__)

_NC = np.full((480, 640, 3), (114, 114, 114), dtype=np.uint8)
cv2.putText(_NC, "Camera not connected", (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
_ret, _NCJ = cv2.imencode(".jpg", _NC)
NO_CAM_JPEG = _NCJ.tobytes() if _ret else b""

pipeline = Pipeline()

def _get_postprocess_mod():
    if "postprocess_mod" in sys.modules:
        return sys.modules["postprocess_mod"]
    return None


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
        self._bad_devices = set()
        self._proc_start_time = 0.0

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
            any_skipped = False
            for dev in devices:
                if self._stopped:
                    return
                if dev in self._bad_devices:
                    any_skipped = True
                    continue
                if self._try_camera(dev):
                    self._bad_devices.clear()
                    print(f"[[CAM]] connected to {dev}", flush=True)
                    return
            if any_skipped:
                time.sleep(1.0)
                self._bad_devices.clear()
            else:
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
                r, _, _ = select.select([proc.stdout], [], [], 1.5)
                if r:
                    self._proc = proc
                    self._proc_start_time = time.monotonic()
                    self.device = dev
                    with self._lock:
                        self.connected = True
                    print(f"[[CAM]] try {dev} variant {idx} OK", flush=True)
                    return True
                proc.kill()
                proc.wait(timeout=1)
                print(f"[[CAM]] try {dev} variant {idx} timeout", flush=True)
            except Exception as e:
                print(f"[[CAM]] try {dev} variant {idx} error: {e}", flush=True)
            time.sleep(0.2)
        return False

    def _read_loop(self):
        buf = b""
        last_frame_time = time.monotonic()
        stat_check = time.monotonic()
        while not self._stopped and self._proc:
            now = time.monotonic()
            if now - stat_check > 3.0:
                stat_check = now
                if self.device:
                    try:
                        r = subprocess.run(["stat", self.device],
                                           capture_output=True, timeout=2)
                        if r.returncode != 0:
                            print("[[CAM]] break: stat failed", flush=True)
                            self._kill_proc()
                            break
                    except FileNotFoundError:
                        print("[[CAM]] break: stat cmd not found", flush=True)
                        self._kill_proc()
                        break
                    except subprocess.TimeoutExpired:
                        print("[[CAM]] break: stat timeout", flush=True)
                        self._kill_proc()
                        break
            if self._proc.poll() is not None:
                runtime = time.monotonic() - self._proc_start_time
                if runtime < 1.5 and self.device:
                    print(f"[[CAM]] break: proc exited after {runtime:.1f}s — marking {self.device} as bad", flush=True)
                    self._bad_devices.add(self.device)
                else:
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
                since_last = time.monotonic() - last_frame_time
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


DOTA_CLASSES = ['plane', 'ship', 'storage-tank', 'storage tank',
    'baseball-diamond', 'baseball diamond', 'tennis-court', 'tennis court',
    'basketball-court', 'basketball court', 'ground-track-field', 'ground track field',
    'harbor', 'bridge', 'large-vehicle', 'large vehicle',
    'small-vehicle', 'small vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'soccer ball field', 'swimming-pool', 'swimming pool']
DOTA_IDS = [0,1,2,2,3,3,4,4,5,5,6,6,7,8,9,9,10,10,11,12,13,13,14,14]
DOTA_CM = dict(zip(DOTA_CLASSES, DOTA_IDS))


def _draw_gt_boxes(img, label_path):
    if not os.path.exists(label_path):
        return img
    h, w = img.shape[:2]
    s = max(h, w) / 1000
    thick = max(1, int(s * 2))
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('imagesource') or line.startswith('gsd'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            cn = ' '.join(parts[8:-1])
            if cn not in DOTA_CM:
                continue
            pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 200), thickness=thick)
    return img


@app.route("/reload")
def reload_pipeline():
    """Reload model and postprocess script. Call after --deploy."""
    try:
        pipeline.reload()
        return "OK", 200
    except Exception as e:
        return f"FAIL: {e}", 500



_TEST_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>test — webcam</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 100%; min-height: 100vh; background: #111; color: #ddd; font-family: sans-serif; }
body { display: flex; flex-direction: column; align-items: center; padding: 20px; gap: 16px; }
#drop-zone {
  width: 100%; max-width: 900px; min-height: 200px;
  border: 2px dashed #555; border-radius: 12px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  cursor: pointer; transition: 0.2s; padding: 24px; gap: 12px;
  font-size: 18px; color: #888;
}
#drop-zone.drag-over { border-color: #4caf50; background: #1a3a1a; color: #4caf50; }
#drop-zone.loading { pointer-events: none; opacity: 0.5; }
#drop-zone input { display: none; }
.btn-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
.btn {
  background: #333; color: #ddd; border: 1px solid #555;
  padding: 10px 24px; border-radius: 8px; cursor: pointer; font-size: 15px;
  transition: 0.15s;
}
.btn:hover { background: #444; border-color: #777; }
.btn.primary { background: #1b5e20; border-color: #4caf50; }
.btn.primary:hover { background: #2e7d32; }
.panels {
  width: 100%; max-width: 1200px;
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
.panel { display: flex; flex-direction: column; gap: 6px; }
.panel .label { font-size: 13px; color: #888; text-align: center; }
.panel img { width: 100%; max-height: 70vh; object-fit: contain; background: #000; border-radius: 8px; }
@media (max-width: 700px) {
  .panels { grid-template-columns: 1fr; }
  #drop-zone { min-height: 140px; font-size: 15px; }
}
.spinner { display: none; width: 28px; height: 28px; border: 3px solid #555; border-top-color: #4caf50; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div id="drop-zone">
  <div class="spinner" id="spinner"></div>
  <span id="dz-text">Drop image here or click to select</span>
  <input type="file" id="file-input" accept="image/*">
</div>
<div class="btn-row">
  <button class="btn primary" id="btn-test">Load test image</button>
</div>
<div class="panels" id="panels" style="display:none">
  <div class="panel">
    <div class="label">Original</div>
    <img id="orig-preview">
  </div>
  <div class="panel">
    <div class="label">Detections</div>
    <img id="result-preview">
  </div>
</div>
<script>
const dz = document.getElementById('drop-zone');
const fi = document.getElementById('file-input');
const dzText = document.getElementById('dz-text');
const spinner = document.getElementById('spinner');
const panels = document.getElementById('panels');
const origPreview = document.getElementById('orig-preview');
const resultPreview = document.getElementById('result-preview');

function setLoading(on) {
  dz.classList.toggle('loading', on);
  spinner.style.display = on ? 'block' : 'none';
  dzText.textContent = on ? 'Processing...' : 'Drop image here or click to select';
}

function processFile(file) {
  const url = URL.createObjectURL(file);
  origPreview.src = url;
  panels.style.display = 'grid';
  setLoading(true);
  const fd = new FormData();
  fd.append('image', file);
  fetch('/process', { method: 'POST', body: fd })
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.blob(); })
    .then(blob => { resultPreview.src = URL.createObjectURL(blob); setLoading(false); })
    .catch(e => { setLoading(false); dzText.textContent = 'Error: ' + e.message; });
}

dz.addEventListener('click', () => fi.click());
fi.addEventListener('change', () => { if (fi.files[0]) processFile(fi.files[0]); });
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) processFile(f);
});

document.getElementById('btn-test').addEventListener('click', () => {
  setLoading(true);
  fetch('/test/raw')
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.blob(); })
    .then(blob => {
      const file = new File([blob], 'test.jpg', { type: 'image/jpeg' });
      const fd = new FormData();
      fd.append('image', file);
      fd.append('with_gt', '1');
      origPreview.src = URL.createObjectURL(blob);
      panels.style.display = 'grid';
      return fetch('/process', { method: 'POST', body: fd });
    })
    .then(r => { if (!r.ok) throw new Error(r.statusText); return r.blob(); })
    .then(blob => { resultPreview.src = URL.createObjectURL(blob); setLoading(false); })
    .catch(e => { setLoading(false); dzText.textContent = 'Error: ' + e.message; });
});
</script>
</body>
</html>"""


if TEST_MODE:

    @app.route("/test")
    def test_page():
        return _TEST_HTML

    @app.route("/test/raw")
    def test_raw():
        img = cv2.imread(TEST_IMG_PATH)
        if img is None:
            return "test image not found", 404
        ret, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(out.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache"})

    @app.route("/process", methods=["POST"])
    def process_upload():
        file = request.files.get("image")
        if file is None:
            return "No image provided", 400
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Invalid image", 400
        frame = pipeline.process(img)
        with_gt = request.form.get("with_gt", "0") == "1"
        if with_gt:
            label_path = os.path.join(os.path.dirname(TEST_IMG_PATH), "test.txt")
            if os.path.exists(label_path):
                frame = _draw_gt_boxes(frame, label_path)
        ret, out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ret:
            return "encode failed", 500
        return Response(out.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


if __name__ == "__main__":
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    finally:
        cam.stop()
