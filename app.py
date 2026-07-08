import glob
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, request

from pipeline import Pipeline

CAM_W, CAM_H = 640, 480
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"
TEST_IMG_PATH = "/app/model/test.jpg"

app = Flask(__name__)

_NC = np.full((480, 640, 3), (114, 114, 114), dtype=np.uint8)
cv2.putText(_NC, "Camera not connected", (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
_ret, _NCJ = cv2.imencode(".jpg", _NC)
NO_CAM_JPEG = _NCJ.tobytes() if _ret else b""

pipeline = Pipeline()


class CameraManager:
    def __init__(self, width=CAM_W, height=CAM_H):
        self.width = width
        self.height = height
        self.latest_processed_jpeg = None
        self.processed_time = 0.0
        self.connected = False
        self.device = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = None
        self._cap = None

    def start(self):
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped = True
        self._release_cap()

    def get_processed_state(self):
        with self._lock:
            return self.latest_processed_jpeg, self.connected, self.processed_time

    def _release_cap(self):
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
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
            self.latest_processed_jpeg = None
            self.device = None
        self._release_cap()
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
        try:
            cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                self._cap = cap
                with self._lock:
                    self.connected = True
                self.device = dev
                return True
            cap.release()
        except Exception as e:
            print(f"[[CAM]] opencv error: {e}", flush=True)
        return False

    def _read_loop(self):
        while not self._stopped and self._cap is not None:
            if self.device and glob.glob(self.device) == []:
                print("[[CAM]] break: device vanished", flush=True)
                break
            ret, frame = self._cap.read()
            if not ret:
                print("[[CAM]] break: cap.read() failed", flush=True)
                break
            try:
                processed = pipeline.process(frame)
            except Exception as e:
                print(f"[[CAM]] process error: {e}", flush=True)
                processed = frame
            ret, out = cv2.imencode(".jpg", processed)
            if ret:
                with self._lock:
                    self.latest_processed_jpeg = out.tobytes()
                    self.processed_time = time.monotonic()


cam = CameraManager()
cam.start()


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
<img id="s">
<script>
const i = document.getElementById('s');
function next() { i.src = '/frame?_=' + performance.now(); }
i.onload = next;
i.onerror = () => setTimeout(next, 500);
next();
</script>
</body>
</html>"""


@app.route("/frame")
def frame():
    jpeg, connected, _ = cam.get_processed_state()
    if not connected or jpeg is None:
        return Response(NO_CAM_JPEG, mimetype="image/jpeg")
    return Response(jpeg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


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
    try:
        model_name = request.args.get("model", "")
        pipeline.reload(model_name=model_name)
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
