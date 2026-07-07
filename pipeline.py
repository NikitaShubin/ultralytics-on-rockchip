import importlib.util
import os
import sys
import threading
from datetime import datetime

import cv2
import numpy as np

NPU_TEMP_PATH = "/sys/class/thermal/thermal_zone6/temp"
TEMP_WARN = 85


class Pipeline:
    def __init__(self, model_dir: str = "/app/model"):
        self.model_dir = model_dir
        self.rknn = None
        self._session = None
        self._postprocess = None
        self._visualize = None
        self.input_size = 640
        self._temp_warned = False
        self._infer_lock = threading.Lock()
        self.backend = os.environ.get("BACKEND", "npu")
        self._load_model()
        self._load_postprocess()

    def _load_model(self):
        if self.backend == "cpu":
            self._load_model_onnx()
        else:
            self._load_model_rknn()

    def _load_model_rknn(self):
        from rknnlite.api import RKNNLite
        model_path = os.path.join(self.model_dir, "model.rknn")
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime: {ret}")
        print(f"[[V4]] pipeline started (RKNN)", flush=True)

    def _load_model_onnx(self):
        import onnxruntime as rt
        model_path = os.path.join(self.model_dir, "model.onnx")
        if not os.path.exists(model_path):
            raise RuntimeError(f"ONNX model not found at {model_path}")
        self._session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        print(f"[[V4]] pipeline started (ONNX CPU)", flush=True)

    def _load_postprocess(self):
        post_path = os.path.join(self.model_dir, "postprocess.py")
        if not os.path.exists(post_path):
            raise RuntimeError(f"postprocess.py not found at {post_path}")
        spec = importlib.util.spec_from_file_location("postprocess_mod", post_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["postprocess_mod"] = mod
        spec.loader.exec_module(mod)
        self._postprocess = getattr(mod, "postprocess", None)
        self._visualize = getattr(mod, "visualize", None)
        if self._postprocess is None:
            raise RuntimeError("postprocess.py must define postprocess()")
        self.input_size = getattr(mod, "INPUT_SIZE", 640)
        print(f"[[V4]] postprocess loaded from {post_path} (input_size={self.input_size})", flush=True)

    def reload(self):
        print("[[V4]] reloading model and postprocess...", flush=True)
        if self.rknn:
            try:
                self.rknn.release()
            except Exception:
                pass
            self.rknn = None
        self._session = None
        self._load_model()

        if "postprocess_mod" in sys.modules:
            del sys.modules["postprocess_mod"]
        self._postprocess = None
        self._visualize = None
        self._load_postprocess()

        print("[[V4]] reload complete", flush=True)

    def _infer(self, tensor):
        with self._infer_lock:
            if self.backend == "cpu":
                return self._session.run(None, {self._input_name: tensor / 255.0})
            nhwc = tensor.transpose(0, 2, 3, 1)
            return self.rknn.inference(inputs=[nhwc])

    def _npu_temp(self) -> int | None:
        try:
            raw = open(NPU_TEMP_PATH).read().strip()
            return int(raw) // 1000
        except Exception:
            return None

    def process(self, frame: np.ndarray) -> np.ndarray:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        temp = self._npu_temp()
        if temp is not None and temp >= TEMP_WARN and not self._temp_warned:
            print(f"[[TEMP]] NPU={temp}°C (≥{TEMP_WARN}) — high temp warning", flush=True)
            self._temp_warned = True

        h, w = frame.shape[:2]
        tensor, scale, pad = self._preprocess(frame)

        try:
            outputs = self._infer(tensor)
        except Exception as e:
            print(f"[[ERR]] inference failed: {e}", flush=True)
            return frame

        if self._postprocess:
            objects = self._postprocess(outputs, scale, pad, (h, w))
            if self._visualize:
                frame = self._visualize(frame, objects)

        return frame

    def process_canvas(self, frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        tensor, scale, pad = self._preprocess(frame)
        canvas = tensor[0].transpose(1, 2, 0).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        try:
            outputs = self._infer(tensor)
        except Exception as e:
            print(f"[[ERR]] inference failed (canvas): {e}", flush=True)
            return canvas, pad, scale

        if self._postprocess:
            objects = self._postprocess(outputs, scale, pad, (h, w))
            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                score = obj["score"]
                label = f"{obj['class_name']}: {score:.2f}"
                cx = (x1 + x2) / 2 * scale + pad[0] / 2
                cy = (y1 + y2) / 2 * scale + pad[1] / 2
                cw = (x2 - x1) * scale
                ch = (y2 - y1) * scale
                cv2.rectangle(
                    canvas,
                    (int(cx - cw / 2), int(cy - ch / 2)),
                    (int(cx + cw / 2), int(cy + ch / 2)),
                    (0, 0, 255), 2,
                )
                cv2.putText(
                    canvas, label,
                    (int(cx - cw / 2), int(cy - ch / 2 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                )
        return canvas, pad, scale

    def _preprocess(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        total_pad_w = self.input_size - new_w
        total_pad_h = self.input_size - new_h
        pad_left = total_pad_w // 2
        pad_top = total_pad_h // 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.copyMakeBorder(
            resized, pad_top, total_pad_h - pad_top,
            pad_left, total_pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
        return tensor, scale, (total_pad_w, total_pad_h)
