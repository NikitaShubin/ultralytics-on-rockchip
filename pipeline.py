from datetime import datetime
import os

import cv2
import numpy as np

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

REG_MAX = 16
BIN_COUNT = 4

NPU_TEMP_PATH = "/sys/class/thermal/thermal_zone6/temp"
TEMP_WARN = 85


class Pipeline:
    def __init__(self, model_path: str = "/app/model/yolov8n_rkopt_o19.rknn"):
        is_onnx = model_path.endswith(".onnx")
        if is_onnx:
            import onnxruntime as rt
            self.sess = rt.InferenceSession(model_path)
            self._infer = self._infer_onnx
            self.input_name = self.sess.get_inputs()[0].name
            print(f"[[V4]] pipeline started (ONNX CPU)", flush=True)
        else:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: {ret}")
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                raise RuntimeError(f"Failed to init runtime: {ret}")
            self._infer = self._infer_rknn
            print(f"[[V4]] pipeline started (RKNN)", flush=True)
        self.input_size = 640
        self.conf_thresh = 0.5
        self.iou_thresh = 0.5
        self.min_wh = 40
        self._temp_warned = False

    def _infer_rknn(self, tensor):
        # RKNN expects NHWC (batch, H, W, C); preprocess returns NCHW
        tensor_nhwc = tensor.transpose(0, 2, 3, 1)
        return self.rknn.inference(inputs=[tensor_nhwc])

    def _infer_onnx(self, tensor):
        return [self.sess.run(None, {self.input_name: tensor / 255.0})[0]]

    def _npu_temp(self) -> int | None:
        try:
            raw = open(NPU_TEMP_PATH).read().strip()
            return int(raw) // 1000
        except Exception:
            return None

    def process(self, frame: np.ndarray) -> np.ndarray:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        temp = self._npu_temp()
        if temp is not None and temp >= TEMP_WARN and not self._temp_warned:
            print(f"[[TEMP]] NPU={temp}°C (≥{TEMP_WARN}) — high temp warning", flush=True)
            self._temp_warned = True
        h, w = frame.shape[:2]
        input_tensor, scale, pad = self._preprocess(frame)
        try:
            outputs = self._infer(input_tensor)
        except Exception as e:
            print(f"[[ERR]] inference failed: {e}", flush=True)
            return frame
        boxes, scores, class_ids = self._postprocess(outputs, scale, pad, (w, h))
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{COCO_CLASSES[cls_id]}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def process_canvas(self, frame: np.ndarray) -> tuple:
        """Returns (canvas_with_detections, input_tensor) for debugging."""
        h, w = frame.shape[:2]
        input_tensor, scale, pad = self._preprocess(frame)
        canvas = input_tensor[0].transpose(1, 2, 0).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        try:
            outputs = self._infer(input_tensor)
        except Exception as e:
            print(f"[[ERR]] inference failed (canvas): {e}", flush=True)
            return canvas, pad, scale
        boxes, scores, class_ids = self._postprocess(outputs, scale, pad, (w, h))
        # Draw boxes in canvas space: reverse the scale (no pad correction since process_canvas uses removed-pad version)
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            cx = (x1 + x2) / 2 * scale + pad[0] / 2
            cy = (y1 + y2) / 2 * scale + pad[1] / 2
            cw = (x2 - x1) * scale
            ch = (y2 - y1) * scale
            cv2.rectangle(canvas, (int(cx - cw / 2), int(cy - ch / 2)), (int(cx + cw / 2), int(cy + ch / 2)), (0, 0, 255), 2)
            label = f"{COCO_CLASSES[cls_id]}: {score:.2f}"
            cv2.putText(canvas, label, (int(cx - cw / 2), int(cy - ch / 2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
        if not hasattr(self, '_saved_canvas'):
            cv2.imwrite("/tmp/debug_input.jpg", canvas)
            self._saved_canvas = True
        input_tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor.transpose(2, 0, 1), axis=0)
        return input_tensor, scale, (total_pad_w, total_pad_h)

    @staticmethod
    def _dfl(position):
        n, c, h, w = position.shape
        mc = c // BIN_COUNT
        y = position.reshape(n, BIN_COUNT, mc, h, w)
        y_exp = np.exp(y - y.max(axis=2, keepdims=True))
        y_softmax = y_exp / y_exp.sum(axis=2, keepdims=True)
        weights = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        return (y_softmax * weights).sum(axis=2)

    @staticmethod
    def _process_head(box_dist, cls_scores, stride, grid_h, grid_w):
        col = np.arange(grid_w, dtype=np.float32).reshape(1, 1, 1, grid_w)
        row = np.arange(grid_h, dtype=np.float32).reshape(1, 1, grid_h, 1)
        grid_x = np.broadcast_to(col, (1, 1, grid_h, grid_w))
        grid_y = np.broadcast_to(row, (1, 1, grid_h, grid_w))
        grid = np.concatenate([grid_x, grid_y], axis=1)

        dbox = Pipeline._dfl(box_dist)
        l = dbox[:, 0:1, :, :]
        t = dbox[:, 1:2, :, :]
        r = dbox[:, 2:3, :, :]
        b = dbox[:, 3:4, :, :]

        x1 = (grid[:, 0:1] + 0.5 - l) * stride
        y1 = (grid[:, 1:2] + 0.5 - t) * stride
        x2 = (grid[:, 0:1] + 0.5 + r) * stride
        y2 = (grid[:, 1:2] + 0.5 + b) * stride

        boxes = np.concatenate([x1, y1, x2, y2], axis=1)
        boxes = boxes.transpose(0, 2, 3, 1).reshape(-1, 4)
        scores = cls_scores.transpose(0, 2, 3, 1).reshape(-1, 80)
        return boxes, scores

    def _postprocess(self, outputs, scale, pad, orig_wh):
        orig_w, orig_h = orig_wh

        if len(outputs) == 1:
            return self._postprocess_old(outputs[0], scale, pad, (orig_w, orig_h))

        strides = [8, 16, 32]
        grid_sizes = [80, 40, 20]

        all_boxes = []
        all_scores = []
        all_cls = []

        for idx in range(3):
            box_dist = outputs[idx * 3]
            cls_scores = outputs[idx * 3 + 1]
            stride = strides[idx]
            gh, gw = grid_sizes[idx], grid_sizes[idx]

            boxes, scores = Pipeline._process_head(box_dist, cls_scores, stride, gh, gw)

            max_cls = scores.max(axis=1)
            cls_ids = scores.argmax(axis=1)

            all_boxes.append(boxes)
            all_scores.append(max_cls)
            all_cls.append(cls_ids)

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        cls_ids = np.concatenate(all_cls)

        mask = scores > self.conf_thresh
        if not mask.any():
            return np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0, dtype=int)
        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        pad_left, pad_top = pad[0] // 2, pad[1] // 2
        x1 = (boxes[:, 0] - pad_left) / scale
        y1 = (boxes[:, 1] - pad_top) / scale
        x2 = (boxes[:, 2] - pad_left) / scale
        y2 = (boxes[:, 3] - pad_top) / scale

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
        boxes, scores, class_ids = boxes[valid], scores[valid], cls_ids[valid]
        if not len(boxes):
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        valid = (bw >= self.min_wh) & (bh >= self.min_wh)
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        if not len(boxes):
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        n_conf = len(boxes)
        if n_conf > 500:
            temp = self._npu_temp()
            t = f" NPU={temp}°C" if temp is not None else ""
            print(f"[[WARN]] {n_conf} raw detections (>500){t} — likely NPU glitch, skipping frame", flush=True)
            return np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0, dtype=int)

        keep = self._nms(boxes, scores, self.iou_thresh)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        keep = self._dedup(boxes, scores, class_ids)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

        if len(boxes) > 1:
            temp = self._npu_temp()
            t = f" NPU={temp}°C" if temp is not None else ""
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int)
            info = ' '.join(f"{COCO_CLASSES[c]}:{s:.2f}({int(w[i])}x{int(h[i])}@{cx[i]},{cy[i]})"
                           for i, (c, s) in enumerate(zip(class_ids, scores)))
            print(f"[[DET]] raw={sum(g*g for g in grid_sizes)} conf>{self.conf_thresh}={n_conf} nms={len(boxes)}{t} {info}", flush=True)
        return boxes, scores, class_ids



    def _postprocess_old(self, output, scale, pad, orig_wh):
        orig_w, orig_h = orig_wh

        output = output.squeeze(0).transpose(1, 0)
        boxes_xywh = output[:, :4]
        scores = output[:, 4:]

        max_cls = scores.max(axis=1)
        mask = max_cls > self.conf_thresh
        if not mask.any():
            return np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0, dtype=int)

        boxes_xywh = boxes_xywh[mask]
        max_cls = max_cls[mask]
        cls_ids = scores[mask].argmax(axis=1)

        pad_left = pad[0] // 2
        pad_top = pad[1] // 2
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - pad_left - bw / 2) / scale
        y1 = (cy - pad_top - bh / 2) / scale
        x2 = (cx - pad_left + bw / 2) / scale
        y2 = (cy - pad_top + bh / 2) / scale
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
        boxes, scores, class_ids = boxes[valid], max_cls[valid], cls_ids[valid]
        if not len(boxes):
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        valid = (bw >= self.min_wh) & (bh >= self.min_wh)
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        if not len(boxes):
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        n_conf = len(boxes)
        if n_conf > 500:
            temp = self._npu_temp()
            t = f" NPU={temp}°C" if temp is not None else ""
            print(f"[[WARN]] {n_conf} raw detections (>500){t} — likely NPU glitch, skipping frame", flush=True)
            return np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0, dtype=int)
        keep = self._nms(boxes, scores, self.iou_thresh)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        keep = self._dedup(boxes, scores, class_ids)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if len(boxes) > 1:
            temp = self._npu_temp()
            t = f" NPU={temp}°C" if temp is not None else ""
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int)
            info = ' '.join(f"{COCO_CLASSES[c]}:{s:.2f}({int(w[i])}x{int(h[i])}@{cx[i]},{cy[i]})"
                           for i, (c, s) in enumerate(zip(class_ids, scores)))
            print(f"[[DET]] raw=8400 conf>{self.conf_thresh}={n_conf} nms={len(boxes)}{t} {info}", flush=True)
        return boxes, scores, class_ids

    def _nms(self, boxes, scores, iou_thresh):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            order = order[1:][iou <= iou_thresh]
        return np.array(keep)

    def _dedup(self, boxes, scores, class_ids):
        """Remove duplicate detections of the same class with similar positions/sizes."""
        if len(boxes) < 2:
            return np.arange(len(boxes))
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        groups = []
        assigned = set()
        for i in range(len(boxes)):
            if i in assigned:
                continue
            group = [i]
            assigned.add(i)
            for j in range(i + 1, len(boxes)):
                if j in assigned:
                    continue
                if class_ids[j] != class_ids[i]:
                    continue
                w_r = min(widths[i], widths[j]) / max(widths[i], widths[j])
                h_r = min(heights[i], heights[j]) / max(heights[i], heights[j])
                center_dist = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                same_center = center_dist < 50
                same_size = w_r > 0.75 and h_r > 0.75
                if same_size or (same_center and center_dist < max(widths[i], widths[j]) * 0.75):
                    group.append(j)
                    assigned.add(j)
            groups.append(group)
        keep = []
        for group in groups:
            best = max(group, key=lambda i: scores[i])
            keep.append(best)
        return np.array(keep)
