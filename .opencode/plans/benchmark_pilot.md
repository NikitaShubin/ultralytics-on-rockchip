# Benchmark Pilot — Implementation Plan

## Files to create

### 1. `benchmark/__init__.py` — empty

### 2. `benchmark/models.py`
```python
class Model(NamedTuple):
    family: str
    size: str
    task: str
    @property
    def name(self) -> str: ...
    @property
    def dataset(self) -> str: ...
    @property
    def input_size(self) -> int: ...

PILOT = [...]  # 10 models: v3u, v5nu, v8n (detect/seg/pose/obb/cls), v11n, v26n, v26n-sem
```

### 3. `benchmark/convert_bench.py` — runs INSIDE `converter` container
- Imports converter functions from `/app/converter/converter.py`
- Calls `resolve_pt(model_name)` → downloads .pt via ultralytics
- Calls `_detect_model_info(pt)` → model metadata
- Calls `export_and_compile(pt, info)` → ONNX + RKNN
- Calls `_generate_postprocess(info, output_dir)` → postprocess.py
- Copies ONNX → `output_dir/{name}.onnx`, RKNN → `output_dir/{name}.rknn`, postprocess → `output_dir/{name}_postprocess.py`
- **Validation**: raw tensor max_diff (ONNX vs RKNN) on 8 test images
- **Cleanup**: removes .pt, .onnx, _rknn_model/, postprocess from CWD, weights/, model/, ultralytics cache
- Outputs JSON to stdout

### 4. `benchmark/fps_bench.py` — runs INSIDE `webcam` container
- Imports `Pipeline` from `/app/pipeline.py`
- Loads model from temp dir
- **CPU**: Pipeline(Backend.CPU) → 30s warmup → 60s measure `detect()` loop → FPS
- **NPU**: Pipeline(Backend.NPU) → 30s warmup → 60s measure `detect()` loop → FPS
- Loads test images from data dir, cycles through them
- Outputs JSON to stdout

### 5. `benchmark/benchmark.py` — HOST orchestrator
For each model in PILOT:
1. `mkdir -p benchmark/temp_models/{name}/`
2. `docker compose -f converter/docker-compose.yml run --rm -v ./benchmark:/app/benchmark --entrypoint python converter /app/benchmark/convert_bench.py {name} {task} /app/benchmark/temp_models/{name}/`
3. Parse validation JSON from stdout
4. `docker compose run --rm -v ./benchmark:/app/benchmark -e TEST_MODE=1 -e WEB_MODE=0 -e CAM_MODE=file -e BACKEND=cpu webcam python3 /app/benchmark/fps_bench.py {name} {task} --temp-dir /app/benchmark/temp_models/{name}/ --data-dir /app/benchmark/data/{dataset}/`
5. Parse FPS JSON from stdout
6. `rm -rf benchmark/temp_models/{name}/`
7. Append result to `benchmark/results/results.json`
After all models: print summary table.

### 6. `benchmark/setup_datasets.py` — runs INSIDE `converter` container once
Downloads test datasets (coco8, coco8-seg, coco8-pose, dota8, imagenet8, cityscapes8) to `/app/benchmark/data/` using urllib + zipfile.

### 7. `benchmark/data/` — created by setup script

## Files to modify

### 8. `docker-compose.yml`
Add volume: `- ./benchmark:/app/benchmark:rw` to webcam service

## Execution order

```
1. mkdir -p benchmark/data benchmark/temp_models benchmark/results
2. docker compose run --rm -v ./benchmark:/app/benchmark --entrypoint python converter /app/benchmark/setup_datasets.py
3. python3 benchmark/benchmark.py
```

## Error handling
- Conversion failure → log FAIL → clean up → next model
- FPS failure → log FAIL → next model
- Guard: temp dir cleanup always runs (try/finally)

## Pilot scope (10 models)

| # | model | task | dataset |
|---|-------|------|---------|
| 1 | yolov3u | detect | coco8 |
| 2 | yolov5nu | detect | coco8 |
| 3 | yolov8n | detect | coco8 |
| 4 | yolov8n-seg | segment | coco8-seg |
| 5 | yolov8n-pose | pose | coco8-pose |
| 6 | yolov8n-obb | obb | dota8 |
| 7 | yolov8n-cls | classify | imagenet8 |
| 8 | yolo11n | detect | coco8 |
| 9 | yolo26n | detect | coco8 |
| 10 | yolo26n-sem | semantic | cityscapes8 |

## Output format

Results accumulated in `benchmark/results/results.json`:
```json
[
  {
    "model": "yolov8n",
    "task": "detect",
    "status": "PASS",
    "validation": {
      "max_diff_avg": 0.0042,
      "max_diff_max": 0.0081,
      "n_images": 8
    },
    "fps": {
      "cpu": 15.3,
      "npu": 22.1
    }
  }
]
```
