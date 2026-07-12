# webcam — YOLO инференс на NPU (RK3588S)

Сервер обнаружения объектов с камеры в реальном времени.
Поддерживает **111 моделей** Ultralytics YOLO (v3/v5/v8/v9/v10/v11/v12/v26)
во всех шести задачах: detect, segment, pose, classify, OBB, semantic segmentation.

**Архитектура**: `CameraSource` → `VideoLoop` (capture → Pipeline) → Flask HTTP + WebSocket → браузер.

**Целевое железо**: Orange Pi 3B (RK3588S), 4 ГБ RAM, NPU 6 TOPS.

## Статус

| Компонент | Состояние |
|-----------|-----------|
| Инференс на NPU (RKNN) | Работает |
| Инференс на CPU (ONNX) | Работает |
| Конвертация YOLO → RKNN | Конвертируются все задачи |
| COCO mAP валидация | Для det/seg/pose (5000 изображений) |
| FPS бенчмарк | 111 моделей |
| Веб-интерфейс | Live viewer + тестовая галерея |
| Камеры: USB/UVC | Работает |
| Камеры: GigE Vision (Basler) | Работает |
| Камеры: RTSP | Работает |
| Камеры: V4L2 | Работает |
| Камеры: RTSP через OpenCV | Работает |
| Камеры: WebRTC | Нестабильно |
| Камеры: GStreamer | Нестабильно |

## Требования

- Docker Engine, пользователь в группе `docker`
- RK3588S с библиотеками NPU: `librknnrt.so`, `librknn_api.so`
- USB камера (MJPEG) или GigE Vision камера (Basler)

## Быстрый старт

```bash
docker compose up -d --build
```

Просмотр: `http://<host>:5000/`
Тестовая страница: `http://<host>:5000/test`

В углу экрана — `SVR:X.X CLT:X.X`: серверный FPS (захват + инференс) и клиентский FPS.

## Режимы камеры

| `CAM_MODE` | Описание |
|------------|----------|
| `auto` | GigE → USB → file (автофоллбэк) |
| `ip` | Только GigE по `CAM_SOURCE` |
| `usb` | Только USB камера |
| `file` | Файл/поток из `CAM_SOURCE` |

```bash
CAM_MODE=auto docker compose up -d
CAM_MODE=ip CAM_SOURCE=192.168.5.1 docker compose up -d
```

При потере GigE камеры автоматический фоллбэк на USB, при восстановлении — обратно.

## Управление моделями

Переключение по имени через `/reload` (без копирования файлов):

```bash
curl "http://localhost:5000/reload?model=yolo26n-sem&backend=cpu"
curl "http://localhost:5000/reload?model=yolo11n-pose"
```

- `?model=` ищет `<name>.rknn`, `<name>.onnx`, `<name>_postprocess.py` в `model/`
- `?backend=cpu` — форсировать ONNX CPU (если RKNN не нужен)
- Без параметров — загружает `model.rknn` / `model.onnx`

## Конвертация новой модели

Скопировать `.pt` в `converter/weights/` и запустить:

```bash
cd converter
docker compose run --rm converter convert my-model.pt
```

Конвертер автоматически:
1. Применяет RKNN-патчи (YOLO26, fuse bug)
2. Экспортирует ONNX + компилирует RKNN
3. Генерирует `postprocess.py` под задачу
4. Копирует результат в `model/`

Параметры:

```bash
docker compose run --rm converter convert <model> [--test] [--deploy]
```

## Бенчмарк

### FPS бенчмарк (все 111 моделей)

Запуск на хосте:

```bash
cd benchmark
python benchmark.py
```

Измеряет:
- **CPU FPS** — ONNX Runtime на CPU
- **NPU FPS** — RKNN на NPU
- **pixel-diff** — попиксельная разница ONNX vs RKNN (валидация конвертации)
- **NPU load** — средняя загрузка NPU за период замера

Результаты: `benchmark/results/summary.tsv`

### COCO mAP валидация

```bash
cd converter
docker compose run --rm converter python eval_coco.py
```

Валидация на 5000 изображений COCO val2017 для задач det/seg/pose.

## Результаты

### FPS (yolov8n-pose, 640×640)

| Метрика | CPU (ONNX) | NPU (RKNN) | Ускорение |
|---------|------------|------------|-----------|
| FPS | 4.09 | 16.29 | **4.0×** |
| FPS std | 0.55 | 2.20 | — |
| NPU load | — | 11.8% | — |

### COCO mAP (yolov8n-pose, 5000 изображений)

| Метрика | CPU ONNX | NPU FP32 | NPU INT8 |
|---------|----------|----------|----------|
| AP@50:95 | **0.499** | **0.499** | 0.463 |
| AP@50 | 0.786 | 0.787 | 0.773 |
| Precision | 0.535 | 0.536 | 0.485 |
| Recall | 0.425 | 0.426 | 0.386 |

**Выводы**:
- NPU FP32 **идентичен** CPU ONNX — потеря точности отсутствует (AP 0.499 vs 0.499)
- NPU INT8 **теряет 7.2%** AP@50:95 относительно FP32
- INT8 конвертация ускоряет инференс на ~17% относительно FP32 на NPU

### Выборка FPS (111 моделей)

| Модель | Задача | Разрешение | CPU FPS | NPU FPS | Ускорение |
|--------|--------|------------|---------|---------|-----------|
| yolov8n | detect | 640 | 6.15 | 7.73 | 1.3× |
| yolov8n-seg | seg | 640 | 3.12 | 9.92 | 3.2× |
| yolov8n-cls | cls | 224 | 85.25 | 67.75 | 0.8× |
| yolov8n-pose | pose | 640 | 5.88 | 7.64 | 1.3× |
| yolov8n-obb | obb | 1024 | 2.56 | 5.50 | 2.2× |
| yolov8s | detect | 640 | 2.43 | 5.38 | 2.2× |
| yolov8m | detect | 640 | 0.96 | 3.87 | 4.0× |
| yolo11n | detect | 640 | 7.33 | 7.28 | 1.0× |
| yolo11n-pose | pose | 640 | 6.45 | 6.88 | 1.1× |
| yolo26n | detect | 640 | 9.26 | 6.59 | 0.7× |

Полная таблица: `benchmark/results/summary.tsv`

## Доступные модели

### Ultralytics YOLO (111 моделей)

| Семейство | Размеры | Задачи | Количество |
|-----------|---------|--------|------------|
| YOLOv3 | u, sppu, tinyu | detect | 3 |
| YOLOv5 | nu/su/mu/lu/xu, n6u/s6u/m6u/l6u/x6u | detect | 10 |
| YOLOv8 | n/s/m/l/x | detect, seg, cls, pose, obb | 25 |
| YOLOv9 | t/s/m/c/e | detect | 5 |
| YOLOv9 | c, e | seg | 2 |
| YOLOv10 | n/s/m/b/l/x | detect | 6 |
| YOLO11 | n/s/m/l/x | detect, seg, cls, pose, obb | 25 |
| YOLO12 | n/s/m/l/x | detect | 5 |
| YOLO26 | n/s/m/l/x | detect, seg, cls, pose, obb, sem | 30 |

### PILOT набор (10 моделей)

`yolov3u`, `yolov5nu`, `yolo11n`, `yolo26n` (detect), `yolov8n` (detect, seg, pose, obb, cls), `yolo26n-sem`

## Эндпоинты

| Route | Метод | Описание |
|-------|-------|----------|
| `/` | GET | Live viewer (HTTP poll, SVR/CLT FPS) |
| `/frame` | GET | Последний обработанный JPEG (2s stale → placeholder) |
| `/objects` | GET | JSON: детекции + fps + frame_id |
| `/reload` | GET | Перезагрузка модели (`?model=name&backend=cpu`) |
| `/test` | GET | Тестовая страница (галерея + drag-drop) |
| `/test/images` | GET | JSON: список тестовых изображений |
| `/test/images/<name>` | GET | Отдача тестового изображения |
| `/test/images/<name>` | DELETE | Удаление из галереи |
| `/test/upload` | POST | Загрузка изображения в галерею |
| `/process` | POST | Обработка загруженного файла |

## Структура

```
webcam/
├── app.py                 # Flask приложение (create_app, endpoints)
├── pipeline.py            # Загрузка модели, инференс, постпроцесс (RLock)
├── main.py                # Точка входа (VideoLoop + HTTP + WebSocket)
├── config.py              # CLI/env конфигурация (WebcamConfig)
├── camera_source.py       # Абстракция камеры (USB/GigE/RTSP/V4L2/WebRTC/GStreamer)
├── video_loop.py          # Async capture → detect → notify loop
├── websocket_server.py    # WebSocket push для live viewer
├── detector_frame.py      # DetectorFrame dataclass
├── utils.py               # letterbox, scale_coords, NMS, DFL
├── camtest.py             # Диагностика USB камер
├── validate_npu.py        # Валидация ONNX vs RKNN
├── docker-compose.yml     # Сервис webcam
├── Dockerfile             # Образ (rknn-toolkit-lite2 + OpenCV + pypylon)
├── model/                 # Активная модель (.rknn/.onnx + _postprocess.py)
├── templates/
│   ├── index.html         # Live viewer (WebSocket + canvas)
│   └── test.html          # Тестовая страница (галерея + drag-drop)
├── test/                  # Тестовые изображения
├── benchmark/
│   ├── benchmark.py       # Оркестратор (конвертация + FPS)
│   ├── convert_bench.py   # Конвертация + pixel-diff валидация
│   ├── fps_bench.py       # FPS бенчмарк (CPU + NPU)
│   ├── models.py          # Списки моделей (FULL: 111, PILOT: 10)
│   └── results/           # Результаты (summary.tsv, eval logs)
├── converter/
│   ├── converter.py       # Конвертер YOLO → ONNX → RKNN
│   ├── postprocess.py     # Постпроцесс (NMS, scale, decode)
│   ├── eval_coco.py       # COCO mAP валидация
│   ├── validate_all.py    # Валидация всех моделей
│   ├── templates/         # Шаблоны постпроцесса
│   ├── patches/           # RKNN-патчи
│   ├── weights/           # Исходные .pt
│   └── model/             # Скомпилированные модели
└── datasets/
    └── coco_val2017/      # 5000 изображений + аннотации
```

## Конфигурация

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `BACKEND` | `npu` | `npu` или `cpu` |
| `MODEL` | `yolo11n` | Имя модели (без расширения) |
| `PORT` | `5000` | HTTP порт |
| `CAM_MODE` | `auto` | Режим камеры |
| `CAM_SOURCE` | — | Источник (IP/USB/path) |
| `CAM_W` / `CAM_H` | `640` / `480` | Разрешение камеры |
| `JPEG_QUALITY` | `85` | Качество JPEG (1–100) |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `MASK_FORMAT` | `contour` | Формат масок (contour/polygon) |
| `TEST_MODE` | `1` | Тестовый режим (файловый ввод) |

## Мониторинг NPU

```bash
cat /sys/kernel/debug/rknpu/load
# NPU load:  Core0: 45%, Core1: 30%, Core2: 0%,

watch -n 1 'cat /sys/kernel/debug/rknpu/load'
```

Для монтирования debugfs (если не смонтирован):

```bash
sudo mount -t debugfs none /sys/kernel/debug
sudo chmod a+rx /sys/kernel/debug
```

## Известные ограничения

- YOLO-NAS, YOLOWorld не поддерживаются (нет Ultralytics-совместимости)
- Компиляция RKNN на aarch64 медленная для больших моделей (>100 MB)
- RT-DETR runtime init падает на NPU (используется CPU)
- `quantize=16` (default) = float-модель; INT8 только с `quantize=8` + calibration
- INT8 калибровка на 64 изображениях (нужно увеличить до 512)
- GigE камера: первый захват может быть медленным (GVCP discovery)
- USB камера на WiFi: HTTP poll может давать ~5fps из-за RTT
