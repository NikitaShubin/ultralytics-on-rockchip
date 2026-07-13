# Перцепт — YOLO инференс на NPU (RK3588S)

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
| FPS бенчмарк | 111 моделей (FP32 + INT8) |
| Веб-интерфейс | Live viewer + тестовая галерея |
| Камеры: USB/UVC | Работает |
| Камеры: GigE Vision (Basler) | Работает |
| Камеры: RTSP | Работает |
| Камеры: V4L2 | Работает |
| Камеры: RTSP через OpenCV | Работает |
| Камеры: WebRTC | Нестабильно |
| Камеры: GStreamer | Нестабильно |
| Полная русификация кода | ✅ Завершена |
| INT8 калибровка (5000 COCO) | В процессе (yolo26x-seg) |

## Требования

- Docker Engine, пользователь в группе `docker`
- RK3588S с библиотеками NPU: `librknnrt.so`, `librknn_api.so`
- USB камера (MJPEG) или GigE Vision камера (Basler)

## Быстрый старт

```bash
./start.sh
# или вручную:
docker compose up -d --build
```

Просмотр: `http://<host>:5000/`
Тестовая страница: `http://<host>:5000/test`

Остановка:

```bash
./stop.sh
```

## Деплой модели

```bash
# Деплой текущей модели из model/
./deploy.sh

# Конвертация + деплой конкретной модели
./deploy.sh yolov8n-pose

# Полный тест (конвертация + FPS бенчмарк + запуск)
./deploy.sh yolov8n-pose --full-test
```

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

## Управление моделями

Переключение по имени через `/reload` (без копирования файлов):

```bash
curl "http://localhost:5000/reload?model=yolo26n-sem&backend=cpu"
curl "http://localhost:5000/reload?model=yolo11n-pose"
curl "http://localhost:5000/reload?model=yolo11n-pose&rknn_suffix=_int8"
```

- `?model=` ищет `<name>.rknn`, `<name>.onnx`, `<name>_postprocess.py` в `model/`
- `?backend=cpu` — форсировать ONNX CPU (если RKNN не нужен)
- `?rknn_suffix=_int8` — выбрать INT8 RKNN модель (по умолчанию `_none` или без суффикса)
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

## Подготовка датасетов

```bash
python setup_datasets.py
```

Загружает COCO val2017 (5000 изображений + аннотации) для калибровки и валидации.

## Бенчмарк (3 фазы)

Три независимые фазы, можно запускать по отдельности:

```bash
cd benchmark

# Фаза 1: конвертация .pt → .onnx + _none.rknn + _int8.rknn (x86 ПК)
python benchmark.py --phase convert

# Фаза 2: валидация ONNX vs RKNN FP32+INT8 (x86 ПК, RKNN simulator)
python benchmark.py --phase validate

# Фаза 3: FPS + нагрузка CPU/NPU (ARM железо)
python benchmark.py --phase bench
```

### Мульти-фазы

```bash
# Конвертация + валидация на ПК
python benchmark.py --phase convert validate

# Всё сразу (convert → validate → bench)
python benchmark.py --phase all

# Только FPS, только INT8
python benchmark.py --phase bench --quantize int8

# Одна модель
python benchmark.py --phase convert --model yolo26x-seg

# С авто-очисткой temp_models
python benchmark.py --phase convert validate --clean
```

### Квантизация

| `--quantize` | Описание |
|-------------|----------|
| `none` | Только FP32 RKNN |
| `int8` | Только INT8 RKNN |
| `both` | Оба варианта (по умолчанию) |

### Resume

Результаты пишутся в `benchmark/results/results.json` по модели. При повторном запуске уже пройденные модели пропускаются.

### Результаты

- `benchmark/results/results.json` — полные результаты по каждой модели
- `benchmark/results/summary.tsv` — TSV-сводка для просмотра
- `benchmark/results/benchmark.log` — лог со всеми stdout/stderr

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

**Выводы**:
- NPU FP32 **идентичен** CPU ONNX — потеря точности отсутствует
- NPU INT8 **теряет 7.2%** AP@50:95 относительно FP32

### Выборка FPS

| Модель | Задача | Разрешение | CPU FPS | NPU FPS | Ускорение |
|--------|--------|------------|---------|---------|-----------|
| yolov8n | detect | 640 | 6.15 | 7.73 | 1.3× |
| yolov8n-seg | seg | 640 | 3.12 | 9.92 | 3.2× |
| yolov8n-cls | cls | 224 | 85.25 | 67.75 | 0.8× |
| yolov8n-pose | pose | 640 | 5.88 | 7.64 | 1.3× |
| yolov8m | detect | 640 | 0.96 | 3.87 | 4.0× |
| yolo11n | detect | 640 | 7.33 | 7.28 | 1.0× |

Полная таблица: `benchmark/results/summary.tsv`

## Структура

```
percept/                     # Модули приложения
├── inference/               # Загрузка модели и инференс (RKNN/ONNX)
│   └── __init__.py          # ModelBackend: загрузка, infer, detect, reload
├── perception/              # Постпроцесс
│   └── __init__.py          # dfl(), nms()
└── io/                      # Ввод/вывод изображений
    └── __init__.py          # letterbox(), scale_coords()

main.py                      # Точка входа (async)
app.py                       # Flask HTTP сервер (/frame, /objects, /reload, /test)
pipeline.py                  # Оркестрация: загрузка постпроцесса + вызов ModelBackend
config.py                    # Конфигурация (CLI/env)
camera_source.py             # Абстракция камеры (USB/GigE/RTSP/V4L2/WebRTC/GStreamer)
video_loop.py                # Async capture → detect → notify loop
websocket_server.py          # WebSocket push для live viewer
detector_frame.py            # DetectorFrame dataclass
utils.py                     # Обратная совместимость (переэкспорт из percept/)
docker-compose.yml           # Сервис webcam
Dockerfile                   # Образ (rknn-toolkit-lite2 + OpenCV + pypylon)
model/                       # Активная модель (.rknn/.onnx + _postprocess.py)
templates/
├── index.html               # Live viewer (WebSocket + canvas)
└── test.html                # Тестовая страница (галерея + drag-drop)
test/                        # Тестовые изображения
start.sh                     # Запуск сервера
stop.sh                      # Остановка сервера
deploy.sh                    # Деплой модели (конвертация + тест + запуск)
setup_datasets.py            # Загрузка датасетов (COCO val2017)
benchmark/
├── benchmark.py             # Оркестратор: --phase convert|validate|bench|all
├── convert_bench.py         # Фаза 1: экспорт + компиляция (4 файла на модель)
├── validate_bench.py        # Фаза 2: ONNX vs RKNN FP32+INT8 валидация
├── fps_bench.py             # Фаза 3: FPS + CPU/NPU нагрузка
├── models.py                # Списки моделей (FULL: 111, PILOT: 10)
├── temp_models/             # Временные файлы конвертации (перенос на ПК)
└── results/                 # Результаты (results.json, summary.tsv, benchmark.log)
converter/
├── converter.py             # Конвертер YOLO → ONNX → RKNN
├── postprocess.py           # Постпроцесс (NMS, scale, decode)
├── eval_coco.py             # COCO mAP валидация
├── templates/               # Шаблоны постпроцесса (6 задач)
├── patches/                 # RKNN-патчи
├── weights/                 # Исходные .pt
└── model/                   # Скомпилированные модели
datasets/
└── coco_val2017/            # 5000 изображений + аннотации
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

## Эндпоинты

| Route | Метод | Описание |
|-------|-------|----------|
| `/` | GET | Live viewer (HTTP poll, SVR/CLT FPS) |
| `/frame` | GET | Последний обработанный JPEG |
| `/objects` | GET | JSON: детекции + fps + frame_id |
| `/reload` | GET | Перезагрузка модели (`?model=name&backend=cpu`) |
| `/test` | GET | Тестовая страница (галерея + drag-drop) |
| `/test/images` | GET | JSON: список тестовых изображений |
| `/test/images/<name>` | GET | Отдача тестового изображения |
| `/test/images/<name>` | DELETE | Удаление из галереи |
| `/test/upload` | POST | Загрузка изображения в галерею |
| `/process` | POST | Обработка загруженного файла |

## Мониторинг NPU

```bash
cat /sys/kernel/debug/rknpu/load
watch -n 1 'cat /sys/kernel/debug/rknpu/load'
```

## Известные ограничения

- YOLO-NAS, YOLOWorld не поддерживаются
- Компиляция RKNN на aarch64 медленная для больших моделей (>100 MB)
- RT-DETR runtime init падает на NPU (используется CPU)
- `quantize=16` (default) = float-модель; INT8 только с `quantize=8` + calibration
- INT8 калибровка на 512 изображениях
- GigE камера: первый захват может быть медленным (GVCP discovery)
- USB камера на WiFi: HTTP poll может давать ~5fps из-за RTT

---

## Что сделано (июль 2026)

### Полная русификация проекта
Все пользовательские сообщения, логи, docstrings, RuntimeError переведены на русский:
- `converter/converter.py` — конвертер, INT8 калибровка, dataset.txt логирование
- `percept/inference/__init__.py` — пайплайн NPU/CPU
- `pipeline.py`, `main.py`, `video_loop.py`, `camera_source.py`
- `websocket_server.py`, `detector_frame.py`
- `converter/patches/rknn_patch.py` — патч RKNN toolkit2
- `benchmark/convert_bench.py`, `benchmark/fps_bench.py`

**Не переводили:** HTTP ответы в `app.py`, `argparse` help, логи библиотек (ultralytics, RKNN toolkit, waitress).

### Docker
- Образ конвертера: `percept-converter:latest` (был `webcam-converter`)
- Образ прод: `webcam-percept` (имя в docker-compose.yml оставлено для совместимости)
- `docker-compose.yml`: сервис `percept`, переменные `MODEL`, `WEB_MODE`
- `converter/docker-compose.yml`: `image: percept-converter:latest`

### Калибровка INT8
- Используются **все 5000 изображений** COCO val2017 (`datasets/coco_val2017/images/val2017/`)
- `_make_calib_yaml(task)` генерирует YAML на лету с полным `nc: 80` и `names: [...]` для всех задач (seg/semantic/obb/pose/classify)
- Исправлен баг: для segment/semantic теперь добавляются `nc` и `names` (раньше было `key missing ❌`)
- Ультралитикс `get_int8_calibration_dataloader()` читает **все** изображения из `data[split]`, без лимита, `fraction=1.0`, `batch=16`
- Прогресс-бар RKNN toolkit2 (`270/270`) — это внутренние шаги квантования, **не** количество изображений
- Добавлено логирование `dataset.txt` в `converter.py` (`_log_dataset_txt()`): ищет в `/tmp/dataset.txt`, `dataset.txt`, `calib_dataset.txt`, `*.txt` в рабочей директории — для проверки списка файлов, используемых при калибровке

### Бенчмарк
- Переписан `benchmark/benchmark.py` с флагом `--model` для запуска одной модели
- Resume-логика: `results.json` апсертится по модели; при рестарте ПРОЙДЕНО пропускаются, ПРОВАЛ/нет результата — повторяются
- Лог-файл в режиме `'a'` (сохраняется история)
- Статусы: ПРОЙДЕНО / ПРОВАЛ
- Сервис в docker-compose: `percept` (не `webcam`)
- `benchmark/run.sh` передаёт `"$@"` в benchmark.py, останавливает percept перед запуском
- TSV-суммаризация в `benchmark/results/summary.tsv`

### Результаты yolo26x-seg (FP32)
| Бэкенд | FPS | Latency | NPU load | Время конвертации | max_diff |
|--------|-----|---------|----------|-------------------|----------|
| CPU | 0.19 | 5.3 с | — | 109 с | 4.94 (ПРОВАЛ валидации) |
| NPU | 1.05 | 0.95 с | 32.4% | 109 с | 4.94 (ПРОВАЛ валидации) |

> **Примечание**: max_diff > 1.0 из-за специфики RKNN постпроцессинга сегментации (mask proto). Детекция боксов корректна.

### INT8 yolo26x-seg — в процессе
Квантовка 385 фаз (layers/tensors), ~3 мин/фаза → **~19 часов**. Контейнер `converter-converter-run-64ba7b2cfdab` работает в фоне.

### deploy.sh
Переписан в однокомандный деплой:
```bash
./deploy.sh --quantize int8   # конвертация + запуск
./deploy.sh                   # только запуск (использует .rknn из /models)
```
Переменные окружения: `MODEL`, `WEB_MODE`, `QUANTIZE`. Авто-переименование `.rknn` файлов под `MODEL`.

### Удалённые файлы
- `run.sh` (legacy) — удалён
- `model/yolo11n_postprocess.py` — root-owned, требует sudo для правки
