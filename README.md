# webcam

Real-time YOLO детекция/сегментация/pose/OBB/классификация/семантическая сегментация на NPU (RK3588S)
с USB/GigE камеры.

**Архитектура**: `CameraSource` → `VideoLoop` (capture → Pipeline) → Flask HTTP + WebSocket → браузер.

## Требования

- Docker Engine, пользователь в группе `docker`
- RK3588S с установленными библиотеками NPU (`librknnrt.so`, `librknn_api.so`)
- USB камера (MJPEG) или GigE Vision камера (Basler, pypylon)
- OpenCV 5.x

## Быстрый старт

```bash
docker compose up -d --build
```

Просмотр: `http://<host>:5000/`
Тестовая страница: `http://<host>:5000/test`

## Режимы камеры

Управляются переменной `CAM_MODE`:

| Значение | Приоритет | Описание |
|----------|-----------|----------|
| `auto` (default) | GigE → USB → file | Сначала ищет GigE камеру через GVCP broadcast, затем USB, затем файл |
| `ip` | GigE only | Только GigE камера, без fallback |
| `usb` | USB only | Только USB камера |
| `file` | file only | Воспроизведение файла/потока по `CAM_SOURCE` |

Пример с GigE камерой:
```bash
CAM_MODE=auto docker compose up -d
# или явно:
CAM_MODE=ip CAM_SOURCE=192.168.5.1 docker compose up -d
```

## Управление моделями

Переключение по имени через `/reload` без копирования файлов:

```bash
curl "http://localhost:5000/reload?model=yolo26n-sem&backend=cpu"
curl "http://localhost:5000/reload?model=yolo11n-pose"
```

`?model=` ищет `<name>.rknn`, `<name>.onnx`, `<name>_postprocess.py` в `model/`.
`?backend=cpu` — форсировать ONNX CPU (если RKNN не нужен или не работает).
Без параметров — загружает `model.rknn` / `model.onnx` / `model/env`.

Поддерживаются задачи: `detect`, `segment`, `pose`, `obb`, `classify`, `sem`.

## Доступные модели

| Модель | Задача | Размер RKNN | Параметры |
|--------|--------|-------------|-----------|
| `yolo11n` | detect | 6.2 MB | 2.9M |
| `yolo11s` | detect | 18 MB | 9.5M |
| `yolo11m` | detect | 42 MB | 20.2M |
| `yolo11n-seg` | segment | 7.3 MB | 2.9M |
| `yolo11s-seg` | segment | 21 MB | 9.5M |
| `yolo11m-seg` | segment | 46 MB | 22.4M |
| `yolo11n-pose` | pose | 6.2 MB | 2.9M |
| `yolo11s-pose` | pose | 21 MB | 9.5M |
| `yolo11m-pose` | pose | 43 MB | 20.2M |
| `yolo11n-obb` | obb | 6.1 MB | 2.9M |
| `yolo11n-cls` | classify | 6.5 MB | 1.6M |
| `yolo12n` | detect | 7.5 MB | 2.6M |
| `yolo12x` | detect | 127 MB | 59.1M |

YOLO26 (reg_max=1, float):

| Модель | Задача | Размер RKNN |
|--------|--------|-------------|
| `yolo26n-seg` | segment | ~7 MB |
| `yolo26n-pose` | pose | ~6 MB |
| `yolo26x-pose` | pose | 116 MB |
| `yolo26n-obb` | obb | 7.2 MB |
| `yolo26n-sem` | semantic seg | 6.7 MB |

## Конвертация новой модели

Скопировать `.pt` в `converter/weights/` и запустить:

```bash
cd converter
docker compose run --rm converter convert my-model.pt
```

Конвертер сам:
1. Применяет патчи для совместимости с RKNN (YOLO26, fuse bug)
2. Экспортирует ONNX + компилирует RKNN
3. Генерирует `postprocess.py` под задачу
4. Копирует результат в `converter/model/` и `../model/`

Параметры:

```bash
docker compose run --rm converter convert <model> [--test] [--deploy]
```

## Тестовая страница

`http://<host>:5000/test` — галерея предустановленных изображений:

- Клик по превью → загрузка + обработка (маски/боксы)
- Drag-and-drop своего изображения → обработка + кнопка "Save to gallery"
- Кнопка × на превью → удаление из галереи
- Два режима отображения: оригинал (слева) и результат (справа)

## Эндпоинты

| Route | Метод | Описание |
|-------|-------|----------|
| `/` | GET | Просмотр с камеры (WebSocket + HTTP fallback) |
| `/frame` | GET | Последний обработанный JPEG |
| `/objects` | GET | JSON с детекциями |
| `/reload` | GET | Перезагрузка модели (`?model=name&backend=cpu`) |
| `/test` | GET | Тестовая страница |
| `/test/images` | GET | JSON список тестовых изображений |
| `/test/images/<name>` | GET | Отдача тестового изображения |
| `/test/images/<name>` | DELETE | Удаление из галереи |
| `/test/upload` | POST | Загрузка изображения в галерею |
| `/process` | POST | Обработка загруженного файла |

## Структура

```
webcam/
├── app.py              # Flask приложение (create_app, endpoints)
├── pipeline.py         # Загрузка модели, инференс, постпроцесс (RLock)
├── main.py             # Точка входа (VideoLoop + HTTP + WebSocket)
├── config.py           # CLI/env конфигурация (WebcamConfig)
├── camera_source.py    # Абстракция камеры (USB/GigE/File)
├── video_loop.py       # Async capture → detect → notify loop
├── websocket_server.py # WebSocket push для live viewer
├── detector_frame.py   # DetectorFrame dataclass
├── utils.py            # letterbox, scale_coords, NMS, DFL
├── camtest.py          # Диагностика USB камер
├── validate_npu.py     # Валидация ONNX vs RKNN
├── docker-compose.yml  # Сервис webcam
├── Dockerfile          # Образ (rknn-toolkit-lite2 + OpenCV + pypylon)
├── model/              # Активная модель (.rknn/.onnx + _postprocess.py)
├── templates/
│   ├── index.html      # Live viewer (WebSocket + canvas)
│   └── test.html       # Тестовая страница (галерея + drag-drop)
├── test/               # Тестовые изображения
└── converter/          # Конвертация .pt → ONNX → RKNN
    ├── converter.py
    ├── templates/      # Шаблоны постпроцесса
    ├── patches/        # RKNN-патчи
    ├── model/          # Скомпилированные модели
    └── weights/        # Исходные .pt
```

## Известные проблемы

- YOLO-NAS, YOLOWorld не поддерживаются
- Компиляция RKNN на aarch64 медленная для больших моделей
- RT-DETR runtime init падает на NPU (используется CPU)
- OBB постпроцесс: автоматическое определение формата выхода (единый или 9 отдельных)
- `quantize=16` (default) = float-модель; INT8 только с `quantize=8` + calibration
- GigE камера: первый захват может быть медленным (GVCP discovery + auto-negotiation packet size)
