# webcam

Real-time YOLO детекция/сегментация/pose/OBB/классификация на NPU (RK3588S) с вебкамеры.

**Архитектура**: OpenCV → Pipeline (RKNN inference + postprocess) → `/frame` endpoint → браузер через canvas-pull.
Низкая задержка (~200-300ms), нет накопления буфера.

## Требования

- Docker Engine, пользователь в группе `docker`
- RK3588S с установленными библиотеками NPU (`librknnrt.so`, `librknn_api.so`)
- USB камера с поддержкой MJPEG

## Быстрый старт

```bash
docker compose up -d --build
```

Просмотр: `http://<host>:5000/`  
Тестовая страница: `http://<host>:5000/test`

## Переключение между моделями

Все скомпилированные модели лежат в `converter/model/`:

```bash
# Посмотреть список доступных моделей
ls converter/model/

# Развернуть нужную (например yolo11s-pose)
cp converter/model/yolo11s-pose_rkopt_o19.rknn  model/model.rknn
cp converter/model/yolo11s-pose_rkopt_o19.onnx   model/model.onnx
cp converter/model/yolo11s-pose_postprocess.py   model/postprocess.py

# Перезагрузить pipeline
curl http://localhost:5000/reload
```

Либо одной командой через converter (если модель уже скомпилирована):
```bash
cd converter
docker compose run --rm converter convert yolo11s-pose.pt --deploy
```

Модели YOLO26 (OBB26, Pose26, Segment26) можно переключать по имени:
```bash
# Положить .rknn, .onnx, _postprocess.py в converter/model/
# Затем перезагрузить с указанием имени:
curl "http://localhost:5000/reload?model=yolo26n-obb"
# Без ?model= загружается model.rknn / MODEL env / yolo11n по умолчанию
```

## Конвертация новой модели

Поддерживаются любые Ultralytics-совместимые `.pt` веса (YOLOv5-v11, YOLO26, RT-DETR, официальные или кастомно обученные)
с задачами `detect`, `segment`, `pose`, `obb`, `classify`.

Скопировать `.pt` файл в `converter/weights/` и запустить:

```bash
cd converter
docker compose run --rm converter convert my-model.pt
```

Если файл лежит в корне `converter/`, передать путь явно:

```bash
docker compose run --rm converter convert ./my-custom-model.pt
```

Конвертер сам:
1. Применяет патчи для совместимости с RKNN (YOLO26, fuse bug workaround, и т.д.)
2. Экспортирует ONNX (RK-opt) + компилирует RKNN
3. Генерирует `postprocess.py` под задачу модели (с учетом `reg_max`, anchor offset и т.д.)
4. Копирует результат в `converter/model/` и `model/`

### Параметры

```bash
docker compose run --rm converter convert <model> [--test] [--deploy]

# --test   — валидация ONNX vs RKNN на coco8 (только для detect)
# --deploy — копировать в model/ и вызвать /reload
```

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

YOLO26 (reg_max=1, no DFL, float — не INT8):
| Модель | Задача | Размер RKNN |
|--------|--------|-------------|
| `yolo26n-seg` | segment | ~7 MB |
| `yolo26n-pose` | pose | ~6 MB |
| `yolo26x-pose` | pose | 116 MB |
| `yolo26n-obb` | obb | 7.2 MB |

## Структура

```
webcam/
├── app.py              # Flask-приложение (CameraManager + endpoints)
├── pipeline.py         # Загрузка модели, pre/post-processing, инференс (RLock, model_name)
├── docker-compose.yml  # Сервис webcam (MODEL env var)
├── Dockerfile          # Образ (rknn-toolkit-lite2 + OpenCV)
├── model/              # Текущая активная модель (model.rknn + postprocess.py)
├── converter/
│   ├── converter.py    # Конвертация .pt → ONNX → RKNN + генерация postprocess
│   ├── templates/      # Шаблоны postprocess (detect/pose/segment/obb/classify/detr)
│   ├── patches/        # RKNN-патчи для разных голов YOLO (+ YOLO26, fuse bug)
│   ├── model/          # Скомпилированные .rknn/.onnx всех моделей
│   ├── weights/        # Исходные .pt файлы
│   ├── docker-compose.yml
│   └── Dockerfile      # Образ с rknn-toolkit2 + onnxruntime + ultralytics
```

## Эндпоинты

| Route | Описание |
|---|---|
| `/` | Просмотр с камеры (canvas pull, `setTimeout(next, 0)`) |
| `/frame` | Последний обработанный JPEG (no-cache) |
| `/test` | Drag-and-drop загрузка изображений |
| `/test/raw` | Тестовое изображение |
| `/process` | POST — обработка загруженного файла |
| `/reload` | Перезагрузить модель (с опцией `?model=name`) |

## Горячее подключение камер

`volumes: /dev:/dev` — камера автоматически обнаруживается при подключении.
При отключении отображается заглушка "Camera not connected".

## Известные проблемы

- YOLO-NAS, YOLOWorld не поддерживаются
- Компиляция RKNN на aarch64 медленная для больших моделей
- RT-DETR: runtime init падает на NPU (используется CPU), GridSample — CPU fallback
- OBB постпроцесс поддерживает два формата выхода модели: единый `(1, C, total_grid)` и 9 отдельных (бокс/класс/угол на stride). Определяется автоматически. Для единого выхода применяет Sigmoid к class scores.
- `quantize=16` (по умолчанию) = float-модель (не INT8). INT8 только с `quantize=8` + calibration dataset.
