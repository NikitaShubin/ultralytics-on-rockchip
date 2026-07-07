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
cp converter/postprocess.py                       model/postprocess.py

# Перезагрузить pipeline
curl http://localhost:5000/reload
```

Либо одной командой через converter (если модель уже скомпилирована):
```bash
cd converter
docker compose run --rm converter convert yolo11s-pose.pt --deploy
```

## Конвертация новой модели

Положить `.pt` файл в `converter/weights/` и запустить:

```bash
cd converter
docker compose run --rm converter convert my-model.pt
```

Конвертер сам:
1. Применяет патчи для совместимости с RKNN
2. Экспортирует ONNX (RK-opt) + компилирует RKNN
3. Генерирует `postprocess.py` под задачу модели
4. Копирует результат в `converter/model/` и `model/`

Поддерживаемые задачи: `detect`, `segment`, `pose`, `obb`, `classify`.

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

## Структура

```
webcam/
├── app.py              # Flask-приложение (CameraManager + endpoints)
├── pipeline.py         # Загрузка модели, pre/post-processing, инференс
├── docker-compose.yml  # Сервис webcam
├── Dockerfile          # Образ (rknn-toolkit-lite2 + OpenCV)
├── model/              # Текущая активная модель (model.rknn + postprocess.py)
├── converter/
│   ├── converter.py    # Конвертация .pt → ONNX → RKNN + генерация postprocess
│   ├── templates/      # Шаблоны postprocess (detect/pose/segment/obb/classify)
│   ├── patches/        # RKNN-патчи для разных голов YOLO
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
| `/reload` | Перезагрузить модель (после ручного копирования) |

## Горячее подключение камер

`volumes: /dev:/dev` — камера автоматически обнаруживается при подключении.
При отключении отображается заглушка "Camera not connected".

## Известные проблемы

- RT-DETR, YOLO-NAS, YOLOWorld не поддерживаются
- Компиляция RKNN на aarch64 медленная для больших моделей
