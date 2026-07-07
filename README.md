# webcam

Real-time YOLO детекция/pose/OBB на NPU (RK3588S) с вебкамеры.

**Архитектура**: OpenCV → Pipeline (RKNN inference + postprocess) → `/frame` endpoint → браузер через canvas-pull.
Низкая задержка (~300ms), нет накопления буфера.

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

## Модели

Конвертация `.pt` → `.onnx` → `.rknn` вынесена в `converter/`:

```bash
cd converter
docker compose run --rm converter convert <model_name>
```

Поддерживаемые задачи: `detect`, `pose`, `obb`.  
Пример: `docker compose run --rm converter convert yolo11n-pose`

Результат (`.rknn` + `postprocess.py`) копируется в `model/`, после чего нужно перезагрузить пайплайн:

```
curl http://<host>:5000/reload
```

### Готовые модели

- `yolo11n` — detect, 80 COCO классов, 2.9M params, ~12 FPS
- `yolo11n-pose` — pose estimation, 17 keypoints, 2.9M params, ~12 FPS

## Структура

```
webcam/
├── app.py              # Flask-приложение (CameraManager + endpoints)
├── pipeline.py         # Загрузка модели, pre/post-processing, инференс
├── docker-compose.yml  # Сервис webcam
├── Dockerfile          # Образ (rknn-toolkit-lite2 + OpenCV)
├── model/              # .rknn + postprocess.py
├── converter/          # Конвертация .pt → .onnx → .rknn
│   ├── templates/      # Шаблоны postprocess (detect/pose/obb)
│   ├── patches/        # RKNN-патчи для разных голов YOLO
│   └── Dockerfile      # Образ с rknn-toolkit2 + onnxruntime
```

## Эндпоинты

| Route | Описание |
|---|---|
| `/` | Просмотр с камеры (canvas pull, `setTimeout(next, 0)`) |
| `/frame` | Последний обработанный JPEG (no-cache) |
| `/test` | Drag-and-drop загрузка изображений |
| `/test/raw` | Тестовое изображение |
| `/process` | POST — обработка загруженного файла |
| `/reload` | Перезагрузить модель (после --deploy) |

## Горячее подключение камер

`volumes: /dev:/dev` — камера автоматически обнаруживается при подключении.
При отключении отображается заглушка "Camera not connected".

## Известные проблемы

- RT-DETR, YOLO-NAS, YOLOWorld не поддерживаются
- Компиляция RKNN на aarch64 медленная для больших моделей
