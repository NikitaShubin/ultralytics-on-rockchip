# webcam

Real-time MJPEG-стрим с вебкамеры + YOLO детекция на NPU (RK3588S).

## Требования

- Docker Engine, пользователь в группе `docker`
- RK3588S с установленными библиотеками NPU (`librknnrt.so`, `librknn_api.so`)

## Быстрый старт

```bash
docker compose up -d --build
```

Стрим на `http://<host>:5000`.

## Модели

По умолчанию используется **yolo11n** (RK-opt ONNX → RKNN).

Готовые `.rknn` лежат в `model/`. Файл модели задаётся в `pipeline.py` (переменная `MODEL_PATH`).

### Конвертация моделей

Конвертер вынесен в отдельное docker-окружение (`converter/`):

```bash
cd converter
docker compose run --rm converter convert <model_name>
```

Например:

```bash
docker compose run --rm converter convert yolov8n
```

Результат (`.rknn` + `.onnx`) копируется в `model/`.

**Пакетная выгрузка ONNX (без компиляции RKNN):**

```bash
docker compose run --rm converter batch-onnx --model-list models_detect_all.txt
```

Список поддерживаемых моделей: `models_detect.txt` (25 моделей), `models_detect_all.txt` (48 моделей).

## Структура проекта

```
webcam/
├── app.py              # Flask-приложение (MJPEG-стрим)
├── pipeline.py         # Загрузка модели, pre/post-processing, инференс
├── docker-compose.yml  # Сервис webcam
├── Dockerfile          # Образ webcam (rknn-toolkit-lite2)
├── model/              # Готовые .rknn и .onnx модели
└── converter/          # Инструмент конвертации .pt → .onnx → .rknn
```

## Известные проблемы

- Модели с нестандартными головами (RT-DETR, YOLO-NAS, YOLOWorld) не поддерживаются
- Компиляция RKNN (`rknn.build`) на aarch64 занимает много времени для больших моделей
