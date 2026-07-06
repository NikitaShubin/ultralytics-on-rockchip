# webcam

Real-time MJPEG-стрим с вебкамеры + YOLOv8 детекция на NPU (RK3588S).

## Внешние зависимости на хосте

Docker Engine (https://docs.docker.com/engine/install/ubuntu/), пользователь в группе `docker`.

### Обновление librknnrt.so

Хостовая библиотека `librknnrt.so` (rknpu2 v1.4.0) несовместима с моделями,
собранными rknn-toolkit2 v2.x — ошибка `Invalid RKNN model version 6`.
Необходимо заменить её версией из репозитория airockchip:

```bash
sudo curl -sLo /usr/lib/librknnrt.so \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
sudo cp /usr/lib/librknnrt.so /usr/lib/librknn_api.so
sudo ldconfig
```

## Сборка и запуск

```bash
docker compose up -d --build
```

Стрим на `http://<host>:5000`.

## Модели

По умолчанию используется **YOLOv8n** (nano) — в 2.3× быстрее v8m (~9 FPS vs ~3 FPS
end-to-end) и меньше ложных срабатываний.

Переключение на YOLOv8m — изменить `model_path` в `pipeline.py` на
`/app/model/yolov8m-rk3588.rknn`.

### Подготовка .rknn модели

Внутри контейнера с rknn-toolkit2 v2.3.2 (arm64) и ultralytics:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # или yolov8m.pt
model.export(format="rknn")
```

Готовый `.rknn` скопировать на хост и положить в `model/`.

## Известные проблемы

### 3×3 тайлинг детекций (YOLOv8m)

Модель YOLOv8m после экспорта через rknn-toolkit2 v2.3.2 выдаёт множественные
детекции одного объекта в сетке 3×3 (артефакт stride-16 head).
Смягчено пост-процессингом в `pipeline.py::_dedup()`: группировка боксов
по классу и соотношению размеров (>75%), сохранение лучшего по score.
