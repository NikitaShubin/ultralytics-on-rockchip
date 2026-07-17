#!/bin/bash
# Деплой модели: конвертация INT8 + запуск сервиса
#
# Использование:
#   ./deploy.sh                    # Деплой текущей модели из model/
#   ./deploy.sh yolo26n-pose       # Конвертация INT8 + деплой
#   ./deploy.sh yolov8n-pose --full-test  # Конвертация + FPS тест
#
set -e

cd "$(dirname "$0")"

MODEL="${1:-}"
FULL_TEST=false

if [ "$2" = "--full-test" ]; then
    FULL_TEST=true
fi

# Проверка датасета для калибровки (автозагрузка при необходимости)
COCO_DIR="datasets/coco_val2017/images/val2017"
if [ ! -d "$COCO_DIR" ] || [ -z "$(ls "$COCO_DIR"/*.jpg 2>/dev/null)" ]; then
    echo "=== Загрузка COCO val2017 (~19 GB) ==="
    python3 setup_datasets.py
fi

# Конвертация INT8 (если указана модель)
if [ -n "$MODEL" ]; then
    if [ -f "model/model.rknn" ]; then
        echo "=== Модель уже сконвертирована, пропуск калибровки ==="
    else
        echo "=== Конвертация $MODEL (INT8 калибровка) ==="
        cd converter
        docker compose run --rm converter convert "$MODEL" --quantize int8
        cd ..
    fi
fi

# Проверка наличия модели
if [ ! -f "model/model.rknn" ] && [ ! -f "model/model.onnx" ]; then
    echo "ОШИБКА: Модель не найдена в model/"
    echo "Укажите имя модели: ./deploy.sh yolo26n-pose"
    exit 1
fi

# FPS тест
if [ "$FULL_TEST" = true ]; then
    echo "=== FPS тест ==="
    cd benchmark
    python benchmark.py
    cd ..
fi

# Запуск сервиса
echo "=== Запуск сервиса ==="
if [ -n "$MODEL" ]; then
    MODEL="$MODEL" WEB_MODE=2 docker compose up -d --build
else
    WEB_MODE=2 docker compose up -d --build
fi

echo ""
echo "Сервис запущен!"
echo "  Просмотр: http://localhost:5000/"
echo "  Тест:     http://localhost:5000/test"
echo ""
echo "Для остановки: ./stop.sh"
