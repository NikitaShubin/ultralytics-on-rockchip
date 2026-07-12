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

# Конвертация INT8 (если указана модель)
if [ -n "$MODEL" ]; then
    echo "=== Конвертация $MODEL (INT8 калибровка) ==="
    cd converter
    docker compose run --rm converter convert "$MODEL" --quantize int8
    cd ..

    # Переименование: {model}_rkopt_o19.rknn → {model}.rknn
    RKNN_VERSIONED="model/${MODEL}_rkopt_o19.rknn"
    RKNN_CANONICAL="model/${MODEL}.rknn"
    if [ -f "$RKNN_VERSIONED" ] && [ ! -f "$RKNN_CANONICAL" ]; then
        mv "$RKNN_VERSIONED" "$RKNN_CANONICAL"
        echo "Переименовано: $RKNN_VERSIONED → $RKNN_CANONICAL"
    fi
fi

# Проверка наличия модели
RKNN_FOUND=false
if [ -n "$MODEL" ] && [ -f "model/${MODEL}.rknn" ]; then
    RKNN_FOUND=true
elif [ -f "model/model.rknn" ]; then
    RKNN_FOUND=true
elif [ -f "model/model.onnx" ]; then
    RKNN_FOUND=true
fi

if [ "$RKNN_FOUND" = false ]; then
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
