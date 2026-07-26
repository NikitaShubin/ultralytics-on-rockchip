#!/bin/bash
# Деплой модели: конвертация + запуск сервиса
#
# Использование:
#   ./deploy.sh yolo26n-pose                # FP16 RKNN (быстрый старт)
#   ./deploy.sh yolo26n-pose --fp16         # FP16 RKNN (то же что без флага)
#   ./deploy.sh yolo26n-pose --int8         # INT8 без калибровки
#   ./deploy.sh yolo26n-pose --int8-calib   # INT8 с калибровкой (нужен датасет)
#   ./deploy.sh yolo26n-pose --onnx         # ONNX на CPU (без NPU)
#   ./deploy.sh yolo26n-pose --full-test    # Конвертация + FPS тест
#   ./deploy.sh                             # Перезапуск текущей модели из model/
#
set -e

cd "$(dirname "$0")"

# Docker создаёт mount-точки от root, если их нет.
# Создаём заранее, чтобы монтировались user-owned директории.
ensure_mountpoints() {
    mkdir -p datasets/coco_val2017 datasets/dotav1 datasets/cityscapes datasets/imagenet model cache
}

MODEL="${1:-}"
FULL_TEST=false
QUANTIZE="fp16"
BACKEND=""

for arg in "$@"; do
    case "$arg" in
        --fp16)         QUANTIZE="fp16" ;;
        --int8)         QUANTIZE="int8" ;;
        --int8-calib)   QUANTIZE="int8-calib" ;;
        --onnx)         QUANTIZE="onnx"; BACKEND="cpu" ;;
        --full-test)    FULL_TEST=true ;;
    esac
done

# Определить нужный датасет по имени модели
detect_dataset() {
    local model="$1"
    case "$model" in
        *-obb)       echo "dota" ;;
        *-sem)       echo "cityscapes" ;;
        *-cls)       echo "imagenet" ;;
        *)           echo "coco" ;;
    esac
}

# Гарантируем mount-точки до любого вызова Docker
ensure_mountpoints

# Скачать датасет только для int8-calib и только нужный
if [ "$QUANTIZE" = "int8-calib" ] && [ -n "$MODEL" ]; then
    DATASET=$(detect_dataset "$MODEL")

    if [ "$DATASET" = "coco" ]; then
        COCO_DIR="datasets/coco_val2017/images/val2017"
        if [ ! -d "$COCO_DIR" ] || [ -z "$(ls "$COCO_DIR"/*.jpg 2>/dev/null)" ]; then
            echo "=== Загрузка COCO val2017 (~19 GB) для калибровки INT8 ==="
            python3 setup_datasets.py --coco
        fi
    elif [ "$DATASET" = "dota" ]; then
        DOTA_DIR="datasets/dotav1/images/val"
        if [ ! -d "$DOTA_DIR" ] || [ -z "$(ls "$DOTA_DIR"/*.png 2>/dev/null)" ]; then
            echo "=== Загрузка DOTAv1 (~1.9 GB) для калибровки INT8 OBB ==="
            python3 setup_datasets.py --dota
        fi
    elif [ "$DATASET" = "cityscapes" ]; then
        CS_DIR="datasets/cityscapes/images/val"
        if [ ! -d "$CS_DIR" ] || [ -z "$(ls "$CS_DIR"/*.png 2>/dev/null)" ]; then
            echo "=== Загрузка Cityscapes val для калибровки INT8 Semantic ==="
            python3 setup_datasets.py --cityscapes
        fi
    elif [ "$DATASET" = "imagenet" ]; then
        IMAGENET_DIR="datasets/imagenet/val"
        if [ ! -d "$IMAGENET_DIR" ] || [ -z "$(ls "$IMAGENET_DIR"/*.JPEG 2>/dev/null)" ]; then
            echo "=== Загрузка ImageNet val (~6.3 GB) для калибровки INT8 Classify ==="
            python3 setup_datasets.py --imagenet
        fi
    fi
fi

# Конвертация (если указана модель)
if [ -n "$MODEL" ]; then
    ensure_mountpoints
    echo "=== Конвертация $MODEL (квантизация: $QUANTIZE) ==="
    cd convert
    docker compose run --rm convert "$MODEL" --"$QUANTIZE"
    cd ..
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
    python benchmark.py --phase bench
    cd ..
fi

# Запуск сервиса
echo "=== Запуск сервиса ==="
ensure_mountpoints
if [ -n "$MODEL" ] && [ -n "$BACKEND" ]; then
    MODEL="$MODEL" BACKEND="$BACKEND" WEB_MODE=2 docker compose up -d --build
elif [ -n "$MODEL" ]; then
    MODEL="$MODEL" WEB_MODE=2 docker compose up -d --build
elif [ -n "$BACKEND" ]; then
    BACKEND="$BACKEND" WEB_MODE=2 docker compose up -d --build
else
    WEB_MODE=2 docker compose up -d --build
fi

echo ""
echo "Сервис запущен!"
echo "  Просмотр: http://localhost:5000/"
echo "  Тест:     http://localhost:5000/test"
echo ""
echo "Для остановки: ./stop.sh"
