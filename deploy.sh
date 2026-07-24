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
    mkdir -p datasets/coco_val2017 datasets/dotav1 datasets/cityscapes datasets/imagenet model
}

MODEL="${1:-}"
FULL_TEST=false
QUANTIZE="none"
BACKEND=""

for arg in "$@"; do
    case "$arg" in
        --fp16)         QUANTIZE="none" ;;
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

# Метка квантизации для model/quantize.txt
quantize_label() {
    case "$1" in
        none)       echo "fp16" ;;
        int8)       echo "int8" ;;
        int8-calib) echo "int8-calib" ;;
        onnx)       echo "onnx" ;;
        *)          echo "$1" ;;
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
    CURRENT_MODEL=""
    if [ -f "model/model_name.txt" ]; then
        CURRENT_MODEL=$(cat "model/model_name.txt" | tr -d '[:space:]')
    fi
    CURRENT_QUANT=""
    if [ -f "model/quantize.txt" ]; then
        CURRENT_QUANT=$(cat "model/quantize.txt" | tr -d '[:space:]')
    fi

    NEED_CONVERT=true
    if [ "$CURRENT_MODEL" = "$MODEL" ] && [ "$CURRENT_QUANT" = "$(quantize_label "$QUANTIZE")" ]; then
        if [ "$QUANTIZE" = "onnx" ] && [ -f "model/model.onnx" ]; then
            echo "=== Модель $MODEL (ONNX) уже сконвертирована, пропуск ==="
            NEED_CONVERT=false
        elif [ "$QUANTIZE" != "onnx" ] && [ -f "model/model.rknn" ]; then
            echo "=== Модель $MODEL ($(quantize_label "$QUANTIZE")) уже сконвертирована, пропуск ==="
            NEED_CONVERT=false
        fi
    fi

    if [ "$NEED_CONVERT" = true ]; then
        if [ -f "model/model_name.txt" ] && [ "$CURRENT_MODEL" != "$MODEL" ]; then
            echo "=== Удаление предыдущей модели ($CURRENT_MODEL) ==="
            rm -f model/model.rknn model/model.onnx model/postprocess.py
        fi
        echo "=== Конвертация $MODEL (квантизация: $QUANTIZE) ==="
        cd converter
        docker compose run --rm converter convert "$MODEL" --quantize "$QUANTIZE"
        cd ..
        echo -n "$MODEL" > model/model_name.txt
        echo -n "$(quantize_label "$QUANTIZE")" > model/quantize.txt
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
    python benchmark.py --phase bench
    cd ..
fi

# Запуск сервиса
echo "=== Запуск сервиса ==="
ensure_mountpoints
ENV_ARGS=""
if [ -n "$BACKEND" ]; then
    ENV_ARGS="BACKEND=$BACKEND"
fi
if [ -n "$MODEL" ]; then
    MODEL="$MODEL" $ENV_ARGS WEB_MODE=2 docker compose up -d --build
else
    $ENV_ARGS WEB_MODE=2 docker compose up -d --build
fi

echo ""
echo "Сервис запущен!"
echo "  Просмотр: http://localhost:5000/"
echo "  Тест:     http://localhost:5000/test"
echo ""
echo "Для остановки: ./stop.sh"
