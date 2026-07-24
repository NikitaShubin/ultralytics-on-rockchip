#!/bin/bash
# Запуск Перцепта
set -e

cd "$(dirname "$0")"

# Docker создаёт mount-точки от root, если их нет.
# Создаём заранее, чтобы монтировались user-owned директории.
mkdir -p datasets/coco_val2017 datasets/dotav1 datasets/cityscapes model

echo "Запуск Перцепта..."
docker compose up -d --build

echo "Сервер запущен на http://localhost:5000/"
echo "Просмотр: http://localhost:5000/"
echo "Тестовая страница: http://localhost:5000/test"
