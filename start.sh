#!/bin/bash
# Запуск Перцепта
set -e

cd "$(dirname "$0")"

echo "Запуск Перцепта..."
docker compose up -d --build

echo "Сервер запущен на http://localhost:5000/"
echo "Просмотр: http://localhost:5000/"
echo "Тестовая страница: http://localhost:5000/test"
