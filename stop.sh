#!/bin/bash
# Остановка Перцепта
set -e

cd "$(dirname "$0")"

echo "Остановка Перцепта..."
docker compose down

echo "Сервер остановлен."
