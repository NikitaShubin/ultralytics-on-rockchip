#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="results"
LOG_FILE="${RESULTS_DIR}/benchmark_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$RESULTS_DIR"

: "${BENCH_WARMUP:=10}"
: "${BENCH_MEASURE:=20}"

export BENCH_WARMUP BENCH_MEASURE

echo "[$(date '+%H:%M:%S')] Запуск бенчмарка"
echo "[$(date '+%H:%M:%S')] Warmup=${BENCH_WARMUP}с  Measure=${BENCH_MEASURE}с"
echo "[$(date '+%H:%M:%S')] Лог: ${LOG_FILE}"

# Остановка percept для освобождения камеры / NPU
docker compose -f ../docker-compose.yml stop percept 2>/dev/null || true

echo ""

# Все аргументы пробрасываются в benchmark.py
# Примеры:
#   ./bench.sh                        # полный бенчмарк
#   ./bench.sh --model yolo26n        # одна модель
#   ./bench.sh --model yolo26n --quantize int8
python3 benchmark.py "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "[$(date '+%H:%M:%S')] Готово. Лог: ${LOG_FILE}"
echo "[$(date '+%H:%M:%S')] TSV-сводка: ${RESULTS_DIR}/summary.tsv"
cat "${RESULTS_DIR}/summary.tsv" 2>/dev/null || true
