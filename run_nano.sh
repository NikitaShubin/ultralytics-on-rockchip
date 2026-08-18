#!/usr/bin/env bash
# Полный бенчмарк yolo26n по всем задачам (det/seg/pose/cls/obb/sem).
# Запускается на HOST из корня проекта. Единственный лог — nano.log в корне.
# При интерактивном запуске сам уходит в фон (nohup), как требует правило проекта
# «долгие процессы — в фон». Принудительно в переднем плане:
#   RUN_NANO_FOREGROUND=1 ./run_nano.sh
set -euo pipefail

cd "$(dirname "$0")"

MODELS=(yolo26n yolo26n-seg yolo26n-pose yolo26n-cls yolo26n-obb yolo26n-sem)

LOG_FILE="nano.log"

if [[ "${RUN_NANO_CHILD:-0}" != "1" ]] && [[ "${RUN_NANO_FOREGROUND:-0}" != "1" ]]; then
    nohup env RUN_NANO_CHILD=1 "$0" "$@" >>"$LOG_FILE" 2>&1 &
    disown || true
    echo "[$(date '+%H:%M:%S')] Бенчмарк yolo26n запущен в фоне (PID $!)"
    echo "Лог: ${LOG_FILE}"
    echo "Статус: tail -f ${LOG_FILE}"
    exit 0
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Полный бенчмарк yolo26n (det/seg/pose/cls/obb/sem)"
log "Старт: $(date '+%Y-%m-%d %H:%M:%S')"
log "Лог: ${LOG_FILE}"
log "Модели: ${MODELS[*]}"

# Освобождаем NPU/камеру от persistent-контейнера percept
docker compose -f docker-compose.yml stop percept 2>/dev/null || true

: "${BENCH_WARMUP:=10}"
: "${BENCH_MEASURE:=30}"
: "${BENCH_REPEATS:=3}"
export BENCH_WARMUP BENCH_MEASURE BENCH_REPEATS
log "Warmup=${BENCH_WARMUP}с  Measure=${BENCH_MEASURE}с  Repeats=${BENCH_REPEATS}"

FAILED=0
for model in "${MODELS[@]}"; do
    log ""
    log "================================"
    log "Запуск: ${model}"
    log "================================"
    if python3 benchmark/benchmark.py --phase all --model "$model"; then
        log "[OK] ${model} — все фазы завершены"
    else
        log "[ПРОВАЛ] ${model} — см. лог выше"
        FAILED=$((FAILED + 1))
    fi
done

log ""
log "================================"
if [ "$FAILED" -eq 0 ]; then
    log "ВСЕ МОДЕЛИ УСПЕШНО ЗАВЕРШЕНЫ"
else
    log "ПРОВАЛОВ: ${FAILED} из ${#MODELS[@]}"
fi
log "Лог: ${LOG_FILE}"
log "Результаты: benchmark/results/results.json"
log "TSV-сводка: benchmark/results/summary.tsv"
log "Завершено: $(date '+%Y-%m-%d %H:%M:%S')"

cat benchmark/results/summary.tsv 2>/dev/null || true

# Честный exit-код для внешнего наблюдателя (CI/nohup-лог)
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0
