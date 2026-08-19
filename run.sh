#!/usr/bin/env bash
# Полный бенчмарк PILOT-моделей: convert → validate → bench → eval
# (yolov3u, yolov5nu, yolov8n, yolov9t, yolov10n, yolo11n, yolo12n, yolo26n)
#
# Запуск из корня проекта. По умолчанию уходит в фон (nohup).
# В переднем плане: RUN_FOREGROUND=1 ./run.sh
# Пропустить convert (--force пересчитать): RUN_FORCE=1 ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

# --- PILOT-модели (benchmark/models.py) ---
MODELS=(
    yolov3u
    yolov5nu
    yolov8n yolov8n-seg yolov8n-pose yolov8n-obb yolov8n-cls
    yolov9t
    yolov10n
    yolo11n yolo11n-seg yolo11n-pose yolo11n-obb yolo11n-cls
    yolo12n
    yolo26n yolo26n-seg yolo26n-pose yolo26n-obb yolo26n-cls yolo26n-sem
)

LOG_FILE="run.log"

# --- Уход в фон ---
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
if [[ "${RUN_CHILD:-0}" != "1" ]] && [[ "${RUN_FOREGROUND:-0}" != "1" ]]; then
    nohup env RUN_CHILD=1 "$SCRIPT" "$@" >>"$LOG_FILE" 2>&1 &
    disown || true
    echo "[$(date '+%H:%M:%S')] Бенчмарк запущен в фоне (PID $!)"
    echo "Лог: tail -f ${LOG_FILE}"
    exit 0
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

FORCE_FLAG=""
if [[ "${RUN_FORCE:-0}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

log "Полный бенчмарк PILOT (${#MODELS[@]} моделей)"
log "Старт: $(date '+%Y-%m-%d %H:%M:%S')"
log "Модели: ${MODELS[*]}"
[[ -n "$FORCE_FLAG" ]] && log "Режим: --force (полный пересчёт)"

# Освобождаем NPU от persistent-контейнера percept
docker compose -f docker-compose.yml stop percept 2>/dev/null || true

: "${BENCH_WARMUP:=10}"
: "${BENCH_MEASURE:=30}"
: "${BENCH_REPEATS:=3}"
export BENCH_WARMUP BENCH_MEASURE BENCH_REPEATS

FAILED=0
for model in "${MODELS[@]}"; do
    log ""
    log "================================"
    log "Запуск: ${model}"
    log "================================"
    if python3 benchmark/benchmark.py --phase all --model "$model" $FORCE_FLAG; then
        log "[OK] ${model}"
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
log "Результаты: benchmark/results/results.json"
log "Сводка: benchmark/results/summary.tsv"
log "Завершено: $(date '+%Y-%m-%d %H:%M:%S')"

cat benchmark/results/summary.tsv 2>/dev/null || true

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0
