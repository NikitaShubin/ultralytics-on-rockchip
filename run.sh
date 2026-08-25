#!/usr/bin/env bash
# Полный бенчмарк ВСЕХ моделей из models.py FULL (111 моделей).
# Запуск из корня проекта. По умолчанию уходит в фон (nohup).
#
# В переднем плане:  RUN_FOREGROUND=1 ./run.sh
# Только PILOT:      RUN_PILOT=1 ./run.sh
# Без автоочистки:   RUN_NO_CLEANUP=1 ./run.sh
# Принудительно:     RUN_FORCE=1 ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

SCRIPT="$(pwd)/run.sh"
LOG_FILE="run.log"

# --- Уход в фон ---
if [[ "${RUN_CHILD:-0}" != "1" ]] && [[ "${RUN_FOREGROUND:-0}" != "1" ]]; then
    nohup env RUN_CHILD=1 "$SCRIPT" "$@" >>"$LOG_FILE" 2>&1 &
    disown || true
    echo "[$(date '+%H:%M:%S')] Бенчмарк запущен в фоне (PID $!)"
    echo "Лог: tail -f ${LOG_FILE}"
    exit 0
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Аргументы ---
EXTRA_FLAGS=()
[[ "${RUN_PILOT:-0}" == "1" ]] && EXTRA_FLAGS+=(--pilot)
[[ "${RUN_FORCE:-0}" == "1" ]] && EXTRA_FLAGS+=(--force)

: "${BENCH_WARMUP:=10}"
: "${BENCH_MEASURE:=30}"
: "${BENCH_REPEATS:=3}"
: "${BENCH_CLEANUP:=1}"
[[ "${RUN_NO_CLEANUP:-0}" == "1" ]] && BENCH_CLEANUP=0
export BENCH_WARMUP BENCH_MEASURE BENCH_REPEATS BENCH_CLEANUP

log "Полный бенчмарк (${EXTRA_FLAGS[*]:-FULL, 111 моделей})"
log "Старт: $(date '+%Y-%m-%d %H:%M:%S')"
log "Warmup=${BENCH_WARMUP}с  Measure=${BENCH_MEASURE}с  Repeats=${BENCH_REPEATS}"

# Освобождаем NPU от persistent-контейнера percept
docker compose -f docker-compose.yml stop percept 2>/dev/null || true

# Единый вызов benchmark.py — он сам итерирует все модели с resume-логикой
if python3 benchmark/benchmark.py --phase all "${EXTRA_FLAGS[@]}"; then
    log "БЕНЧМАРК ЗАВЕРШЁН УСПЕШНО"
else
    log "БЕНЧМАРК ЗАВЕРШЁН С ОШИБКАМИ (см. лог выше)"
fi

log "Результаты: benchmark/results/results.json"
log "Сводка: benchmark/results/summary.tsv"
log "Завершено: $(date '+%Y-%m-%d %H:%M:%S')"

cat benchmark/results/summary.tsv 2>/dev/null || true

# Честный exit-код
python3 -c "
import json, sys
r = json.load(open('benchmark/results/results.json'))
ok = sum(1 for e in r if e.get('status') == 'ПРОЙДЕНО')
total = len(r)
failed = total - ok
if failed:
    print(f'ПРОВАЛОВ: {failed} из {total}')
    sys.exit(1)
print(f'ВСЕ {total} МОДЕЛЕЙ ПРОЙДЕНЫ')
" || exit 1
exit 0
