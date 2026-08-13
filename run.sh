#!/usr/bin/env bash
set -euo pipefail

# Полный бенчмарк ВСЕХ поддерживаемых моделей (111 шт, список FULL в benchmark/models.py).
#
# Устойчив к прерываниям: benchmark.py сохраняет results.json после каждого шага
# (конвертация/валидация/fps/eval по вариантам) и при повторном запуске выполняет
# только незавершённые задачи. Прерванную задачу (напр. eval fp16) можно просто
# доработать повторным запуском ./run.sh.
#
# Запускайте в фоне/в tmux, т.к. полный прогон занимает недели:
#   tmux new -s bench './run.sh'
# Прогресс: benchmark/results/results.json, лог: benchmark/results/benchmark_*.log

cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
    set -- --phase all --quantize all
fi

exec benchmark/bench.sh "$@"
