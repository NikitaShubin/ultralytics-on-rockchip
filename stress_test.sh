#!/usr/bin/env bash
# Стресс-тест Orange Pi (RK3588S): CPU + NPU + RAM + температура.
# Ctrl+C корректно останавливает всю нагрузку и завершает скрипт.
#
# Использование:
#   ./stress_test.sh                          # всё, до Ctrl+C
#   ./stress_test.sh -d 600                   # всё, 10 минут
#   ./stress_test.sh --npu-only               # только NPU
#   ./stress_test.sh --cpu-only               # только CPU+RAM
#   ./stress_test.sh -c 0,1 -s 4              # 2 ядра × 4 потока NPU
#   ./stress_test.sh -m yolo11n_none.rknn     # другая модель
#   ./stress_test.sh -l results.csv           # лог в CSV

set -uo pipefail

# ─── Аргументы ─────────────────────────────────────────────────────────────

DURATION=0
NPU_ONLY=0
CPU_ONLY=0
NO_RAM=0
NPU_CORES="0,1,2"
NPU_STREAMS=8
MODEL="model/yolo26x-seg.rknn"
INTERVAL=2
LOG_FILE=""
RAM_MB=1024

usage() {
    cat <<EOF
Использование: $(basename "$0") [опции]

Опции:
  -d, --duration SEC     Длительность в секундах (0 = до Ctrl+C, по умолчанию)
  -c, --npu-cores LIST   NPU ядра через запятую (по умолчанию: 0,1,2)
  -s, --npu-streams N    Потоков инференса на ядро (по умолчанию: 8)
  -m, --model FILE       RKNN модель (по умолчанию: model/yolo26x-seg.rknn)
  -i, --interval SEC     Интервал мониторинга (по умолчанию: 2)
  -l, --log FILE         CSV лог результатов
  --npu-only             Только NPU
  --cpu-only             Только CPU + RAM
  --no-ram               Без нагрузки на RAM
  -h, --help             Помощь
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--duration)    DURATION="$2"; shift 2 ;;
        -c|--npu-cores)   NPU_CORES="$2"; shift 2 ;;
        -s|--npu-streams) NPU_STREAMS="$2"; shift 2 ;;
        -m|--model)       MODEL="$2"; shift 2 ;;
        -i|--interval)    INTERVAL="$2"; shift 2 ;;
        -l|--log)         LOG_FILE="$2"; shift 2 ;;
        --npu-only)       NPU_ONLY=1; shift ;;
        --cpu-only)       CPU_ONLY=1; shift ;;
        --no-ram)         NO_RAM=1; shift ;;
        -h|--help)        usage ;;
        *) echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/$MODEL"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ОШИБКА] Модель не найдена: $MODEL_PATH"
    exit 1
fi

# ─── Термозоны ──────────────────────────────────────────────────────────────

declare -a THERMAL_NAMES=()
declare -a THERMAL_FILES=()

init_thermal() {
    local i=0
    for zone_dir in /sys/class/thermal/thermal_zone*; do
        [[ -f "$zone_dir/type" && -f "$zone_dir/temp" ]] || continue
        THERMAL_NAMES[$i]=$(cat "$zone_dir/type")
        THERMAL_FILES[$i]="$zone_dir/temp"
        ((i++))
    done
}

get_temps() {
    local result=""
    for i in "${!THERMAL_NAMES[@]}"; do
        local raw
        raw=$(cat "${THERMAL_FILES[$i]}" 2>/dev/null || echo "0")
        local temp=$((raw / 1000))
        result+="${THERMAL_NAMES[$i]}=${temp}° "
    done
    echo "$result"
}

# ─── CPU load ───────────────────────────────────────────────────────────────

CPU_STATE_FILE="/tmp/.stress_cpu_state"

read_cpu_load() {
    # Читает /proc/stat, считает дельту между вызовами.
    # Состояние хранится в файле (обход subshell-проблемы bash $(...))
    local vals
    read -ra vals < <(head -1 /proc/stat)
    local user=${vals[1]:-0}
    local nice=${vals[2]:-0}
    local system=${vals[3]:-0}
    local idle=${vals[4]:-0}
    local iowait=${vals[5]:-0}
    local irq=${vals[6]:-0}
    local softirq=${vals[7]:-0}
    local steal=${vals[8]:-0}

    local cur_idle=$((idle + iowait))
    local cur_total=$((user + nice + system + idle + iowait + irq + softirq + steal))

    if [[ ! -f "$CPU_STATE_FILE" ]]; then
        echo "${cur_idle} ${cur_total}" > "$CPU_STATE_FILE"
        echo 0
        return
    fi

    local prev_idle prev_total
    read -r prev_idle prev_total < "$CPU_STATE_FILE"
    echo "${cur_idle} ${cur_total}" > "$CPU_STATE_FILE"

    local d_idle=$((cur_idle - prev_idle))
    local d_total=$((cur_total - prev_total))

    if [[ $d_total -eq 0 ]]; then
        echo 0
        return
    fi

    echo $(( (d_total - d_idle) * 100 / d_total ))
}

# ─── NPU load ───────────────────────────────────────────────────────────────

read_npu_load_cores() {
    local raw
    raw=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo "")
    if [[ -z "$raw" ]]; then
        echo ""
        return
    fi
    echo "$raw" | grep -oP 'Core\d+:\s*\d+%' | sed 's/Core/C/g;s/: /=/' | tr '\n' ' '
}

npu_load_avg() {
    local raw
    raw=$(read_npu_load_cores)
    if [[ -z "$raw" ]]; then
        echo 0
        return
    fi
    local sum=0 cnt=0
    for token in $raw; do
        local v="${token#*=}"
        v="${v%%%}"
        if [[ "$v" =~ ^[0-9]+$ ]]; then
            sum=$((sum + v))
            cnt=$((cnt + 1))
        fi
    done
    if [[ $cnt -gt 0 ]]; then
        echo $((sum / cnt))
    else
        echo 0
    fi
}

# ─── RAM ────────────────────────────────────────────────────────────────────

read_ram() {
    awk '/^MemTotal:/{total=$2} /^MemAvailable:/{avail=$2} END{printf "%d %d", total/1024, (total-avail)/1024}' /proc/meminfo
}

# ─── Управление процессами ─────────────────────────────────────────────────

STRESS_PIDS=()
NPU_PID=""
NPU_SCRIPT=""
STOP=0
CLEANUP_DONE=0

cleanup() {
    [[ $CLEANUP_DONE -eq 1 ]] && return
    CLEANUP_DONE=1

    echo ""
    echo "[ОСТАНОВКА] Завершаю все процессы..."

    # Убить stress
    for pid in "${STRESS_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    pkill -f 'stress --cpu' 2>/dev/null || true
    pkill -f 'stress --vm' 2>/dev/null || true

    # Остановить NPU Python процесс (SIGTERM → 3с → SIGKILL)
    if [[ -n "$NPU_PID" ]] && kill -0 "$NPU_PID" 2>/dev/null; then
        kill -TERM "$NPU_PID" 2>/dev/null || true
        local waited=0
        while kill -0 "$NPU_PID" 2>/dev/null && [[ $waited -lt 3 ]]; do
            sleep 1
            waited=$((waited + 1))
        done
        kill -9 "$NPU_PID" 2>/dev/null || true
    fi

    wait 2>/dev/null || true
    [[ -n "$NPU_SCRIPT" ]] && rm -f "$NPU_SCRIPT"
    rm -f "$CPU_STATE_FILE"

    echo "[ГОТОВО] Нагрузка снята"
}

on_signal() {
    STOP=1
    # Передать сигнал NPU Python процессу, чтобы он остановил потоки
    [[ -n "$NPU_PID" ]] && kill -INT "$NPU_PID" 2>/dev/null || true
}

trap on_signal INT TERM
trap 'cleanup; exit 0' EXIT

# ─── CPU/RAM stress ────────────────────────────────────────────────────────

start_cpu_stress() {
    local n_workers
    n_workers=$(nproc)
    nice -n 19 stress --cpu "$n_workers" --timeout 999d &>/dev/null &
    STRESS_PIDS+=($!)
    echo "[CPU] stress: $n_workers воркеров (PID $!)"
}

start_ram_stress() {
    local n_workers=$(( $(nproc) / 2 ))
    [[ $n_workers -lt 1 ]] && n_workers=1
    nice -n 19 stress --vm "$n_workers" --vm-bytes "${RAM_MB}M" --vm-keep --timeout 999d &>/dev/null &
    STRESS_PIDS+=($!)
    echo "[RAM] stress: ${n_workers} × ${RAM_MB} МБ (PID $!)"
}

# ─── NPU stress (Python только для rknnlite) ───────────────────────────────

start_npu_stress() {
    NPU_SCRIPT=$(mktemp /tmp/npu_stress_XXXXXX.py)
    cat > "$NPU_SCRIPT" <<'PYEOF'
import sys, signal, threading, time, numpy as np

CORES_MAP = {
    0: 'NPU_CORE_0', 1: 'NPU_CORE_1', 2: 'NPU_CORE_2',
    3: 'NPU_CORE_0_1', 4: 'NPU_CORE_0_1_2',
}

model_path = sys.argv[1]
core_ids = [int(x) for x in sys.argv[2].split(',')]
streams = int(sys.argv[3])
stop = threading.Event()

def run(core_id, idx):
    from rknnlite.api import RKNNLite
    core_name = CORES_MAP.get(core_id, f'NPU_CORE_{core_id}')
    mask = getattr(RKNNLite, core_name, None)
    if mask is None:
        return
    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        print(f'[NPU] core{core_id}/s{idx}: load failed', file=sys.stderr); return
    if rknn.init_runtime(core_mask=mask) != 0:
        print(f'[NPU] core{core_id}/s{idx}: init failed', file=sys.stderr); return
    dummy = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)
    print(f'[NPU] core{core_id}/s{idx}: started', flush=True)
    while not stop.is_set():
        try:
            rknn.inference(inputs=[dummy])
        except Exception as e:
            print(f'[NPU] core{core_id}/s{idx}: {e}', file=sys.stderr)
            time.sleep(1)
    rknn.release()
    print(f'[NPU] core{core_id}/s{idx}: stopped', flush=True)

for cid in core_ids:
    for s in range(streams):
        t = threading.Thread(target=run, args=(cid, s), daemon=True)
        t.start()

signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
while True:
    time.sleep(60)
PYEOF
    python3 "$NPU_SCRIPT" "$MODEL_PATH" "$NPU_CORES" "$NPU_STREAMS" &
    NPU_PID=$!
    echo "[NPU] ядра=$NPU_CORES × $NPU_STREAMS потоков модель=$(basename "$MODEL_PATH") (PID $NPU_PID)"
}

# ─── Мониторинг ─────────────────────────────────────────────────────────────

monitor() {
    local elapsed=0

    [[ -n "$LOG_FILE" ]] && echo "time,elapsed,cpu%,npu%,npu_cores,ram_used_mb,temps" > "$LOG_FILE"

    printf "%-9s %5s %5s  %-24s %7s  %s\n" \
        "ВРЕМЯ" "CPU%" "NPU%" "NPU_LOAD" "RAM_МБ" "ТЕМПЕРАТУРЫ"
    printf "%-9s %5s %5s  %-24s %7s  %s\n" \
        "" "/proc" "/sys" "(C0=XX% C1=XX% C2=XX%)" "(used)" "(big/little/npu/soc)"
    echo "──────────────────────────────────────────────────────────────────────────────────────"

    while true; do
        sleep "$INTERVAL"
        [[ $STOP -eq 1 ]] && break

        elapsed=$((elapsed + INTERVAL))

        # Проверка таймаута
        if [[ $DURATION -gt 0 && $elapsed -ge $DURATION ]]; then
            break
        fi

        local ts
        ts=$(date +%H:%M:%S)
        local cpu_pct; cpu_pct=$(read_cpu_load)
        local npu_pct; npu_pct=$(npu_load_avg)
        local npu_cores; npu_cores=$(read_npu_load_cores)
        read -r _ ram_used <<< "$(read_ram)"
        local temps; temps=$(get_temps)

        printf "%-9s %4s%% %4s%%  %-24s %6s  %s\n" \
            "$ts" "$cpu_pct" "$npu_pct" "$npu_cores" "$ram_used" "$temps"

        [[ -n "$LOG_FILE" ]] && \
            echo "${ts},${elapsed},${cpu_pct},${npu_pct},\"${npu_cores}\",${ram_used},\"${temps}\"" >> "$LOG_FILE"
    done
}

# ─── Результаты ─────────────────────────────────────────────────────────────

print_results() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  РЕЗУЛЬТАТЫ"
    echo "════════════════════════════════════════════════════════"
    echo ""
    for i in "${!THERMAL_NAMES[@]}"; do
        local raw
        raw=$(cat "${THERMAL_FILES[$i]}" 2>/dev/null || echo "0")
        printf "  %-25s %3d°C\n" "${THERMAL_NAMES[$i]}" $((raw / 1000))
    done
    local npu_cores; npu_cores=$(read_npu_load_cores)
    [[ -n "$npu_cores" ]] && echo -e "\nNPU load: $npu_cores"
    local rt ru; read -r rt ru <<< "$(read_ram)"
    echo "RAM: ${ru}/${rt} МБ"
    [[ -n "$LOG_FILE" ]] && echo "Лог: $LOG_FILE"
}

# ─── main ───────────────────────────────────────────────────────────────────

init_thermal

echo "════════════════════════════════════════════════════════"
if [[ $DURATION -gt 0 ]]; then
    echo "  СТРЕСС-ТЕСТ: ${DURATION}с"
else
    echo "  СТРЕСС-ТЕСТ: до Ctrl+C"
fi
[[ $NPU_ONLY -eq 0 ]] && echo "  CPU: $(nproc) ядер (stress, nice=19)"
[[ $CPU_ONLY -eq 0 ]] && echo "  NPU: $NPU_CORES × $NPU_STREAMS потоков"
echo "  Модель: $MODEL"
echo "════════════════════════════════════════════════════════"
echo "════════════════════════════════════════════════════════"
echo ""

if [[ $NPU_ONLY -eq 0 ]]; then
    start_cpu_stress
    [[ $NO_RAM -eq 0 ]] && start_ram_stress
fi

if [[ $CPU_ONLY -eq 0 ]]; then
    start_npu_stress
    sleep 5
fi

monitor

print_results
