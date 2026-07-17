#!/usr/bin/env bash
# Полноценный бенчмарк с разгоном CPU/NPU/GPU.
# Цикл: FPS до (stock) → разгон → FPS после → сравнение.
#
# Использование:
#   ./bench_overclock.sh                          # текущая модель, все бэкенды
#   ./bench_overclock.sh --model yolo11x-pose     # конкретная модель
#   ./bench_overclock.sh --model yolo11x-pose --measure 30  # дольше измерение
#   ./bench_overclock.sh --revert                 # откат к стоковым частотам
set -uo pipefail
cd "$(dirname "$0")"

# ─── Аргументы ─────────────────────────────────────────────────────────────

CONTAINER="percept"
TASK="detect"
WARMUP=10
MEASURE=20
QUANTIZE="both"
REVERT=0

OC_DIR="$(dirname "$0")/overclock"
RESULTS_DIR="$(dirname "$0")/results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
CSV_FILE="$RESULTS_DIR/bench_overclock_${TIMESTAMP}.csv"

# Доступные частоты (фильтруются по доступным на устройстве)
OC_CPU_STEPS=(1800000 2016000 2208000 2256000 2304000 2352000 2400000)
OC_NPU_STEPS=(600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000)
OC_GPU_STEPS=(400000000 500000000 600000000 700000000 800000000 900000000 1000000000 1050000000 1100000000)

FPS_DROP_THRESHOLD=5

# Модель: аргумент → env var → yolo11n
detect_model() {
    local name_file="${MODEL_DIR:-model}/model_name.txt"
    if [[ -f "$name_file" ]]; then
        cat "$name_file" | tr -d '[:space:]'
        return
    fi
    if [[ -n "${MODEL:-}" ]]; then
        echo "$MODEL"
        return
    fi
    echo "yolo11n"
}

MODEL=""

for arg in "$@"; do
    case "$arg" in
        --model=*)      MODEL="${arg#--model=}" ;;
        --task=*)       TASK="${arg#--task=}" ;;
        --warmup=*)     WARMUP="${arg#--warmup=}" ;;
        --measure=*)    MEASURE="${arg#--measure=}" ;;
        --quantize=*)   QUANTIZE="${arg#--quantize=}" ;;
        --revert)       REVERT=1 ;;
        --setup)        SETUP=1 ;;
        -h|--help)
            echo "Использование: $(basename "$0") [опции]"
            echo ""
            echo "  --model=NAME      Модель (по умолчанию: текущая развёрнутая)"
            echo "  --task=TASK       Задача (detect, pose, seg)"
            echo "  --warmup=SEC      Прогрев (по умолчанию: 10)"
            echo "  --measure=SEC     Измерение (по умолчанию: 20)"
            echo "  --quantize=MODE   none/int8/both (по умолчанию: both)"
            echo "  --revert          Откат к стоковым частотам"
            echo "  --setup           Установить DTB overlay для разгона + reboot"
            exit 0
            ;;
    esac
done

MODEL="${MODEL:-$(detect_model)}"

# ─── Утилиты ────────────────────────────────────────────────────────────────

die() { echo "[ОШИБКА] $*"; exit 1; }

log() {
    local ts
    ts=$(date '+%H:%M:%S')
    echo "[$ts] $*" >&2
}

require_docker() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || die "Контейнер $CONTAINER не запущен"
}

service_pause() {
    docker pause "$CONTAINER" 2>/dev/null || true
}

service_unpause() {
    docker unpause "$CONTAINER" 2>/dev/null || true
}

do_setup() {
    if cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null | grep -q "2352000"; then
        log "Overclock уже установлен (2352 МГц доступен)"
        return 0
    fi

    local env_file="/boot/orangepiEnv.txt"
    [[ -f "$env_file" ]] || die "Файл $env_file не найден"
    [[ -f "/boot/dtb/rockchip/overlay/rk3588-opi5pro-oc.dtbo" ]] || die "Overlay не найден"

    log "Установка DTB overlay для разгона..."
    echo 'orangepi' | sudo -S sed -i 's/^overlays=.*/overlays=rk3588-opi5pro-oc/' "$env_file" 2>/dev/null
    if ! grep -q "rk3588-opi5pro-oc" "$env_file" 2>/dev/null; then
        echo 'orangepi' | sudo -S bash -c "echo 'overlays=rk3588-opi5pro-oc' >> $env_file" 2>/dev/null
    fi
    log "Reboot через 3 секунды..."
    sleep 3
    echo 'orangepi' | sudo -S reboot 2>/dev/null
}

freq_to_mhz() {
    local f=$1
    if [[ $f -ge 10000000 ]]; then
        echo $((f / 1000000))
    elif [[ $f -ge 1000 ]]; then
        echo $((f / 1000))
    else
        echo "$f"
    fi
}

get_max_temp() {
    local max=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        local t
        t=$(cat "$zone" 2>/dev/null || echo 0)
        [[ $t -gt $max ]] && max=$t
    done
    echo $((max / 1000))
}

# ─── Управление частотами ───────────────────────────────────────────────────

set_cpu_freq() {
    local freq=$1
    echo 'orangepi' | sudo -S bash -c "
        for p in /sys/devices/system/cpu/cpufreq/policy{4,6}; do
            echo userspace > \$p/scaling_governor 2>/dev/null
            echo $freq > \$p/scaling_setspeed 2>/dev/null
        done
    " 2>/dev/null
}

set_npu_freq() {
    local freq=$1
    echo 'orangepi' | sudo -S bash -c "
        echo userspace > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null
        echo $freq > /sys/class/devfreq/fdab0000.npu/min_freq 2>/dev/null
        echo $freq > /sys/class/devfreq/fdab0000.npu/max_freq 2>/dev/null
    " 2>/dev/null
}

set_gpu_freq() {
    local freq=$1
    echo 'orangepi' | sudo -S bash -c "
        echo userspace > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null
        echo $freq > /sys/class/devfreq/fb000000.gpu/min_freq 2>/dev/null
        echo $freq > /sys/class/devfreq/fb000000.gpu/max_freq 2>/dev/null
    " 2>/dev/null
}

restore_governors() {
    echo 'orangepi' | sudo -S bash -c "
        for p in /sys/devices/system/cpu/cpufreq/policy{0,4,6}; do
            echo ondemand > \$p/scaling_governor 2>/dev/null
        done
        echo rknpu_ondemand > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null
        echo simple_ondemand > /sys/class/devfreq/fb000000.gpu/governor 2>/dev/null
    " 2>/dev/null
}

# ─── Замер FPS ──────────────────────────────────────────────────────────────

# Замер FPS инференса (файлы с диска, воспроизводимо) — через pause/unpause
FPS_CMD="python3 /app/benchmark/fps_bench.py $MODEL $TASK --temp-dir /app/benchmark/temp_models/$MODEL --data-dir /app/coco_val2017/images/val2017 --warmup $WARMUP --measure $MEASURE"

measure_fps_inference() {
    local quantize_flag=""
    case "$QUANTIZE" in
        none) quantize_flag="--quantize none" ;;
        int8) quantize_flag="--quantize int8" ;;
        both) quantize_flag="--quantize both" ;;
    esac

    service_pause
    local result
    result=$(docker exec "$CONTAINER" $FPS_CMD $quantize_flag 2>/dev/null | grep -E '^\{' | tail -1)
    service_unpause

    if [[ -z "$result" ]]; then
        echo '{"cpu":{"fps":0},"npu_none":{"fps":0},"npu_int8":{"fps":0}}'
        return
    fi
    echo "$result"
}

# Замер FPS камеры (реальная работа сервиса) — через curl
measure_fps_camera() {
    local result
    result=$(curl -s --max-time 5 http://localhost:5000/objects 2>/dev/null)
    if [[ -z "$result" ]]; then
        echo "0"
        return
    fi
    echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('fps',0))" 2>/dev/null || echo "0"
}

parse_fps() {
    echo "$1" | python3 -c "
import sys,json
d=json.load(sys.stdin)
key='$2'
b=d.get(key,{})
print(f'{b.get(\"fps\",0):.2f}')" 2>/dev/null || echo "0"
}

check_stability() {
    local baseline=$1
    local current=$2
    python3 -c "
b=$baseline; c=$current
if b <= 0: exit(1)
drop = (b - c) / b * 100
exit(0 if drop < $FPS_DROP_THRESHOLD else 1)
" 2>/dev/null
}

# ─── Фильтрация доступных частот ────────────────────────────────────────────

filter_freqs() {
    local component=$1
    local available=""
    case "$component" in
        cpu)
            available=$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_available_frequencies 2>/dev/null)
            ;;
        npu)
            available=$(cat /sys/class/devfreq/fdab0000.npu/available_frequencies 2>/dev/null)
            ;;
        gpu)
            available=$(cat /sys/class/devfreq/fb000000.gpu/available_frequencies 2>/dev/null)
            ;;
    esac

    local var_name="OC_${component^^}_STEPS"
    eval "local steps=(\"\${${var_name}[@]}\")"

    local result=()
    for freq in "${steps[@]}"; do
        if echo "$available" | grep -qw "$freq"; then
            result+=("$freq")
        fi
    done
    echo "${result[@]}"
}

# ─── Запись CSV ─────────────────────────────────────────────────────────────

write_csv_header() {
    mkdir -p "$RESULTS_DIR"
    echo "phase,component,freq_mhz,camera_fps,fps_cpu,fps_npu_fp32,fps_npu_int8,temp_c,timestamp" > "$CSV_FILE"
}

write_csv_row() {
    local phase=$1 component=$2 freq=$3 cam_fps=$4 fps_cpu=$5 fps_fp32=$6 fps_int8=$7 temp=$8
    local mhz
    mhz=$(freq_to_mhz "$freq")
    echo "$phase,$component,$mhz,$cam_fps,$fps_cpu,$fps_fp32,$fps_int8,$temp,$(date '+%H:%M:%S')" >> "$CSV_FILE"
}

# ─── Конвертация ────────────────────────────────────────────────────────────

do_convert() {
    local temp_dir="benchmark/temp_models/$MODEL"
    local rknn_none="$temp_dir/${MODEL}_none.rknn"
    local rknn_int8="$temp_dir/${MODEL}_int8.rknn"
    local onnx="$temp_dir/${MODEL}.onnx"

    if [[ -f "$rknn_none" ]] || [[ -f "$rknn_int8" ]] || [[ -f "$onnx" ]]; then
        log "Модель $MODEL уже сконвертирована"
        return 0
    fi

    log "Конвертация $MODEL..."
    cd benchmark
    python3 benchmark.py --phase convert --model "$MODEL" --quantize "$QUANTIZE" 2>&1 | while IFS= read -r line; do
        log "[convert] $line"
    done
    cd ..
    log "Конвертация завершена"
}

# ─── Замер всех FPS (инференс, воспроизводимо) ──────────────────────────────

benchmark_inference() {
    local label=$1
    log "Инференс FPS ($label)..."

    local json
    json=$(measure_fps_inference)

    local fps_cpu fps_fp32 fps_int8
    fps_cpu=$(parse_fps "$json" "cpu")
    fps_fp32=$(parse_fps "$json" "npu_none")
    fps_int8=$(parse_fps "$json" "npu_int8")
    local temp
    temp=$(get_max_temp)

    log "  CPU ONNX:   $fps_cpu FPS"
    log "  NPU FP32:   $fps_fp32 FPS"
    log "  NPU INT8:   $fps_int8 FPS"
    log "  Темп:       ${temp}°C"

    echo "$fps_cpu $fps_fp32 $fps_int8 $temp"
}

# ─── Тест одной частоты ─────────────────────────────────────────────────────

test_freq() {
    local component=$1
    local freq=$2
    local baseline_cpu=$3
    local baseline_fp32=$4
    local baseline_int8=$5

    case "$component" in
        cpu) set_cpu_freq "$freq" ;;
        npu) set_npu_freq "$freq" ;;
        gpu) set_gpu_freq "$freq" ;;
    esac
    sleep 2

    local json
    json=$(measure_fps_inference)

    local fps_cpu fps_fp32 fps_int8 temp
    fps_cpu=$(parse_fps "$json" "cpu")
    fps_fp32=$(parse_fps "$json" "npu_none")
    fps_int8=$(parse_fps "$json" "npu_int8")
    temp=$(get_max_temp)

    local mhz
    mhz=$(freq_to_mhz "$freq")

    if check_stability "$baseline_fp32" "$fps_fp32"; then
        local gain
        gain=$(python3 -c "b=$baseline_fp32; c=$fps_fp32; print(f'{(c-b)/b*100:+.1f}%' if b>0 else '?')" 2>/dev/null)
        echo "PASS $fps_cpu $fps_fp32 $fps_int8"
        printf "  %5s МГц: FP32=%s FPS (%s) CPU=%s INT8=%s  %3d°C  ✓\n" \
            "$mhz" "$fps_fp32" "$gain" "$fps_cpu" "$fps_int8" "$temp" >&2
        write_csv_row "test" "$component" "$freq" "-" "$fps_cpu" "$fps_fp32" "$fps_int8" "$temp"
    else
        local drop
        drop=$(python3 -c "b=$baseline_fp32; c=$fps_fp32; print(f'{(c-b)/b*100:+.1f}%' if b>0 else '?')" 2>/dev/null)
        echo "FAIL $fps_cpu $fps_fp32 $fps_int8"
        printf "  %5s МГц: FP32=%s FPS (%s) CPU=%s INT8=%s  %3d°C  ✗ НЕСТАБИЛЬНО\n" \
            "$mhz" "$fps_fp32" "$drop" "$fps_cpu" "$fps_int8" "$temp" >&2
        write_csv_row "test_fail" "$component" "$freq" "-" "$fps_cpu" "$fps_fp32" "$fps_int8" "$temp"
    fi
}

# ─── ОТКАТ ──────────────────────────────────────────────────────────────────

cmd_revert() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  ОТКАТ К СТОКОВЫМ ЧАСТОТАМ"
    echo "══════════════════════════════════════════════════════════════"
    restore_governors
    log "Governors восстановлены (ondemand)"
    echo "  CPU6: $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq) kHz"
    echo "  NPU:  $(cat /sys/class/devfreq/fdab0000.npu/cur_freq) Hz"
    echo "  GPU:  $(cat /sys/class/devfreq/fb000000.gpu/cur_freq) Hz"
    echo "══════════════════════════════════════════════════════════════"
}

# ─── ОСНОВНАЯ ПРОГРАММА ─────────────────────────────────────────────────────

# Откат по флагу
if [[ $REVERT -eq 1 ]]; then
    cmd_revert
    exit 0
fi

echo "══════════════════════════════════════════════════════════════"
echo "  БЕНЧМАРК С РАЗГОНОМ: $MODEL ($TASK)"
echo "  Warmup: ${WARMUP}с | Измерение: ${MEASURE}с | Квантизация: $QUANTIZE"
echo "══════════════════════════════════════════════════════════════"
echo ""

require_docker

# Setup
if [[ ${SETUP:-0} -eq 1 ]]; then
    do_setup
    exit 0
fi

# Определяем доступные частоты
read -ra AVAIL_CPU <<< "$(filter_freqs cpu)"
read -ra AVAIL_NPU <<< "$(filter_freqs npu)"
read -ra AVAIL_GPU <<< "$(filter_freqs gpu)"

log "Доступные частоты:"
log "  CPU: ${AVAIL_CPU[*]:-нет}"
log "  NPU: ${AVAIL_NPU[*]:-нет}"
log "  GPU: ${AVAIL_GPU[*]:-нет}"
echo ""

# Конвертация
do_convert
echo ""

# CSV
write_csv_header

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 1: FPS ДО (stock)
# ═══════════════════════════════════════════════════════════════════
echo "── FPS ДО (stock ondemand) ──"
restore_governors
sleep 2

# Камерный FPS (реальная работа сервиса)
CAMERA_BEFORE=$(measure_fps_camera)
log "  Камерный FPS: $CAMERA_BEFORE"

# Инференс FPS (воспроизводимо, pause/unpause)
BEFORE=$(benchmark_inference "STOCK")
read -r BEFORE_CPU BEFORE_FP32 BEFORE_INT8 BEFORE_TEMP <<< "$BEFORE"
write_csv_row "before" "stock" "0" "$CAMERA_BEFORE" "$BEFORE_CPU" "$BEFORE_FP32" "$BEFORE_INT8" "$BEFORE_TEMP"
echo ""

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 2: Разгон CPU
# ═══════════════════════════════════════════════════════════════════
BEST_CPU_FREQ=0
BEST_CPU_FP32=0

if [[ ${#AVAIL_CPU[@]} -gt 0 ]]; then
    echo "── Разгон CPU ──"
    for freq in "${AVAIL_CPU[@]}"; do
        result=$(test_freq "cpu" "$freq" "$BEFORE_CPU" "$BEFORE_FP32" "$BEFORE_INT8")
        status=$(echo "$result" | head -1 | awk '{print $1}')
        if [[ "$status" == "PASS" ]]; then
            BEST_CPU_FREQ=$freq
            BEST_CPU_FP32=$(echo "$result" | head -1 | awk '{print $3}')
        else
            break
        fi
    done
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 3: Разгон NPU
# ═══════════════════════════════════════════════════════════════════
BEST_NPU_FREQ=0
BEST_NPU_FP32=0

if [[ ${#AVAIL_NPU[@]} -gt 0 ]]; then
    echo "── Разгон NPU ──"
    # Используем лучший CPU для теста NPU
    if [[ $BEST_CPU_FREQ -gt 0 ]]; then
        set_cpu_freq "$BEST_CPU_FREQ"
        sleep 1
    fi

    for freq in "${AVAIL_NPU[@]}"; do
        result=$(test_freq "npu" "$freq" "$BEFORE_CPU" "$BEFORE_FP32" "$BEFORE_INT8")
        status=$(echo "$result" | head -1 | awk '{print $1}')
        if [[ "$status" == "PASS" ]]; then
            BEST_NPU_FREQ=$freq
            BEST_NPU_FP32=$(echo "$result" | head -1 | awk '{print $3}')
        else
            break
        fi
    done
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 4: Разгон GPU (мониторинг)
# ═══════════════════════════════════════════════════════════════════
BEST_GPU_FREQ=0

if [[ ${#AVAIL_GPU[@]} -gt 0 ]]; then
    echo "── Разгон GPU (мониторинг) ──"
    for freq in "${AVAIL_GPU[@]}"; do
        set_gpu_freq "$freq"
        sleep 2
        temp=$(get_max_temp)
        mhz=$(freq_to_mhz "$freq")
        printf "  %5s МГц:  %3d°C  (GPU не влияет на FPS YOLO)\n" "$mhz" "$temp"
        write_csv_row "test_gpu" "gpu" "$freq" "-" "-" "-" "-" "$temp"
        BEST_GPU_FREQ=$freq
    done
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 5: FPS ПОСЛЕ (разгон) — разгон оставлен включённым
# ═══════════════════════════════════════════════════════════════════
echo "── Установка разогнанных частот ──"
if [[ $BEST_CPU_FREQ -gt 0 ]]; then
    set_cpu_freq "$BEST_CPU_FREQ"
    log "  CPU: $(freq_to_mhz "$BEST_CPU_FREQ") МГц"
fi
if [[ $BEST_NPU_FREQ -gt 0 ]]; then
    set_npu_freq "$BEST_NPU_FREQ"
    log "  NPU: $(freq_to_mhz "$BEST_NPU_FREQ") МГц"
fi
if [[ $BEST_GPU_FREQ -gt 0 ]]; then
    set_gpu_freq "$BEST_GPU_FREQ"
    log "  GPU: $(freq_to_mhz "$BEST_GPU_FREQ") МГц"
fi
sleep 3

echo ""
echo "── FPS ПОСЛЕ (разгон) ──"

# Камерный FPS (реальная работа сервиса с разогнанным железом)
CAMERA_AFTER=$(measure_fps_camera)
log "  Камерный FPS: $CAMERA_AFTER"

# Инференс FPS (воспроизводимо, pause/unpause)
AFTER=$(benchmark_inference "OVERCLOCK")
read -r AFTER_CPU AFTER_FP32 AFTER_INT8 AFTER_TEMP <<< "$AFTER"
write_csv_row "after" "oc" "0" "$CAMERA_AFTER" "$AFTER_CPU" "$AFTER_FP32" "$AFTER_INT8" "$AFTER_TEMP"
echo ""

# ═══════════════════════════════════════════════════════════════════
# ЭТАП 6: Итоговая таблица
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  РЕЗУЛЬТАТЫ: $MODEL ($TASK)"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  Разгон:"
[[ $BEST_CPU_FREQ -gt 0 ]] && echo "    CPU: $(freq_to_mhz "$BEST_CPU_FREQ") МГц" || echo "    CPU: сток"
[[ $BEST_NPU_FREQ -gt 0 ]] && echo "    NPU: $(freq_to_mhz "$BEST_NPU_FREQ") МГц" || echo "    NPU: сток"
[[ $BEST_GPU_FREQ -gt 0 ]] && echo "    GPU: $(freq_to_mhz "$BEST_GPU_FREQ") МГц" || echo "    GPU: сток"
echo ""

# Камерный FPS
gain_cam=$(python3 -c "
b=$CAMERA_BEFORE; a=$CAMERA_AFTER
if b > 0: print(f'{(a-b)/b*100:+.1f}%')
else: print('—')
" 2>/dev/null)
echo "  ┌─────────────┬──────────┬──────────┬──────────┐"
printf "  │ Камера      │ %7s  │ %7s  │ %7s  │\n" "$CAMERA_BEFORE" "$CAMERA_AFTER" "$gain_cam"

# Инференс FPS
echo "  ├─────────────┼──────────┼──────────┼──────────┤"

gain_cpu=$(python3 -c "
b=$BEFORE_CPU; a=$AFTER_CPU
if b > 0: print(f'{(a-b)/b*100:+.1f}%')
else: print('—')
" 2>/dev/null)
printf "  │ CPU ONNX    │ %7s  │ %7s  │ %7s  │\n" "$BEFORE_CPU" "$AFTER_CPU" "$gain_cpu"

gain_fp32=$(python3 -c "
b=$BEFORE_FP32; a=$AFTER_FP32
if b > 0: print(f'{(a-b)/b*100:+.1f}%')
else: print('—')
" 2>/dev/null)
printf "  │ NPU FP32    │ %7s  │ %7s  │ %7s  │\n" "$BEFORE_FP32" "$AFTER_FP32" "$gain_fp32"

gain_int8=$(python3 -c "
b=$BEFORE_INT8; a=$AFTER_INT8
if b > 0: print(f'{(a-b)/b*100:+.1f}%')
else: print('—')
" 2>/dev/null)
printf "  │ NPU INT8    │ %7s  │ %7s  │ %7s  │\n" "$BEFORE_INT8" "$AFTER_INT8" "$gain_int8"

echo "  └─────────────┴──────────┴──────────┴──────────┘"
echo ""
echo "  Температура: ${BEFORE_TEMP}°C → ${AFTER_TEMP}°C"
echo "  CSV: $CSV_FILE"
echo ""
echo "  Разгон ОСТАВЛЁН включённым."
echo "  Для отката: ./bench_overclock.sh --revert"
echo "══════════════════════════════════════════════════════════════"
